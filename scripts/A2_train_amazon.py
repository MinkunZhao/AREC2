"""Stage A2 (Amazon): fine-tune the AREC²/OneRec model with Strategy 3.

Implements the transfer-learning training of paper §6.3 using Strategy 3
(Text-Augmented Itemic Tokens). Supports two regimes (§6.3.2):
  - --regime single : domain-specific (one model per category)
  - --regime joint  : multi-domain joint (one model over all given categories)

The model is fine-tuned with LoRA, mirroring scripts/03_train_sft.py (same
LoRA targets, collator, loss-mask-on-answer semantics). The base model is
configurable (default: the merged AREC² checkpoint), per your instruction.

Itemic vocabulary note: Amazon items reuse the existing OneRec SID vocabulary
(<s_a_0..8191>, <s_b_*>, <s_c_*>), which the OneRec tokenizer already contains
(verified in try.py). No vocabulary expansion is required for Strategy 3 since
we PRESERVE the 3-layer tokens and only append plain-text keywords.

Usage (domain-specific, one category):
    python scripts/A2_train_amazon.py \
        --regime single --category Toys_and_Games \
        --base-model ./models/arec2-merged

Usage (multi-domain joint, all 10):
    python scripts/A2_train_amazon.py \
        --regime joint --base-model ./models/arec2-merged
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from arec2.amazon.data_amazon import AMAZON_CATEGORIES, AmazonDomain
from arec2.amazon.dataset import AmazonSFTDataset
from arec2.amazon.item_repr import Strategy3Representer
from arec2.amazon.model_loader import load_base_with_sft_adapter
from arec2.training.data_loader import format_sample_for_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("A2_train_amazon")


def load_artifacts(cache_dir: Path, category: str):
    path = cache_dir / category / "strategy3_artifacts.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Artifacts for '{category}' not found at {path}. "
            f"Run scripts/A1_build_amazon_tokens.py first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def artifacts_to_domain(art: dict) -> AmazonDomain:
    """Rebuild a minimal AmazonDomain from cached artifacts (no item_meta needed)."""
    return AmazonDomain(
        category=art["category"],
        item_ids=art["item_ids"],
        item_meta={},  # not needed downstream (codes + keywords already cached)
        user_sequences={u: h for u, h in art["train_histories"].items()},
        train_histories=art["train_histories"],
        val_targets=art["val_targets"],
        test_targets=art["test_targets"],
    )


def make_representer(art: dict, n_keywords: int) -> Strategy3Representer:
    return Strategy3Representer(
        asin2code=art["asin2code"],
        asin2keywords=art["asin2keywords"],
        n_keywords=n_keywords,
    )


class FormattedDataset(torch.utils.data.Dataset):
    """Pre-tokenizes Amazon SFT samples once (mirrors 03_train_sft.py)."""

    def __init__(self, base_dataset, tokenizer, max_length: int = 2048):
        import os
        from tqdm import tqdm

        # Under torchrun every rank builds its own cache (the Trainer's
        # DistributedSampler then hands each rank a disjoint slice). Only rank 0
        # prints the progress bar / logs so the output isn't N-way interleaved.
        is_main = int(os.environ.get("RANK", "0")) == 0
        self.cache: list[dict] = []
        skipped = 0
        n = len(base_dataset)
        if is_main:
            logger.info("Pre-formatting (tokenizing) %d samples ...", n)
        t0 = time.time()
        log_every = max(1, n // 20)  # ~20 log lines even without a TTY progress bar
        iterator = tqdm(
            range(n), desc="Tokenizing", total=n, dynamic_ncols=True, disable=not is_main
        )
        for i in iterator:
            formatted = format_sample_for_training(base_dataset[i], tokenizer, max_length)
            if formatted is not None:
                self.cache.append(formatted)
            else:
                skipped += 1
            if is_main and (i + 1) % log_every == 0:
                logger.info("  ...formatted %d/%d (%.0f%%)", i + 1, n, 100.0 * (i + 1) / n)
        if is_main:
            logger.info(
                "Pre-formatting done in %.1fs: %d valid, %d skipped",
                time.time() - t0, len(self.cache), skipped,
            )

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        return self.cache[idx]


class SFTDataCollator:
    """Pads input_ids/attention_mask/labels; labels padded with -100."""

    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_train_dataset(args, cache_dir: Path, categories: list[str], tokenizer):
    domains: list[AmazonDomain] = []
    representers: dict[str, Strategy3Representer] = {}
    for cat in categories:
        art = load_artifacts(cache_dir, cat)
        domains.append(artifacts_to_domain(art))
        representers[cat] = make_representer(art, args.n_keywords)

    base = AmazonSFTDataset(
        domains=domains,
        representers=representers,
        with_keywords=not args.no_keywords,  # Strategy 3 by default
        split="train",
        augment_subsequences=not args.no_augment,
        max_subsequences_per_user=args.max_subseq,
    )
    return FormattedDataset(base, tokenizer, max_length=args.max_length)


def setup_model_and_tokenizer(args):
    # Load OneRec base, merge the AREC² SFT LoRA adapter into it, then attach a
    # FRESH LoRA for the Amazon transfer fine-tune (same pattern as 07_train_dpo).
    model, tokenizer = load_base_with_sft_adapter(
        base_model=args.base_model,
        sft_adapter=args.sft_adapter,
        torch_dtype=torch.bfloat16,
        for_training=True,   # DDP-aware: full model per GPU under torchrun
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model, tokenizer


def train_one(args, cache_dir: Path, categories: list[str], output_dir: Path):
    model, tokenizer = setup_model_and_tokenizer(args)
    train_dataset = build_train_dataset(args, cache_dir, categories, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=20,
        save_steps=args.save_steps,
        save_total_limit=2,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        report_to="none",
        disable_tqdm=False,   # show the per-step training progress bar
        ddp_find_unused_parameters=False,  # LoRA: no unused params; faster + avoids DDP error
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=SFTDataCollator(tokenizer),
    )

    logger.info("=" * 60)
    logger.info("Training | regime=%s | categories=%s", args.regime, categories)
    logger.info("  samples=%d | epochs=%d | eff_bsz=%d | lr=%.1e | strategy3=%s",
                len(train_dataset), args.epochs,
                args.batch_size * args.grad_accum, args.lr, not args.no_keywords)
    logger.info("=" * 60)

    trainer.train()

    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Saved LoRA adapter to %s", final_dir)
    return final_dir


def main():
    parser = argparse.ArgumentParser(description="Amazon Strategy-3 transfer training")
    parser.add_argument("--regime", choices=["single", "joint"], default="joint",
                        help="single = per-domain (paper SD); joint = multi-domain (paper MD)")
    parser.add_argument("--category", type=str, default=None,
                        help="Required for --regime single.")
    parser.add_argument("--categories", type=str, nargs="*", default=None,
                        help="Categories for joint regime; default = all 10.")
    parser.add_argument("--cache-dir", type=str, default="./caches/amazon")
    parser.add_argument("--base-model", type=str, default="OpenOneRec/OneRec-8B",
                        help="OneRec base model (HF name or local path).")
    parser.add_argument("--sft-adapter", type=str, default="./full_lora_8b",
                        help="Your AREC² SFT LoRA adapter dir to merge into the base "
                             "before Amazon fine-tuning. Set '' to use the base as-is "
                             "(or if --base-model is already a merged model).")
    parser.add_argument("--output-root", type=str, default="./checkpoints/amazon")
    # Strategy / ablation toggles
    parser.add_argument("--no-keywords", action="store_true",
                        help="Ablation: itemic tokens only (drops Strategy 3 keywords).")
    parser.add_argument("--n-keywords", type=int, default=5)
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable next-item subsequence augmentation.")
    parser.add_argument("--max-subseq", type=int, default=0)
    # LoRA / training
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers per process (per GPU under DDP).")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output_root = Path(args.output_root)
    t0 = time.time()

    tag = "itemonly" if args.no_keywords else "strategy3"

    if args.regime == "single":
        if not args.category:
            parser.error("--regime single requires --category")
        output_dir = output_root / f"{tag}_single_{args.category}"
        train_one(args, cache_dir, [args.category], output_dir)
    else:
        categories = args.categories or AMAZON_CATEGORIES
        output_dir = output_root / f"{tag}_joint"
        train_one(args, cache_dir, categories, output_dir)

    logger.info("All training done in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()