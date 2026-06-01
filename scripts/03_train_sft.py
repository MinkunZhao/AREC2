"""Phase 4: SFT Training with Agentic Context Enrichment.

Trains OpenOneRec-1.7B with LoRA on enriched RecIF + General_SFT data.

v3 throughput fixes (no change to learned objective):
  1. FormattedDataset now CACHES the tokenized result during pre-filtering and
     serves it from memory at __getitem__ time. Previously every sample was
     tokenized once during pre-filter AND again every epoch in __getitem__
     (double tokenization + no real use of dataloader workers). Now each sample
     is tokenized exactly once.
  2. Pre-filtering/formatting is parallelized across processes (--format-workers).
     This is a one-time cost, not per-epoch.
  3. Optional sequence PACKING (--packing): multiple short samples are
     concatenated up to max_length, with per-sample label masking preserved and
     cross-sample attention blocked via position_ids reset + segment-aware mask.
     RecIF answers are short, so packing greatly raises token utilization.
  4. max_length / num_workers are read from config; set max_length to the real
     p99 sequence length to cut padding compute and allow larger batches.

The loss mask semantics are IDENTICAL to before: loss is computed only on the
final assistant answer (prompt tokens are -100). Packing concatenates already
correctly-masked samples, so it does not change which tokens contribute to loss.
"""

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from arec2.agents.executor_agent import ExecutorAgent
from arec2.agents.planner_agent import PlannerAgent
from arec2.retrieval.stores import (
    CollaborativeStore,
    ItemTextStore,
    LabelBehaviorStore,
    ProfileStore,
)
from arec2.training.data_loader import (
    CombinedDataset,
    GeneralSFTDataset,
    RecIFDataset,
    format_sample_for_training,
    truncate_preserving_answer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_sft")


# ---------------------------------------------------------------------------
# Module-level globals for the process pool. ProcessPoolExecutor pickles the
# worker function; tokenizers (fast) are picklable, but to be safe and fast we
# initialize a per-worker tokenizer once via an initializer.
# ---------------------------------------------------------------------------
_WORKER_TOKENIZER = None
_WORKER_MAX_LENGTH = 4096


def _format_worker_init(tokenizer_path: str, max_length: int):
    global _WORKER_TOKENIZER, _WORKER_MAX_LENGTH
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    _WORKER_MAX_LENGTH = max_length


def _format_worker(sample: dict):
    return format_sample_for_training(sample, _WORKER_TOKENIZER, _WORKER_MAX_LENGTH)


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_pid2sid(path: str) -> dict:
    """Load PID→SID mapping from parquet."""
    logger.info("Loading PID→SID mapping from %s", path)
    pid2sid = {}
    df = pd.read_parquet(path)
    for _, row in df.iterrows():
        pid2sid[int(row["pid"])] = list(row["sid"])
    logger.info("Loaded %d PID→SID mappings", len(pid2sid))
    return pid2sid


def load_stores(config: dict):
    """Load offline retrieval stores."""
    logger.info("Loading offline stores...")
    stores_config = config["stores"]

    profile_store = ProfileStore.load(stores_config["profile_store_path"])
    label_store = LabelBehaviorStore.load(stores_config["label_store_path"])
    collab_store = CollaborativeStore.load(stores_config["collab_store_path"])
    text_store = ItemTextStore.load(stores_config["text_store_path"])

    logger.info("All stores loaded")
    return profile_store, label_store, collab_store, text_store


def prepare_datasets(config: dict, planner, executor, pid2sid, card_source: str = "heuristic"):
    """Prepare training datasets."""
    logger.info("Preparing datasets (card_source=%s)...", card_source)
    data_config = config["data"]

    # Load product PID->SID if available
    pid2sid_product = {}
    product_path = data_config.get("pid2sid_product_path", "")
    if product_path and Path(product_path).exists():
        logger.info("Loading product PID->SID mapping from %s", product_path)
        df = pd.read_parquet(product_path)
        for _, row in df.iterrows():
            pid2sid_product[int(row["pid"])] = list(row["sid"])
        logger.info("Loaded %d product PID->SID mappings", len(pid2sid_product))

    # Determine enrichment setup based on card_source
    cards_v2_dir = data_config.get("cards_v2_dir", None)
    use_planner = card_source == "heuristic"

    # RecIF dataset
    recif_dataset = RecIFDataset(
        data_dir=data_config["recif_data_dir"],
        tasks=data_config["recif_tasks"],
        planner=planner if use_planner else None,
        executor=executor if use_planner else None,
        pid2sid=pid2sid,
        pid2sid_product=pid2sid_product,
        card_source=card_source,
        cards_v2_dir=cards_v2_dir,
        max_samples=data_config.get("max_recif_samples"),
        seed=data_config["seed"],
    )

    # General SFT dataset
    general_sft_dataset = GeneralSFTDataset(
        data_dir=data_config["general_sft_data_dir"],
        max_samples=data_config.get("max_general_samples"),
        seed=data_config["seed"],
    )

    # Combined dataset
    combined_dataset = CombinedDataset(
        recif_dataset=recif_dataset,
        general_sft_dataset=general_sft_dataset,
        recif_ratio=data_config["recif_ratio"],
        seed=data_config["seed"],
    )

    logger.info("Datasets prepared: %d total samples", len(combined_dataset))
    return combined_dataset


def setup_model_and_tokenizer(config: dict):
    """Load base model, apply LoRA, and load tokenizer."""
    logger.info("Loading base model and tokenizer...")
    model_config = config["model"]
    lora_config_dict = config["lora"]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["base_model_path"],
        trust_remote_code=True,
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_config["base_model_path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_config_dict["r"],
        lora_alpha=lora_config_dict["lora_alpha"],
        lora_dropout=lora_config_dict["lora_dropout"],
        bias=lora_config_dict["bias"],
        task_type=lora_config_dict["task_type"],
        target_modules=lora_config_dict["target_modules"],
    )

    model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    logger.info("Model and tokenizer loaded")
    return model, tokenizer


def create_training_args(config: dict) -> TrainingArguments:
    """Create TrainingArguments from config."""
    training_config = config["training"]
    model_config = config["model"]

    return TrainingArguments(
        output_dir=model_config["output_dir"],
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        warmup_ratio=training_config["warmup_ratio"],
        weight_decay=training_config["weight_decay"],
        max_grad_norm=training_config["max_grad_norm"],
        optim=training_config["optim"],
        bf16=training_config["bf16"],
        gradient_checkpointing=training_config["gradient_checkpointing"],
        logging_steps=training_config["logging_steps"],
        save_steps=training_config["save_steps"],
        save_total_limit=training_config["save_total_limit"],
        dataloader_num_workers=training_config["dataloader_num_workers"],
        dataloader_pin_memory=training_config.get("dataloader_pin_memory", True),
        remove_unused_columns=training_config["remove_unused_columns"],
        report_to=training_config.get("report_to", "none"),
    )


class FormattedDataset(torch.utils.data.Dataset):
    """Pre-formats samples ONCE and serves the cached tokenized result.

    Previously this class tokenized every sample during pre-filtering and then
    tokenized it AGAIN on every __getitem__ (i.e. every epoch). Now the
    tokenized dict is cached in memory and returned directly, so each sample is
    tokenized exactly once. Pre-formatting can be parallelized across processes.
    """

    def __init__(self, base_dataset, tokenizer, max_length: int = 4096,
                 tokenizer_path: str | None = None, format_workers: int = 1):
        self.max_length = max_length
        self.cache: list[dict] = []
        self.skipped = 0

        logger.info(
            "Pre-formatting dataset (max_length=%d, workers=%d)...",
            max_length, format_workers,
        )
        t0 = time.time()

        # Materialize raw samples first (this triggers enrichment / dict lookups).
        # For card_source='llm_optimized' this is cheap; for 'heuristic' it is the
        # bulk of the cost and is unavoidable, but at least it now happens once.
        raw_samples = [base_dataset[i] for i in range(len(base_dataset))]

        if format_workers > 1 and tokenizer_path is not None:
            with ProcessPoolExecutor(
                max_workers=format_workers,
                initializer=_format_worker_init,
                initargs=(tokenizer_path, max_length),
            ) as pool:
                for formatted in pool.map(_format_worker, raw_samples, chunksize=64):
                    if formatted is not None:
                        self.cache.append(formatted)
                    else:
                        self.skipped += 1
        else:
            for sample in raw_samples:
                formatted = format_sample_for_training(sample, tokenizer, max_length)
                if formatted is not None:
                    self.cache.append(formatted)
                else:
                    self.skipped += 1

        logger.info(
            "Pre-formatting complete in %.1fs: %d valid, %d skipped (%.1f%% filtered)",
            time.time() - t0,
            len(self.cache),
            self.skipped,
            100.0 * self.skipped / max(len(raw_samples), 1),
        )

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        return self.cache[idx]


class PackedDataset(torch.utils.data.Dataset):
    """Greedily packs pre-formatted samples up to max_length.

    Each packed example concatenates several samples. Labels keep their original
    per-sample masking (prompt = -100, answer = real ids), so the supervised
    objective is unchanged. To prevent tokens of one sample from attending to a
    different sample, we emit `position_ids` that reset at each segment boundary
    and a `segment_ids` tensor the collator turns into a block-diagonal mask.

    NOTE: This requires the model's attention to honor a 2D/4D attention mask.
    If your model/transformers version does not support custom block masks,
    disable packing (--no-packing) and rely on caching + reduced data instead.
    """

    def __init__(self, formatted_samples: list[dict], max_length: int = 4096):
        self.max_length = max_length
        self.packs: list[dict] = []
        self._build(formatted_samples)
        logger.info(
            "Packing: %d samples -> %d packed sequences (avg %.1f samples/pack)",
            len(formatted_samples), len(self.packs),
            len(formatted_samples) / max(len(self.packs), 1),
        )

    def _build(self, samples: list[dict]):
        cur_ids, cur_labels, cur_pos, cur_seg = [], [], [], []
        seg_id = 0

        def flush():
            nonlocal cur_ids, cur_labels, cur_pos, cur_seg, seg_id
            if cur_ids:
                self.packs.append({
                    "input_ids": cur_ids,
                    "labels": cur_labels,
                    "position_ids": cur_pos,
                    "segment_ids": cur_seg,
                    "attention_mask": [1] * len(cur_ids),
                })
            cur_ids, cur_labels, cur_pos, cur_seg = [], [], [], []
            seg_id = 0

        for s in samples:
            ids = s["input_ids"]
            labels = s["labels"]
            n = len(ids)
            if n > self.max_length:
                # Shouldn't happen (filtered upstream) but guard.
                ids = ids[:self.max_length]
                labels = labels[:self.max_length]
                n = self.max_length

            if len(cur_ids) + n > self.max_length and cur_ids:
                flush()

            cur_ids.extend(ids)
            cur_labels.extend(labels)
            cur_pos.extend(range(n))
            cur_seg.extend([seg_id] * n)
            seg_id += 1

        flush()

    def __len__(self):
        return len(self.packs)

    def __getitem__(self, idx):
        return self.packs[idx]


class SFTDataCollator:
    """Pads input_ids, attention_mask, and labels. Labels are padded with -100."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class PackedDataCollator:
    """Collator for PackedDataset.

    Builds a 4D block-diagonal attention mask so each segment only attends to
    its own (causal) tokens, plus padded position_ids. Falls back gracefully if
    the model ignores 4D masks (then it behaves like normal causal attention,
    which slightly leaks across segment boundaries — acceptable but not ideal).
    """

    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        bsz = len(features)

        input_ids = torch.full((bsz, max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((bsz, max_len), -100, dtype=torch.long)
        position_ids = torch.zeros((bsz, max_len), dtype=torch.long)

        # 4D additive mask: (bsz, 1, max_len, max_len), 0 = attend, -inf = block.
        neg_inf = torch.finfo(torch.bfloat16).min
        attn_4d = torch.full((bsz, 1, max_len, max_len), neg_inf, dtype=torch.bfloat16)

        for b, f in enumerate(features):
            ids = f["input_ids"]
            n = len(ids)
            input_ids[b, :n] = torch.tensor(ids, dtype=torch.long)
            labels[b, :n] = torch.tensor(f["labels"], dtype=torch.long)
            position_ids[b, :n] = torch.tensor(f["position_ids"], dtype=torch.long)

            seg = torch.tensor(f["segment_ids"], dtype=torch.long)
            # Allow attention only within the same segment AND causal (j <= i).
            for i in range(n):
                same_seg = (seg[: i + 1] == seg[i])
                idxs = torch.arange(i + 1)[same_seg]
                attn_4d[b, 0, i, idxs] = 0.0

        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "attention_mask": attn_4d,
        }


def main():
    t0 = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(description="SFT training for AREC2")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--ablation", type=str, default=None, choices=["a0"],
                        help="a0: disable context enrichment (baseline)")
    parser.add_argument("--card_source", type=str, default="heuristic",
                        choices=["heuristic", "llm_optimized", "none"],
                        help="Context card source: heuristic (rule-based on-the-fly), "
                             "llm_optimized (pre-computed from TextGrad), none (no enrichment)")
    parser.add_argument("--planner", type=str, default="rule",
                        choices=["rule", "llm"],
                        help="Planner type for heuristic card_source: rule (PlannerAgent) "
                             "or llm (LLMPlanner, requires LLM config)")
    parser.add_argument("--format-workers", type=int, default=8,
                        help="Parallel processes for one-time tokenization/pre-filtering.")
    parser.add_argument("--packing", action="store_true", default=False,
                        help="Enable sequence packing (big speedup for short answers).")
    parser.add_argument("--no-packing", dest="packing", action="store_false",
                        help="Disable sequence packing.")
    args = parser.parse_args()

    config_path = args.config
    logger.info("Loading config from %s", config_path)
    config = load_config(config_path)

    # A0 ablation overrides card_source to "none"
    card_source = args.card_source
    if args.ablation == "a0":
        card_source = "none"
        logger.info("Ablation A0: context enrichment DISABLED (card_source=none)")

    # Load PID->SID mapping
    pid2sid = load_pid2sid(config["data"]["pid2sid_path"])

    # Load stores (needed for heuristic enrichment)
    planner = None
    executor = None
    if card_source == "heuristic":
        profile_store, label_store, collab_store, text_store = load_stores(config)

        if args.planner == "llm":
            from arec2.agents.llm_client import LLMClient
            from arec2.agents.llm_planner import LLMPlanner

            llm_config = config.get("llm", {})
            llm = LLMClient(
                model=llm_config.get("model", "gpt-4o-mini"),
                base_url=llm_config.get("base_url", "https://yunwu.ai/v1"),
                api_key=llm_config.get("api_key", ""),
                cache_dir=llm_config.get("cache_dir", "./caches/llm_cache"),
                max_concurrency=llm_config.get("max_concurrency", 8),
            )
            planner_path = config.get("prompts", {}).get(
                "planner_init", "prompts/planner_v0.txt"
            )
            logger.info("Using LLMPlanner with instruction from %s", planner_path)
            planner = LLMPlanner(llm, instruction_path=planner_path)
        else:
            logger.info("Using rule-based PlannerAgent")
            planner = PlannerAgent(token_budget=config["data"]["token_budget"])

        executor = ExecutorAgent(
            profile_store=profile_store,
            label_store=label_store,
            collab_store=collab_store,
            text_store=text_store,
            token_budget=config["data"]["token_budget"],
            card_style=config["data"].get("card_style", "natural"),
        )
    else:
        logger.info("card_source=%s: skipping store/planner/executor initialization", card_source)

    # Prepare datasets
    combined_dataset = prepare_datasets(config, planner, executor, pid2sid, card_source=card_source)

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)

    # Wrap dataset with formatting (cached + optionally parallel)
    max_length = config["training"].get("max_length", 4096)
    tokenizer_path = config["model"]["base_model_path"]

    # Parallel formatting only makes sense when enrichment is cheap (llm_optimized
    # / none): the raw sample materialization happens in the main process. For
    # heuristic enrichment, raw materialization is the bottleneck and runs in the
    # main process regardless, so we keep workers but they only do tokenization.
    formatted = FormattedDataset(
        combined_dataset, tokenizer, max_length=max_length,
        tokenizer_path=tokenizer_path, format_workers=max(1, args.format_workers),
    )

    # Optional packing
    if args.packing:
        train_dataset = PackedDataset(formatted.cache, max_length=max_length)
        data_collator = PackedDataCollator(tokenizer)
        logger.info("Sequence packing ENABLED.")
    else:
        train_dataset = formatted
        data_collator = SFTDataCollator(tokenizer=tokenizer)
        logger.info("Sequence packing disabled (use --packing to enable).")

    # Training arguments
    training_args = create_training_args(config)

    # Trainer
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("  Total training units: %d", len(train_dataset))
    logger.info("  Batch size: %d", training_args.per_device_train_batch_size)
    logger.info("  Gradient accumulation: %d", training_args.gradient_accumulation_steps)
    logger.info("  Effective batch size: %d", training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
    logger.info("  Epochs: %d", training_args.num_train_epochs)
    logger.info("  Learning rate: %.2e", training_args.learning_rate)
    logger.info("  Max length: %d | Packing: %s", max_length, args.packing)
    logger.info("=" * 60)

    trainer.train()

    # Save final model
    logger.info("Saving final LoRA adapters...")
    final_output_dir = Path(config["model"]["output_dir"]) / "final"
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    logger.info("=" * 60)
    logger.info("Training complete! (%.1f hours)", (time.time() - t0) / 3600)
    logger.info("LoRA adapters saved to: %s", final_output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()