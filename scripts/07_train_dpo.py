"""Phase D: DPO Training with RecPO preference pairs (v2).

Loads SFT v2 checkpoint, applies fresh LoRA (r=8), trains with DPO
on the (now per-task-balanced + binary-for-label_pred) preference pairs from
scripts/06_generate_preferences.py.

v2 fixes:
  - Passes rpo_alpha (chosen-NLL anchoring) and a lower beta to suppress
    cross-task negative transfer (ad/product/label_pred regression).
  - Hard-asserts reference equivalence so a non-identity fresh adapter fails
    fast instead of silently corrupting the DPO reference (which especially
    hurts the low-logit-margin label_pred judgement task).

Usage:
    python scripts/07_train_dpo.py --config configs/dpo_config.yaml
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
import yaml
from datasets import Dataset

from arec2.rl.rec_dpo_trainer import (
    create_dpo_trainer,
    load_sft_model_for_dpo,
    verify_reference_equivalence,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_dpo")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="DPO training for AREC2 (v2)")
    parser.add_argument("--config", type=str, default="configs/dpo_config.yaml")
    parser.add_argument("--ref-check-tol", type=float, default=1e-3,
                        help="Max allowed mean |logit diff| between adapter-enabled "
                             "and disabled at init. Fail fast if exceeded.")
    parser.add_argument("--skip-ref-assert", action="store_true",
                        help="Only warn (do not abort) if reference equivalence check fails.")
    args = parser.parse_args()

    t0 = time.time()
    config = load_config(args.config)

    # Load preference pairs
    pairs_path = Path(config["data"]["preferences_dir"]) / "all_pairs.parquet"
    logger.info("Loading preference pairs from %s", pairs_path)
    df = pd.read_parquet(pairs_path)
    logger.info("Loaded %d preference pairs", len(df))

    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_pandas(df[["prompt", "chosen", "rejected"]])

    # Optional eval split
    eval_dataset = None
    eval_size = config["training"].get("eval_size", 0)
    if eval_size > 0 and len(train_dataset) > eval_size:
        split = train_dataset.train_test_split(test_size=eval_size, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        logger.info("Split: %d train, %d eval", len(train_dataset), len(eval_dataset))

    # Load model
    model_config = config["model"]
    lora_config = config["lora"]

    model, tokenizer = load_sft_model_for_dpo(
        sft_checkpoint_path=model_config["sft_checkpoint_path"],
        base_model_path=model_config["base_model_path"],
        lora_r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
    )

    # Create trainer
    training_config = config["training"]
    trainer = create_dpo_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=model_config["output_dir"],
        beta=training_config.get("beta", 0.05),
        loss_type=training_config.get("loss_type", "sigmoid"),
        learning_rate=training_config["learning_rate"],
        num_train_epochs=training_config["num_train_epochs"],
        per_device_batch_size=training_config["per_device_train_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        max_length=training_config["max_length"],
        max_prompt_length=training_config["max_prompt_length"],
        rpo_alpha=training_config.get("rpo_alpha", 1.0),
    )

    # Hard reference-equivalence check: fresh adapter must be identity at init.
    ref_diff = verify_reference_equivalence(trainer)
    if ref_diff > args.ref_check_tol:
        msg = (
            f"Reference equivalence FAILED: max |logit diff| = {ref_diff:.3e} "
            f"> tol {args.ref_check_tol:.1e}. The fresh DPO adapter is NOT identity "
            f"at init, so the DPO reference != merged-SFT model. This silently "
            f"corrupts low-margin tasks (e.g. label_pred AUC)."
        )
        if args.skip_ref_assert:
            logger.warning(msg + " (continuing due to --skip-ref-assert)")
        else:
            raise RuntimeError(msg)
    else:
        logger.info("Reference equivalence OK (diff=%.3e <= tol=%.1e).", ref_diff, args.ref_check_tol)

    # Train
    logger.info("=" * 60)
    logger.info("Starting DPO training...")
    logger.info("  Pairs: %d", len(train_dataset))
    logger.info("  Beta: %.3f", training_config.get("beta", 0.05))
    logger.info("  rpo_alpha: %.3f", training_config.get("rpo_alpha", 1.0))
    logger.info("  Loss: %s", training_config.get("loss_type", "sigmoid"))
    logger.info("  LR: %.2e", training_config["learning_rate"])
    logger.info("  LoRA rank: %d", lora_config["r"])
    logger.info("=" * 60)

    trainer.train()

    # Save
    final_dir = Path(model_config["output_dir"]) / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    logger.info("=" * 60)
    logger.info("DPO training complete! (%.1f hours)", (time.time() - t0) / 3600)
    logger.info("Adapter saved to: %s", final_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()