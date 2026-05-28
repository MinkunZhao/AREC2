"""RecPO DPO Trainer: wraps TRL DPOTrainer for recommendation-specific training.

Key adaptations / fixes:
  - Loads SFT v2 merged model as base, applies a FRESH LoRA (r=8) for DPO.
    Because the fresh adapter initializes to identity (LoRA B = 0), the TRL
    "disable-adapter" reference equals the merged-SFT model — exactly the
    reference we want. No separate ref_model needed.
  - `max_prompt_length` is now actually used, and `truncation_mode="keep_end"`
    keeps the tail of the prompt (context card + final user question), instead
    of silently truncating the answer or the card.
  - Default loss_type is "sigmoid" (standard DPO). IPO is more aggressive on the
    very short SID answers in RecIF; switch to it only after sigmoid is verified
    not to regress.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

logger = logging.getLogger(__name__)


def load_sft_model_for_dpo(
    sft_checkpoint_path: str,
    base_model_path: str,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    device_map: str = "auto",
) -> tuple:
    """Load SFT LoRA, merge into base, then attach a fresh LoRA for DPO."""
    logger.info("Loading tokenizer from %s", base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model from %s", base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )

    sft_path = Path(sft_checkpoint_path)
    if sft_path.exists():
        logger.info("Merging SFT LoRA from %s ...", sft_checkpoint_path)
        model = PeftModel.from_pretrained(model, sft_checkpoint_path)
        model = model.merge_and_unload()
        logger.info("SFT LoRA merged. This merged model is the DPO reference.")
    else:
        logger.warning(
            "SFT checkpoint not found at %s; using raw base model as DPO start. "
            "This is almost certainly NOT what you want for RecPO.",
            sft_checkpoint_path,
        )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    return model, tokenizer


def create_dpo_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset=None,
    output_dir: str = "./checkpoints/arec2-dpo-r8",
    beta: float = 0.1,
    loss_type: str = "sigmoid",
    learning_rate: float = 5e-6,
    num_train_epochs: int = 1,
    per_device_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_length: int = 2048,
    max_prompt_length: int = 1792,
) -> DPOTrainer:
    """Create a configured DPO trainer.

    Args:
        max_length: Max total (prompt + completion) length.
        max_prompt_length: Max prompt length. The completion (SID list + turn
            end) is short, so set this close to max_length to avoid dropping the
            context card. With truncation_mode="keep_end", over-long prompts keep
            their tail (card + final question), which is what matters most.
    """
    if max_prompt_length >= max_length:
        max_prompt_length = max_length - 64  # always leave room for completion

    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        beta=beta,
        loss_type=loss_type,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        truncation_mode="keep_end",   # keep card + final question, not the system header
        label_pad_token_id=-100,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    return trainer


@torch.no_grad()
def verify_reference_equivalence(trainer: DPOTrainer, n_check: int = 2) -> None:
    """Sanity check: the fresh adapter must start as identity so that the DPO
    reference (adapter disabled) == merged SFT model.

    Logs the mean absolute logit difference between adapter-enabled and
    adapter-disabled forward passes on a few training examples. At init it
    should be ~0. Call this once before trainer.train().
    """
    model = trainer.model
    if not isinstance(model, PeftModel):
        logger.info("Model is not a PeftModel; skipping reference check.")
        return

    ds = trainer.train_dataset
    if ds is None or len(ds) == 0:
        return

    tok = trainer.processing_class
    device = next(model.parameters()).device
    max_diff = 0.0
    for i in range(min(n_check, len(ds))):
        row = ds[i]
        if "prompt" in row and isinstance(row["prompt"], str):
            text = row["prompt"] + row["chosen"]
            ids = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
        else:
            all_ids = (row["prompt_input_ids"] + row["chosen_input_ids"])[:2048]
            ids = {
                "input_ids": torch.tensor([all_ids], device=device),
                "attention_mask": torch.ones(1, len(all_ids), device=device, dtype=torch.long),
            }

        model.eval()
        enabled = model(**ids).logits

        with model.disable_adapter():
            disabled = model(**ids).logits

        diff = (enabled.float() - disabled.float()).abs().mean().item()
        max_diff = max(max_diff, diff)

    logger.info(
        "[ref-check] max |logit(enabled) - logit(disabled)| = %.3e "
        "(should be ~0 at init; large => fresh adapter is not identity)",
        max_diff,
    )