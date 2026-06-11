"""Model-loading helper for the Amazon transfer experiment.

Your AREC² supervised fine-tune is saved as an *unmerged* LoRA adapter
(full_lora_8b/ contains adapter_config.json + adapter_model.safetensors + the
tokenizer files), on top of the OneRec-8B base. The Amazon scripts therefore
need to: load the OneRec-8B base, merge your AREC² adapter into it, and only
THEN proceed (train a fresh Amazon LoRA, or evaluate).

This mirrors the project's own scripts/07_train_dpo.py::load_sft_model_for_dpo,
which merges the SFT LoRA before attaching a fresh adapter.

Two entry points:
  - resolve_model_path: turn a path/HF-name into something from_pretrained can load.
  - load_base_with_sft_adapter: load base + (optionally) merge an SFT adapter,
    returning (model, tokenizer). Tokenizer is taken from the adapter dir when
    present (it carries the itemic-token vocab), else from the base.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_model_path(model_arg: str) -> str:
    """Return a local resolved path if it exists, else the HF name as-is."""
    p = Path(model_arg)
    return str(p.resolve()) if p.exists() else model_arg


def _looks_like_adapter(path: str) -> bool:
    """True if the directory holds a PEFT adapter (not a full model)."""
    d = Path(path)
    return d.is_dir() and (d / "adapter_config.json").exists()


def _distributed_info():
    """Return (is_distributed, local_rank, world_size) from torchrun env vars."""
    import os

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size > 1, local_rank, world_size


def load_base_with_sft_adapter(
    base_model: str,
    sft_adapter: str | None,
    device_map: str = "auto",
    torch_dtype=None,
    for_training: bool = False,
) -> tuple:
    """Load the base model and merge an SFT LoRA adapter into it.

    Args:
        base_model: OneRec base (e.g. "OpenOneRec/OneRec-8B" or a local path).
        sft_adapter: your AREC² SFT adapter dir (e.g. "./full_lora_8b").
            If None/empty, the base model is returned unmerged.
        device_map: HF device_map for the NON-distributed / inference case.
        torch_dtype: model dtype (defaults to bfloat16).
        for_training: set True for the training script. Under torchrun (DDP)
            this pins the ENTIRE model to the current process's local-rank GPU
            (data parallelism), instead of sharding layers across GPUs with
            "auto" (model parallelism), which is incompatible with DDP.

    Returns:
        (model, tokenizer). The model is a plain (merged) AutoModelForCausalLM,
        ready to either attach a fresh LoRA (training) or run inference.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch_dtype is None:
        torch_dtype = torch.bfloat16

    base_resolved = resolve_model_path(base_model)

    # Decide the device placement.
    is_dist, local_rank, world_size = _distributed_info()
    if for_training and is_dist:
        # DDP: one full model replica per GPU, placed on this rank's device.
        effective_device_map = {"": local_rank}
        logger.info(
            "Distributed training detected (world_size=%d); placing full model "
            "on cuda:%d (DDP data parallel).", world_size, local_rank,
        )
    elif for_training:
        # Single-GPU training: put the whole model on cuda:0 (not sharded), so
        # the HF Trainer's optimizer/backward see one device.
        effective_device_map = {"": 0} if torch.cuda.is_available() else None
        logger.info("Single-GPU training; placing full model on cuda:0.")
    else:
        # Inference / eval: keep the requested device_map (e.g. "auto").
        effective_device_map = device_map

    # Prefer the adapter dir's tokenizer: it carries the expanded itemic-token
    # vocab (added_tokens.json / vocab.json), which the base may not have if you
    # point at a vanilla checkpoint.
    tok_src = base_resolved
    if sft_adapter and _looks_like_adapter(sft_adapter):
        adapter_dir = resolve_model_path(sft_adapter)
        if (Path(adapter_dir) / "tokenizer_config.json").exists():
            tok_src = adapter_dir

    logger.info("Loading tokenizer from %s", tok_src)
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model from %s", base_resolved)
    model = AutoModelForCausalLM.from_pretrained(
        base_resolved,
        torch_dtype=torch_dtype,
        device_map=effective_device_map,
        trust_remote_code=True,
    )

    # If the adapter added itemic tokens beyond the base vocab, resize embeddings
    # so merging does not index out of range.
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        logger.info(
            "Resizing token embeddings %d -> %d to match adapter tokenizer.",
            model.get_input_embeddings().weight.shape[0], len(tokenizer),
        )
        model.resize_token_embeddings(len(tokenizer))

    if sft_adapter and _looks_like_adapter(sft_adapter):
        from peft import PeftModel

        adapter_dir = resolve_model_path(sft_adapter)
        logger.info("Merging AREC² SFT LoRA adapter from %s ...", adapter_dir)
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
        logger.info("SFT adapter merged into base. This is now the AREC² model.")
    elif sft_adapter:
        logger.warning(
            "sft_adapter=%s is not a PEFT adapter dir (no adapter_config.json); "
            "treating --base-model as an already-merged model and ignoring it.",
            sft_adapter,
        )
    else:
        logger.info("No SFT adapter provided; using base model as-is.")

    return model, tokenizer