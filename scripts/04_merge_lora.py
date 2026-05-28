"""Merge LoRA adapters into base model weights.

Usage:
    python scripts/04_merge_lora.py \
        --base_model ./models/1.7B-pretrain \
        --lora_path ./checkpoints/arec2-lora-r16/final \
        --output_dir ./models/arec2-merged
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base model")
    parser.add_argument("--base_model", type=str, default="./models/1.7B-pretrain",
                        help="Path to base model")
    parser.add_argument("--lora_path", type=str, default="./checkpoints/arec2-lora-r16/final",
                        help="Path to LoRA adapter directory")
    parser.add_argument("--output_dir", type=str, default="./models/arec2-merged",
                        help="Output directory for merged model")
    args = parser.parse_args()

    print(f"Loading base model from {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    print(f"Loading LoRA adapters from {args.lora_path}")
    model = PeftModel.from_pretrained(model, args.lora_path)

    print("Merging adapters into base model...")
    model = model.merge_and_unload()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Done! Merged model saved.")


if __name__ == "__main__":
    main()
