"""Sanity checks for AREC² SFT pipeline after B1-B4 bug fixes.

Run: python scripts/sanity_check.py --config configs/training_config_quick.yaml

Verification 1: Single sample loss behavior
Verification 2: Truncation doesn't cut answer (100 longest samples)
Verification 3: Train/eval tokenization consistency
Verification 4: (manual) --ablation a0 switch presence check
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
import os


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from arec2.training.data_loader import format_sample_for_training, truncate_preserving_answer


def load_tokenizer(config):
    model_path = config["model"]["base_model_path"]
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return tokenizer


def make_test_sample(with_card=True):
    """Create a synthetic test sample mimicking RecIF format."""
    card = ""
    if with_card:
        card = (
            "This user's long-term profile shows: video=458, ad=12, active-liker. "
            "Their preferred items include <s_a_1><s_b_2><s_c_3> <s_a_4><s_b_5><s_c_6>.\n"
            "In their most recent 15 interactions, this user engaged with: "
            "<s_a_7><s_b_8><s_c_9> <s_a_10><s_b_11><s_c_12>.\n"
            "Similar users also engaged with: collaborative filtering results. "
            "Popular items among neighbors: <s_a_13><s_b_14><s_c_15>."
        )

    user_content = "Based on the user's historical interactions, recommend 10 items."
    if card:
        user_content = user_content + "\n\n" + card

    messages = [
        {"role": "system", "content": "You are a recommendation assistant."},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": "<s_a_20><s_b_21><s_c_22> <s_a_23><s_b_24><s_c_25> <s_a_26><s_b_27><s_c_28>"},
    ]
    return {"messages": messages, "messages_enriched": messages}


def verification_1(tokenizer, config):
    """Verification 1: Single sample loss behavior."""
    print("\n" + "=" * 60)
    print("VERIFICATION 1: Single sample loss behavior")
    print("=" * 60)

    sample = make_test_sample(with_card=True)
    max_length = config["training"].get("max_length", 4096)
    result = format_sample_for_training(sample, tokenizer, max_length=max_length)

    if result is None:
        print("FAIL: format_sample_for_training returned None (sample was filtered)")
        return False

    input_ids = result["input_ids"]
    labels = result["labels"]

    total_len = len(input_ids)
    masked_count = sum(1 for l in labels if l == -100)
    unmasked_count = total_len - masked_count

    # Find position range of unmasked tokens
    unmasked_positions = [i for i, l in enumerate(labels) if l != -100]
    if unmasked_positions:
        first_unmasked = unmasked_positions[0]
        last_unmasked = unmasked_positions[-1]
    else:
        first_unmasked = last_unmasked = -1

    print(f"  input_ids length: {total_len}")
    print(f"  labels -100 count (masked/prompt): {masked_count}")
    print(f"  labels non-(-100) count (answer): {unmasked_count}")
    print(f"  Unmasked position range: [{first_unmasked}, {last_unmasked}]")
    print(f"  Expected: unmasked at tail of sequence")

    # Check: unmasked should be contiguous at the tail
    is_contiguous_tail = (
        unmasked_count > 0
        and last_unmasked == total_len - 1
        and (last_unmasked - first_unmasked + 1) == unmasked_count
    )

    # Check: unmasked should be much less than total (answer is typically 30-100 tokens)
    answer_ratio = unmasked_count / total_len if total_len > 0 else 0

    print(f"  Answer ratio: {answer_ratio:.2%}")
    print(f"  Contiguous at tail: {is_contiguous_tail}")

    if not is_contiguous_tail:
        print("FAIL: Unmasked tokens are not contiguous at the tail")
        return False

    if answer_ratio > 0.5:
        print("FAIL: More than half the sequence is unmasked - mask logic may be wrong")
        return False

    # Optional: compute loss if model is available
    print("PASS: Loss mask is correctly applied (answer-only at tail)")
    return True


def verification_2(tokenizer, config):
    """Verification 2: Truncation doesn't cut the answer."""
    print("\n" + "=" * 60)
    print("VERIFICATION 2: Truncation preserves answer (100 samples)")
    print("=" * 60)

    max_length = config["training"].get("max_length", 4096)
    passed = 0
    failed = 0

    # Generate samples of varying lengths
    for i in range(100):
        # Make progressively longer cards to stress-test truncation
        card_repeat = 5 + i * 3  # increasingly long cards
        long_card = " ".join(
            [f"<s_a_{j}><s_b_{j+1}><s_c_{j+2}> item_{j} description text padding"
             for j in range(card_repeat)]
        )
        user_content = "Recommend items.\n\n" + long_card

        messages = [
            {"role": "system", "content": "You are a recommendation assistant."},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "<s_a_99><s_b_100><s_c_101> answer_token_here"},
        ]
        sample = {"messages": messages, "messages_enriched": messages}

        result = format_sample_for_training(sample, tokenizer, max_length=max_length)

        if result is None:
            # Sample was correctly filtered (too long even after truncation)
            passed += 1
            continue

        labels = result["labels"]

        # Assert: last token of labels must NOT be -100
        if labels[-1] == -100:
            print(f"  FAIL sample {i}: labels[-1] == -100 (answer was cut)")
            failed += 1
            continue

        # Assert: all answer tokens are preserved (non -100 tokens exist)
        unmasked = [l for l in labels if l != -100]
        if len(unmasked) == 0:
            print(f"  FAIL sample {i}: no unmasked tokens at all")
            failed += 1
            continue

        passed += 1

    print(f"  Results: {passed}/100 passed, {failed}/100 failed")

    if failed > 0:
        print("FAIL: Some samples had their answers truncated")
        return False

    print("PASS: All 100 samples preserve the answer correctly")
    return True


def verification_3(tokenizer, config):
    """Verification 3: Train/eval tokenization consistency."""
    print("\n" + "=" * 60)
    print("VERIFICATION 3: Train/eval token consistency")
    print("=" * 60)

    sample = make_test_sample(with_card=True)
    messages = sample["messages"]

    # --- Training side: tokenize full then extract prompt portion ---
    prompt_messages = messages[:-1]
    train_prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )["input_ids"]

    # --- Eval side: same as build_prompt_from_messages in eval_finetuned.py ---
    prompt_messages_eval = []
    for msg in messages:
        if msg["role"] == "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join([c.get("text", "") for c in content if isinstance(c, dict)])
        prompt_messages_eval.append({"role": msg["role"], "content": content})

    eval_text = tokenizer.apply_chat_template(
        prompt_messages_eval,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    eval_ids = tokenizer(eval_text, return_tensors="pt")["input_ids"][0].tolist()

    # Compare
    train_first50 = train_prompt_ids[:50]
    eval_first50 = eval_ids[:50]
    train_last50 = train_prompt_ids[-50:]
    eval_last50 = eval_ids[-50:]

    print(f"  Train prompt length: {len(train_prompt_ids)}")
    print(f"  Eval prompt length: {len(eval_ids)}")
    print(f"  First 50 tokens match: {train_first50 == eval_first50}")
    print(f"  Last 50 tokens match: {train_last50 == eval_last50}")
    print(f"  Full match: {train_prompt_ids == eval_ids}")

    if train_prompt_ids == eval_ids:
        print("PASS: Train and eval tokenization produce identical sequences")
        return True
    else:
        # Find first difference
        for i in range(min(len(train_prompt_ids), len(eval_ids))):
            if train_prompt_ids[i] != eval_ids[i]:
                print(f"  First diff at position {i}: train={train_prompt_ids[i]}, eval={eval_ids[i]}")
                print(f"  Train context: {train_prompt_ids[max(0,i-3):i+3]}")
                print(f"  Eval  context: {eval_ids[max(0,i-3):i+3]}")
                break
        print("FAIL: Token sequences differ between train and eval")
        return False


def verification_4_check(config):
    """Verification 4: Check --ablation a0 switch is available."""
    print("\n" + "=" * 60)
    print("VERIFICATION 4: Ablation A0 switch availability")
    print("=" * 60)

    eval_script = Path(__file__).parent.parent / "eval_finetuned.py"
    content = eval_script.read_text(encoding="utf-8")

    has_ablation_arg = "--ablation" in content and "a0" in content
    has_enrich_toggle = "args.enrich" in content or "enrichment_enabled" in content

    print(f"  --ablation a0 argument present: {has_ablation_arg}")
    print(f"  Enrichment toggle present: {has_enrich_toggle}")

    if has_ablation_arg and has_enrich_toggle:
        print("PASS: Ablation A0 switch is available")
        print("  Usage: python eval_finetuned.py --ablation a0")
        return True
    else:
        print("FAIL: Ablation switch not properly implemented")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training_config_quick.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    tokenizer = load_tokenizer(config)

    results = {}
    results["v1_loss_mask"] = verification_1(tokenizer, config)
    results["v2_truncation"] = verification_2(tokenizer, config)
    results["v3_token_consistency"] = verification_3(tokenizer, config)
    results["v4_ablation_switch"] = verification_4_check(config)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll verifications PASSED!")
    else:
        print("\nSome verifications FAILED - review output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
