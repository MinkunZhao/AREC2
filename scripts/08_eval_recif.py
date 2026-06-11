"""Evaluate OneRec/OpenOneRec checkpoints on RecIF.

This script mirrors the public OpenOneRec benchmark protocol:

* prompts are built with ``tokenizer.apply_chat_template`` and the
  ``qwen3_soft_switch.jinja2`` template;
* recommendation tasks append ``<|sid_begin|>`` and generate SID candidates
  via beam search (128 beams by default, matching official vLLM config);
* ``label_pred`` uses next-token probabilities for ``是`` / ``否`` instead of
  parsing generated text;
* item-understanding and recommendation-reason scores use an external LLM
  judge and are therefore optional.

Examples:
    python scripts/08_eval_recif.py --model ./models/1.7B --tasks video ad product label_cond interactive label_pred
    python scripts/08_eval_recif.py --model ./models/1.7B --tasks video --max-samples 100 --batch-size 2
    python scripts/08_eval_recif.py --model ./models/1.7B --tasks rec_reason --judge true
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("HF_HOME", "D:/hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "D:/hf_cache/datasets")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = None


SID_BEGIN = "<|sid_begin|>"
SID_END = "<|sid_end|>"
SID_RE = re.compile(r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")
CODE_MULTIPLIER_1 = 8192 * 8192
CODE_MULTIPLIER_2 = 8192

RECALL_TASKS = ["video", "ad", "product", "label_cond", "interactive"]
AUC_TASKS = ["label_pred"]
LLM_SCORE_TASKS = ["item_understand", "rec_reason"]
ALL_TASKS = RECALL_TASKS + AUC_TASKS + LLM_SCORE_TASKS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official-style RecIF evaluation")
    parser.add_argument("--model", type=str, default="OpenOneRec/OneRec-8B")
    parser.add_argument("--adapter", type=str, default="./checkpoints/arec2-dpo-8B-r16-v2/final", help="Optional LoRA adapter to merge before eval")
    parser.add_argument("--benchmark-dir", type=str, default="./data/recif/benchmark_data")
    parser.add_argument("--tasks", type=str, nargs="*", default=None, help="Default: all RecIF tasks")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16, help="HF beam-search batch size")
    parser.add_argument("--num-beams", type=int, default=128,
                        help="Beam width for SID generation (official: max(16, 128)=128)")
    parser.add_argument("--num-return-sequences", type=int, default=128,
                        help="Number of SID candidates to return (official: 128)")
    parser.add_argument("--beam-chunk-size", type=int, default=8,
                        help="Process beams in chunks of this size to limit memory. "
                             "Results are exact regardless of chunk size.")
    parser.add_argument("--max-input-tokens", type=int, default=None, help="Default: no truncation, same as official eval")
    parser.add_argument("--rec-reason-batch-size", type=int, default=8,
                        help="Batch size for rec_reason generation. Default: 1 to avoid long-context OOM.")
    parser.add_argument("--rec-reason-max-input-tokens", type=int, default=8192,
                        help="Left-truncate rec_reason prompts to this many tokens. 0 disables truncation.")
    parser.add_argument("--rec-reason-max-new-tokens", type=int, default=2000,
                        help="Max new tokens for rec_reason generation.")
    parser.add_argument("--rec-reason-oom-fallback-input-tokens", type=int, default=4096,
                        help="Fallback left-truncation length for a rec_reason sample after OOM. 0 disables truncation.")
    parser.add_argument("--rec-reason-oom-fallback-new-tokens", type=int, default=1024,
                        help="Fallback max new tokens for a rec_reason sample after OOM.")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge", type=str, choices=["true", "false"], default="true",
                        help="Run official external LLM judge for item_understand/rec_reason")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--judge-config", type=str,
                        default="./configs/judge_llm_config.json",
                        help="Path to judge LLM config (OpenAI-compatible format)")
    parser.add_argument("--enrich", type=str, choices=["true", "false"], default="true",
                        help="Kept for compatibility. true changes the benchmark prompt and is not paper-comparable.")
    return parser.parse_args()


args = parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

selected_tasks = args.tasks if args.tasks else ALL_TASKS
benchmark_dir = Path(args.benchmark_dir)


def load_benchmark() -> dict[str, pd.DataFrame]:
    print(f"Loading benchmark data from {benchmark_dir} ...")
    data: dict[str, pd.DataFrame] = {}
    for task in selected_tasks:
        test_file = benchmark_dir / task / f"{task}_test.parquet"
        if not test_file.exists():
            print(f"  [WARN] missing {test_file}, skip {task}")
            continue
        df = pd.read_parquet(test_file)
        if args.max_samples:
            df = df.head(args.max_samples)
        data[task] = df
        print(f"  {task}: {len(df)} samples")
    return data


def resolve_model_path(model_arg: str) -> str:
    p = Path(model_arg)
    return str(p.resolve()) if p.exists() else model_arg


def load_model_and_tokenizer():
    model_path = resolve_model_path(args.model)
    print(f"\nLoading model: {model_path}")

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    template_path = PROJECT_ROOT / "configs" / "qwen3_soft_switch.jinja2"
    if template_path.exists():
        tok.chat_template = template_path.read_text(encoding="utf-8")
        print(f"Loaded chat template: {template_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if args.adapter:
        from peft import PeftModel
        print(f"Loading LoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()

    model.eval()
    print(f"Model loaded, parameters: {model.num_parameters():,}")
    return tok, model


_t0 = time.time()
benchmark_data = load_benchmark()
_time_benchmark_load = time.time() - _t0

_t0 = time.time()
tokenizer, model = load_model_and_tokenizer()
_time_model_load = time.time() - _t0

print(f"\n[Timing] Benchmark load: {_time_benchmark_load:.2f}s, Model load: {_time_model_load:.2f}s")


def model_device() -> torch.device:
    return next(model.parameters()).device


def chunked(items: list[Any], n: int):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def parse_json_maybe(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def flatten_messages(raw_messages: str | list[dict[str, Any]]) -> list[dict[str, str]]:
    messages = parse_json_maybe(raw_messages)
    formatted = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        formatted.append({"role": msg.get("role", "user"), "content": str(content)})
    return formatted


def build_prompt(raw_messages: str | list[dict[str, Any]], enable_thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        flatten_messages(raw_messages),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def get_metadata(row: pd.Series) -> dict[str, Any]:
    return parse_json_maybe(row.get("metadata", "{}"))


def clean_decoded(text: str) -> str:
    for ctrl in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
        text = text.replace(ctrl, "")
    return text.strip()


def decode_new_tokens(sequence: torch.Tensor, prompt_len: int) -> str:
    token_ids = sequence[prompt_len:].tolist()
    return clean_decoded(tokenizer.decode(token_ids, skip_special_tokens=False))


def normalize_token_limit(value: int | None) -> int | None:
    """Treat None/0 as no truncation for per-call token limits."""
    if value is None or value <= 0:
        return None
    return value


def tokenize_prompts(prompts: list[str], max_input_tokens: int | None = None):
    kwargs = {
        "return_tensors": "pt",
        "padding": True,
    }
    token_limit = normalize_token_limit(max_input_tokens if max_input_tokens is not None else args.max_input_tokens)
    old_truncation_side = tokenizer.truncation_side
    if token_limit:
        kwargs.update({"truncation": True, "max_length": token_limit})
        tokenizer.truncation_side = "left"
    try:
        return tokenizer(prompts, **kwargs).to(model_device())
    finally:
        tokenizer.truncation_side = old_truncation_side


def tokenize_single(prompt: str, max_input_tokens: int | None = None):
    kwargs = {"return_tensors": "pt"}
    token_limit = normalize_token_limit(max_input_tokens if max_input_tokens is not None else args.max_input_tokens)
    old_truncation_side = tokenizer.truncation_side
    if token_limit:
        kwargs.update({"truncation": True, "max_length": token_limit})
        tokenizer.truncation_side = "left"
    try:
        return tokenizer(prompt, **kwargs).to(model_device())
    finally:
        tokenizer.truncation_side = old_truncation_side


def extract_sid(text: str) -> str:
    """Return the first SID body, e.g. ``<s_a_1><s_b_2><s_c_3>``."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    if SID_BEGIN in text:
        after = text.split(SID_BEGIN, 1)[1]
        if SID_END in after:
            text = after.split(SID_END, 1)[0]
    match = SID_RE.search(text)
    if not match:
        return ""
    a, b, c = match.groups()
    return f"<s_a_{a}><s_b_{b}><s_c_{c}>"


def extract_sids(text: str) -> list[str]:
    seen = set()
    result = []
    for a, b, c in SID_RE.findall(text):
        sid = f"<s_a_{a}><s_b_{b}><s_c_{c}>"
        if sid not in seen:
            seen.add(sid)
            result.append(sid)
    return result


def sid_to_code(sid: str) -> int | None:
    match = SID_RE.search(sid)
    if not match:
        return None
    a, b, c = (int(x) for x in match.groups())
    return a * CODE_MULTIPLIER_1 + b * CODE_MULTIPLIER_2 + c


def load_sid_pid_mapping(task_name: str) -> dict[int, list[dict[str, int]]]:
    filename = "sid2iid.json" if task_name == "product" else "sid2pid.json"
    path = benchmark_dir / filename
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


def choose_pid(info_list: list[dict[str, int]]) -> int:
    if not info_list:
        return 0
    max_count = max(info.get("count_after_downsample", 0) for info in info_list)
    tied = [info.get("pid", info.get("iid", 0)) for info in info_list if info.get("count_after_downsample", 0) == max_count]
    return random.choice(tied) if tied else 0


def sid_to_pid(sid: str, mapping: dict[int, list[dict[str, int]]]) -> int:
    code = sid_to_code(sid)
    if code is None:
        return 0
    return choose_pid(mapping.get(code, []))


def unique_nonzero(values: list[int]) -> list[int]:
    seen = set()
    result = []
    for value in values:
        value = int(value)
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def compute_ranking_metrics(pred_ids: list[Any], gt_ids: list[Any], k_values=(1, 32)) -> dict[str, float | bool]:
    gt = list(dict.fromkeys(gt_ids))
    gt_set = set(gt)
    first_gt = gt[0] if gt else None
    metrics: dict[str, float | bool] = {}
    for k in k_values:
        top_k = [x for x in pred_ids[:k] if x]
        top_k_set = set(top_k)
        metrics[f"pass@{k}"] = bool(top_k_set & gt_set)
        metrics[f"position1_pass@{k}"] = bool(first_gt is not None and first_gt in top_k_set)
        metrics[f"recall@{k}"] = (len(top_k_set & gt_set) / len(gt_set)) if gt_set else 0.0
    return metrics


def expand_cache(prefix_cache, repeats: int):
    """Return a NEW cache with the batch dimension repeated ``repeats`` times.

    Works with ``DynamicCache`` (transformers >= 4.38) and legacy tuple caches.
    ``repeat_interleave`` always allocates, so the original is never modified.
    """
    if hasattr(prefix_cache, "key_cache") and hasattr(prefix_cache, "value_cache"):
        try:
            from transformers.cache_utils import DynamicCache
        except ImportError:
            from transformers import DynamicCache
        new = DynamicCache()
        if hasattr(prefix_cache, "_seen_tokens"):
            new._seen_tokens = prefix_cache._seen_tokens
        new.key_cache = [k.repeat_interleave(repeats, dim=0) for k in prefix_cache.key_cache]
        new.value_cache = [v.repeat_interleave(repeats, dim=0) for v in prefix_cache.value_cache]
        return new
    if isinstance(prefix_cache, (tuple, list)):
        return tuple(
            tuple(t.repeat_interleave(repeats, dim=0) for t in layer)
            for layer in prefix_cache
        )
    raise TypeError(f"Unsupported cache type: {type(prefix_cache)}")


def custom_three_token_beam_search(prompt: str, beam_width: int, chunk_size: int = 32) -> list[str]:
    """Memory-efficient EXACT beam search for 3 SID tokens.

    Identical results to the previous implementation, but never materializes a
    full (beam_width, vocab) log-prob tensor. Within each chunk we immediately
    top-k-truncate to `beam_width` candidate tokens per beam, so the peak memory
    is O(beam_width^2) instead of O(beam_width * vocab). This is exact because
    the global top-`beam_width` extension of any beam must lie within that beam's
    own top-`beam_width` next tokens.

    Matches the official OpenOneRec vLLM evaluation:
      beam_width = max(num_beams=16, num_return_sequences=128) = 128
    """
    device = model_device()
    W = beam_width

    # Phase 1 — prefill prompt + <|sid_begin|>, get top-W first-token scores
    inputs = tokenize_single(prompt + SID_BEGIN)
    with torch.no_grad():
        out0 = model(**inputs, use_cache=True)
    prefix_cache = out0.past_key_values
    lp0 = torch.log_softmax(out0.logits[0, -1, :].float(), dim=-1)
    scores_1, tokens_1 = torch.topk(lp0, k=W)           # (W,), (W,)
    del out0, lp0

    # Phase 2 — for every first token, keep only top-W second tokens (chunked).
    # We store per-beam (value, token-id) instead of the full vocab row.
    s2_vals_chunks: list[torch.Tensor] = []   # each (chunk, W)
    s2_toks_chunks: list[torch.Tensor] = []   # each (chunk, W)
    for s in range(0, W, chunk_size):
        e = min(s + chunk_size, W)
        chunk_tok = tokens_1[s:e]
        cs = e - s
        cache_c = expand_cache(prefix_cache, cs)
        with torch.no_grad():
            o1 = model(input_ids=chunk_tok.unsqueeze(1),
                       past_key_values=cache_c, use_cache=True)
        lp = torch.log_softmax(o1.logits[:, -1, :].float(), dim=-1)  # (cs, V)
        tv, tt = torch.topk(lp, k=W, dim=-1)                          # (cs, W) each
        s2_vals_chunks.append(tv)
        s2_toks_chunks.append(tt)
        del cache_c, o1, lp, tv, tt

    s2_vals = torch.cat(s2_vals_chunks, dim=0)   # (W, W)
    s2_toks = torch.cat(s2_toks_chunks, dim=0)   # (W, W)
    del s2_vals_chunks, s2_toks_chunks

    # Combine first-token score with each beam's top-W second tokens.
    combined_2 = scores_1.unsqueeze(1) + s2_vals          # (W, W)
    flat_2, idx_2 = torch.topk(combined_2.reshape(-1), k=W)
    parent_2 = idx_2.div(W, rounding_mode="floor")        # which first-token beam
    local_2 = idx_2.remainder(W)                          # which of its top-W second tokens
    sel_t1 = tokens_1[parent_2]                           # (W,) first tokens
    tok_2 = s2_toks[parent_2, local_2]                    # (W,) second tokens
    del combined_2, s2_vals, s2_toks

    # Phase 3 — for every (t1, t2) pair, keep only top-W third tokens (chunked).
    s3_vals_chunks: list[torch.Tensor] = []
    s3_toks_chunks: list[torch.Tensor] = []
    for s in range(0, W, chunk_size):
        e = min(s + chunk_size, W)
        cs = e - s
        t1c = sel_t1[s:e]
        t2c = tok_2[s:e]
        cache_c = expand_cache(prefix_cache, cs)
        with torch.no_grad():
            oa = model(input_ids=t1c.unsqueeze(1),
                       past_key_values=cache_c, use_cache=True)
            ob = model(input_ids=t2c.unsqueeze(1),
                       past_key_values=oa.past_key_values, use_cache=True)
        lp = torch.log_softmax(ob.logits[:, -1, :].float(), dim=-1)   # (cs, V)
        tv, tt = torch.topk(lp, k=W, dim=-1)                           # (cs, W)
        s3_vals_chunks.append(tv)
        s3_toks_chunks.append(tt)
        del cache_c, oa, ob, lp, tv, tt

    s3_vals = torch.cat(s3_vals_chunks, dim=0)   # (W, W)
    s3_toks = torch.cat(s3_toks_chunks, dim=0)   # (W, W)
    del s3_vals_chunks, s3_toks_chunks

    combined_3 = flat_2.unsqueeze(1) + s3_vals            # (W, W)
    flat_3, idx_3 = torch.topk(combined_3.reshape(-1), k=W)
    parent_3 = idx_3.div(W, rounding_mode="floor")
    local_3 = idx_3.remainder(W)

    final_t1 = sel_t1[parent_3]
    final_t2 = tok_2[parent_3]
    final_t3 = s3_toks[parent_3, local_3]
    del prefix_cache, combined_3, s3_vals, s3_toks

    return [
        clean_decoded(tokenizer.decode(
            [t1.item(), t2.item(), t3.item()], skip_special_tokens=False))
        for t1, t2, t3 in zip(final_t1, final_t2, final_t3)
    ]


def batch_generate_recommendations(prompts: list[str], _sample_times: list[float] | None = None) -> list[list[str]]:
    if args.num_return_sequences != args.num_beams:
        raise ValueError("The custom RecIF beam search expects --num-return-sequences == --num-beams")
    results = []
    for prompt in prompts:
        t0 = time.time()
        r = custom_three_token_beam_search(prompt, args.num_beams, args.beam_chunk_size)
        if _sample_times is not None:
            _sample_times.append(time.time() - t0)
        results.append(r)
    return results


def evaluate_recall_task(task_name: str) -> dict[str, Any] | None:
    df = benchmark_data.get(task_name)
    if df is None:
        return None

    print(f"\n--- {task_name}: official-style beam candidates, n={len(df)} ---")

    t_prep = time.time()
    prompts = [build_prompt(row["messages"], enable_thinking=False) for _, row in df.iterrows()]
    metadata = [get_metadata(row) for _, row in df.iterrows()]
    t_prompt_build = time.time() - t_prep

    t_map = time.time()
    sid_pid_mapping = load_sid_pid_mapping(task_name)
    t_mapping_load = time.time() - t_map

    print(f"  Prep: prompt_build={t_prompt_build:.2f}s, sid_pid_mapping={t_mapping_load:.2f}s")

    sid_metric_sums = {name: 0.0 for name in ("pass@1", "position1_pass@1", "recall@1", "pass@32", "position1_pass@32", "recall@32")}
    pid_metric_sums = {f"pid_{name}": 0.0 for name in sid_metric_sums}
    total_samples = len(df)
    sample_times: list[float] = []

    for batch_indices in tqdm(list(chunked(list(range(len(prompts))), args.batch_size)), desc=task_name):
        batch_prompts = [prompts[i] for i in batch_indices]
        batch_generations = batch_generate_recommendations(batch_prompts, _sample_times=sample_times)

        for idx, generations in zip(batch_indices, batch_generations):
            pred_sids = [extract_sid(gen) for gen in generations]
            gt_sids = extract_sids(metadata[idx].get("answer", ""))
            if not gt_sids:
                continue

            sid_metrics = compute_ranking_metrics(pred_sids, gt_sids)
            for key, value in sid_metrics.items():
                sid_metric_sums[key] += float(value)

            if sid_pid_mapping:
                pred_pids = [sid_to_pid(sid, sid_pid_mapping) for sid in pred_sids]
                answer_key = "answer_iid" if task_name == "product" else "answer_pid"
                gt_pids = unique_nonzero(metadata[idx].get(answer_key, []))
                pid_metrics = compute_ranking_metrics(pred_pids, gt_pids)
                for key, value in pid_metrics.items():
                    pid_metric_sums[f"pid_{key}"] += float(value)

    results = {key: round(value / total_samples, 4) for key, value in sid_metric_sums.items()} if total_samples else {}
    if sid_pid_mapping and total_samples:
        results.update({key: round(value / total_samples, 4) for key, value in pid_metric_sums.items()})
    results["num_samples"] = total_samples
    if sample_times:
        for i, dt in enumerate(sample_times):
            print(f"    [{task_name}] sample {i}/{total_samples}: {dt:.3f}s")
        avg_t = sum(sample_times) / len(sample_times)
        print(f"  Inference avg: {avg_t:.3f}s/sample, total: {sum(sample_times):.1f}s")
    results["_timing"] = {
        "prompt_build": round(t_prompt_build, 3),
        "sid_pid_mapping": round(t_mapping_load, 3),
        "inference_total": round(sum(sample_times), 3),
        "inference_avg": round(sum(sample_times) / len(sample_times), 3) if sample_times else 0,
    }
    print(f"  pass@1={results.get('pass@1', 0):.4f}, pass@32={results.get('pass@32', 0):.4f}, recall@32={results.get('recall@32', 0):.4f}, total={total_samples}")
    return results


def target_token_id(token: str) -> int:
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) != 1:
        print(f"[WARN] token {token!r} encoded to {token_ids}; using first token")
    return token_ids[0]


def batch_label_scores(prompts: list[str]) -> list[float]:
    pos_id = target_token_id("是")
    neg_id = target_token_id("否")
    inputs = tokenize_prompts(prompts)
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]
    pair_logits = torch.stack([logits[:, pos_id], logits[:, neg_id]], dim=-1).float()
    probs = torch.softmax(pair_logits, dim=-1)[:, 0]
    return probs.detach().cpu().tolist()


def evaluate_auc_task(task_name: str) -> dict[str, Any] | None:
    df = benchmark_data.get(task_name)
    if df is None:
        return None

    print(f"\n--- {task_name}: next-token 是/否 AUC, n={len(df)} ---")

    t_prep = time.time()
    prompts = [build_prompt(row["messages"], enable_thinking=False) for _, row in df.iterrows()]
    labels = [1 if get_metadata(row).get("answer", "").strip() == "是" else 0 for _, row in df.iterrows()]
    t_prompt_build = time.time() - t_prep
    print(f"  Prep: prompt_build={t_prompt_build:.2f}s")

    scores = []
    sample_times: list[float] = []
    for batch_indices in tqdm(list(chunked(list(range(len(prompts))), max(args.batch_size, 1))), desc=task_name):
        batch_prompts = [prompts[i] for i in batch_indices]
        t_inf = time.time()
        scores.extend(batch_label_scores(batch_prompts))
        dt = time.time() - t_inf
        per_sample = dt / len(batch_indices)
        for i in batch_indices:
            sample_times.append(per_sample)
            print(f"    [{task_name}] sample {i}/{len(prompts)}: ~{per_sample:.3f}s")

    auc = roc_auc_score(labels, scores) if len(set(labels)) == 2 else 0.5
    if sample_times:
        avg_t = sum(sample_times) / len(sample_times)
        print(f"  Inference avg: {avg_t:.3f}s/sample, total: {sum(sample_times):.1f}s")
    print(f"  AUC={auc:.4f}, positives={sum(labels)}, negatives={len(labels) - sum(labels)}")
    return {
        "auc": round(float(auc), 4),
        "num_samples": len(labels),
        "_timing": {
            "prompt_build": round(t_prompt_build, 3),
            "inference_total": round(sum(sample_times), 3),
            "inference_avg": round(sum(sample_times) / len(sample_times), 3) if sample_times else 0,
        },
    }


def clear_cuda_memory() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        torch.cuda.empty_cache()


def is_oom_error(exc: BaseException) -> bool:
    oom_type = getattr(torch, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: memory" in message


def batch_generate_text(
    prompts: list[str],
    max_new_tokens: int,
    do_sample: bool,
    repetition_penalty: float = 1.0,
    max_input_tokens: int | None = None,
    use_cache: bool = True,
) -> list[str]:
    inputs = None
    outputs = None
    inputs = tokenize_prompts(prompts, max_input_tokens=max_input_tokens)
    prompt_len = inputs["input_ids"].shape[1]
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_return_sequences": 1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": repetition_penalty,
        "use_cache": use_cache,
    }
    if do_sample:
        gen_kwargs.update({"temperature": 0.7, "top_p": 0.95})
    try:
        with torch.inference_mode():
            outputs = model.generate(**inputs, **gen_kwargs)
        return [decode_new_tokens(seq, prompt_len) for seq in outputs]
    finally:
        del outputs
        del inputs
        clear_cuda_memory()


def oom_safe_batch_generate_text(
    prompts: list[str],
    max_new_tokens: int,
    do_sample: bool,
    repetition_penalty: float,
    max_input_tokens: int | None,
    fallback_input_tokens: int | None,
    fallback_new_tokens: int,
) -> list[str]:
    try:
        return batch_generate_text(
            prompts,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            max_input_tokens=max_input_tokens,
        )
    except Exception as exc:
        if not is_oom_error(exc):
            raise
        clear_cuda_memory()
        if len(prompts) > 1:
            print(f"  [WARN] OOM for batch of {len(prompts)} rec_reason prompts; splitting batch.")
            mid = len(prompts) // 2
            return (
                oom_safe_batch_generate_text(
                    prompts[:mid],
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    max_input_tokens=max_input_tokens,
                    fallback_input_tokens=fallback_input_tokens,
                    fallback_new_tokens=fallback_new_tokens,
                )
                + oom_safe_batch_generate_text(
                    prompts[mid:],
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    max_input_tokens=max_input_tokens,
                    fallback_input_tokens=fallback_input_tokens,
                    fallback_new_tokens=fallback_new_tokens,
                )
            )

        print(
            "  [WARN] OOM for one rec_reason prompt; retrying with "
            f"max_input_tokens={fallback_input_tokens or 'none'}, max_new_tokens={fallback_new_tokens}."
        )
        try:
            return batch_generate_text(
                prompts,
                max_new_tokens=fallback_new_tokens,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                max_input_tokens=fallback_input_tokens,
            )
        except Exception as fallback_exc:
            if not is_oom_error(fallback_exc):
                raise
            clear_cuda_memory()

        print("  [WARN] rec_reason fallback still OOM; retrying with use_cache=False.")
        try:
            return batch_generate_text(
                prompts,
                max_new_tokens=fallback_new_tokens,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                max_input_tokens=fallback_input_tokens,
                use_cache=False,
            )
        except Exception as final_exc:
            if not is_oom_error(final_exc):
                raise
            clear_cuda_memory()
            print("  [WARN] rec_reason sample failed after OOM fallbacks; recording empty prediction.")
            return ["" for _ in prompts]


def generate_llm_task(task_name: str) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    df = benchmark_data[task_name]
    enable_thinking = task_name == "rec_reason"
    max_new_tokens = args.rec_reason_max_new_tokens if task_name == "rec_reason" else 128
    repetition_penalty = 1.1 if task_name == "rec_reason" else 1.0
    batch_size = max(args.rec_reason_batch_size if task_name == "rec_reason" else args.batch_size, 1)
    max_input_tokens = (
        normalize_token_limit(args.rec_reason_max_input_tokens)
        if task_name == "rec_reason" else None
    )
    fallback_input_tokens = normalize_token_limit(args.rec_reason_oom_fallback_input_tokens)
    fallback_new_tokens = max(args.rec_reason_oom_fallback_new_tokens, 1)

    t_prep = time.time()
    prompts = [build_prompt(row["messages"], enable_thinking=enable_thinking) for _, row in df.iterrows()]
    refs = {str(i): get_metadata(row).get("answer", "") for i, (_, row) in enumerate(df.iterrows())}
    t_prompt_build = time.time() - t_prep
    print(f"  Prep: prompt_build={t_prompt_build:.2f}s")

    preds: dict[str, str] = {}
    sample_times: list[float] = []

    if task_name == "rec_reason":
        print(
            "  rec_reason memory guard: "
            f"batch_size={batch_size}, max_input_tokens={max_input_tokens or 'none'}, "
            f"max_new_tokens={max_new_tokens}, fallback_input_tokens={fallback_input_tokens or 'none'}, "
            f"fallback_new_tokens={fallback_new_tokens}"
        )

    for batch_indices in tqdm(list(chunked(list(range(len(prompts))), batch_size)), desc=f"{task_name}-gen"):
        batch_prompts = [prompts[i] for i in batch_indices]
        t_inf = time.time()
        if task_name == "rec_reason":
            texts = oom_safe_batch_generate_text(
                batch_prompts,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=repetition_penalty,
                max_input_tokens=max_input_tokens,
                fallback_input_tokens=fallback_input_tokens,
                fallback_new_tokens=fallback_new_tokens,
            )
        else:
            texts = batch_generate_text(
                batch_prompts,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=repetition_penalty,
            )
        dt = time.time() - t_inf
        per_sample = dt / len(batch_indices)
        for idx, text in zip(batch_indices, texts):
            sample_times.append(per_sample)
            print(f"    [{task_name}] sample {idx}/{len(prompts)}: ~{per_sample:.3f}s")
            preds[str(idx)] = text

    if sample_times:
        avg_t = sum(sample_times) / len(sample_times)
        print(f"  Inference avg: {avg_t:.3f}s/sample, total: {sum(sample_times):.1f}s")
    timing = {
        "prompt_build": round(t_prompt_build, 3),
        "inference_total": round(sum(sample_times), 3),
        "inference_avg": round(sum(sample_times) / len(sample_times), 3) if sample_times else 0,
    }
    return preds, refs, timing


class _OpenAIJudgeClient:
    """Minimal OpenAI-compatible judge client with retry, matching the
    official ``BaseLLMClient`` interface (``generate(prompt) -> str``)."""

    def __init__(self, api_key: str, base_url: str, model_name: str, *,
                 max_new_tokens: int = 10000, temperature: float = 0.01,
                 max_retries: int = 3, retry_delay: int = 2, **_kw):
        from openai import OpenAI
        self.model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_new_tokens
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, temperature: float | None = None,
                 max_tokens: int | None = None, **_kw) -> str:
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                if attempt > 0:
                    delay = self._retry_delay * (2 ** (attempt - 1))
                    time.sleep(delay + random.uniform(0, delay * 0.3))
                resp = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature if temperature is not None else self._temperature,
                    max_tokens=max_tokens or self._max_tokens,
                )
                content = resp.choices[0].message.content
                if content:
                    return content
                raise RuntimeError("API returned empty response")
            except Exception as exc:
                last_err = exc
                msg = str(exc).lower()
                retryable = any(k in msg for k in ("503", "429", "500", "timeout", "rate limit"))
                if attempt == self._max_retries - 1 or not retryable:
                    raise
        raise RuntimeError(f"Max retries reached: {last_err}")


def _load_judge_client(judge_model: str, config_path: Path) -> _OpenAIJudgeClient:
    """Create judge client from project-local config (OpenAI-compatible)."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"Judge config not found: {config_path}\n"
            f"Create it with api_key / base_url / model_name fields."
        )
    all_config = json.loads(config_path.read_text(encoding="utf-8"))
    model_key = judge_model.lower()
    if model_key not in all_config:
        raise ValueError(
            f"Judge model '{model_key}' not in {config_path}. "
            f"Available: {', '.join(all_config.keys())}"
        )
    return _OpenAIJudgeClient(**all_config[model_key])


_LLM_JUDGE_PROMPT = """You are an expert evaluator. Rate the quality of the following response on a scale of 1-5.

**Task**: {task_desc}

**Ground Truth Reference**:
{reference}

**Model Response**:
{response}

Rate the response quality (1=very poor, 2=poor, 3=acceptable, 4=good, 5=excellent).
Consider: accuracy, relevance, completeness, and coherence.

Score (just the number):"""

_TASK_DESCRIPTIONS = {
    "item_understand": "Understand and describe the given item based on its content and attributes.",
    "rec_reason": "Explain why this item is recommended to the user based on their preferences and history.",
}


def _judge_score_samples(
    preds: dict[str, str],
    refs: dict[str, str],
    judge_client: _OpenAIJudgeClient,
    task_name: str,
    save_dir: Path,
) -> dict[str, Any]:
    task_desc = _TASK_DESCRIPTIONS.get(task_name, "")
    scores: list[int] = []
    details: list[dict[str, Any]] = []

    for key in sorted(preds.keys(), key=int):
        pred = preds[key][:500]
        ref = refs.get(key, "")[:500]
        prompt = _LLM_JUDGE_PROMPT.format(task_desc=task_desc, reference=ref, response=pred)
        try:
            judge_reply = judge_client.generate(prompt)
            match = re.search(r"[1-5]", judge_reply)
            score = int(match.group()) if match else 3
        except Exception as exc:
            print(f"  [WARN] judge failed for sample {key}: {exc}")
            score = 3
        scores.append(score)
        details.append({"key": key, "score": score, "pred": pred, "ref": ref})

    avg = float(np.mean(scores)) if scores else 0.0
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "judge_details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  {task_name} LLM Score = {avg:.4f} ({len(scores)} samples)")

    if task_name == "rec_reason":
        return {"llm_score": round(avg, 4)}
    return {"macro_wip_double_weighted_f1": round(avg, 4)}


def evaluate_llm_task(task_name: str, api=None) -> dict[str, Any] | None:
    if task_name not in benchmark_data:
        return None

    print(f"\n--- {task_name}: generation with optional external judge ---")
    preds, refs, gen_timing = generate_llm_task(task_name)
    result: dict[str, Any] = {"num_samples": len(preds), "judge_enabled": args.judge == "true", "_timing": gen_timing}

    if args.judge != "true":
        print("  Judge disabled. Score requires --judge true and judge LLM config.")
        result["note"] = "generated only; external judge not run"
        return result

    save_dir = PROJECT_ROOT / "results" / "official_judge" / task_name

    try:
        judge_config_path = Path(args.judge_config)
        if not judge_config_path.is_absolute():
            judge_config_path = PROJECT_ROOT / judge_config_path
        judge_client = _load_judge_client(args.judge_model, judge_config_path)
        metrics = _judge_score_samples(preds, refs, judge_client, task_name, save_dir)
        result.update(metrics)
    except Exception as exc:
        print(f"  [WARN] judge failed: {exc}")
        result["judge_error"] = str(exc)

    return result


def main() -> None:
    if args.enrich == "true":
        print("[WARN] --enrich true is not paper-comparable; official RecIF eval uses raw benchmark prompts.")

    start = time.time()
    print("\n" + "=" * 72)
    print("RecIF Official-Style Evaluation")
    print(f"Model: {args.model}")
    print(f"Tasks: {selected_tasks}")
    print(f"Batch size: {args.batch_size}, beams: {args.num_beams}, returns: {args.num_return_sequences}")
    print("=" * 72)

    all_results: dict[str, Any] = {}
    for task in selected_tasks:
        if task in RECALL_TASKS:
            res = evaluate_recall_task(task)
        elif task in AUC_TASKS:
            res = evaluate_auc_task(task)
        elif task in LLM_SCORE_TASKS:
            res = evaluate_llm_task(task)
        else:
            print(f"[WARN] unknown task {task}, skipped")
            res = None
        if res is not None:
            all_results[task] = res

    elapsed = time.time() - start
    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    for task, res in all_results.items():
        if task in RECALL_TASKS:
            print(
                f"{task:<16} pass@1={res.get('pass@1', 0):.4f} "
                f"pass@32={res.get('pass@32', 0):.4f} "
                f"recall@32={res.get('recall@32', 0):.4f} "
                f"samples={res.get('num_samples', 0)}"
            )
        elif task == "label_pred":
            print(f"{task:<16} auc={res.get('auc', 0):.4f} samples={res.get('num_samples', 0)}")
        elif task == "rec_reason":
            print(f"{task:<16} llm_score={res.get('llm_score', 'NA')} samples={res.get('num_samples', 0)}")
        else:
            print(
                f"{task:<16} macro_wip_double_weighted_f1="
                f"{res.get('macro_wip_double_weighted_f1', 'NA')} samples={res.get('num_samples', 0)}"
            )

    print("\n" + "-" * 72)
    print("Timing Breakdown")
    print("-" * 72)
    print(f"  Benchmark data load:    {_time_benchmark_load:.2f}s")
    print(f"  Model + tokenizer load: {_time_model_load:.2f}s")
    for task, res in all_results.items():
        t = res.get("_timing", {})
        if not t:
            continue
        parts = [f"prompt_build={t.get('prompt_build', 0):.3f}s"]
        if "sid_pid_mapping" in t:
            parts.append(f"sid_pid_mapping={t['sid_pid_mapping']:.3f}s")
        parts.append(f"inference_total={t.get('inference_total', 0):.1f}s")
        parts.append(f"inference_avg={t.get('inference_avg', 0):.3f}s/sample")
        print(f"  {task:<16} {', '.join(parts)}")
    print(f"  Total wall time:        {elapsed:.1f}s")
    print("-" * 72)

    output_path = Path(args.output or f"results/eval_official_{Path(args.model).name.replace('/', '_')}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timing_summary = {
        "benchmark_load": round(_time_benchmark_load, 2),
        "model_load": round(_time_model_load, 2),
        "per_task": {task: res["_timing"] for task, res in all_results.items() if "_timing" in res},
    }
    serializable_results = {task: {k: v for k, v in res.items() if k != "_timing"} for task, res in all_results.items()}
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "adapter": args.adapter,
                "official_style": True,
                "elapsed_seconds": round(elapsed, 1),
                "timing": timing_summary,
                "tasks": serializable_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
