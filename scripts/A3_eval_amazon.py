"""Stage A3 (Amazon): evaluate transfer performance (Recall@K, NDCG@K).

Implements the §6.1 Amazon protocol: leave-one-out, report Recall@{5,10} and
NDCG@{5,10}. Predictions are produced by candidate-constrained scoring: for
each test user we score every in-corpus item's itemic-token string under the
model conditioned on the user's Strategy-3 history, then rank by log-prob.

Because scoring all items for all users is expensive on large domains, we use
the standard sampled-candidate protocol: rank the ground-truth target against
``--num-negatives`` sampled negatives (default 99 -> the common 1+99 setup).
Set ``--full-ranking`` to rank against the entire item corpus (slow, exact).

The candidate STRINGS are itemic-tokens-only (target_repr), so matching is by
canonical SID equality, consistent with arec2.amazon.metrics.

Usage:
    python scripts/A3_eval_amazon.py \
        --base-model ./models/arec2-merged \
        --adapter ./checkpoints/amazon/strategy3_joint/final \
        --categories Baby Beauty Toys_and_Games
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from arec2.amazon.data_amazon import AMAZON_CATEGORIES
from arec2.amazon.dataset import SYSTEM_PROMPT, USER_TEMPLATE
from arec2.amazon.item_repr import Strategy3Representer, itemic_token_str
from arec2.amazon.metrics import MetricAccumulator, canonical_sid
from arec2.base_model.openonerec_wrapper import TOKENS_PER_SID, ScoredCandidate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("A3_eval_amazon")


def _expand_kv_cache(cache, batch_size: int):
    """Expand a batch-1 KV cache to *batch_size* via .expand (no copy)."""
    if hasattr(cache, "key_cache"):
        from transformers.cache_utils import DynamicCache
        expanded = DynamicCache()
        for i in range(len(cache.key_cache)):
            expanded.update(
                cache.key_cache[i].expand(batch_size, -1, -1, -1),
                cache.value_cache[i].expand(batch_size, -1, -1, -1),
                i,
            )
        return expanded
    return tuple(
        (k.expand(batch_size, -1, -1, -1), v.expand(batch_size, -1, -1, -1))
        for k, v in cache
    )


def score_candidates_batched(
    wrapper,
    messages: list[dict],
    candidate_sids: list[str],
    candidate_batch_size: int = 0,
    enable_thinking: bool = False,
) -> list[ScoredCandidate]:
    """Drop-in replacement for wrapper.score_candidates with batched forward passes."""
    tokenizer = wrapper.tokenizer
    model = wrapper.model

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    prompt_ids = tokenizer.encode(text, add_special_tokens=False)

    all_cand_token_ids: list[list[int]] = []
    for sid_str in candidate_sids:
        toks = tokenizer.encode(sid_str, add_special_tokens=False)
        assert len(toks) == TOKENS_PER_SID
        all_cand_token_ids.append(toks)

    N = len(all_cand_token_ids)
    if N == 0:
        return []

    input_ids = torch.tensor([prompt_ids], device=model.device)
    with torch.no_grad():
        prefix_out = model(input_ids=input_ids, use_cache=True)
    prefix_cache = prefix_out.past_key_values
    last_logits = prefix_out.logits[0, -1, :]  # [V]

    cand_ids = torch.tensor(all_cand_token_ids, device=model.device)  # [N, 5]
    bs = candidate_batch_size if candidate_batch_size > 0 else N
    log_probs_list: list[torch.Tensor] = []

    for start in range(0, N, bs):
        chunk = cand_ids[start : start + bs]  # [B, 5]
        B = chunk.shape[0]
        expanded_cache = _expand_kv_cache(prefix_cache, B)
        with torch.no_grad():
            cand_out = model(
                input_ids=chunk[:, :-1],  # [B, 4]
                past_key_values=expanded_cache,
                use_cache=False,
            )
        all_logits = torch.cat(
            [last_logits.unsqueeze(0).unsqueeze(0).expand(B, -1, -1),
             cand_out.logits],
            dim=1,
        )  # [B, 5, V]
        lp = F.log_softmax(all_logits, dim=-1)
        target_lp = lp.gather(2, chunk.unsqueeze(-1)).squeeze(-1)  # [B, 5]
        log_probs_list.append(target_lp.sum(dim=1))  # [B]

    all_log_probs = torch.cat(log_probs_list)  # [N]
    scored = [
        ScoredCandidate(sid_str=s, log_prob=lp.item())
        for s, lp in zip(candidate_sids, all_log_probs)
    ]
    scored.sort(key=lambda x: x.log_prob, reverse=True)
    return scored


def load_artifacts(cache_dir: Path, category: str) -> dict:
    path = cache_dir / category / "strategy3_artifacts.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Artifacts for '{category}' not found at {path}. Run A1 first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model(base_model: str, sft_adapter: str | None, adapter: str | None):
    """Load OneRec base, merge the AREC² SFT adapter, then the Amazon adapter.

    Loading order (each merge bakes the LoRA into the weights so the next
    PeftModel.from_pretrained starts from a clean base):
        OneRec base  ->  + AREC² SFT adapter (full_lora_8b)  ->  + Amazon adapter
    """
    from arec2.base_model.openonerec_wrapper import OpenOneRecWrapper
    from arec2.amazon.model_loader import _looks_like_adapter, resolve_model_path

    logger.info(
        "Loading model: base=%s sft_adapter=%s amazon_adapter=%s",
        base_model, sft_adapter or "(none)", adapter or "(none)",
    )
    wrapper = OpenOneRecWrapper(model_path=base_model, device="auto", torch_dtype="bfloat16")

    from peft import PeftModel

    # Align the tokenizer + embedding size with the AREC² adapter, which carries
    # the itemic-token vocab. This prevents index-out-of-range during merge if
    # the base checkpoint has a smaller vocab than the adapter expects.
    if sft_adapter and _looks_like_adapter(sft_adapter):
        from transformers import AutoTokenizer
        sft_dir_for_tok = resolve_model_path(sft_adapter)
        if (Path(sft_dir_for_tok) / "tokenizer_config.json").exists():
            wrapper.tokenizer = AutoTokenizer.from_pretrained(
                sft_dir_for_tok, trust_remote_code=True
            )
            emb_rows = wrapper.model.get_input_embeddings().weight.shape[0]
            if len(wrapper.tokenizer) > emb_rows:
                logger.info(
                    "Resizing token embeddings %d -> %d to match SFT adapter tokenizer.",
                    emb_rows, len(wrapper.tokenizer),
                )
                wrapper.model.resize_token_embeddings(len(wrapper.tokenizer))
    if sft_adapter and _looks_like_adapter(sft_adapter):
        sft_dir = resolve_model_path(sft_adapter)
        wrapper.model = PeftModel.from_pretrained(wrapper.model, sft_dir)
        wrapper.model = wrapper.model.merge_and_unload()
        logger.info("Merged AREC² SFT adapter from %s.", sft_dir)

    # 2) Merge the Amazon transfer adapter (from A2) on top, if provided.
    if adapter and _looks_like_adapter(adapter):
        amazon_dir = resolve_model_path(adapter)
        wrapper.model = PeftModel.from_pretrained(wrapper.model, amazon_dir)
        wrapper.model = wrapper.model.merge_and_unload()
        logger.info("Merged Amazon transfer adapter from %s.", amazon_dir)

    wrapper.model.eval()
    return wrapper


def build_eval_messages(rep: Strategy3Representer, domain_short: str, hist: list[str]) -> list[dict]:
    """Build the chat prompt (no assistant turn) for a test user."""
    history_str = rep.sequence_repr(hist, with_keywords=True)
    user_content = USER_TEMPLATE.format(domain=domain_short, history=history_str)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def evaluate_category(
    wrapper,
    art: dict,
    n_keywords: int,
    num_negatives: int,
    full_ranking: bool,
    max_users: int | None,
    seed: int,
    candidate_batch_size: int = 0,
) -> dict:
    rng = random.Random(seed)
    category = art["category"]
    rep = Strategy3Representer(
        asin2code=art["asin2code"],
        asin2keywords=art["asin2keywords"],
        n_keywords=n_keywords,
    )

    # Short name for the prompt's Domain field (consistent with training).
    from arec2.amazon.data_amazon import CATEGORY_SHORT
    domain_short = CATEGORY_SHORT.get(category, category)

    # Candidate string pool: every in-corpus item rendered as itemic-tokens-only.
    all_asins = art["item_ids"]
    asin2sidstr = {a: rep.target_repr(a) for a in all_asins}
    # Map canonical SID -> asin for de-dup / matching.
    sidstr_pool = [asin2sidstr[a] for a in all_asins if asin2sidstr[a]]
    sidstr_pool = list(dict.fromkeys(sidstr_pool))  # de-dup collided codes

    test_targets = art["test_targets"]
    train_hist = art["train_histories"]
    val_targets = art["val_targets"]

    users = list(test_targets.keys())
    if max_users:
        users = users[:max_users]

    acc = MetricAccumulator(ks=(5, 10))

    for uid in tqdm(users, desc=f"Eval {category}"):
        target_asin = test_targets[uid]
        target_str = asin2sidstr.get(target_asin)
        if not target_str:
            continue
        target_sid = canonical_sid(target_str)

        # Eval history = training history + validation target (standard LOO:
        # the test target is predicted from everything before it).
        hist = list(train_hist.get(uid, []))
        if uid in val_targets:
            hist = hist + [val_targets[uid]]
        if not hist:
            continue

        messages = build_eval_messages(rep, domain_short, hist)

        # Build the candidate set to score.
        if full_ranking:
            candidates = sidstr_pool
        else:
            seen_codes = {target_str}
            negs: list[str] = []
            tries = 0
            max_tries = num_negatives * 20
            while len(negs) < num_negatives and tries < max_tries:
                cand = rng.choice(sidstr_pool)
                tries += 1
                if cand in seen_codes:
                    continue
                seen_codes.add(cand)
                negs.append(cand)
            candidates = [target_str] + negs
            rng.shuffle(candidates)

        # Score candidates by log-prob (batched forward passes, KV-cached prefix).
        scored = score_candidates_batched(
            wrapper, messages, candidates,
            candidate_batch_size=candidate_batch_size, enable_thinking=False,
        )
        ranked = [canonical_sid(s.sid_str) for s in scored]
        ranked = [r for r in ranked if r is not None]

        acc.update(ranked, target_sid)

    result = acc.result()
    result["category"] = category
    result["short_name"] = domain_short
    result["protocol"] = "full" if full_ranking else f"sampled_1+{num_negatives}"
    return result


def main():
    parser = argparse.ArgumentParser(description="Amazon Strategy-3 transfer evaluation")
    parser.add_argument("--base-model", type=str, default="OpenOneRec/OneRec-8B",
                        help="OneRec base model (HF name or local path).")
    parser.add_argument("--sft-adapter", type=str, default="./full_lora_8b",
                        help="Your AREC² SFT LoRA adapter, merged into the base first. "
                             "Set '' if --base-model is already a merged model.")
    parser.add_argument("--adapter", type=str, default="./checkpoints/amazon/strategy3_joint/final",
                        help="Amazon transfer adapter from A2 (e.g. "
                             "checkpoints/amazon/strategy3_joint/final). Omit to measure "
                             "the AREC² model's ZERO-SHOT transfer (no Amazon fine-tune).")
    parser.add_argument("--cache-dir", type=str, default="./caches/amazon")
    parser.add_argument("--categories", type=str, nargs="*", default=None,
                        help="Default = all 10.")
    parser.add_argument("--n-keywords", type=int, default=5)
    parser.add_argument("--num-negatives", type=int, default=99,
                        help="Sampled negatives per user (1+N protocol). Ignored with --full-ranking.")
    parser.add_argument("--full-ranking", action="store_true",
                        help="Rank against the entire item corpus (exact but slow).")
    parser.add_argument("--candidate-batch-size", type=int, default=0,
                        help="Batch size for candidate scoring forward passes. "
                             "0 (default) = all candidates in one pass. "
                             "Reduce if OOM on full-ranking.")
    parser.add_argument("--max-users", type=int, default=None,
                        help="Cap users per category for quick runs.")
    parser.add_argument("--output", type=str, default="./results/amazon/eval.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    categories = args.categories or AMAZON_CATEGORIES
    cache_dir = Path(args.cache_dir)
    adapter = args.adapter if args.adapter else None
    sft_adapter = args.sft_adapter if args.sft_adapter else None

    wrapper = load_model(args.base_model, sft_adapter, adapter)

    t0 = time.time()
    all_results: dict[str, dict] = {}
    for cat in categories:
        try:
            art = load_artifacts(cache_dir, cat)
        except FileNotFoundError as e:
            logger.error("%s", e)
            continue
        res = evaluate_category(
            wrapper, art, args.n_keywords, args.num_negatives,
            args.full_ranking, args.max_users, args.seed,
            candidate_batch_size=args.candidate_batch_size,
        )
        all_results[cat] = res
        logger.info(
            "  %-30s R@5=%.4f R@10=%.4f N@5=%.4f N@10=%.4f (users=%d, %s)",
            cat, res["recall@5"], res["recall@10"], res["ndcg@5"], res["ndcg@10"],
            res["num_users"], res["protocol"],
        )

    # Averages across categories (paper reports per-domain + avg improvement).
    if all_results:
        avg = {}
        for m in ["recall@5", "recall@10", "ndcg@5", "ndcg@10"]:
            avg[m] = round(float(np.mean([r[m] for r in all_results.values()])), 4)
    else:
        avg = {}

    elapsed = time.time() - t0
    logger.info("=" * 80)
    logger.info("Amazon Transfer Results (Strategy 3 = Text-Augmented Itemic Tokens)")
    logger.info("  base=%s | sft_adapter=%s | amazon_adapter=%s",
                args.base_model, sft_adapter or "(none)", adapter or "(none, zero-shot)")
    logger.info("-" * 80)
    logger.info("%-28s %-8s %-8s %-8s %-8s", "Domain", "R@5", "R@10", "N@5", "N@10")
    for cat, r in all_results.items():
        logger.info("%-28s %-8.4f %-8.4f %-8.4f %-8.4f",
                    r["short_name"], r["recall@5"], r["recall@10"], r["ndcg@5"], r["ndcg@10"])
    if avg:
        logger.info("-" * 80)
        logger.info("%-28s %-8.4f %-8.4f %-8.4f %-8.4f",
                    "AVERAGE", avg["recall@5"], avg["recall@10"], avg["ndcg@5"], avg["ndcg@10"])
    logger.info("  total time: %.1f min", elapsed / 60)
    logger.info("=" * 80)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model": args.base_model,
                "sft_adapter": sft_adapter,
                "adapter": adapter,
                "strategy": "text_augmented_itemic_tokens",
                "per_domain": all_results,
                "average": avg,
                "elapsed_seconds": round(elapsed, 1),
            },
            f, indent=2, ensure_ascii=False,
        )
    logger.info("Saved results to %s", out_path)


if __name__ == "__main__":
    main()