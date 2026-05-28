"""Phase 0 smoke test: run baseline OpenOneRec-1.7B on 10 video-rec samples.

Measures:
  1. Free generation: does the model output valid SIDs?
  2. Candidate-constrained scoring: Recall@K on a candidate set.

Usage:
    python scripts/00_smoke_test.py [--n_samples 10] [--model_path OpenOneRec/OneRec-1.7B]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("smoke_test")


def load_pid2sid(path: str) -> dict[int, list[int]]:
    df = pd.read_parquet(path)
    return {row["pid"]: row["sid"] for _, row in df.iterrows()}


def build_candidate_pool(
    gt_sids: list[str],
    all_sid_strs: list[str],
    pool_size: int = 64,
) -> list[str]:
    """Build a candidate pool: ground-truth SIDs + random negatives."""
    import random

    pool = set(gt_sids)
    candidates = list(set(all_sid_strs) - pool)
    n_neg = min(pool_size - len(pool), len(candidates))
    pool.update(random.sample(candidates, n_neg))
    return list(pool)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--model_path", type=str, default="OpenOneRec/OneRec-1.7B")
    parser.add_argument("--task", type=str, default="video")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)

    from arec2.base_model.openonerec_wrapper import OpenOneRecWrapper

    logger.info("Loading model from %s ...", args.model_path)
    t0 = time.time()
    wrapper = OpenOneRecWrapper(model_path=args.model_path)
    logger.info("Model loaded in %.1fs", time.time() - t0)

    task_path = f"data/recif/benchmark_data/{args.task}/{args.task}_test.parquet"
    df = pd.read_parquet(task_path)
    logger.info("Loaded %d samples from %s", len(df), task_path)

    pid2sid_path = "data/recif/video_ad_pid2sid.parquet"
    pid2sid = load_pid2sid(pid2sid_path)
    logger.info("Loaded %d pid->sid mappings", len(pid2sid))

    all_sid_strs = [
        wrapper.sid_triple_to_str(*s) for s in pid2sid.values()
    ]

    subset = df.sample(n=args.n_samples, random_state=args.seed).reset_index(drop=True)

    # ----- Test 1: Free Generation -----
    logger.info("=== Test 1: Free Generation (no thinking) ===")
    gen_valid_count = 0
    gen_hit_count = 0
    for i, row in subset.iterrows():
        messages = wrapper.format_messages_from_benchmark(row["messages"])
        meta = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
        gt_answer = meta["answer"]
        gt_sids = set(wrapper.parse_sids_from_text(gt_answer))

        outputs = wrapper.generate(
            messages, max_new_tokens=256, enable_thinking=False,
            temperature=0.6, top_k=20, top_p=0.95,
        )
        generated_text = outputs[0]
        pred_sids = wrapper.parse_sids_from_text(generated_text)

        n_valid = len(pred_sids)
        gen_valid_count += min(n_valid, 1)
        hits = len(set(pred_sids) & gt_sids)
        gen_hit_count += min(hits, 1)

        logger.info(
            "  Sample %d: generated %d valid SIDs, %d hits (gt has %d SIDs) | first pred: %s",
            i, n_valid, hits, len(gt_sids),
            wrapper.sid_triple_to_str(*pred_sids[0]) if pred_sids else "NONE",
        )

    logger.info(
        "  Free-gen summary: %d/%d generated valid SIDs, %d/%d had at least one hit",
        gen_valid_count, args.n_samples, gen_hit_count, args.n_samples,
    )

    # ----- Test 2: Candidate-Constrained Scoring -----
    logger.info("=== Test 2: Candidate-Constrained Scoring ===")
    recall_at_k = {1: 0, 5: 0, 10: 0, 32: 0}
    for i, row in subset.iterrows():
        messages = wrapper.format_messages_from_benchmark(row["messages"])
        meta = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
        gt_answer = meta["answer"]
        gt_sids_parsed = wrapper.parse_sids_from_text(gt_answer)
        gt_sid_strs = [wrapper.sid_triple_to_str(*s) for s in gt_sids_parsed]

        candidates = build_candidate_pool(gt_sid_strs, all_sid_strs, pool_size=64)
        random.shuffle(candidates)

        scored = wrapper.score_candidates(
            messages, candidates, enable_thinking=False,
        )

        for k in recall_at_k:
            top_k_sids = {s.sid_str for s in scored[:k]}
            hits = len(top_k_sids & set(gt_sid_strs))
            recall_at_k[k] += hits / len(gt_sid_strs) if gt_sid_strs else 0

        logger.info(
            "  Sample %d: top-1=%s | gt[0]=%s | match=%s",
            i,
            scored[0].sid_str if scored else "?",
            gt_sid_strs[0] if gt_sid_strs else "?",
            scored[0].sid_str in gt_sid_strs if scored else False,
        )

    logger.info("=== RESULTS ===")
    for k in recall_at_k:
        avg = recall_at_k[k] / args.n_samples
        logger.info("  Recall@%d = %.4f (pool=64)", k, avg)

    logger.info("=== SMOKE TEST COMPLETE ===")


if __name__ == "__main__":
    main()
