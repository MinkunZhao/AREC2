"""Stage A1 (Amazon): build Strategy-3 item artifacts per domain.

For each Amazon category, this script:
  1. Loads meta/reviews JSON, applies 5-core, builds leave-one-out splits.
  2. Embeds item text with Qwen3-Embedding (§4.1.1).
  3. Fits a 3-layer RQ-Kmeans tokenizer (codebook 8192) and encodes items.
  4. Extracts 5 distinctive keywords per item (TF-IDF).
  5. Caches everything to caches/amazon/<category>/ for reuse by training/eval.

The cached artifacts decouple the (expensive) embedding+quantization step from
the (repeatable) training/eval runs, mirroring how the project caches offline
stores in scripts/01_build_offline_stores.py.

Usage:
    python scripts/A1_build_amazon_tokens.py \
        --data-dir ./data/amazon14 \
        --cache-dir ./caches/amazon \
        --embed-model Qwen/Qwen3-Embedding-0.6B \
        --categories Baby Beauty Toys_and_Games
    # default: all 10 categories
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arec2.amazon.data_amazon import AMAZON_CATEGORIES, load_amazon_domain
from arec2.amazon.embeddings import embed_items, extract_keywords
from arec2.amazon.rqkmeans import build_item_codes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("A1_build_amazon_tokens")


def build_one_category(
    category: str,
    data_dir: str,
    cache_dir: Path,
    embed_model: str,
    codebook_size: int,
    n_keywords: int,
    max_hist_len: int,
    batch_size: int,
    device: str,
    seed: int,
    overwrite: bool,
) -> dict:
    out_dir = cache_dir / category
    artifact_path = out_dir / "strategy3_artifacts.pkl"
    if artifact_path.exists() and not overwrite:
        logger.info("[%s] artifacts already exist at %s (skip; use --overwrite)", category, artifact_path)
        with open(artifact_path, "rb") as f:
            meta = pickle.load(f).get("stats", {})
        return {"category": category, "skipped": True, **meta}

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. Load + split.
    domain = load_amazon_domain(
        data_dir=data_dir, category=category, max_hist_len=max_hist_len
    )

    # 2. Embed item text.
    embs, texts = embed_items(
        domain.item_ids, domain.item_meta,
        model_name=embed_model, device=device, batch_size=batch_size,
    )

    # 3. RQ-Kmeans 3-layer codes.
    asin2code, tokenizer, raw_collision = build_item_codes(
        domain.item_ids, embs, codebook_size=codebook_size, seed=seed
    )

    # 4. 5 distinctive keywords.
    asin2keywords = extract_keywords(domain.item_ids, texts, top_k=n_keywords)

    # Strategy-3 collision rate: (code-tuple, keyword-set) duplicates.
    s3_seen: set[tuple] = set()
    s3_collided = 0
    for iid in domain.item_ids:
        key = (asin2code[iid], tuple(asin2keywords.get(iid, [])))
        if key in s3_seen:
            s3_collided += 1
        s3_seen.add(key)
    s3_collision = s3_collided / max(len(domain.item_ids), 1)

    stats = {
        "n_users": domain.n_users,
        "n_items": domain.n_items,
        "n_interactions": domain.n_interactions,
        "raw_3layer_collision_rate": round(raw_collision, 6),
        "strategy3_collision_rate": round(s3_collision, 6),
        "build_seconds": round(time.time() - t0, 1),
    }

    # 5. Persist. We save the splits + codes + keywords (the heavy embeddings
    #    are not needed downstream once codes exist, so we drop them to save disk).
    artifacts = {
        "category": category,
        "item_ids": domain.item_ids,
        "asin2code": asin2code,
        "asin2keywords": asin2keywords,
        "train_histories": domain.train_histories,
        "val_targets": domain.val_targets,
        "test_targets": domain.test_targets,
        "stats": stats,
    }
    with open(artifact_path, "wb") as f:
        pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)
    tokenizer.save(str(out_dir / "rqkmeans_tokenizer.pkl"))

    logger.info(
        "[%s] done in %.1fs | items=%d users=%d | raw_collision=%.2f%% s3_collision=%.2f%%",
        category, stats["build_seconds"], stats["n_items"], stats["n_users"],
        raw_collision * 100, s3_collision * 100,
    )
    return {"category": category, "skipped": False, **stats}


def main():
    parser = argparse.ArgumentParser(description="Build Strategy-3 Amazon item artifacts")
    parser.add_argument("--data-dir", type=str, default="./data/amazon14")
    parser.add_argument("--cache-dir", type=str, default="./caches/amazon")
    parser.add_argument("--categories", type=str, nargs="*", default=None,
                        help="Subset of categories; default = all 10.")
    parser.add_argument("--embed-model", type=str, default="Qwen/Qwen3-Embedding-0.6B",
                        help="Qwen3-Embedding id (use Qwen/Qwen3-Embedding-8B for full fidelity)")
    parser.add_argument("--codebook-size", type=int, default=0,
                        help="Per-layer RQ-Kmeans K. 0 (default) = auto-resolve from "
                             "item count so the 3-layer hierarchy stays non-degenerate. "
                             "Set a positive value to force it (capped at N//2).")
    parser.add_argument("--n-keywords", type=int, default=5)
    parser.add_argument("--max-hist-len", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    categories = args.categories or AMAZON_CATEGORIES
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Building Amazon Strategy-3 artifacts")
    logger.info("  categories: %s", categories)
    logger.info("  embed model: %s | codebook: %s | keywords: %d",
                args.embed_model,
                "auto" if args.codebook_size == 0 else str(args.codebook_size),
                args.n_keywords)
    logger.info("=" * 70)

    summary = []
    for cat in categories:
        try:
            res = build_one_category(
                cat, args.data_dir, cache_dir, args.embed_model,
                args.codebook_size, args.n_keywords, args.max_hist_len,
                args.batch_size, args.device, args.seed, args.overwrite,
            )
            summary.append(res)
        except FileNotFoundError as e:
            logger.error("[%s] %s", cat, e)
            summary.append({"category": cat, "error": str(e)})

    # Write a compact summary.
    summary_path = cache_dir / "build_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 70)
    logger.info("Summary (collision rates should be low; Strategy-3 << raw):")
    for s in summary:
        if "error" in s:
            logger.info("  %-30s ERROR: %s", s["category"], s["error"])
        else:
            logger.info(
                "  %-30s items=%-7d raw=%.2f%% s3=%.2f%%",
                s["category"], s.get("n_items", 0),
                s.get("raw_3layer_collision_rate", 0) * 100,
                s.get("strategy3_collision_rate", 0) * 100,
            )
    logger.info("Summary saved to %s", summary_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()