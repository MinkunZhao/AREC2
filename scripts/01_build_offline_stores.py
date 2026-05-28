"""Build all offline stores for the AREC² pipeline.

Reads onerec_bench_release.parquet and PID→SID mappings, builds:
  1. ProfileStore     -> caches/profile_store.pkl
  2. LabelBehaviorStore -> caches/label_behavior_store.pkl
  3. CollaborativeStore -> caches/collaborative_store  (.npz + .meta.pkl)
  4. ItemTextStore    -> caches/item_text_store.pkl

Usage:
    python scripts/01_build_offline_stores.py [--data_dir ./data/recif] [--cache_dir ./caches]
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from arec2.retrieval.stores import (
    CollaborativeStore,
    ItemTextStore,
    LabelBehaviorStore,
    ProfileStore,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_stores")


def load_pid2sid(path: str) -> dict[int, list[int]]:
    logger.info("Loading PID→SID mapping from %s ...", path)
    df = pd.read_parquet(path)
    mapping = {}
    for _, row in df.iterrows():
        mapping[int(row["pid"])] = list(row["sid"])
    logger.info("  Loaded %d mappings", len(mapping))
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Build AREC² offline stores")
    parser.add_argument("--data_dir", type=str, default="./data/recif")
    parser.add_argument("--cache_dir", type=str, default="./caches")
    parser.add_argument(
        "--stores", type=str, nargs="*",
        default=["profile", "label_behavior", "collaborative", "item_text"],
        help="Which stores to build (default: all)",
    )
    parser.add_argument("--cooc_window", type=int, default=10)
    parser.add_argument("--cooc_decay", type=float, default=0.9)
    parser.add_argument("--cooc_min_freq", type=int, default=3)
    parser.add_argument("--cooc_max_vocab", type=int, default=500000)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    main_data = str(data_dir / "onerec_bench_release.parquet")

    t_start = time.time()

    pid2sid_video = load_pid2sid(str(data_dir / "video_ad_pid2sid.parquet"))
    pid2sid_product = load_pid2sid(str(data_dir / "product_pid2sid.parquet"))

    if "profile" in args.stores:
        logger.info("=" * 60)
        logger.info("Building ProfileStore ...")
        t0 = time.time()
        store = ProfileStore()
        store.build(main_data, pid2sid_video, pid2sid_product)
        store.save(str(cache_dir / "profile_store.pkl"))
        logger.info("ProfileStore done in %.1fs", time.time() - t0)

    if "label_behavior" in args.stores:
        logger.info("=" * 60)
        logger.info("Building LabelBehaviorStore ...")
        t0 = time.time()
        store = LabelBehaviorStore()
        store.build(main_data, pid2sid_video)
        store.save(str(cache_dir / "label_behavior_store.pkl"))
        logger.info("LabelBehaviorStore done in %.1fs", time.time() - t0)

    if "collaborative" in args.stores:
        logger.info("=" * 60)
        logger.info("Building CollaborativeStore ...")
        t0 = time.time()
        store = CollaborativeStore()
        store.build(
            main_data, pid2sid_video,
            window_size=args.cooc_window,
            decay_factor=args.cooc_decay,
            min_freq=args.cooc_min_freq,
            max_vocab=args.cooc_max_vocab,
        )
        store.save(str(cache_dir / "collaborative_store"))
        logger.info("CollaborativeStore done in %.1fs", time.time() - t0)

    if "item_text" in args.stores:
        logger.info("=" * 60)
        logger.info("Building ItemTextStore ...")
        t0 = time.time()
        store = ItemTextStore()
        benchmark_dir = str(data_dir / "benchmark_data")
        store.build(main_data, pid2sid_video, pid2sid_product, benchmark_dir=benchmark_dir)
        store.save(str(cache_dir / "item_text_store.pkl"))
        logger.info("ItemTextStore done in %.1fs", time.time() - t0)

    logger.info("=" * 60)
    logger.info("All stores built in %.1fs", time.time() - t_start)
    logger.info("Cache directory: %s", cache_dir.resolve())
    for f in sorted(cache_dir.iterdir()):
        logger.info("  %s (%.1f MB)", f.name, f.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
