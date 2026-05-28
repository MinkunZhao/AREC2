"""Pre-compute enriched context cards for all RecIF samples.

Saves enriched data to parquet so training can skip real-time enrichment.
Run this once, then use training_config_full_precomputed.yaml for fast full training.

Usage:
    python scripts/05_precompute_enrichment.py [--max_samples N] [--workers N]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml
from tqdm import tqdm

from arec2.agents.executor_agent import ExecutorAgent
from arec2.agents.planner_agent import PlannerAgent
from arec2.retrieval.stores import (
    CollaborativeStore,
    ItemTextStore,
    LabelBehaviorStore,
    ProfileStore,
)
from arec2.training.data_loader import RecIFDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("precompute")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--output_dir", type=str, default="./data/recif/enriched")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_save", type=int, default=10000,
                        help="Save every N samples to avoid losing progress")
    args = parser.parse_args()

    t0 = time.time()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load PID→SID mapping
    logger.info("Loading PID→SID mapping...")
    pid2sid = {}
    df = pd.read_parquet(config["data"]["pid2sid_path"])
    for _, row in df.iterrows():
        pid2sid[int(row["pid"])] = list(row["sid"])
    logger.info("Loaded %d PID→SID mappings", len(pid2sid))

    # Load stores
    logger.info("Loading stores...")
    stores_cfg = config["stores"]
    profile_store = ProfileStore.load(stores_cfg["profile_store_path"])
    label_store = LabelBehaviorStore.load(stores_cfg["label_store_path"])
    collab_store = CollaborativeStore.load(stores_cfg["collab_store_path"])
    text_store = ItemTextStore.load(stores_cfg["text_store_path"])

    # Initialize planner and executor
    planner = PlannerAgent(token_budget=config["data"]["token_budget"])
    executor = ExecutorAgent(
        profile_store=profile_store,
        label_store=label_store,
        collab_store=collab_store,
        text_store=text_store,
        token_budget=config["data"]["token_budget"],
    )

    # Load RecIF dataset WITHOUT enrichment (just raw samples)
    logger.info("Loading RecIF samples (no enrichment)...")
    dataset = RecIFDataset(
        data_dir=config["data"]["recif_data_dir"],
        tasks=config["data"]["recif_tasks"],
        planner=None,
        executor=None,
        pid2sid=pid2sid,
        enrich_context=False,
        max_samples=args.max_samples,
        seed=config["data"]["seed"],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process and save in batches
    enriched_records = []
    total = len(dataset)
    failed = 0
    batch_idx = 0

    logger.info("Enriching %d samples...", total)

    for i in tqdm(range(total), desc="Enriching"):
        sample = dataset[i]

        # Parse metadata/messages
        if isinstance(sample.get("metadata"), str):
            sample["metadata"] = json.loads(sample["metadata"])
        if isinstance(sample.get("messages"), str):
            sample["messages"] = json.loads(sample["messages"])

        # Enrich
        metadata = sample.get("metadata", {})
        uid = int(metadata.get("uid", 0))
        task_type = sample.get("task_type", "video")

        try:
            hist_pids = dataset._extract_hist_pids(sample)
            hist_labels = dataset._extract_hist_labels(sample)

            plan = planner.plan(
                uid=uid,
                task_type=task_type,
                hist_pids=hist_pids,
                hist_labels=hist_labels,
                pid2sid=pid2sid,
            )
            context_card = executor.execute(plan)
        except Exception as e:
            context_card = ""
            failed += 1

        enriched_records.append({
            "index": i,
            "task_type": task_type,
            "context_card": context_card,
            "messages": json.dumps(sample.get("messages", []), ensure_ascii=False),
        })

        # Batch save
        if len(enriched_records) >= args.batch_save:
            save_path = output_dir / f"enriched_batch_{batch_idx:04d}.parquet"
            pd.DataFrame(enriched_records).to_parquet(save_path)
            logger.info("Saved batch %d (%d records) to %s", batch_idx, len(enriched_records), save_path)
            enriched_records = []
            batch_idx += 1

    # Save remaining
    if enriched_records:
        save_path = output_dir / f"enriched_batch_{batch_idx:04d}.parquet"
        pd.DataFrame(enriched_records).to_parquet(save_path)
        logger.info("Saved final batch (%d records) to %s", len(enriched_records), save_path)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Pre-computation complete!")
    logger.info("  Total samples: %d", total)
    logger.info("  Failed: %d (%.1f%%)", failed, 100 * failed / max(total, 1))
    logger.info("  Output: %s", output_dir)
    logger.info("  Time: %.1f minutes (%.1f hours)", elapsed / 60, elapsed / 3600)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
