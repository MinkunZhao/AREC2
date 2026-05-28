"""Test data loading and context enrichment pipeline.

Quick sanity check before full training.
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml
from transformers import AutoTokenizer

from arec2.agents.executor_agent import ExecutorAgent
from arec2.agents.planner_agent import PlannerAgent
from arec2.retrieval.stores import (
    CollaborativeStore,
    ItemTextStore,
    LabelBehaviorStore,
    ProfileStore,
)
from arec2.training.data_loader import (
    CombinedDataset,
    GeneralSFTDataset,
    RecIFDataset,
    format_sample_for_training,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_data_pipeline")


def main():
    t0 = time.time()

    # Load config
    with open("configs/training_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load PID→SID mapping (small subset for testing)
    logger.info("Loading PID→SID mapping...")
    pid2sid = {}
    df = pd.read_parquet(config["data"]["pid2sid_path"])
    for i, row in enumerate(df.iterrows()):
        if i >= 100000:  # Only load 100K for quick test
            break
        _, r = row
        pid2sid[int(r["pid"])] = list(r["sid"])
    logger.info("Loaded %d PID→SID mappings", len(pid2sid))

    # Load stores
    logger.info("Loading stores...")
    profile_store = ProfileStore.load(config["stores"]["profile_store_path"])
    label_store = LabelBehaviorStore.load(config["stores"]["label_store_path"])
    collab_store = CollaborativeStore.load(config["stores"]["collab_store_path"])
    text_store = ItemTextStore.load(config["stores"]["text_store_path"])

    # Initialize planner and executor
    logger.info("Initializing planner and executor...")
    planner = PlannerAgent(token_budget=2048)
    executor = ExecutorAgent(
        profile_store=profile_store,
        label_store=label_store,
        collab_store=collab_store,
        text_store=text_store,
        token_budget=2048,
    )

    # Test RecIF dataset (100 samples)
    logger.info("=" * 60)
    logger.info("TEST 1: RecIF dataset with context enrichment")
    recif_dataset = RecIFDataset(
        data_dir=config["data"]["recif_data_dir"],
        tasks=["video", "label_cond"],
        planner=planner,
        executor=executor,
        pid2sid=pid2sid,
        enrich_context=True,
        max_samples=100,
        seed=42,
    )
    logger.info("RecIF dataset loaded: %d samples", len(recif_dataset))

    # Check a few samples
    for i in [0, 10, 50]:
        sample = recif_dataset[i]
        logger.info("Sample %d:", i)
        logger.info("  Task: %s", sample.get("task_type"))
        logger.info("  UID: %s", sample.get("metadata", {}).get("uid"))
        logger.info("  Context card length: %d chars", len(sample.get("context_card", "")))
        logger.info("  Context card preview: %s...", sample.get("context_card", "")[:200])

    # Test General SFT dataset (100 samples)
    logger.info("=" * 60)
    logger.info("TEST 2: General SFT dataset")
    general_dataset = GeneralSFTDataset(
        data_dir=config["data"]["general_sft_data_dir"],
        max_samples=100,
        seed=42,
    )
    logger.info("General SFT dataset loaded: %d samples", len(general_dataset))

    sample = general_dataset[0]
    logger.info("Sample 0:")
    logger.info("  Source: %s", sample.get("source"))
    logger.info("  Messages: %d turns", len(sample.get("messages", [])))

    # Test Combined dataset
    logger.info("=" * 60)
    logger.info("TEST 3: Combined dataset (80%% RecIF + 20%% General)")
    combined_dataset = CombinedDataset(
        recif_dataset=recif_dataset,
        general_sft_dataset=general_dataset,
        recif_ratio=0.8,
        seed=42,
    )
    logger.info("Combined dataset: %d samples", len(combined_dataset))

    # Count sources
    recif_count = sum(1 for i in range(len(combined_dataset)) if combined_dataset[i].get("source") == "recif")
    general_count = len(combined_dataset) - recif_count
    logger.info("  RecIF: %d (%.1f%%)", recif_count, 100 * recif_count / len(combined_dataset))
    logger.info("  General: %d (%.1f%%)", general_count, 100 * general_count / len(combined_dataset))

    # Test formatting
    logger.info("=" * 60)
    logger.info("TEST 4: Format samples for training")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model_path"], trust_remote_code=True)

    for i in [0, 50, 100]:
        sample = combined_dataset[i]
        formatted = format_sample_for_training(sample, tokenizer)
        text = formatted["text"]
        tokens = tokenizer.encode(text)

        logger.info("Sample %d:", i)
        logger.info("  Source: %s", sample.get("source"))
        logger.info("  Text length: %d chars", len(text))
        logger.info("  Token count: %d", len(tokens))
        logger.info("  Text preview: %s...", text[:300])

    logger.info("=" * 60)
    logger.info("ALL DATA PIPELINE TESTS PASSED (%.1fs)", time.time() - t0)


if __name__ == "__main__":
    main()
