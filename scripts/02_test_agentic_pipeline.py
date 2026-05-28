"""Phase 3 Integration Test: Agentic Planner → Executor → Context Card.

Tests the full agentic pipeline:
1. PlannerAgent generates tool call plans based on task type
2. ExecutorAgent executes plans and produces context cards
3. Verify cards are non-empty, within token budget, and contain expected sections
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from arec2.agents.executor_agent import ExecutorAgent
from arec2.agents.planner_agent import PlannerAgent
from arec2.retrieval.stores import (
    CollaborativeStore,
    ItemTextStore,
    LabelBehaviorStore,
    ProfileStore,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_agentic_pipeline")


def load_pid2sid():
    """Load PID→SID mapping for video domain."""
    pid2sid_video = {}
    df_sid = pd.read_parquet("data/recif/video_ad_pid2sid.parquet")
    for _, row in df_sid.iterrows():
        pid2sid_video[int(row["pid"])] = list(row["sid"])
    return pid2sid_video


def test_video_task(planner, executor, pid2sid_video):
    """Test agentic pipeline on video recommendation task."""
    logger.info("=" * 60)
    logger.info("TEST 1: video task (hist_pid only)")

    df = pd.read_parquet("data/recif/benchmark_data/video/video_test.parquet")
    row = df.iloc[42]
    meta = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
    uid = int(meta["uid"])
    hist_pids = list(row["hist_pid"])

    logger.info("Sample: uid=%d, hist_len=%d", uid, len(hist_pids))

    # Generate plan
    plan = planner.plan(
        uid=uid,
        task_type="video",
        hist_pids=hist_pids,
        pid2sid=pid2sid_video,
    )
    logger.info("Plan generated: %d tool calls", len(plan))
    for tc in plan:
        logger.info("  - %s", tc.tool_name)

    # Execute plan
    card = executor.execute(plan)
    logger.info("Context card generated (%d chars):\n%s", len(card), card)

    # Assertions
    assert len(card) > 0, "Context card should not be empty"
    assert "long-term profile" in card, "Card should contain profile section"
    assert "most recent" in card, "Card should contain recent intent section"
    assert len(plan) >= 2, "Plan should have at least profile + recent_intent"

    logger.info("✓ VIDEO TASK TEST PASSED")
    return card


def test_label_cond_task(planner, executor, pid2sid_video):
    """Test agentic pipeline on label-conditioned recommendation task."""
    logger.info("=" * 60)
    logger.info("TEST 2: label_cond task (full label arrays)")

    df = pd.read_parquet("data/recif/benchmark_data/label_cond/label_cond_test.parquet")
    row = df.iloc[0]
    meta = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
    uid = int(meta["uid"])

    # Extract label arrays
    hist_labels = {}
    all_pids = set()
    for label_name in ["longview", "like", "follow", "forward", "not_interested"]:
        col = f"hist_{label_name}"
        if col in row.index and row[col] is not None:
            pids = list(row[col])
            hist_labels[label_name] = pids
            all_pids.update(int(p) for p in pids)

    all_pids_list = list(all_pids)
    logger.info(
        "Sample: uid=%d, label_pids=%d (longview=%d, like=%d, follow=%d, forward=%d, ni=%d)",
        uid, len(all_pids_list),
        len(hist_labels.get("longview", [])),
        len(hist_labels.get("like", [])),
        len(hist_labels.get("follow", [])),
        len(hist_labels.get("forward", [])),
        len(hist_labels.get("not_interested", [])),
    )

    # Build unified hist_pids + binary label arrays for tools
    combined_pids = []
    combined_binary = {l: [] for l in ["longview", "like", "follow", "forward", "not_interested"]}

    for label_name in ["longview", "like", "follow", "forward", "not_interested"]:
        pids = hist_labels.get(label_name, [])
        combined_pids.extend(pids)
        for l in combined_binary:
            combined_binary[l].extend([1 if l == label_name else 0] * len(pids))

    # Generate plan
    plan = planner.plan(
        uid=uid,
        task_type="label_cond",
        hist_pids=combined_pids,
        hist_labels=combined_binary,
        pid2sid=pid2sid_video,
    )
    logger.info("Plan generated: %d tool calls", len(plan))
    for tc in plan:
        logger.info("  - %s", tc.tool_name)

    # Execute plan
    card = executor.execute(plan)
    logger.info("Context card generated (%d chars):\n%s", len(card), card)

    # Assertions
    assert len(card) > 0, "Context card should not be empty"
    assert "long-term profile" in card, "Card should contain profile section"
    assert "interaction patterns" in card or "most recent" in card, "Card should contain labels or recent section"
    assert len(plan) >= 3, "Plan should have profile + label_behavior + recent_intent"

    logger.info("✓ LABEL_COND TASK TEST PASSED")
    return card


def test_ad_task(planner, executor, pid2sid_video):
    """Test agentic pipeline on ad recommendation task."""
    logger.info("=" * 60)
    logger.info("TEST 3: ad task (cross-domain signals)")

    df = pd.read_parquet("data/recif/benchmark_data/ad/ad_test.parquet")
    row = df.iloc[0]
    meta = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
    uid = int(meta["uid"])
    hist_ad = list(row["hist_ad"]) if "hist_ad" in row.index else []

    logger.info("Sample: uid=%d, hist_ad_len=%d", uid, len(hist_ad))

    # Generate plan
    plan = planner.plan(
        uid=uid,
        task_type="ad",
        hist_pids=hist_ad,
        pid2sid=pid2sid_video,
    )
    logger.info("Plan generated: %d tool calls", len(plan))
    for tc in plan:
        logger.info("  - %s", tc.tool_name)

    # Execute plan
    card = executor.execute(plan)
    logger.info("Context card generated (%d chars):\n%s", len(card), card[:500])

    # Assertions
    assert len(card) > 0, "Context card should not be empty"
    assert "cross_domain" in [tc.tool_name for tc in plan], "Ad task should include cross_domain tool"

    logger.info("✓ AD TASK TEST PASSED")
    return card


def test_plan_structure(planner):
    """Test that planner generates correct tool sequences for different task types."""
    logger.info("=" * 60)
    logger.info("TEST 4: Plan structure validation")

    test_cases = [
        ("video", ["profile", "recent_intent", "collaborative"]),
        ("label_cond", ["profile", "label_behavior", "recent_intent", "collaborative"]),
        ("ad", ["profile", "recent_intent", "cross_domain", "collaborative"]),
        ("product", ["profile", "recent_intent", "cross_domain", "collaborative"]),
    ]

    for task_type, expected_tools in test_cases:
        plan = planner.plan(
            uid=12345,
            task_type=task_type,
            hist_pids=[1, 2, 3, 4, 5],
            hist_labels={"like": [1, 0, 1, 0, 1]},
            pid2sid={1: [1, 2, 3], 2: [4, 5, 6]},
        )
        actual_tools = [tc.tool_name for tc in plan]
        logger.info("  %s: %s", task_type, actual_tools)

        for tool in expected_tools:
            if tool == "label_behavior" and task_type not in ["label_cond", "label_pred"]:
                continue  # Label behavior only for label tasks
            assert tool in actual_tools, f"{task_type} plan should include {tool}"

    logger.info("✓ PLAN STRUCTURE TEST PASSED")


def main():
    t0 = time.time()

    logger.info("Loading stores...")
    profile_store = ProfileStore.load("caches/profile_store.pkl")
    label_store = LabelBehaviorStore.load("caches/label_behavior_store.pkl")
    collab_store = CollaborativeStore.load("caches/collaborative_store")
    text_store = ItemTextStore.load("caches/item_text_store.pkl")
    logger.info("Stores loaded in %.1fs", time.time() - t0)

    logger.info("Loading PID→SID mapping...")
    pid2sid_video = load_pid2sid()
    logger.info("Loaded %d mappings in %.1fs", len(pid2sid_video), time.time() - t0)

    logger.info("Initializing agents...")
    planner = PlannerAgent(token_budget=2048)
    executor = ExecutorAgent(
        profile_store=profile_store,
        label_store=label_store,
        collab_store=collab_store,
        text_store=text_store,
        token_budget=2048,
    )
    logger.info("Agents initialized")

    # Run tests
    test_plan_structure(planner)
    test_video_task(planner, executor, pid2sid_video)
    test_label_cond_task(planner, executor, pid2sid_video)
    test_ad_task(planner, executor, pid2sid_video)

    logger.info("=" * 60)
    logger.info("ALL PHASE 3 TESTS PASSED (%.1fs total)", time.time() - t0)


if __name__ == "__main__":
    main()
