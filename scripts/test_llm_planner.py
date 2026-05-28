"""Quick sanity test for LLM-Planner."""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arec2.agents.llm_client import LLMClient
from arec2.agents.llm_planner import LLMPlanner

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_llm_planner():
    """Test LLM-Planner with a few sample inputs."""

    # Initialize client and planner
    llm = LLMClient(model="gpt-4o-mini", cache_dir="./caches/llm_cache")
    planner = LLMPlanner(llm, instruction_path="prompts/planner_v0.txt")

    # Test cases
    test_cases = [
        {
            "name": "Video task with history",
            "uid": 12345,
            "task_type": "video",
            "hist_pids": [101, 102, 103, 104, 105],
            "hist_labels": None,
        },
        {
            "name": "Label_cond task with labels",
            "uid": 67890,
            "task_type": "label_cond",
            "hist_pids": [201, 202, 203],
            "hist_labels": {"like": [201, 203], "follow": [202], "longview": [201, 202, 203]},
        },
        {
            "name": "Product task (cross-domain)",
            "uid": 11111,
            "task_type": "product",
            "hist_pids": [301, 302, 303, 304],
            "hist_labels": None,
        },
        {
            "name": "Label_pred with candidate pool",
            "uid": 22222,
            "task_type": "label_pred",
            "hist_pids": [401, 402],
            "hist_labels": {"like": [401], "not_interested": [402]},
            "candidate_pool": ["<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>"],
        },
        {
            "name": "Short history (should skip collaborative)",
            "uid": 33333,
            "task_type": "video",
            "hist_pids": [501, 502],
            "hist_labels": None,
        },
    ]

    print("\n" + "=" * 80)
    print("LLM-PLANNER SANITY TEST")
    print("=" * 80 + "\n")

    success_count = 0
    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}] {test['name']}")
        print("-" * 80)

        try:
            tool_calls = planner.plan(**test)

            print(f"[OK] Generated {len(tool_calls)} tool calls:")
            for tc in tool_calls:
                # Show params without full data dumps
                clean_params = {k: v for k, v in tc.params.items() if k not in ["hist_pids", "pid2sid", "hist_labels", "pid2sid_video", "pid2sid_product"]}
                print(f"  - {tc.tool_name}: {clean_params}")

            success_count += 1

        except Exception as e:
            print(f"[FAIL] {e}")

    print("\n" + "=" * 80)
    print(f"Results: {success_count}/{len(test_cases)} tests passed")

    # Show stats
    stats = planner.get_stats()
    print(f"Planner stats: {stats}")

    # Show cost
    llm_stats = llm.get_stats()
    print(f"LLM stats: {llm_stats}")
    print("=" * 80 + "\n")

    # Check fallback rate
    if stats["fallback_rate"] > 0.1:
        print(f"WARNING: High fallback rate ({stats['fallback_rate']:.1%})")
        return False

    if success_count == len(test_cases):
        print("[PASS] All tests passed!")
        return True
    else:
        print(f"[FAIL] {len(test_cases) - success_count} tests failed")
        return False


if __name__ == "__main__":
    success = test_llm_planner()
    sys.exit(0 if success else 1)
