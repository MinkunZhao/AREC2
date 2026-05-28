"""Run TextGrad optimization to improve planner and compiler prompts.

This script:
1. Loads offline stores (Profile, LabelBehavior, Collaborative, ItemText)
2. Samples dev train/val splits from onerec_bench_release.parquet
3. Defines forward_fn that runs: LLMPlanner → ExecutorAgent → LLMCompiler → OpenOneRec.score_candidates
4. Computes R@10 rewards
5. Runs TextGradEngine.optimize() for N iterations
6. Saves optimized prompts to prompts/optimized/

Usage:
    python scripts/05_run_textgrad.py --config configs/textgrad_config.yaml
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arec2.agents.executor_agent import ExecutorAgent
from arec2.agents.llm_client import LLMClient
from arec2.agents.llm_compiler import LLMCompiler
from arec2.agents.llm_planner import LLMPlanner
from arec2.base_model.openonerec_wrapper import OpenOneRecWrapper
from arec2.retrieval.stores import (
    CollaborativeStore,
    ItemTextStore,
    LabelBehaviorStore,
    ProfileStore,
)
from arec2.textgrad.engine import TextGradEngine, Trajectory
from arec2.textgrad.text_module import TextVariable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("textgrad")


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_stores(config: dict) -> dict:
    """Load offline stores from cache."""
    cache_dir = Path(config["stores"]["cache_dir"])
    logger.info("Loading offline stores from %s ...", cache_dir)

    stores = {}
    stores["profile"] = ProfileStore.load(str(cache_dir / "profile_store.pkl"))
    stores["label"] = LabelBehaviorStore.load(str(cache_dir / "label_behavior_store.pkl"))
    stores["collab"] = CollaborativeStore.load(str(cache_dir / "collaborative_store"))
    stores["text"] = ItemTextStore.load(str(cache_dir / "item_text_store.pkl"))

    logger.info("Stores loaded: profile=%d users, label=%d users, collab=%d items",
                len(stores["profile"].profiles),
                len(stores["label"].behaviors),
                len(stores["collab"].idx2sid))
    return stores


def load_pid2sid_mappings(config: dict) -> tuple[dict, dict]:
    """Load PID→SID mappings."""
    logger.info("Loading PID→SID mappings...")
    df_video = pd.read_parquet(config["stores"]["pid2sid_video"])
    pid2sid_video = {int(row["pid"]): list(row["sid"]) for _, row in df_video.iterrows()}

    df_product = pd.read_parquet(config["stores"]["pid2sid_product"])
    pid2sid_product = {int(row["pid"]): list(row["sid"]) for _, row in df_product.iterrows()}

    logger.info("Loaded %d video mappings, %d product mappings",
                len(pid2sid_video), len(pid2sid_product))
    return pid2sid_video, pid2sid_product


def sample_dev_data(config: dict, pid2sid_video: dict) -> tuple[list, list]:
    """Sample train and val splits from onerec_bench_release.parquet.

    Returns:
        Tuple of (train_samples, val_samples), each a list of dicts with:
        {uid, task_type, hist_pids, hist_labels, gt_sids, candidate_pool}
    """
    data_path = config["dev_data"]["source"]
    train_size = config["dev_data"]["train_size"]
    val_size = config["dev_data"]["val_size"]
    seed = config["dev_data"]["seed"]

    logger.info("Sampling dev data from %s ...", data_path)
    df = pd.read_parquet(data_path)

    # Filter to users with sufficient history
    df = df[df["hist_video_pid"].apply(lambda x: x is not None and len(x) >= 5)]
    logger.info("Filtered to %d users with >=5 history items", len(df))

    # Sample
    random.seed(seed)
    np.random.seed(seed)
    sampled = df.sample(n=min(train_size + val_size, len(df)), random_state=seed)

    train_df = sampled.iloc[:train_size]
    val_df = sampled.iloc[train_size:train_size + val_size]

    # Pre-compute negative SID pool once (avoid per-row copy+shuffle)
    all_sid_strs = [
        f"<|sid_begin|><s_a_{t[0]}><s_b_{t[1]}><s_c_{t[2]}><|sid_end|>"
        for t in pid2sid_video.values()
    ]

    def df_to_samples(df_subset, task_type="video"):
        samples = []
        for _, row in df_subset.iterrows():
            uid = int(row["uid"])
            hist_pids = [int(p) for p in row["hist_video_pid"]] if row["hist_video_pid"] is not None else []

            # Build hist_labels if available
            hist_labels = {}
            for label in ["like", "longview", "follow", "forward", "not_interested"]:
                col = f"hist_video_{label}"
                if col in row and row[col] is not None:
                    hist_labels[label] = [int(x) for x in row[col]]

            # Ground truth: last item in history (simulate next-item prediction)
            if len(hist_pids) < 2:
                continue
            gt_pid = int(hist_pids[-1])
            hist_pids_train = hist_pids[:-1]  # Use all but last for context

            gt_sid_tuple = pid2sid_video.get(gt_pid)
            if not gt_sid_tuple:
                continue
            gt_sids = f"<|sid_begin|><s_a_{gt_sid_tuple[0]}><s_b_{gt_sid_tuple[1]}><s_c_{gt_sid_tuple[2]}><|sid_end|>"

            # Build candidate pool: ground truth + 19 random negatives
            negatives = random.sample(all_sid_strs, min(40, len(all_sid_strs)))
            candidate_pool = [gt_sids] + [s for s in negatives if s != gt_sids][:19]

            samples.append({
                "uid": uid,
                "task_type": task_type,
                "hist_pids": hist_pids_train,
                "hist_labels": hist_labels if hist_labels else None,
                "gt_sids": gt_sids,
                "candidate_pool": candidate_pool,
            })
        return samples

    train_samples = df_to_samples(train_df)
    val_samples = df_to_samples(val_df)

    logger.info("Sampled %d train, %d val samples", len(train_samples), len(val_samples))
    return train_samples, val_samples


def build_forward_fn(
    stores: dict,
    pid2sid_video: dict,
    pid2sid_product: dict,
    model: OpenOneRecWrapper,
    llm: LLMClient,
):
    """Build forward function for TextGrad.

    Args:
        stores: Dict of offline stores.
        pid2sid_video: Video PID→SID mapping.
        pid2sid_product: Product PID→SID mapping.
        model: OpenOneRec model for scoring.
        llm: LLM client for planner/compiler.

    Returns:
        Callable: (planner_instr, compiler_instr, batch) -> [Trajectory]
    """

    def forward_fn(planner_instr: str, compiler_instr: str, batch: list) -> list[Trajectory]:
        """Run forward pass on a batch of samples.

        Args:
            planner_instr: Planner instruction text.
            compiler_instr: Compiler instruction text.
            batch: List of sample dicts.

        Returns:
            List of Trajectory objects with rewards.
        """
        # Create temporary instruction files
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(planner_instr)
            planner_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(compiler_instr)
            compiler_path = f.name

        try:
            # Initialize agents
            planner = LLMPlanner(llm, instruction_path=planner_path)
            compiler = LLMCompiler(llm, instruction_path=compiler_path, max_chars=3000)
            executor = ExecutorAgent(
                profile_store=stores["profile"],
                label_store=stores["label"],
                collab_store=stores["collab"],
                text_store=stores["text"],
                token_budget=1024,
                card_style="natural",
            )

            trajectories = []
            for sample in batch:
                try:
                    uid = sample["uid"]
                    task_type = sample["task_type"]
                    hist_pids = sample["hist_pids"]
                    hist_labels = sample["hist_labels"]
                    gt_sids = sample["gt_sids"]
                    candidate_pool = sample["candidate_pool"]

                    # Plan
                    plan = planner.plan(
                        uid=uid,
                        task_type=task_type,
                        hist_pids=hist_pids,
                        hist_labels=hist_labels,
                        pid2sid=pid2sid_video,
                        pid2sid_video=pid2sid_video,
                        pid2sid_product=pid2sid_product,
                    )

                    # Execute tools
                    tool_results = []
                    results_dict = {}
                    for tool_call in plan:
                        tool = executor.tool_registry.get(tool_call.tool_name)
                        if tool is None:
                            continue
                        params = executor._resolve_params(tool_call.params, results_dict)
                        try:
                            result = tool.run(**params)
                            results_dict[tool_call.tool_name] = result
                            tool_results.append(result)
                        except Exception as e:
                            logger.debug(f"Tool {tool_call.tool_name} failed: {e}")
                            continue

                    # Compile card
                    card = compiler.compile(tool_results, task_type=task_type, candidate_sids=None)

                    # Build prompt
                    messages = [
                        {"role": "system", "content": "你是一个推荐系统助手。"},
                        {"role": "user", "content": f"{card}\n\n请推荐下一个用户可能感兴趣的内容。"},
                    ]

                    # Score candidates
                    scored = model.score_candidates(messages, candidate_pool, enable_thinking=False)
                    pred_sids = scored[0].sid_str if scored else ""

                    # Compute reward (R@10)
                    reward = 1.0 if gt_sids in [s.sid_str for s in scored[:10]] else 0.0

                    # Build history summary
                    hist_summary = f"{len(hist_pids)} items"
                    if hist_labels:
                        label_counts = {k: len(v) for k, v in hist_labels.items() if v}
                        hist_summary += f" | labels: {label_counts}"

                    # Build plan JSON
                    plan_json = [{"tool": tc.tool_name, "params": {k: v for k, v in tc.params.items() if k not in ["hist_pids", "pid2sid", "hist_labels", "uid"]}} for tc in plan]

                    trajectories.append(Trajectory(
                        uid=uid,
                        task_type=task_type,
                        history_summary=hist_summary,
                        plan_json=plan_json,
                        card=card,
                        pred=pred_sids,
                        gt=gt_sids,
                        reward=reward,
                    ))

                except Exception as e:
                    logger.warning(f"Sample uid={sample['uid']} failed: {e}")
                    continue

            return trajectories

        finally:
            # Clean up temp files
            Path(planner_path).unlink(missing_ok=True)
            Path(compiler_path).unlink(missing_ok=True)

    return forward_fn


def main():
    parser = argparse.ArgumentParser(description="Run TextGrad optimization")
    parser.add_argument("--config", type=str, default="../configs/textgrad_config.yaml")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    logger.info("Config loaded from %s", args.config)

    # Load stores
    stores = load_stores(config)
    pid2sid_video, pid2sid_product = load_pid2sid_mappings(config)

    # Sample dev data
    train_samples, val_samples = sample_dev_data(config, pid2sid_video)

    # Initialize LLM client
    llm_config = config["llm"]
    llm = LLMClient(
        model=llm_config["model"],
        base_url=llm_config["base_url"],
        api_key=llm_config["api_key"],
        cache_dir=llm_config["cache_dir"],
        max_concurrency=llm_config["max_concurrency"],
    )

    # Load base model for reward computation
    logger.info("Loading base model for reward computation...")
    model = OpenOneRecWrapper(
        model_path=config["reward"]["base_model_path"],
        device="auto",
        torch_dtype="bfloat16",
    )

    # Build forward function
    forward_fn = build_forward_fn(stores, pid2sid_video, pid2sid_product, model, llm)

    # Load initial prompts
    planner_init = Path(config["prompts"]["planner_init"]).read_text(encoding="utf-8")
    compiler_init = Path(config["prompts"]["compiler_init"]).read_text(encoding="utf-8")

    planner_var = TextVariable(
        name="planner_instruction",
        value=planner_init,
        role="instruction",
        requires_grad=True,
    )
    compiler_var = TextVariable(
        name="compiler_instruction",
        value=compiler_init,
        role="instruction",
        requires_grad=True,
    )

    # Initialize TextGrad engine
    textgrad_config = config["textgrad"]
    engine = TextGradEngine(
        llm=llm,
        forward_fn=forward_fn,
        save_dir=config["prompts"]["output_dir"],
        n_iterations=textgrad_config["n_iterations"],
        train_batch_size=textgrad_config["train_batch_size"],
        val_batch_size=textgrad_config["val_batch_size"],
        commit_threshold_pp=textgrad_config["commit_threshold_pp"],
    )

    # Run optimization
    logger.info("\n" + "=" * 80)
    logger.info("Starting TextGrad optimization...")
    logger.info("=" * 80 + "\n")

    t_start = time.time()
    planner_opt, compiler_opt, history = engine.optimize(
        planner_var, compiler_var, train_samples, val_samples
    )
    elapsed = time.time() - t_start

    # Save history
    history_path = Path(config["prompts"]["output_dir"]) / "history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 80)
    logger.info("TextGrad optimization complete!")
    logger.info("=" * 80)
    logger.info("Total time: %.1f minutes", elapsed / 60)
    logger.info("Total cost: $%.3f", llm.total_cost)
    logger.info("Final prompts saved to: %s", config["prompts"]["output_dir"])
    logger.info("History saved to: %s", history_path)
    logger.info("=" * 80 + "\n")

    # Print history summary
    print("\nOptimization History:")
    print("-" * 60)
    for entry in history:
        status = "[ACCEPTED]" if entry["accepted"] else "[REJECTED]"
        print(f"Iter {entry['iter']+1}: val={entry['val']:.4f} | "
              f"improvement={entry['improvement_pp']:+.2f}pp | {status}")
    print("-" * 60)


if __name__ == "__main__":
    main()
