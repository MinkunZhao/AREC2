"""Pre-compute LLM-optimized context cards for SFT v2 training.

Iterates over onerec_bench_release.parquet, uses the TextGrad-optimized
LLMPlanner + LLMCompiler to generate context cards, and saves them to
data/cards_v2/<task_type>.parquet.

Supports resume: checks which (uid, task_type) pairs already exist and
skips them.

Parallelism strategy:
  Each sample requires 2 LLM calls (planner + compiler), both I/O-bound.
  We use a ThreadPoolExecutor so N samples run concurrently, giving ~Nx
  throughput. The LLMClient's diskcache is thread-safe, and its internal
  OpenAI client handles concurrent HTTP connections.

Usage:
    python scripts/05b_precompute_cards.py --config configs/textgrad_config.yaml
    python scripts/05b_precompute_cards.py --config configs/textgrad_config.yaml --workers 32
"""

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arec2.agents.executor_agent import ExecutorAgent
from arec2.agents.llm_client import LLMClient
from arec2.agents.llm_compiler import LLMCompiler
from arec2.agents.llm_planner import LLMPlanner
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
logger = logging.getLogger("precompute_cards")


def load_stores(config: dict) -> dict:
    cache_dir = Path(config["stores"]["cache_dir"])
    logger.info("Loading offline stores from %s ...", cache_dir)
    stores = {
        "profile": ProfileStore.load(str(cache_dir / "profile_store.pkl")),
        "label": LabelBehaviorStore.load(str(cache_dir / "label_behavior_store.pkl")),
        "collab": CollaborativeStore.load(str(cache_dir / "collaborative_store")),
        "text": ItemTextStore.load(str(cache_dir / "item_text_store.pkl")),
    }
    logger.info(
        "Stores loaded: profile=%d users, label=%d users, collab=%d items",
        len(stores["profile"].profiles),
        len(stores["label"].behaviors),
        len(stores["collab"].idx2sid),
    )
    return stores


def load_pid2sid(path: str) -> dict:
    df = pd.read_parquet(path)
    return {int(row["pid"]): list(row["sid"]) for _, row in df.iterrows()}


def load_existing_cards(output_dir: Path) -> set[tuple[int, str]]:
    """Load already-computed (uid, task_type) pairs for resume.

    Also recovers any unmerged chunk files from interrupted runs.
    """
    # First, recover any orphaned chunks into main files
    for chunk in sorted(output_dir.glob(".chunk_*.parquet")):
        # Parse task_type from filename: .chunk_{task_type}_{id}.parquet
        parts = chunk.stem.split("_")
        # parts = ['.chunk', task_type..., chunk_id]
        task_type = "_".join(parts[1:-1])
        try:
            chunk_df = pd.read_parquet(chunk)
            out_path = output_dir / f"{task_type}.parquet"
            if out_path.exists():
                existing = pd.read_parquet(out_path)
                combined = pd.concat([existing, chunk_df], ignore_index=True)
                combined.drop_duplicates(subset=["uid", "task_type"], keep="last", inplace=True)
                combined.to_parquet(out_path, index=False)
            else:
                chunk_df.to_parquet(out_path, index=False)
            chunk.unlink()
            logger.info("Recovered orphaned chunk: %s (%d rows)", chunk.name, len(chunk_df))
        except Exception as e:
            logger.warning("Failed to recover chunk %s: %s", chunk.name, e)

    # Now load all completed pairs
    done = set()
    for pq in output_dir.glob("*.parquet"):
        if pq.stem.startswith(".chunk"):
            continue
        task_type = pq.stem
        try:
            df = pd.read_parquet(pq, columns=["uid"])
            for uid in df["uid"]:
                done.add((int(uid), task_type))
        except Exception:
            pass
    return done


def extract_user_samples(df: pd.DataFrame, pid2sid_video: dict, pid2sid_product: dict) -> list[dict]:
    """Extract per-user, per-task samples from onerec_bench_release.parquet."""
    task_types = ["video", "ad", "product", "label_cond", "label_pred"]

    samples = []
    for _, row in df.iterrows():
        uid = int(row["uid"])
        hist_video_pid = row.get("hist_video_pid")
        if hist_video_pid is None:
            continue
        hist_pids = [int(p) for p in hist_video_pid]
        if len(hist_pids) < 3:
            continue

        hist_labels = {}
        for label in ["like", "longview", "follow", "forward", "not_interested"]:
            col = f"hist_video_{label}"
            val = row.get(col)
            if val is not None:
                hist_labels[label] = [int(x) for x in val]

        for task_type in task_types:
            samples.append({
                "uid": uid,
                "task_type": task_type,
                "hist_pids": hist_pids,
                "hist_labels": hist_labels if hist_labels else None,
            })

    return samples


def compute_card_for_sample(
    sample: dict,
    planner: LLMPlanner,
    executor: ExecutorAgent,
    compiler: LLMCompiler,
    pid2sid_video: dict,
    pid2sid_product: dict,
) -> tuple[dict, str | None]:
    """Run LLMPlanner -> ExecutorAgent tools -> LLMCompiler for one sample.

    Returns (sample, card_or_None) so the caller can identify which sample
    this result belongs to.
    """
    uid = sample["uid"]
    task_type = sample["task_type"]
    hist_pids = sample["hist_pids"]
    hist_labels = sample["hist_labels"]

    try:
        plan = planner.plan(
            uid=uid,
            task_type=task_type,
            hist_pids=hist_pids,
            hist_labels=hist_labels,
            pid2sid=pid2sid_video,
            pid2sid_video=pid2sid_video,
            pid2sid_product=pid2sid_product,
        )

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
                logger.debug("Tool %s failed for uid=%d: %s", tool_call.tool_name, uid, e)

        if not tool_results:
            return sample, None

        card = compiler.compile(tool_results, task_type=task_type, candidate_sids=None)
        return sample, card

    except Exception as e:
        logger.warning("Failed to compute card for uid=%d task=%s: %s", uid, task_type, e)
        return sample, None


def flush_batch(output_dir: Path, batch_rows: dict[str, list[dict]], chunk_id: int):
    """Save batch as atomic chunk files, then merge into main parquet.

    Strategy: write to a temp chunk file first, then append to main file.
    This ensures no data loss on crash — chunk files are proof of work.
    """
    for task_type, rows in batch_rows.items():
        if not rows:
            continue
        new_df = pd.DataFrame(rows)

        # Write atomic chunk (survives crashes)
        chunk_path = output_dir / f".chunk_{task_type}_{chunk_id:06d}.parquet"
        new_df.to_parquet(chunk_path, index=False)

        # Merge into main file
        out_path = output_dir / f"{task_type}.parquet"
        if out_path.exists():
            existing = pd.read_parquet(out_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.to_parquet(out_path, index=False)
        else:
            new_df.to_parquet(out_path, index=False)

        # Remove chunk after successful merge
        chunk_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute LLM-optimized context cards")
    parser.add_argument("--config", type=str, default="../configs/textgrad_config.yaml")
    parser.add_argument("--workers", type=int, default=32,
                        help="Number of parallel worker threads")
    parser.add_argument("--flush-every", type=int, default=500,
                        help="Flush to parquet every N computed cards")
    args = parser.parse_args()

    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    stores = load_stores(config)

    pid2sid_video = load_pid2sid(config["stores"]["pid2sid_video"])
    pid2sid_product = load_pid2sid(config["stores"]["pid2sid_product"])

    output_dir = Path(config.get("cards_v2_dir", "../data/cards_v2"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume: load already-done pairs
    done_set = load_existing_cards(output_dir)
    logger.info("Resume: %d (uid, task) pairs already computed", len(done_set))

    # Load training data
    data_path = config["dev_data"]["source"]
    logger.info("Loading user data from %s ...", data_path)
    df = pd.read_parquet(data_path)
    logger.info("Loaded %d users", len(df))

    all_samples = extract_user_samples(df, pid2sid_video, pid2sid_product)
    logger.info("Extracted %d total (user, task) samples", len(all_samples))

    # Filter out already-done
    pending = [s for s in all_samples if (s["uid"], s["task_type"]) not in done_set]
    logger.info("After resume filter: %d samples to compute", len(pending))

    if not pending:
        logger.info("All cards already computed. Nothing to do.")
        return

    # Initialize LLM client
    llm_config = config["llm"]
    llm = LLMClient(
        model=llm_config["model"],
        base_url=llm_config["base_url"],
        api_key=llm_config["api_key"],
        cache_dir=llm_config["cache_dir"],
        max_concurrency=args.workers,
    )

    # Load optimized prompts
    planner_path = config["prompts"].get("planner_optimized",
                                         str(Path(config["prompts"]["output_dir"]) / "planner_final.txt"))
    compiler_path = config["prompts"].get("compiler_optimized",
                                          str(Path(config["prompts"]["output_dir"]) / "compiler_final.txt"))

    if not Path(planner_path).exists():
        logger.error("Optimized planner prompt not found at %s. Run TextGrad first.", planner_path)
        sys.exit(1)
    if not Path(compiler_path).exists():
        logger.error("Optimized compiler prompt not found at %s. Run TextGrad first.", compiler_path)
        sys.exit(1)

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

    logger.info("=" * 60)
    logger.info("Starting card pre-computation (%d workers)...", args.workers)
    logger.info("  Pending samples: %d", len(pending))
    logger.info("  Planner prompt: %s", planner_path)
    logger.info("  Compiler prompt: %s", compiler_path)
    logger.info("  Output dir: %s", output_dir)
    logger.info("  Flush every: %d", args.flush_every)
    logger.info("=" * 60)

    t_start = time.time()
    computed = 0
    failed = 0
    processed = 0
    flush_count = 0
    batch_rows: dict[str, list[dict]] = {}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                compute_card_for_sample,
                sample, planner, executor, compiler,
                pid2sid_video, pid2sid_product,
            ): sample
            for sample in pending
        }

        for future in as_completed(futures):
            processed += 1
            try:
                sample, card = future.result()
            except Exception as e:
                logger.warning("Worker exception: %s", e)
                failed += 1
                continue

            if card is not None:
                task_type = sample["task_type"]
                batch_rows.setdefault(task_type, []).append({
                    "uid": sample["uid"],
                    "task_type": task_type,
                    "context_card": card,
                })
                computed += 1
            else:
                failed += 1

            # Periodic flush
            total_in_batch = sum(len(v) for v in batch_rows.values())
            if total_in_batch >= args.flush_every:
                flush_count += 1
                flush_batch(output_dir, batch_rows, flush_count)
                batch_rows = {}
                elapsed = time.time() - t_start
                rate = processed / elapsed if elapsed > 0 else 0
                logger.info(
                    "[Progress] %d/%d done | %d computed, %d failed | "
                    "%.1f samples/s | cost=$%.3f",
                    processed, len(pending), computed, failed, rate, llm.total_cost,
                )

    # Final flush
    flush_count += 1
    flush_batch(output_dir, batch_rows, flush_count)

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("Pre-computation complete!")
    logger.info("  Computed: %d cards", computed)
    logger.info("  Failed: %d samples", failed)
    logger.info("  Time: %.1f minutes", elapsed / 60)
    logger.info("  Cost: $%.3f", llm.total_cost)
    logger.info("  Output: %s", output_dir)
    logger.info("=" * 60)

    # Print per-task stats
    for pq in sorted(output_dir.glob("*.parquet")):
        df_task = pd.read_parquet(pq)
        logger.info("  %s: %d cards", pq.stem, len(df_task))


if __name__ == "__main__":
    main()
