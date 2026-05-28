"""Generate DPO preference pairs for RecPO.

Strategies (see arec2/rl/preference_pair_gen.py):
  P1 - On-policy hard negative (default workhorse): greedy decode, rejected =
       model's own top-ranked non-GT SIDs.
  P2 - Score-based hard negative (optional, --use-scoring): rejected = highest
       log-prob non-GT candidates from a per-sample candidate pool.

All chosen/rejected are canonicalized + length-matched so DPO learns item
identity, not formatting. prompt + chosen reconstructs an SFT-consistent
sequence.

RECOMMENDED FIRST RUN (to verify DPO no longer regresses):
    python scripts/06_generate_preferences.py --config configs/dpo_config.yaml \
        --tasks video --max-pairs 20000

Output: data/preferences/all_pairs.parquet  ({"prompt","chosen","rejected"})

Usage:
    python scripts/06_generate_preferences.py --config configs/dpo_config.yaml
"""

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arec2.agents.executor_agent import ExecutorAgent
from arec2.agents.planner_agent import PlannerAgent
from arec2.base_model.openonerec_wrapper import OpenOneRecWrapper
from arec2.retrieval.stores import (
    CollaborativeStore,
    ItemTextStore,
    LabelBehaviorStore,
    ProfileStore,
)
from arec2.rl.preference_pair_gen import (
    generate_onpolicy_pairs_batch,
    generate_score_hardneg_pair,
    pair_to_dict,
)
from arec2.rl.rollout_beam import parse_sid_list
from arec2.training.data_loader import RecIFDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gen_preferences")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_sid_pool(pid2sid: dict) -> list[str]:
    """All canonical single-SID strings, used as filler / candidate negatives."""
    pool = []
    for sid in pid2sid.values():
        if sid and len(sid) >= 3:
            pool.append(f"<|sid_begin|><s_a_{sid[0]}><s_b_{sid[1]}><s_c_{sid[2]}><|sid_end|>")
    return pool


def flatten_messages(messages: list[dict]) -> list[dict]:
    flat = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
        flat.append({"role": msg["role"], "content": content})
    return flat


def main():
    parser = argparse.ArgumentParser(description="Generate DPO preference pairs (RecPO)")
    parser.add_argument("--config", type=str, default="configs/dpo_config.yaml")
    parser.add_argument("--tasks", type=str, nargs="*",
                        default=["video", "ad", "product", "label_cond"],
                        help="Start with just 'video' to validate Gate D first.")
    parser.add_argument("--max-pairs", type=int, default=50000)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--use-scoring", action="store_true",
                        help="Enable P2 score-based hard negatives as a fallback "
                             "when on-policy yields too few non-GT SIDs.")
    parser.add_argument("--cand-pool-size", type=int, default=64,
                        help="Candidate pool size for P2 scoring (GT + negatives).")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for P1 greedy generation (higher = faster, more VRAM).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    rng = random.Random(args.seed)
    random.seed(args.seed)

    # ---- Load SFT model (merged) ----
    base_path = config["model"]["base_model_path"]
    sft_ckpt = config["model"]["sft_checkpoint_path"]
    logger.info("Loading model (base=%s)...", base_path)
    model = OpenOneRecWrapper(model_path=base_path, device="auto", torch_dtype="bfloat16")

    sft_path = Path(sft_ckpt)
    if sft_path.exists():
        from peft import PeftModel
        logger.info("Merging SFT LoRA from %s ...", sft_ckpt)
        model.model = PeftModel.from_pretrained(model.model, sft_ckpt)
        model.model = model.model.merge_and_unload()
        model.model.eval()
        logger.info("SFT LoRA merged.")
    else:
        logger.warning("SFT checkpoint not found at %s; generating from base model.", sft_ckpt)

    # ---- Stores + enrichment pipeline (must mirror training) ----
    stores_config = config["stores"]
    profile_store = ProfileStore.load(stores_config["profile_store_path"])
    label_store = LabelBehaviorStore.load(stores_config["label_store_path"])
    collab_store = CollaborativeStore.load(stores_config["collab_store_path"])
    text_store = ItemTextStore.load(stores_config["text_store_path"])

    planner = PlannerAgent(token_budget=1024)
    executor = ExecutorAgent(
        profile_store=profile_store,
        label_store=label_store,
        collab_store=collab_store,
        text_store=text_store,
        token_budget=1024,
        card_style="natural",
    )

    # ---- PID->SID ----
    df_pid = pd.read_parquet(config["data"]["pid2sid_path"])
    pid2sid = {int(row["pid"]): list(row["sid"]) for _, row in df_pid.iterrows()}
    sid_pool = build_sid_pool(pid2sid)
    logger.info("Loaded %d PID->SID; SID pool size=%d", len(pid2sid), len(sid_pool))

    # ---- Dataset (enriched, same cards as SFT) ----
    logger.info("Loading RecIF dataset with enrichment (tasks=%s)...", args.tasks)
    recif_dataset = RecIFDataset(
        data_dir=config["data"]["recif_data_dir"],
        tasks=args.tasks,
        planner=planner,
        executor=executor,
        pid2sid=pid2sid,
        card_source="heuristic",
        max_samples=config["data"].get("max_samples", args.max_pairs * 3),
        seed=args.seed,
    )
    logger.info("Dataset has %d samples", len(recif_dataset))

    output_dir = Path(config["data"]["preferences_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = list(range(len(recif_dataset)))
    rng.shuffle(indices)

    all_pairs = []
    n_onpolicy = 0
    n_score = 0
    t_start = time.time()
    last_log = 0
    last_ckpt = 0

    # -- Collect valid samples into batches, then generate in one call --
    batch_buf: list[tuple[list[dict], str, list[str]]] = []

    def _flush_batch():
        nonlocal n_onpolicy, n_score, last_log, last_ckpt
        if not batch_buf:
            return
        try:
            pairs = generate_onpolicy_pairs_batch(
                model, batch_buf,
                filler_pool=sid_pool, rng=rng,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as e:
            logger.debug("Batch generation failed: %s", e)
            batch_buf.clear()
            return

        for (prompt_msgs, gt_answer, gt_list), pair in zip(batch_buf, pairs):
            if len(all_pairs) >= args.max_pairs:
                break
            if pair is not None:
                all_pairs.append(pair_to_dict(pair))
                n_onpolicy += 1
            elif args.use_scoring:
                try:
                    pool = list(dict.fromkeys(gt_list))
                    while len(pool) < args.cand_pool_size and sid_pool:
                        cand = rng.choice(sid_pool)
                        if cand not in pool:
                            pool.append(cand)
                    p2 = generate_score_hardneg_pair(
                        model, prompt_msgs, gt_answer, pool, rng=rng,
                    )
                    if p2 is not None:
                        all_pairs.append(pair_to_dict(p2))
                        n_score += 1
                except Exception as e:
                    logger.debug("P2 fallback failed: %s", e)

        batch_buf.clear()

        cur = len(all_pairs)
        elapsed = time.time() - t_start
        if cur // 500 > last_log and cur:
            last_log = cur // 500
            logger.info(
                "[progress] %d pairs (onpolicy=%d, score=%d) | %.1f pairs/min",
                cur, n_onpolicy, n_score, cur / elapsed * 60,
            )
        if cur // 2000 > last_ckpt and cur:
            last_ckpt = cur // 2000
            pd.DataFrame(all_pairs).to_parquet(output_dir / "all_pairs.parquet", index=False)
            logger.info("Checkpoint saved: %d pairs", cur)

    for idx in indices:
        if len(all_pairs) >= args.max_pairs:
            break

        sample = recif_dataset[idx]
        messages = sample.get("messages_enriched") or sample.get("messages", [])
        flat = flatten_messages(messages)
        if len(flat) < 2 or flat[-1]["role"] != "assistant":
            continue

        gt_answer = flat[-1]["content"]
        gt_list = parse_sid_list(gt_answer)
        if not gt_list:
            continue

        batch_buf.append((flat[:-1], gt_answer, gt_list))
        if len(batch_buf) >= args.batch_size:
            _flush_batch()

    _flush_batch()

    if all_pairs:
        pd.DataFrame(all_pairs).to_parquet(output_dir / "all_pairs.parquet", index=False)

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("Preference generation complete!")
    logger.info("  On-policy hard-neg pairs: %d", n_onpolicy)
    logger.info("  Score hard-neg pairs:     %d", n_score)
    logger.info("  Total:                    %d", len(all_pairs))
    logger.info("  Time: %.1f min", elapsed / 60)
    logger.info("  Saved to: %s", output_dir / "all_pairs.parquet")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()