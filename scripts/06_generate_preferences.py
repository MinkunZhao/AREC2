"""Generate DPO preference pairs for RecPO (v2, fixes ad/product/label_pred regression).

Strategies (see arec2/rl/preference_pair_gen.py):
  P1 - On-policy hard negative (default for SID tasks): greedy decode, rejected =
       model's own top-ranked non-GT SIDs. For ad/product (ambiguous candidate
       space) we take rank [need, 2*need) instead of the single top non-GT.
  P2 - Score-based hard negative (optional, --use-scoring).
  B  - Binary judgement pairs for label_pred (是/否). NO SID tokens, so DPO does
       not steal first-token mass from 是/否 (which AUC eval reads).

KEY v2 FIXES:
  1. Per-task quota: --max-pairs is split evenly across --tasks so video can't
     starve ad/product/label_pred. Avoids cross-task negative transfer from an
     all-video preference set.
  2. card_source defaults to "llm_optimized" (reads data/cards_v2) so the
     enrichment prompt EXACTLY matches SFT and eval. Use --card-source heuristic
     only if cards_v2 is unavailable (data_loader now extracts hist per task and
     passes video/product mappings, so heuristic is also领域-correct).
  3. label_pred uses binary pairs, never SID pairs.

RECOMMENDED FIRST RUN (verify no regression):
    python scripts/06_generate_preferences.py --config configs/dpo_config.yaml \
        --tasks video --max-pairs 20000

FULL RUN (all 5 trainable tasks, balanced):
    python scripts/06_generate_preferences.py --config configs/dpo_config.yaml \
        --tasks video ad product label_cond label_pred --max-pairs 100000

Output: data/preferences/all_pairs.parquet  ({"prompt","chosen","rejected"})
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
    BINARY_TASKS,
    generate_binary_pair,
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
    parser = argparse.ArgumentParser(description="Generate DPO preference pairs (RecPO v2)")
    parser.add_argument("--config", type=str, default="configs/dpo_config.yaml")
    parser.add_argument("--tasks", type=str, nargs="*",
                        default=["video", "ad", "product", "label_cond", "label_pred"],
                        help="Tasks to generate pairs for. Each gets an equal quota of --max-pairs.")
    parser.add_argument("--max-pairs", type=int, default=100000,
                        help="Total pair budget, split EVENLY across --tasks (per-task quota).")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--card-source", type=str, default="llm_optimized",
                        choices=["llm_optimized", "heuristic"],
                        help="MUST match what SFT used. Default llm_optimized reads data/cards_v2.")
    parser.add_argument("--cards-v2-dir", type=str, default=None,
                        help="Override cards_v2 dir; default reads from config['cards_v2_dir'] or ./data/cards_v2.")
    parser.add_argument("--use-scoring", action="store_true",
                        help="Enable P2 score-based hard negatives as a fallback "
                             "when on-policy yields too few non-GT SIDs.")
    parser.add_argument("--cand-pool-size", type=int, default=64,
                        help="Candidate pool size for P2 scoring (GT + negatives).")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for P1 greedy generation (higher = faster, more VRAM).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    rng = random.Random(args.seed)
    random.seed(args.seed)

    tasks = list(dict.fromkeys(args.tasks))  # de-dup, keep order
    per_task_cap = max(1, args.max_pairs // max(1, len(tasks)))
    logger.info("Per-task quota: %d pairs/task across tasks=%s", per_task_cap, tasks)

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

    # ---- PID->SID ----
    df_pid = pd.read_parquet(config["data"]["pid2sid_path"])
    pid2sid = {int(row["pid"]): list(row["sid"]) for _, row in df_pid.iterrows()}

    pid2sid_product = {}
    product_path = config["data"].get("pid2sid_product_path", "")
    if not product_path:
        product_path = "./data/recif/product_pid2sid.parquet"
    if Path(product_path).exists():
        df_prod = pd.read_parquet(product_path)
        pid2sid_product = {int(row["pid"]): list(row["sid"]) for _, row in df_prod.iterrows()}
        logger.info("Loaded %d product PID->SID mappings from %s",
                    len(pid2sid_product), product_path)
    else:
        logger.warning("product pid2sid not found at %s; product task will yield 0 samples",
                       product_path)

    sid_pool = build_sid_pool(pid2sid)
    logger.info("Loaded %d PID->SID; SID pool size=%d", len(pid2sid), len(sid_pool))

    # ---- Stores + enrichment pipeline (only needed for heuristic card_source) ----
    planner = None
    executor = None
    cards_v2_dir = args.cards_v2_dir or config.get("cards_v2_dir") or "./data/cards_v2"

    if args.card_source == "heuristic":
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
        logger.info("Heuristic enrichment pipeline loaded.")
    else:
        if not Path(cards_v2_dir).exists():
            logger.error(
                "card_source=llm_optimized but cards_v2 dir %s missing. "
                "Run scripts/05b_precompute_cards.py first, or use --card-source heuristic.",
                cards_v2_dir,
            )
            sys.exit(1)
        logger.info("Using pre-computed cards from %s (matches SFT).", cards_v2_dir)

    output_dir = Path(config["data"]["preferences_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pairs: list[dict] = []
    per_task_counts: dict[str, int] = {t: 0 for t in tasks}
    n_onpolicy = 0
    n_score = 0
    n_binary = 0
    t_start = time.time()

    def _save_checkpoint():
        if all_pairs:
            pd.DataFrame(all_pairs).to_parquet(output_dir / "all_pairs.parquet", index=False)

    # ---- Process one task at a time so quotas are respected independently ----
    for task in tasks:
        logger.info("=" * 60)
        logger.info("Generating pairs for task=%s (quota=%d)", task, per_task_cap)

        recif_dataset = RecIFDataset(
            data_dir=config["data"]["recif_data_dir"],
            tasks=[task],
            planner=planner,
            executor=executor,
            pid2sid=pid2sid,
            pid2sid_product=pid2sid_product,
            card_source=args.card_source,
            cards_v2_dir=cards_v2_dir if args.card_source == "llm_optimized" else None,
            max_samples=per_task_cap * 3,
            seed=args.seed,
        )
        logger.info("  Dataset(%s) has %d samples", task, len(recif_dataset))

        indices = list(range(len(recif_dataset)))
        rng.shuffle(indices)

        is_binary = task in BINARY_TASKS

        # Buffer for batched P1 (SID tasks only)
        batch_buf: list[tuple[list[dict], str, list[str]]] = []

        def _flush_batch():
            nonlocal n_onpolicy, n_score
            if not batch_buf:
                return
            try:
                pairs = generate_onpolicy_pairs_batch(
                    model, batch_buf,
                    filler_pool=sid_pool, rng=rng,
                    max_new_tokens=args.max_new_tokens,
                    task_types=[task] * len(batch_buf),
                )
            except Exception as e:
                logger.debug("Batch generation failed: %s", e)
                batch_buf.clear()
                return

            for (prompt_msgs, gt_answer, gt_list), pair in zip(batch_buf, pairs):
                if per_task_counts[task] >= per_task_cap:
                    break
                if pair is not None:
                    all_pairs.append(pair_to_dict(pair))
                    per_task_counts[task] += 1
                    n_onpolicy += 1
                elif args.use_scoring:
                    try:
                        pool = list(dict.fromkeys(gt_list))
                        while len(pool) < args.cand_pool_size and sid_pool:
                            cand = rng.choice(sid_pool)
                            if cand not in pool:
                                pool.append(cand)
                        p2 = generate_score_hardneg_pair(
                            model, prompt_msgs, gt_answer, pool, rng=rng, task_type=task,
                        )
                        if p2 is not None:
                            all_pairs.append(pair_to_dict(p2))
                            per_task_counts[task] += 1
                            n_score += 1
                    except Exception as e:
                        logger.debug("P2 fallback failed: %s", e)
            batch_buf.clear()

        last_log = 0
        for idx in indices:
            if per_task_counts[task] >= per_task_cap:
                break

            sample = recif_dataset[idx]
            messages = sample.get("messages_enriched") or sample.get("messages", [])
            flat = flatten_messages(messages)
            if len(flat) < 2 or flat[-1]["role"] != "assistant":
                continue

            gt_answer = flat[-1]["content"]
            prompt_msgs = flat[:-1]

            if is_binary:
                # Binary judgement pair: no SID parsing, no generation needed.
                try:
                    bp = generate_binary_pair(model, prompt_msgs, gt_answer, task_type=task)
                except Exception as e:
                    logger.debug("Binary pair failed: %s", e)
                    bp = None
                if bp is not None:
                    all_pairs.append(pair_to_dict(bp))
                    per_task_counts[task] += 1
                    n_binary += 1
            else:
                gt_list = parse_sid_list(gt_answer)
                if not gt_list:
                    continue
                batch_buf.append((prompt_msgs, gt_answer, gt_list))
                if len(batch_buf) >= args.batch_size:
                    _flush_batch()

            # progress / checkpoint
            cur = per_task_counts[task]
            if cur // 500 > last_log and cur:
                last_log = cur // 500
                elapsed = time.time() - t_start
                logger.info(
                    "  [%s] %d/%d | total=%d (onpolicy=%d, score=%d, binary=%d) | %.1f pairs/min",
                    task, cur, per_task_cap, len(all_pairs),
                    n_onpolicy, n_score, n_binary, len(all_pairs) / elapsed * 60,
                )
                _save_checkpoint()

        # flush any remaining SID batch for this task
        if not is_binary:
            _flush_batch()

        logger.info("  Task %s done: %d pairs", task, per_task_counts[task])
        _save_checkpoint()

    _save_checkpoint()

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("Preference generation complete!")
    for t in tasks:
        logger.info("  %-12s : %d pairs", t, per_task_counts[t])
    logger.info("  On-policy hard-neg pairs: %d", n_onpolicy)
    logger.info("  Score hard-neg pairs:     %d", n_score)
    logger.info("  Binary judgement pairs:   %d", n_binary)
    logger.info("  Total:                    %d", len(all_pairs))
    logger.info("  Time: %.1f min", elapsed / 60)
    logger.info("  Saved to: %s", output_dir / "all_pairs.parquet")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()