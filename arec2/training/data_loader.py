"""Data loaders for AREC² SFT training.

Combines RecIF training data (from onerec_bench_release.parquet, enriched with
agentic context cards) and General_SFT data.

IMPORTANT: RecIF training samples are constructed from the release parquet
(disjoint from test UIDs) to avoid test-set leakage.

v2 fix (ad/product DPO regression):
  - Context enrichment now extracts hist_pids per task using the CORRECT
    history column (ad -> hist_ad, product -> hist_goods, etc.) instead of a
    fixed priority list that always fell back to video PIDs. The planner is
    also given pid2sid_video / pid2sid_product so cross_domain works.
  This guarantees the enrichment card used at preference-generation time
  matches the card used at SFT/eval time for the SAME task.

v3 fix (SFT throughput / balanced sampling):
  - `max_recif_samples` is now interpreted as a GLOBAL budget that is split
    EVENLY across the requested tasks (per-task quota), instead of generating
    all-of-task-A then random-sampling globally (which let `video` dominate
    and could early-exit before `ad`/`product`/`label_pred` were even built).
    This keeps the task mix balanced when you reduce the dataset for speed,
    so per-task eval metrics do not regress from a skewed SFT mix.
  - Per-task generation also stops as soon as that task's quota is met, so
    reducing `max_recif_samples` directly reduces wall-clock build time.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

from arec2.agents.executor_agent import ExecutorAgent
from arec2.agents.planner_agent import PlannerAgent

logger = logging.getLogger(__name__)

CardSource = Literal["heuristic", "llm_optimized", "none"]


# Per-task history column used to seed enrichment. Ordered fallbacks per task.
TASK_HIST_COLS = {
    "video": ["hist_pid", "hist_longview"],
    "ad": ["hist_ad", "hist_pid", "hist_longview"],
    "product": ["hist_goods", "hist_pid", "hist_longview"],
    "label_cond": ["hist_longview", "hist_pid"],
    "label_pred": ["hist_longview", "hist_pid"],
}


# ─── Task-specific prompt templates (extracted from the official RecIF benchmark) ───

_TASK_TEMPLATES = {
    "video": {
        "system": "你是一位视频推荐系统专家，擅长捕捉用户的兴趣演变。请根据历史序列推荐后续视频。",
        "user_format": "{hist_sids}\n推荐后续视频：",
        "fields": {"hist": "hist_video_pid", "target": "target_video_pid"},
        "pid2sid_type": "video_ad",
    },
    "ad": {
        "system": "你是一个广告点击预测专家，擅长分析用户的观看习惯和广告点击偏好，预测用户的广告兴趣。",
        "user_format": "用户的行为历史：\n用户感兴趣的视频：{hist_longview_sids}\n用户感兴趣的广告视频：{hist_ad_sids}\n根据用户的视频观看和广告点击习惯，推荐用户可能点击的广告视频。",
        "fields": {"hist_longview": "hist_video_longview", "hist_ad": "hist_ad_pid", "target": "target_ad_pid"},
        "pid2sid_type": "video_ad",
    },
    "product": {
        "system": "你是一个跨域推荐专家，擅长分析用户的观看习惯和购物偏好，预测用户的商品兴趣。",
        "user_format": "以下是用户的观看和购物记录：\n用户长时间观看的视频：{hist_longview_sids}\n用户浏览过的商品：{hist_goods_sids}\n请推荐用户接下来可能感兴趣并点击的商品。",
        "fields": {"hist_longview": "hist_video_longview", "hist_goods": "hist_goods_pid", "target": "target_goods_pid"},
        "pid2sid_type": "mixed",
    },
    "label_cond": {
        "system": "你是一个内容推荐引擎，通过学习用户的互动历史，预测用户对新内容的反应。",
        "user_format": "观察到用户有以下互动行为：\n用户深度浏览过以下内容：{hist_longview_sids}\n用户喜欢的内容：{hist_like_sids}\n用户转发过以下内容：{hist_forward_sids}\n基于以上互动记录，预测用户会完整观看的内容。",
        "fields": {
            "hist_longview": "hist_video_longview",
            "hist_like": "hist_video_like",
            "hist_forward": "hist_video_forward",
            "target": "target_video_longview",
        },
        "pid2sid_type": "video_ad",
    },
    "label_pred": {
        "system": "你是一个内容推荐专家，擅长分析用户的互动模式，预测用户的内容偏好。",
        "user_format": (
            "用户的内容互动历史：\n"
            "用户深度浏览过以下内容：{hist_longview_sids}\n"
            "用户表示认可的内容：{hist_like_sids}\n"
            '用户会花时间仔细观看视频{candidate_sid}吗？你的回答只需包含"是"或"否"。'
        ),
        "fields": {
            "hist_longview": "hist_video_longview",
            "hist_like": "hist_video_like",
            "candidate": "target_video_pid",
            "label": "target_video_longview",
        },
        "pid2sid_type": "video_ad",
    },
}


def _pids_to_sids_str(pids, pid2sid: dict) -> str:
    """Convert a list of PIDs to SID string tokens."""
    parts = []
    for pid in pids:
        pid = int(pid)
        sid = pid2sid.get(pid)
        if sid is None:
            continue
        parts.append(
            f"<|sid_begin|><s_a_{sid[0]}><s_b_{sid[1]}><s_c_{sid[2]}><|sid_end|>"
        )
    return "".join(parts)


class RecIFDataset(Dataset):
    """RecIF training dataset built from onerec_bench_release.parquet.

    Constructs SFT samples (system + user + assistant messages) from raw user
    interaction data, using PID→SID mappings to produce the same token format
    as the official RecIF benchmark — but with TRAINING users only (no test
    set leakage).
    """

    SUPPORTED_TASKS = ["video", "ad", "product", "label_cond", "label_pred"]

    def __init__(
        self,
        data_dir: str = "data/recif",
        tasks: list[str] | None = None,
        planner: Optional[PlannerAgent] = None,
        executor: Optional[ExecutorAgent] = None,
        pid2sid: dict | None = None,
        pid2sid_product: dict | None = None,
        card_source: CardSource = "heuristic",
        cards_v2_dir: str | None = None,
        enrich_context: bool = True,
        max_samples: int | None = None,
        seed: int = 42,
    ):
        """Initialize RecIF dataset from onerec_bench_release.parquet.

        Args:
            data_dir: Path to RecIF data directory (contains onerec_bench_release.parquet).
            tasks: List of tasks to generate. If None, uses all supported tasks.
            planner: PlannerAgent for generating tool call plans.
            executor: ExecutorAgent for executing plans and generating context cards.
            pid2sid: PID->SID mapping dict for video/ad items.
            pid2sid_product: PID->SID mapping dict for product items.
            card_source: How to produce context cards:
                "heuristic" - on-the-fly rule-based planner+compiler (default)
                "llm_optimized" - load pre-computed cards from cards_v2_dir
                "none" - skip enrichment entirely
            cards_v2_dir: Path to pre-computed card parquets (required when card_source="llm_optimized").
            enrich_context: Legacy flag. Ignored when card_source is explicitly set.
            max_samples: GLOBAL max samples to build. Split EVENLY across `tasks`
                (per-task quota) so the task mix stays balanced when reduced.
            seed: Random seed for sampling.
        """
        self.data_dir = Path(data_dir)
        self.tasks = [t for t in (tasks or self.SUPPORTED_TASKS) if t in self.SUPPORTED_TASKS]
        self.planner = planner
        self.executor = executor
        self.pid2sid = pid2sid or {}
        self.pid2sid_product = pid2sid_product or {}
        self.card_source = card_source
        self.seed = seed

        # Load pre-computed card cache if using llm_optimized
        self._card_cache: dict[str, dict[int, str]] = {}
        if card_source == "llm_optimized":
            if cards_v2_dir is None:
                raise ValueError("cards_v2_dir is required when card_source='llm_optimized'")
            self._load_card_cache(cards_v2_dir)

        random.seed(seed)

        self.samples = self._load_and_build_samples(max_samples)
        logger.info("RecIFDataset loaded: %d samples from %d tasks (card_source=%s)",
                     len(self.samples), len(self.tasks), card_source)

    def _load_card_cache(self, cards_v2_dir: str):
        """Load pre-computed cards into memory as nested dict: cache[task_type][uid] -> card."""
        cards_dir = Path(cards_v2_dir)
        if not cards_dir.exists():
            raise FileNotFoundError(f"cards_v2 directory not found: {cards_dir}")

        total = 0
        for pq in cards_dir.glob("*.parquet"):
            task_type = pq.stem
            df = pd.read_parquet(pq)
            task_cache = {}
            for _, row in df.iterrows():
                task_cache[int(row["uid"])] = str(row["context_card"])
            self._card_cache[task_type] = task_cache
            total += len(task_cache)
            logger.info("  Loaded %d pre-computed cards for task '%s'", len(task_cache), task_type)

        logger.info("Card cache loaded: %d total cards across %d tasks", total, len(self._card_cache))

    def _load_and_build_samples(self, max_samples: int | None) -> list[dict]:
        """Load onerec_bench_release.parquet and build SFT samples.

        Per-task quota strategy:
          - If `max_samples` is set, it is the GLOBAL budget and is split evenly
            across `self.tasks` (the last task absorbs any remainder). Each task
            stops building as soon as its own quota is met.
          - If `max_samples` is None, all tasks build all available samples.
        """
        release_path = self.data_dir / "onerec_bench_release.parquet"
        if not release_path.exists():
            logger.error("onerec_bench_release.parquet not found at %s", release_path)
            return []

        logger.info("Loading onerec_bench_release.parquet...")
        df = pd.read_parquet(release_path)
        logger.info("Loaded %d users from release data", len(df))

        n_tasks = max(1, len(self.tasks))
        if max_samples:
            base_quota = max_samples // n_tasks
            # Distribute remainder to the last task so totals match max_samples.
            quotas = {t: base_quota for t in self.tasks}
            if self.tasks:
                quotas[self.tasks[-1]] += max_samples - base_quota * n_tasks
            logger.info("Per-task quota (global budget=%d): %s", max_samples, quotas)
            # We do not know up-front how many users yield a valid sample for a
            # task, so we process users until the quota is hit. A user yields at
            # most one sample per task, so to fill quota Q we may need up to
            # ~2-3x users. Process the whole frame but break early per task.
        else:
            quotas = {t: None for t in self.tasks}

        all_samples: list[dict] = []

        for task in self.tasks:
            template = _TASK_TEMPLATES.get(task)
            if template is None:
                logger.warning("No template for task: %s, skipping", task)
                continue

            quota = quotas.get(task)
            task_samples = self._build_task_samples(df, task, template, quota)
            logger.info("  Task '%s': generated %d samples (quota=%s)",
                         task, len(task_samples), quota)
            all_samples.extend(task_samples)

        # Shuffle the balanced mix so batches interleave tasks.
        random.shuffle(all_samples)
        return all_samples

    def _build_task_samples(
        self, df: pd.DataFrame, task: str, template: dict, quota: int | None
    ) -> list[dict]:
        """Build SFT samples for a specific task, stopping at `quota` if set."""
        samples = []
        fields = template["fields"]

        for _, row in df.iterrows():
            sample = self._build_single_sample(row, task, template, fields)
            if sample is not None:
                samples.append(sample)
                if quota is not None and len(samples) >= quota:
                    break

        return samples

    def _build_single_sample(self, row, task: str, template: dict, fields: dict) -> dict | None:
        """Build a single SFT sample from a user row."""
        uid = int(row["uid"])

        if task == "video":
            return self._build_video_sample(row, uid, template)
        elif task == "ad":
            return self._build_ad_sample(row, uid, template)
        elif task == "product":
            return self._build_product_sample(row, uid, template)
        elif task == "label_cond":
            return self._build_label_cond_sample(row, uid, template)
        elif task == "label_pred":
            return self._build_label_pred_samples(row, uid, template)
        return None

    def _build_video_sample(self, row, uid: int, template: dict) -> dict | None:
        hist_pids = row.get("hist_video_pid")
        target_pids = row.get("target_video_pid")
        if hist_pids is None or target_pids is None:
            return None
        hist_pids = np.asarray(hist_pids)
        target_pids = np.asarray(target_pids)
        if len(hist_pids) == 0 or len(target_pids) == 0:
            return None

        hist_sids = _pids_to_sids_str(hist_pids, self.pid2sid)
        target_sids = _pids_to_sids_str(target_pids, self.pid2sid)
        if not hist_sids or not target_sids:
            return None

        user_content = template["user_format"].format(hist_sids=hist_sids)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": template["system"]}]},
            {"role": "user", "content": [{"type": "text", "text": user_content}]},
            {"role": "assistant", "content": [{"type": "text", "text": target_sids}]},
        ]

        return {
            "messages": messages,
            "metadata": {"uid": uid, "answer": target_sids, "answer_pid": list(int(p) for p in target_pids)},
            "task_type": "video",
            "hist_pid": list(int(p) for p in hist_pids),
        }

    def _build_ad_sample(self, row, uid: int, template: dict) -> dict | None:
        hist_longview = row.get("hist_video_longview")
        hist_video_pid = row.get("hist_video_pid")
        hist_ad_pid = row.get("hist_ad_pid")
        target_ad_pid = row.get("target_ad_pid")

        if hist_video_pid is None or hist_ad_pid is None or target_ad_pid is None:
            return None

        hist_video_pid = np.asarray(hist_video_pid)
        hist_ad_pid = np.asarray(hist_ad_pid)
        target_ad_pid = np.asarray(target_ad_pid)
        if hist_longview is not None:
            hist_longview = np.asarray(hist_longview)

        if len(hist_ad_pid) == 0 or len(target_ad_pid) == 0:
            return None

        # Use longview-filtered PIDs for the "interested videos" section
        if hist_longview is not None and len(hist_longview) == len(hist_video_pid):
            longview_mask = np.asarray(hist_longview, dtype=bool)
            interested_video_pids = hist_video_pid[longview_mask]
        else:
            interested_video_pids = hist_video_pid[:100]

        hist_longview_sids = _pids_to_sids_str(interested_video_pids, self.pid2sid)
        hist_ad_sids = _pids_to_sids_str(hist_ad_pid, self.pid2sid)
        target_sids = _pids_to_sids_str(target_ad_pid, self.pid2sid)

        if not hist_ad_sids or not target_sids:
            return None

        user_content = template["user_format"].format(
            hist_longview_sids=hist_longview_sids,
            hist_ad_sids=hist_ad_sids,
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": template["system"]}]},
            {"role": "user", "content": [{"type": "text", "text": user_content}]},
            {"role": "assistant", "content": [{"type": "text", "text": target_sids}]},
        ]

        return {
            "messages": messages,
            "metadata": {"uid": uid, "answer": target_sids, "answer_pid": list(int(p) for p in target_ad_pid)},
            "task_type": "ad",
            "hist_longview": list(int(p) for p in interested_video_pids),
            "hist_ad": list(int(p) for p in hist_ad_pid),
        }

    def _build_product_sample(self, row, uid: int, template: dict) -> dict | None:
        hist_longview = row.get("hist_video_longview")
        hist_video_pid = row.get("hist_video_pid")
        hist_goods_pid = row.get("hist_goods_pid")
        target_goods_pid = row.get("target_goods_pid")

        if hist_video_pid is None or hist_goods_pid is None or target_goods_pid is None:
            return None

        hist_video_pid = np.asarray(hist_video_pid)
        hist_goods_pid = np.asarray(hist_goods_pid)
        target_goods_pid = np.asarray(target_goods_pid)
        if hist_longview is not None:
            hist_longview = np.asarray(hist_longview)

        if len(hist_goods_pid) == 0 or len(target_goods_pid) == 0:
            return None

        # Longview-filtered videos
        if hist_longview is not None and len(hist_longview) == len(hist_video_pid):
            longview_mask = np.asarray(hist_longview, dtype=bool)
            interested_video_pids = hist_video_pid[longview_mask]
        else:
            interested_video_pids = hist_video_pid[:100]

        hist_longview_sids = _pids_to_sids_str(interested_video_pids, self.pid2sid)
        hist_goods_sids = _pids_to_sids_str(hist_goods_pid, self.pid2sid_product)
        target_sids = _pids_to_sids_str(target_goods_pid, self.pid2sid_product)

        if not hist_goods_sids or not target_sids:
            return None

        user_content = template["user_format"].format(
            hist_longview_sids=hist_longview_sids,
            hist_goods_sids=hist_goods_sids,
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": template["system"]}]},
            {"role": "user", "content": [{"type": "text", "text": user_content}]},
            {"role": "assistant", "content": [{"type": "text", "text": target_sids}]},
        ]

        return {
            "messages": messages,
            "metadata": {"uid": uid, "answer": target_sids, "answer_pid": list(int(p) for p in target_goods_pid)},
            "task_type": "product",
            "hist_longview": list(int(p) for p in interested_video_pids),
            "hist_goods": list(int(p) for p in hist_goods_pid),
        }

    def _build_label_cond_sample(self, row, uid: int, template: dict) -> dict | None:
        hist_video_pid = row.get("hist_video_pid")
        hist_longview = row.get("hist_video_longview")
        hist_like = row.get("hist_video_like")
        hist_forward = row.get("hist_video_forward")
        target_longview = row.get("target_video_longview")
        target_video_pid = row.get("target_video_pid")

        if hist_video_pid is None or target_video_pid is None or target_longview is None:
            return None

        hist_video_pid = np.asarray(hist_video_pid)
        target_video_pid = np.asarray(target_video_pid)
        target_longview = np.asarray(target_longview)

        if len(hist_video_pid) == 0 or len(target_video_pid) == 0:
            return None

        # Build per-label PID lists from history
        if hist_longview is not None:
            hist_longview = np.asarray(hist_longview)
            longview_pids = hist_video_pid[hist_longview.astype(bool)] if len(hist_longview) == len(hist_video_pid) else []
        else:
            longview_pids = []

        if hist_like is not None:
            hist_like = np.asarray(hist_like)
            like_pids = hist_video_pid[hist_like.astype(bool)] if len(hist_like) == len(hist_video_pid) else []
        else:
            like_pids = []

        if hist_forward is not None:
            hist_forward = np.asarray(hist_forward)
            forward_pids = hist_video_pid[hist_forward.astype(bool)] if len(hist_forward) == len(hist_video_pid) else []
        else:
            forward_pids = []

        # Target: PIDs where target_longview == 1
        target_positive_pids = target_video_pid[target_longview.astype(bool)]
        if len(target_positive_pids) == 0:
            return None

        hist_longview_sids = _pids_to_sids_str(longview_pids, self.pid2sid)
        hist_like_sids = _pids_to_sids_str(like_pids, self.pid2sid)
        hist_forward_sids = _pids_to_sids_str(forward_pids, self.pid2sid)
        target_sids = _pids_to_sids_str(target_positive_pids, self.pid2sid)

        if not target_sids:
            return None

        user_content = template["user_format"].format(
            hist_longview_sids=hist_longview_sids,
            hist_like_sids=hist_like_sids,
            hist_forward_sids=hist_forward_sids,
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": template["system"]}]},
            {"role": "user", "content": [{"type": "text", "text": user_content}]},
            {"role": "assistant", "content": [{"type": "text", "text": target_sids}]},
        ]

        return {
            "messages": messages,
            "metadata": {"uid": uid, "answer": target_sids, "answer_pid": list(int(p) for p in target_positive_pids)},
            "task_type": "label_cond",
            "hist_longview": list(int(p) for p in longview_pids),
        }

    def _build_label_pred_samples(self, row, uid: int, template: dict) -> dict | None:
        """Build label_pred sample: binary yes/no prediction for a candidate item."""
        hist_video_pid = row.get("hist_video_pid")
        hist_longview = row.get("hist_video_longview")
        hist_like = row.get("hist_video_like")
        target_video_pid = row.get("target_video_pid")
        target_longview = row.get("target_video_longview")

        if hist_video_pid is None or target_video_pid is None or target_longview is None:
            return None

        hist_video_pid = np.asarray(hist_video_pid)
        target_video_pid = np.asarray(target_video_pid)
        target_longview = np.asarray(target_longview)

        if len(hist_video_pid) == 0 or len(target_video_pid) == 0:
            return None

        # Build history label lists
        if hist_longview is not None:
            hist_longview = np.asarray(hist_longview)
            longview_pids = hist_video_pid[hist_longview.astype(bool)] if len(hist_longview) == len(hist_video_pid) else hist_video_pid[:10]
        else:
            longview_pids = hist_video_pid[:10]

        if hist_like is not None:
            hist_like = np.asarray(hist_like)
            like_pids = hist_video_pid[hist_like.astype(bool)] if len(hist_like) == len(hist_video_pid) else []
        else:
            like_pids = []

        # Pick a random target item for binary prediction
        idx = random.randint(0, len(target_video_pid) - 1)
        candidate_pid = int(target_video_pid[idx])
        label = int(target_longview[idx])
        answer = "是" if label == 1 else "否"

        candidate_sid = _pids_to_sids_str([candidate_pid], self.pid2sid)
        if not candidate_sid:
            return None

        hist_longview_sids = _pids_to_sids_str(longview_pids, self.pid2sid)
        hist_like_sids = _pids_to_sids_str(like_pids, self.pid2sid)

        user_content = template["user_format"].format(
            hist_longview_sids=hist_longview_sids,
            hist_like_sids=hist_like_sids,
            candidate_sid=candidate_sid,
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": template["system"]}]},
            {"role": "user", "content": [{"type": "text", "text": user_content}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]

        return {
            "messages": messages,
            "metadata": {"uid": uid, "answer": answer},
            "task_type": "label_pred",
            "hist_longview": list(int(p) for p in longview_pids),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample with optional context enrichment."""
        sample = self.samples[idx].copy()

        if self.card_source == "none":
            return sample
        elif self.card_source == "llm_optimized":
            return self._enrich_from_cache(sample)
        elif self.card_source == "heuristic" and self.planner and self.executor:
            return self._enrich_sample(sample)

        return sample

    def _extract_hist_pids_for_task(self, sample: dict, task_type: str) -> list[int]:
        """Pick the correct history column for the given task.

        This is the v2 fix: previously a fixed priority list always hit a video
        column for ad/product, so the card described video preferences even for
        ad/product targets. Now we use a per-task ordered fallback.
        """
        cols = TASK_HIST_COLS.get(task_type, ["hist_pid", "hist_longview", "hist_ad", "hist_goods"])
        for col in cols:
            if col in sample and sample[col] is not None and len(sample[col]) > 0:
                return [int(p) for p in sample[col]]
        return []

    def _enrich_from_cache(self, sample: dict) -> dict:
        """Inject pre-computed card from cache. Zero LLM calls."""
        metadata = sample.get("metadata", {})
        uid = int(metadata.get("uid", 0))
        task_type = sample.get("task_type", "video")

        task_cache = self._card_cache.get(task_type, {})
        card = task_cache.get(uid, "")

        if card:
            messages = [dict(m) for m in sample["messages"]]
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    user_content = messages[i].get("content", "")
                    if isinstance(user_content, list):
                        user_content = " ".join(
                            [c.get("text", "") for c in user_content if isinstance(c, dict)]
                        )
                    messages[i]["content"] = user_content + "\n\n" + card
                    break
            sample["messages_enriched"] = messages
            sample["context_card"] = card
        else:
            sample["messages_enriched"] = sample["messages"]
            sample["context_card"] = ""

        return sample

    def _enrich_sample(self, sample: dict) -> dict:
        """Enrich sample with agentic context card."""
        try:
            metadata = sample.get("metadata", {})
            uid = int(metadata.get("uid", 0))
            task_type = sample.get("task_type", "video")

            # v2 fix: task-aware history extraction
            hist_pids = self._extract_hist_pids_for_task(sample, task_type)

            hist_labels = None

            # v2 fix: pass video/product mappings so cross_domain works for ad/product
            plan = self.planner.plan(
                uid=uid,
                task_type=task_type,
                hist_pids=hist_pids,
                hist_labels=hist_labels,
                pid2sid=self.pid2sid,
                pid2sid_video=self.pid2sid,
                pid2sid_product=self.pid2sid_product,
            )

            context_card = self.executor.execute(plan)

            # Inject context card into the user message (before the assistant answer)
            messages = [dict(m) for m in sample["messages"]]
            last_user_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    last_user_idx = i
                    break

            if last_user_idx is not None:
                user_content = messages[last_user_idx].get("content", "")
                if isinstance(user_content, list):
                    user_content = " ".join([c.get("text", "") for c in user_content if isinstance(c, dict)])
                messages[last_user_idx]["content"] = user_content + "\n\n" + context_card

            sample["messages_enriched"] = messages
            sample["context_card"] = context_card

        except Exception as e:
            logger.warning("Failed to enrich sample (uid=%s): %s", uid, e)
            sample["messages_enriched"] = sample["messages"]
            sample["context_card"] = ""

        return sample


class GeneralSFTDataset(Dataset):
    """General SFT dataset (non-recommendation tasks)."""

    def __init__(
        self,
        data_dir: str = "data/general_sft",
        max_samples: int | None = None,
        seed: int = 42,
    ):
        """Initialize General SFT dataset.

        Args:
            data_dir: Path to general_sft directory.
            max_samples: Maximum samples to load.
            seed: Random seed for sampling.
        """
        self.data_dir = Path(data_dir)
        self.seed = seed

        random.seed(seed)

        # Load samples
        self.samples = self._load_samples(max_samples)
        logger.info("GeneralSFTDataset loaded: %d samples", len(self.samples))

    def _load_samples(self, max_samples: int | None) -> list[dict]:
        """Load samples from general_sft parquet files."""
        parquet_files = list(self.data_dir.glob("filtered_*.parquet"))
        logger.info("Found %d general_sft parquet files", len(parquet_files))

        all_samples = []

        # If max_samples specified, sample files first to avoid loading all 301 files
        if max_samples:
            # Estimate samples per file (~7K average)
            files_needed = min(len(parquet_files), (max_samples // 7000) + 1)
            parquet_files = random.sample(parquet_files, files_needed)

        for file_path in tqdm(parquet_files, desc="Loading general_sft"):
            df = pd.read_parquet(file_path)
            for _, row in df.iterrows():
                all_samples.append(row.to_dict())

                if max_samples and len(all_samples) >= max_samples:
                    break

            if max_samples and len(all_samples) >= max_samples:
                break

        # Final sampling if needed
        if max_samples and len(all_samples) > max_samples:
            all_samples = random.sample(all_samples, max_samples)

        return all_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample."""
        sample = self.samples[idx].copy()

        # Parse metadata if string
        if isinstance(sample.get("metadata"), str):
            sample["metadata"] = json.loads(sample["metadata"])

        # Parse messages if string
        if isinstance(sample.get("messages"), str):
            sample["messages"] = json.loads(sample["messages"])

        return sample


class CombinedDataset(Dataset):
    """Combined dataset: 80% RecIF + 20% General_SFT."""

    def __init__(
        self,
        recif_dataset: RecIFDataset,
        general_sft_dataset: GeneralSFTDataset,
        recif_ratio: float = 0.8,
        seed: int = 42,
    ):
        """Initialize combined dataset.

        Args:
            recif_dataset: RecIF dataset instance.
            general_sft_dataset: General SFT dataset instance.
            recif_ratio: Ratio of RecIF samples (default 0.8 for 80%).
            seed: Random seed for shuffling.
        """
        self.recif_dataset = recif_dataset
        self.general_sft_dataset = general_sft_dataset
        self.recif_ratio = recif_ratio
        self.seed = seed

        random.seed(seed)

        # Build index mapping
        self.indices = self._build_indices()
        logger.info(
            "CombinedDataset: %d samples (RecIF=%d, General=%d)",
            len(self.indices),
            sum(1 for src, _ in self.indices if src == "recif"),
            sum(1 for src, _ in self.indices if src == "general"),
        )

    def _build_indices(self) -> list[tuple[str, int]]:
        """Build shuffled index mapping."""
        # Calculate target counts
        total_recif = len(self.recif_dataset)
        total_general = len(self.general_sft_dataset)

        # Target: recif_ratio of total samples should be RecIF
        # If recif_ratio=0.8: recif / (recif + general) = 0.8
        # => general = recif * (1 - 0.8) / 0.8 = recif * 0.25
        target_general = int(total_recif * (1 - self.recif_ratio) / self.recif_ratio)
        target_general = min(target_general, total_general)

        # Build indices
        indices = []
        indices.extend([("recif", i) for i in range(total_recif)])
        indices.extend([("general", i) for i in range(target_general)])

        # Shuffle
        random.shuffle(indices)

        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample from either dataset."""
        source, source_idx = self.indices[idx]

        if source == "recif":
            sample = self.recif_dataset[source_idx]
            sample["source"] = "recif"
        else:
            sample = self.general_sft_dataset[source_idx]
            sample["source"] = "general_sft"

        return sample


def format_sample_for_training(sample: dict, tokenizer, max_length: int = 4096) -> dict | None:
    """Format a sample for training (apply chat template and tokenize).

    Produces labels with -100 masking on the prompt portion so that loss is
    computed only on the last assistant turn (the answer).

    NOTE: This function does NOT mutate the input `sample`. It builds a local
    copy of the messages (flattening any list-typed content) so that repeated
    calls on the same sample are idempotent and worker-safe.

    Args:
        sample: Sample dict with 'messages' or 'messages_enriched' field.
        tokenizer: HuggingFace tokenizer with chat template.
        max_length: Maximum sequence length for tokenization.

    Returns:
        Dict with 'input_ids', 'attention_mask', and 'labels' lists,
        or None if the sample should be skipped (prompt alone exceeds max_length).
    """
    raw_messages = sample.get("messages_enriched") or sample.get("messages", [])

    # Build a local, flattened copy WITHOUT mutating the source sample.
    messages = []
    for msg in raw_messages:
        content = msg.get("content")
        if isinstance(content, list):
            content = " ".join([c.get("text", "") for c in content if isinstance(c, dict)])
        messages.append({"role": msg.get("role", "user"), "content": content})

    if len(messages) < 2:
        return None

    # Split: prompt = all messages except last assistant; full = everything
    prompt_messages = messages[:-1]

    # Tokenize prompt-only (with generation prompt appended) to find boundary
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )["input_ids"]

    prompt_len = len(prompt_ids)

    # If prompt alone exceeds max_length, skip this sample entirely
    if prompt_len >= max_length:
        return None

    # Tokenize full sequence (prompt + assistant answer)
    full_encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
    )

    input_ids = full_encoded["input_ids"]

    # Custom truncation preserving answer (see truncate_preserving_answer)
    answer_len = len(input_ids) - prompt_len
    if len(input_ids) > max_length:
        input_ids = truncate_preserving_answer(
            input_ids, prompt_len, answer_len, max_length
        )
        if input_ids is None:
            return None
        # Recalculate: answer is always at the tail
        prompt_len = len(input_ids) - answer_len

    attention_mask = [1] * len(input_ids)

    # Build labels: -100 for prompt, real ids for answer
    labels = [-100] * prompt_len + input_ids[prompt_len:]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def truncate_preserving_answer(
    input_ids: list[int],
    prompt_len: int,
    answer_len: int,
    max_len: int,
) -> list[int] | None:
    """Truncate from the middle of the prompt, never touching the answer.

    Strategy: the answer (tail) is always preserved. If the sequence exceeds
    max_len, tokens are removed from the prompt portion. The first ~128 tokens
    (system prompt header) and last tokens of the prompt (closest to answer) are
    kept; middle content is dropped.

    Returns None if even after removing all droppable prompt content the answer
    still doesn't fit.
    """
    if answer_len > max_len:
        return None

    total = prompt_len + answer_len
    excess = total - max_len
    if excess <= 0:
        return input_ids[:total]

    # Keep the first 128 tokens of prompt (system header) and the tail
    head_keep = min(128, prompt_len)
    tail_start = prompt_len  # answer starts here
    # The prompt tail (close to answer) is more valuable — keep it
    prompt_tail_keep = max(0, prompt_len - head_keep - excess)
    new_prompt_len = head_keep + prompt_tail_keep

    if new_prompt_len + answer_len > max_len:
        # Still too long — trim more from prompt tail
        new_prompt_len = max_len - answer_len
        if new_prompt_len < head_keep:
            return None

    # Reconstruct: [head_keep tokens] + [last prompt_tail_keep tokens of prompt] + [answer]
    prompt_tail_start = prompt_len - (new_prompt_len - head_keep)
    result = (
        input_ids[:head_keep]
        + input_ids[prompt_tail_start:prompt_len]
        + input_ids[prompt_len:prompt_len + answer_len]
    )
    return result