"""RecIF evaluation runner.

Evaluates a model checkpoint on all 8 RecIF benchmark tasks.
CRITICAL: Dynamically generates context cards for test samples using the
same Planner + Executor + offline stores used during training to prevent
distribution shift collapse.

Metrics:
  - SID-based tasks (video, ad, product, label_cond, interactive):
      PASS@1, PASS@32, Recall@32
  - Binary tasks (label_pred): Accuracy
  - Free-form tasks (rec_reason, item_understand): ROUGE-L / exact match
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from arec2.agents.executor_agent import ExecutorAgent
from arec2.agents.planner_agent import PlannerAgent
from arec2.base_model.openonerec_wrapper import OpenOneRecWrapper

logger = logging.getLogger(__name__)

SID_TASKS = {"video", "ad", "product", "label_cond", "interactive"}
BINARY_TASKS = {"label_pred"}
FREEFORM_TASKS = {"rec_reason", "item_understand"}

_SID_TOKEN_RE = re.compile(r"<s_[abc]_\d+>")


@dataclass
class TaskResult:
    task: str
    n_samples: int
    metrics: dict[str, float] = field(default_factory=dict)


class RecIFRunner:
    """Unified RecIF benchmark evaluator."""

    def __init__(
        self,
        model: OpenOneRecWrapper,
        planner: Optional[PlannerAgent] = None,
        executor: Optional[ExecutorAgent] = None,
        pid2sid: Optional[dict] = None,
        pid2sid_product: Optional[dict] = None,
        enrich_test: bool = True,
        benchmark_dir: str = "data/recif/benchmark_data",
        batch_size: int = 1,
    ):
        self.model = model
        self.planner = planner
        self.executor = executor
        self.pid2sid = pid2sid or {}
        self.pid2sid_product = pid2sid_product or {}
        self.enrich_test = enrich_test
        self.benchmark_dir = Path(benchmark_dir)
        self.batch_size = batch_size

    def evaluate_all(
        self,
        tasks: list[str] | None = None,
        max_samples_per_task: int | None = None,
    ) -> list[TaskResult]:
        """Evaluate on all (or specified) tasks.

        Args:
            tasks: List of task names. None = all available.
            max_samples_per_task: Cap per task (for quick testing).

        Returns:
            List of TaskResult objects.
        """
        if tasks is None:
            tasks = sorted(
                [d.stem for d in self.benchmark_dir.iterdir() if d.is_dir()]
            )

        results = []
        for task in tasks:
            task_dir = self.benchmark_dir / task
            test_file = task_dir / f"{task}_test.parquet"
            if not test_file.exists():
                logger.warning("Test file not found: %s, skipping", test_file)
                continue

            logger.info("Evaluating task: %s", task)
            result = self._evaluate_task(task, test_file, max_samples_per_task)
            results.append(result)
            logger.info("  %s: %s", task, result.metrics)

        return results

    def _evaluate_task(
        self, task: str, test_file: Path, max_samples: int | None
    ) -> TaskResult:
        """Evaluate a single task."""
        df = pd.read_parquet(test_file)
        if max_samples:
            df = df.head(max_samples)

        if task in SID_TASKS:
            return self._eval_sid_task(task, df)
        elif task in BINARY_TASKS:
            return self._eval_binary_task(task, df)
        else:
            return self._eval_freeform_task(task, df)

    def _eval_sid_task(self, task: str, df: pd.DataFrame) -> TaskResult:
        """Evaluate SID-based tasks via PASS@1, PASS@32, Recall@32.

        Single greedy generation per sample. The model outputs a ranked list
        of SIDs in one response; metrics are computed over that list:
          PASS@1    = 1 if top-1 predicted SID is in GT
          PASS@32   = 1 if any of the top-32 predicted SIDs is in GT
          Recall@32 = |top-32 pred SIDs ∩ GT| / |GT|
        """
        samples = []
        for _, row in df.iterrows():
            metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
            raw_messages = json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"]

            gt_answer = metadata.get("answer", "")
            gt_sids = self._parse_sids(gt_answer)
            if not gt_sids:
                continue

            messages = self.model.format_messages_from_benchmark(raw_messages)
            if self.enrich_test and self.planner and self.executor:
                messages = self._enrich_messages(messages, row, task)
            if messages[-1]["role"] == "assistant":
                messages = messages[:-1]

            samples.append({"messages": messages, "gt_sids": gt_sids})

        pass1_scores, pass32_scores, recall32_scores = [], [], []

        for i in tqdm(range(0, len(samples), self.batch_size), desc=f"Eval {task}"):
            batch = samples[i: i + self.batch_size]
            msgs_batch = [s["messages"] for s in batch]

            if len(msgs_batch) == 1:
                preds = self.model.generate(
                    msgs_batch[0], max_new_tokens=256,
                    temperature=None, do_sample=False,
                    num_return=1, num_beams=1,
                )
            else:
                preds = self.model.generate_batch(
                    msgs_batch, max_new_tokens=256,
                    temperature=None, do_sample=False,
                    num_return=1, num_beams=1,
                )

            for j, pred_text in enumerate(preds):
                gt_set = set(batch[j]["gt_sids"])
                pred_sids = self._parse_sids(pred_text)

                pass1_scores.append(1.0 if pred_sids and pred_sids[0] in gt_set else 0.0)
                top32 = pred_sids[:32]
                pass32_scores.append(1.0 if any(s in gt_set for s in top32) else 0.0)
                recall32_scores.append(
                    len(set(top32) & gt_set) / len(gt_set) if gt_set else 0.0
                )

        return TaskResult(
                task=task,
                n_samples=len(pass1_scores),
                metrics={
                    "PASS@1": np.mean(pass1_scores) if pass1_scores else 0.0,
                    "PASS@32": np.mean(pass32_scores) if pass32_scores else 0.0,
                    "Recall@32": np.mean(recall32_scores) if recall32_scores else 0.0,
                },
            )

    def _eval_binary_task(self, task: str, df: pd.DataFrame) -> TaskResult:
        """Evaluate binary prediction tasks (batched)."""
        samples = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Prep {task}"):
            metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
            raw_messages = json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"]

            gt_answer = metadata.get("answer", "")
            messages = self.model.format_messages_from_benchmark(raw_messages)
            if self.enrich_test and self.planner and self.executor:
                messages = self._enrich_messages(messages, row, task)
            if messages[-1]["role"] == "assistant":
                messages = messages[:-1]

            samples.append({"messages": messages, "gt_answer": gt_answer})

        correct = 0
        total = 0
        for i in tqdm(range(0, len(samples), self.batch_size), desc=f"Eval {task}"):
            batch = samples[i : i + self.batch_size]
            msgs_batch = [s["messages"] for s in batch]

            if len(msgs_batch) == 1:
                preds = self.model.generate(msgs_batch[0], max_new_tokens=10, temperature=0.0, do_sample=False, num_return=1)
            else:
                preds = self.model.generate_batch(msgs_batch, max_new_tokens=10, temperature=0.0, do_sample=False)

            for j, pred in enumerate(preds):
                pred = pred.strip()
                gt = batch[j]["gt_answer"]
                if pred in gt or gt in pred:
                    correct += 1
                total += 1

        return TaskResult(
            task=task,
            n_samples=total,
            metrics={"accuracy": correct / total if total > 0 else 0.0},
        )

    def _eval_freeform_task(self, task: str, df: pd.DataFrame) -> TaskResult:
        """Evaluate free-form generation tasks (batched)."""
        samples = []
        for _, row in df.iterrows():
            metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
            raw_messages = json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"]

            gt_answer = metadata.get("answer", "")
            messages = self.model.format_messages_from_benchmark(raw_messages)
            if messages[-1]["role"] == "assistant":
                messages = messages[:-1]

            samples.append({"messages": messages, "gt_answer": gt_answer})

        rouge_scores = []
        for i in tqdm(range(0, len(samples), self.batch_size), desc=f"Eval {task}"):
            batch = samples[i : i + self.batch_size]
            msgs_batch = [s["messages"] for s in batch]

            if len(msgs_batch) == 1:
                preds = self.model.generate(msgs_batch[0], max_new_tokens=512, temperature=0.1, do_sample=False, num_return=1)
            else:
                preds = self.model.generate_batch(msgs_batch, max_new_tokens=512, temperature=0.1, do_sample=False)

            for j, pred in enumerate(preds):
                rouge_scores.append(self._rouge_l(pred, batch[j]["gt_answer"]))

        return TaskResult(
            task=task,
            n_samples=len(rouge_scores),
            metrics={"ROUGE-L": np.mean(rouge_scores) if rouge_scores else 0.0},
        )

    def _enrich_messages(self, messages: list[dict], row, task: str) -> list[dict]:
        """Enrich test messages with context card using planner + executor."""
        try:
            hist_pids = []
            for col in ["hist_pid", "hist_longview", "hist_ad", "hist_goods"]:
                if col in row.index and row[col] is not None:
                    hist_pids = [int(p) for p in row[col]]
                    break

            metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
            uid = int(metadata.get("uid", 0))

            plan = self.planner.plan(
                uid=uid,
                task_type=task,
                hist_pids=hist_pids,
                pid2sid=self.pid2sid,
            )

            card = self.executor.execute(plan)

            # Inject card before the last user message
            enriched = list(messages)
            for i in range(len(enriched) - 1, -1, -1):
                if enriched[i]["role"] == "user":
                    enriched[i] = dict(enriched[i])
                    enriched[i]["content"] = enriched[i]["content"] + "\n\n" + card
                    break

            return enriched

        except Exception as e:
            logger.debug("Enrichment failed for task=%s: %s", task, e)
            return messages

    @staticmethod
    def _parse_sids(text: str) -> list[str]:
        """Parse SID triples from text.

        Matches <s_a_X><s_b_Y><s_c_Z> tokens and groups them into triples.
        Does not require <|sid_begin|>/<|sid_end|> wrappers (they get stripped
        by skip_special_tokens=True during decoding).
        """
        tokens = _SID_TOKEN_RE.findall(text)
        items = []
        for i in range(0, len(tokens) - 2, 3):
            items.append(tokens[i] + tokens[i + 1] + tokens[i + 2])
        return items

    @staticmethod
    def _rouge_l(pred: str, ref: str) -> float:
        """Compute ROUGE-L F1 score (character-level for Chinese)."""
        if not pred or not ref:
            return 0.0
        pred_chars = list(pred)
        ref_chars = list(ref)

        m, n = len(pred_chars), len(ref_chars)
        if m == 0 or n == 0:
            return 0.0

        # LCS via DP (truncate for efficiency)
        max_len = 500
        pred_chars = pred_chars[:max_len]
        ref_chars = ref_chars[:max_len]
        m, n = len(pred_chars), len(ref_chars)

        prev = [0] * (n + 1)
        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            for j in range(1, n + 1):
                if pred_chars[i - 1] == ref_chars[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(curr[j - 1], prev[j])
            prev = curr

        lcs_len = prev[n]
        precision = lcs_len / m
        recall = lcs_len / n
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
