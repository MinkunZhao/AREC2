"""Ranking metrics for Amazon sequential recommendation (paper §6.1).

We report Recall@K and NDCG@K for K in {5, 10}. In the leave-one-out protocol
there is exactly ONE ground-truth target per user, so:

  - Recall@K = 1 if the target is in the top-K ranked predictions, else 0
    (averaged over users; equals HitRate@K for single-target LOO).
  - NDCG@K = 1 / log2(rank + 1) if the target is at position `rank` (1-indexed)
    within the top-K, else 0.

Predictions are an ordered list of candidate itemic-token strings (the model's
ranking). The evaluator constrains the candidate space to in-corpus items, so
matching is by canonical SID string equality.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

# Canonical SID body matcher, wrapper-agnostic (mirrors the project's parser).
_SID_BODY_RE = re.compile(r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")


def canonical_sid(text: str) -> str | None:
    """Extract the first (c1,c2,c3) SID body from text -> '<s_a_..><s_b_..><s_c_..>'."""
    m = _SID_BODY_RE.search(text)
    if not m:
        return None
    a, b, c = m.groups()
    return f"<s_a_{a}><s_b_{b}><s_c_{c}>"


def parse_ranked_sids(text: str) -> list[str]:
    """Parse a generation into an ordered, de-duplicated list of SID bodies."""
    out: list[str] = []
    seen: set[str] = set()
    for a, b, c in _SID_BODY_RE.findall(text):
        sid = f"<s_a_{a}><s_b_{b}><s_c_{c}>"
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def recall_at_k(ranked: list[str], target: str, k: int) -> float:
    return 1.0 if target in ranked[:k] else 0.0


def ndcg_at_k(ranked: list[str], target: str, k: int) -> float:
    top = ranked[:k]
    for rank, sid in enumerate(top, start=1):
        if sid == target:
            return 1.0 / math.log2(rank + 1)
    return 0.0


@dataclass
class MetricAccumulator:
    """Accumulates Recall@K / NDCG@K over users for K in ks."""

    ks: tuple[int, ...] = (5, 10)
    recall_sums: dict[int, float] = field(default_factory=dict)
    ndcg_sums: dict[int, float] = field(default_factory=dict)
    n: int = 0

    def __post_init__(self):
        self.recall_sums = {k: 0.0 for k in self.ks}
        self.ndcg_sums = {k: 0.0 for k in self.ks}

    def update(self, ranked: list[str], target: str):
        self.n += 1
        for k in self.ks:
            self.recall_sums[k] += recall_at_k(ranked, target, k)
            self.ndcg_sums[k] += ndcg_at_k(ranked, target, k)

    def result(self) -> dict[str, float]:
        if self.n == 0:
            return {f"recall@{k}": 0.0 for k in self.ks} | {
                f"ndcg@{k}": 0.0 for k in self.ks
            }
        out: dict[str, float] = {}
        for k in self.ks:
            out[f"recall@{k}"] = round(self.recall_sums[k] / self.n, 4)
            out[f"ndcg@{k}"] = round(self.ndcg_sums[k] / self.n, 4)
        out["num_users"] = self.n
        return out
    