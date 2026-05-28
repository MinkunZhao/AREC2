"""LabelBehaviorTool: slices history by interaction type (like, follow, etc.).

Can use the pre-built LabelBehaviorStore or build label-conditioned SID lists
on-the-fly from benchmark sample fields (hist_pids + label arrays).
"""

from __future__ import annotations

from typing import Optional

from arec2.retrieval.stores import LabelBehaviorStore
from arec2.tools.tool_base import Tool, ToolResult


class LabelBehaviorTool(Tool):
    name = "label_behavior"
    description = "Slice user history by interaction labels"
    node_type = "label_condition"

    def __init__(self, store: Optional[LabelBehaviorStore] = None):
        self.store = store

    def run(
        self,
        uid: int,
        labels: list[str] | None = None,
        top_k: int = 10,
        hist_pids: list | None = None,
        hist_labels: dict | None = None,
        pid2sid: dict | None = None,
        **kwargs,
    ) -> ToolResult:
        if labels is None:
            labels = ["longview", "like", "follow", "forward", "not_interested"]

        all_behaviors = None
        if self.store is not None:
            all_behaviors = self.store.query(uid)

        if all_behaviors is None and hist_pids is not None and pid2sid is not None:
            all_behaviors = self._build_inline_behaviors(hist_pids, hist_labels, pid2sid)

        if all_behaviors is None:
            return ToolResult(
                tool_name=self.name,
                node_type=self.node_type,
                data={"behaviors": {}, "summary": "No behavior data"},
                confidence=0.0,
            )

        behaviors = {}
        behavior_strs = {}
        total_signals = 0
        for label in labels:
            sids = all_behaviors.get(label, [])
            truncated = sids[-top_k:] if len(sids) > top_k else sids
            behaviors[label] = truncated
            behavior_strs[label] = self.sids_to_compact(truncated, max_items=top_k)
            total_signals += len(truncated)

        dominant = max(labels, key=lambda l: len(behaviors.get(l, [])))

        return ToolResult(
            tool_name=self.name,
            node_type=self.node_type,
            data={
                "behaviors": behaviors,
                "behavior_strs": behavior_strs,
                "counts": {l: len(behaviors.get(l, [])) for l in labels},
                "dominant_label": dominant,
                "summary": f"dominant={dominant}|signals={total_signals}",
            },
            confidence=1.0,
            token_cost=total_signals * 5 + 20,
        )

    def _build_inline_behaviors(
        self, hist_pids: list, hist_labels: dict | None, pid2sid: dict,
    ) -> dict:
        label_map = {
            "longview": "longview",
            "like": "like",
            "follow": "follow",
            "forward": "forward",
            "not_interested": "not_interested",
        }
        result = {l: [] for l in label_map}
        labels = hist_labels or {}
        n = len(hist_pids)

        for label_key, out_key in label_map.items():
            arr = labels.get(label_key)
            if arr is None:
                continue
            for i in range(min(n, len(arr))):
                if arr[i]:
                    pid = int(hist_pids[i])
                    sid = pid2sid.get(pid)
                    if sid:
                        result[out_key].append(list(sid))

        if any(len(v) > 0 for v in result.values()):
            return result
        return None
