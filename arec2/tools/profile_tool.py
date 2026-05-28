"""ProfileTool: extracts user long-term preferences.

Can use the pre-built ProfileStore (for users in the main data) or build a
profile on-the-fly from benchmark sample fields (hist_pids + label arrays).
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from arec2.retrieval.stores import ProfileStore
from arec2.tools.tool_base import Tool, ToolResult


class ProfileTool(Tool):
    name = "profile"
    description = "Extract user long-term preference profile"
    node_type = "long_term_profile"

    def __init__(self, store: Optional[ProfileStore] = None):
        self.store = store

    def run(
        self,
        uid: int,
        top_k: int = 10,
        hist_pids: list | None = None,
        hist_labels: dict | None = None,
        pid2sid: dict | None = None,
        **kwargs,
    ) -> ToolResult:
        if self.store is not None:
            profile = self.store.query(uid)
            if profile is not None:
                return self._format_result(profile, top_k)

        if hist_pids is not None and pid2sid is not None:
            profile = self._build_inline_profile(hist_pids, hist_labels, pid2sid)
            return self._format_result(profile, top_k)

        return ToolResult(
            tool_name=self.name,
            node_type=self.node_type,
            data={"summary": "No profile data", "top_sids": [], "stats": {}},
            confidence=0.0,
        )

    def _build_inline_profile(
        self, hist_pids, hist_labels: dict | None, pid2sid: dict,
    ) -> dict:
        n = len(hist_pids)
        sid_counts = Counter()
        positive_sid_counts = Counter()
        longview_count = like_count = follow_count = forward_count = 0

        labels = hist_labels or {}
        likes = labels.get("like")
        longview = labels.get("longview")
        follow = labels.get("follow")
        forward = labels.get("forward")

        for i in range(n):
            pid = int(hist_pids[i])
            sid = pid2sid.get(pid)
            if not sid:
                continue
            sid_tuple = tuple(sid)
            sid_counts[sid_tuple] += 1

            if longview is not None and i < len(longview) and longview[i]:
                longview_count += 1
            if likes is not None and i < len(likes) and likes[i]:
                like_count += 1
                positive_sid_counts[sid_tuple] += 1
            if follow is not None and i < len(follow) and follow[i]:
                follow_count += 1
                positive_sid_counts[sid_tuple] += 1
            if forward is not None and i < len(forward) and forward[i]:
                forward_count += 1
                positive_sid_counts[sid_tuple] += 1

        return {
            "top_sids": [list(s) for s, _ in sid_counts.most_common(20)],
            "top_positive_sids": [list(s) for s, _ in positive_sid_counts.most_common(20)],
            "interaction_stats": {
                "total_video": n,
                "longview_rate": longview_count / n if n > 0 else 0,
                "like_rate": like_count / n if n > 0 else 0,
                "follow_rate": follow_count / n if n > 0 else 0,
                "forward_rate": forward_count / n if n > 0 else 0,
            },
            "domain_distribution": {"video": n, "ad": 0, "product": 0},
        }

    def _format_result(self, profile: dict, top_k: int) -> ToolResult:
        top_sids = profile["top_sids"][:top_k]
        positive_sids = profile["top_positive_sids"][:top_k]
        stats = profile["interaction_stats"]
        domain = profile["domain_distribution"]

        summary_parts = []
        summary_parts.append(
            f"video={domain['video']},ad={domain['ad']},product={domain['product']}"
        )
        if stats["longview_rate"] > 0.3:
            summary_parts.append("high-engagement")
        if stats["like_rate"] > 0.05:
            summary_parts.append("active-liker")
        if stats["follow_rate"] > 0.01:
            summary_parts.append("follows-creators")

        return ToolResult(
            tool_name=self.name,
            node_type=self.node_type,
            data={
                "top_sids": top_sids,
                "positive_sids": positive_sids,
                "stats": stats,
                "domain_distribution": domain,
                "summary": "|".join(summary_parts),
                "top_sids_str": self.sids_to_compact(top_sids, max_items=top_k),
                "positive_sids_str": self.sids_to_compact(positive_sids, max_items=top_k),
            },
            confidence=1.0,
            token_cost=len(top_sids) * 5 + len(positive_sids) * 5 + 20,
        )
