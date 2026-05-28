"""PlannerAgent: rule-based tool selection for dependency-aware context enrichment.

Maps task types to ordered tool call sequences based on:
1. Task requirements (video/ad/product/label_cond/etc.)
2. Available data (hist_pids, hist_labels, candidate_pool)
3. Dependency order (foundation → refinement → expansion)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """Represents a single tool invocation with parameters."""
    tool_name: str
    params: dict[str, Any] = field(default_factory=dict)


class PlannerAgent:
    """Rule-based planner that selects tools based on task type and available data."""

    # Task type → required tool names (in dependency order)
    TASK_TOOL_MAP = {
        "video": ["profile", "recent_intent", "collaborative"],
        "ad": ["profile", "recent_intent", "cross_domain", "collaborative"],
        "product": ["profile", "recent_intent", "cross_domain", "collaborative"],
        "label_cond": ["profile", "label_behavior", "recent_intent", "collaborative"],
        "label_pred": ["profile", "label_behavior", "collaborative"],
        "interactive": ["profile", "recent_intent", "label_behavior"],
        "rec_reason": ["profile", "recent_intent", "label_behavior", "collaborative"],
        "item_understand": [],  # Item-centric, not user-centric
    }

    def __init__(self, token_budget: int = 1024):
        self.token_budget = token_budget

    def plan(
        self,
        uid: int,
        task_type: str = "video",
        hist_pids: list | None = None,
        hist_labels: dict | None = None,
        pid2sid: dict | None = None,
        pid2sid_video: dict | None = None,
        pid2sid_product: dict | None = None,
        candidate_pool: list | None = None,
        **kwargs,
    ) -> list[ToolCall]:
        """Generate ordered tool call plan based on task type and available data.

        Args:
            uid: User ID.
            task_type: One of video/ad/product/label_cond/label_pred/interactive/rec_reason/item_understand.
            hist_pids: User history PIDs (for inline profile building).
            hist_labels: Dict of label arrays (longview/like/follow/forward/not_interested).
            pid2sid: PID→SID mapping (primary, usually video).
            pid2sid_video: Video-specific PID→SID mapping.
            pid2sid_product: Product-specific PID→SID mapping.
            candidate_pool: Candidate SIDs for scoring (optional).
            **kwargs: Additional parameters.

        Returns:
            Ordered list of ToolCall objects.
        """
        tool_names = self.TASK_TOOL_MAP.get(task_type, ["profile", "recent_intent"])

        # Filter tools based on data availability
        if "label_behavior" in tool_names and not hist_labels:
            tool_names = [t for t in tool_names if t != "label_behavior"]

        if "collaborative" in tool_names and (not hist_pids or len(hist_pids) < 3):
            tool_names = [t for t in tool_names if t != "collaborative"]

        # Build tool calls with appropriate parameters
        tool_calls = []
        for tool_name in tool_names:
            params = self._build_params(
                tool_name=tool_name,
                uid=uid,
                task_type=task_type,
                hist_pids=hist_pids,
                hist_labels=hist_labels,
                pid2sid=pid2sid or pid2sid_video,
                pid2sid_video=pid2sid_video,
                pid2sid_product=pid2sid_product,
                **kwargs,
            )
            tool_calls.append(ToolCall(tool_name=tool_name, params=params))

        return tool_calls

    def _build_params(
        self,
        tool_name: str,
        uid: int,
        task_type: str,
        hist_pids: list | None,
        hist_labels: dict | None,
        pid2sid: dict | None,
        pid2sid_video: dict | None,
        pid2sid_product: dict | None,
        **kwargs,
    ) -> dict[str, Any]:
        """Build parameters for a specific tool call."""
        base_params = {
            "uid": uid,
            "hist_pids": hist_pids,
            "pid2sid": pid2sid,
        }

        if tool_name == "profile":
            return {
                **base_params,
                "top_k": 10,
                "hist_labels": hist_labels,
            }

        if tool_name == "recent_intent":
            return {
                **base_params,
                "recent_n": 15,
            }

        if tool_name == "label_behavior":
            return {
                **base_params,
                "top_k": 5,
                "hist_labels": hist_labels,
            }

        if tool_name == "cross_domain":
            target_domain = "video"
            if task_type in ["ad", "product"]:
                target_domain = task_type
            return {
                **base_params,
                "target_domain": target_domain,
                "pid2sid_video": pid2sid_video,
                "pid2sid_product": pid2sid_product,
                "top_k": 8,
            }

        if tool_name == "collaborative":
            # Deferred parameter: query_sids will be resolved from recent_intent output
            return {
                "uid": uid,
                "query_sids": "$recent_intent.recent_sids[:3]",
                "top_k": 5,
                "max_queries": 5,
            }

        return base_params
