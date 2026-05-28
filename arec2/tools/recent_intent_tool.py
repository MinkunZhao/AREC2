"""RecentIntentTool: extracts short-term intent from recent interaction history."""

from __future__ import annotations

from arec2.tools.tool_base import Tool, ToolResult


class RecentIntentTool(Tool):
    name = "recent_intent"
    description = "Extract user short-term intent from recent history"
    node_type = "recent_intent"

    def run(
        self,
        uid: int,
        hist_pids: list | None = None,
        pid2sid: dict | None = None,
        recent_n: int = 15,
        **kwargs,
    ) -> ToolResult:
        """Extract recent intent from the last N items in user history.

        Args:
            uid: User ID.
            hist_pids: Raw PID history array from the benchmark sample.
            pid2sid: PID→SID mapping dict.
            recent_n: Number of recent items to consider.
        """
        if hist_pids is None or len(hist_pids) == 0:
            return ToolResult(
                tool_name=self.name,
                node_type=self.node_type,
                data={"recent_sids": [], "summary": "No history"},
                confidence=0.0,
            )

        recent_pids = list(hist_pids[-recent_n:])
        recent_sids = []
        if pid2sid is not None:
            for pid in recent_pids:
                sid = pid2sid.get(int(pid))
                if sid:
                    recent_sids.append(sid)

        return ToolResult(
            tool_name=self.name,
            node_type=self.node_type,
            data={
                "recent_sids": recent_sids,
                "recent_pids": recent_pids,
                "window_size": recent_n,
                "recent_sids_str": self.sids_to_compact(recent_sids, max_items=recent_n),
            },
            confidence=1.0,
            token_cost=len(recent_sids) * 5 + 10,
        )
