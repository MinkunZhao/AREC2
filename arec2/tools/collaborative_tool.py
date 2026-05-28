"""CollaborativeTool: item-item co-occurrence neighbor lookup."""

from __future__ import annotations

from arec2.retrieval.stores import CollaborativeStore
from arec2.tools.tool_base import Tool, ToolResult


class CollaborativeTool(Tool):
    name = "collaborative"
    description = "Find collaborative item-item co-occurrence neighbors"
    node_type = "collaborative_neighbors"

    def __init__(self, store: CollaborativeStore):
        self.store = store

    def run(
        self,
        uid: int,
        query_sids: list[list[int] | tuple] | None = None,
        top_k: int = 5,
        max_queries: int = 5,
        **kwargs,
    ) -> ToolResult:
        """Find items that co-occur with the query SIDs.

        Args:
            uid: User ID (for logging).
            query_sids: SIDs to look up neighbors for.
            top_k: Number of neighbors per query SID.
            max_queries: Max query SIDs to process.
        """
        if not query_sids:
            return ToolResult(
                tool_name=self.name,
                node_type=self.node_type,
                data={"neighbors": {}, "summary": "No query SIDs"},
                confidence=0.0,
            )

        all_neighbors = {}
        merged_set = {}

        for sid in query_sids[:max_queries]:
            sid_tuple = tuple(sid)
            neighbors = self.store.query_neighbors(sid_tuple, top_k=top_k)
            if neighbors:
                sid_str = self.sid_to_str(sid_tuple)
                all_neighbors[sid_str] = [
                    {"sid": list(n_sid), "sid_str": self.sid_to_str(n_sid), "score": score}
                    for n_sid, score in neighbors
                ]
                for n_sid, score in neighbors:
                    n_key = tuple(n_sid)
                    if n_key not in merged_set or merged_set[n_key] < score:
                        merged_set[n_key] = score

        top_merged = sorted(merged_set.items(), key=lambda x: x[1], reverse=True)[:top_k * 2]
        top_merged_sids = [list(s) for s, _ in top_merged]

        return ToolResult(
            tool_name=self.name,
            node_type=self.node_type,
            data={
                "per_query_neighbors": all_neighbors,
                "merged_neighbors": top_merged_sids,
                "merged_scores": {self.sid_to_str(s): sc for s, sc in top_merged},
                "merged_sids_str": self.sids_to_compact(top_merged_sids, max_items=top_k * 2),
                "summary": f"queries={len(all_neighbors)}|merged={len(top_merged)}",
            },
            confidence=0.8 if top_merged else 0.1,
            token_cost=len(top_merged) * 5 + 15,
        )
