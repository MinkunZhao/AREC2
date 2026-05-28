"""CrossDomainTool: extracts cross-domain transfer signals (video↔ad↔product).

Can use the pre-built ProfileStore or infer domain distribution from benchmark
sample fields when the user is not in the store.
"""

from __future__ import annotations

from typing import Optional

from arec2.retrieval.stores import ProfileStore
from arec2.tools.tool_base import Tool, ToolResult


class CrossDomainTool(Tool):
    name = "cross_domain"
    description = "Extract cross-domain transfer signals between video/ad/product"
    node_type = "cross_domain_signal"

    def __init__(self, store: Optional[ProfileStore] = None):
        self.store = store

    def run(
        self,
        uid: int,
        target_domain: str = "video",
        hist_pids: list | None = None,
        pid2sid_video: dict | None = None,
        pid2sid_product: dict | None = None,
        pid2sid: dict | None = None,
        top_k: int = 8,
        **kwargs,
    ) -> ToolResult:
        profile = None
        if self.store is not None:
            profile = self.store.query(uid)

        if profile is None and hist_pids is not None:
            profile = self._build_inline_cross_domain(
                hist_pids, target_domain, pid2sid_video, pid2sid_product, pid2sid,
            )

        if profile is None:
            return ToolResult(
                tool_name=self.name,
                node_type=self.node_type,
                data={"cross_sids": [], "summary": "No profile"},
                confidence=0.0,
            )

        domain_dist = profile["domain_distribution"]
        active_domains = [d for d, c in domain_dist.items() if c > 0 and d != target_domain]

        if not active_domains:
            return ToolResult(
                tool_name=self.name,
                node_type=self.node_type,
                data={
                    "cross_sids": [],
                    "source_domains": [],
                    "summary": f"User only active in {target_domain}",
                },
                confidence=0.2,
            )

        cross_sids = profile["top_sids"][:top_k]
        positive_cross = profile["top_positive_sids"][:top_k]

        return ToolResult(
            tool_name=self.name,
            node_type=self.node_type,
            data={
                "source_domains": active_domains,
                "target_domain": target_domain,
                "cross_sids": cross_sids,
                "positive_cross_sids": positive_cross,
                "domain_distribution": domain_dist,
                "cross_sids_str": self.sids_to_compact(cross_sids, max_items=top_k),
                "summary": f"domains={','.join(active_domains)}|items={len(cross_sids)}",
            },
            confidence=0.7 if len(active_domains) > 0 else 0.2,
            token_cost=len(cross_sids) * 5 + 15,
        )

    def _build_inline_cross_domain(
        self,
        hist_pids: list,
        target_domain: str,
        pid2sid_video: dict | None,
        pid2sid_product: dict | None,
        pid2sid: dict | None,
    ) -> dict | None:
        video_sids = []
        product_sids = []

        p2s_video = pid2sid_video or (pid2sid if target_domain == "video" else None)
        p2s_product = pid2sid_product or (pid2sid if target_domain == "product" else None)

        for pid in hist_pids:
            pid = int(pid)
            if p2s_video and pid in p2s_video:
                video_sids.append(list(p2s_video[pid]))
            if p2s_product and pid in p2s_product:
                product_sids.append(list(p2s_product[pid]))

        domain_dist = {
            "video": len(video_sids),
            "ad": 0,
            "product": len(product_sids),
        }

        all_sids = video_sids + product_sids
        if not all_sids:
            return None

        return {
            "top_sids": all_sids[:20],
            "top_positive_sids": [],
            "domain_distribution": domain_dist,
        }
