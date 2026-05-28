"""Graph Compiler: converts tool outputs into an Evidence Graph and serializes
it into a compact Textual Context Card for the generative model.

The compiler:
1. Creates typed nodes from ToolResults.
2. Infers edges based on node types (dependency-aware).
3. Prunes redundant/low-confidence nodes.
4. Serializes into a compact card within a token budget.
"""

from __future__ import annotations

import logging
from typing import Optional

from arec2.enrichment.evidence_graph import (
    Edge,
    EdgeType,
    EvidenceGraph,
    Node,
    NodeType,
)
from arec2.tools.tool_base import ToolResult

logger = logging.getLogger(__name__)

NODE_TYPE_MAP = {
    "long_term_profile": NodeType.LONG_TERM_PROFILE,
    "recent_intent": NodeType.RECENT_INTENT,
    "label_condition": NodeType.LABEL_CONDITION,
    "cross_domain_signal": NodeType.CROSS_DOMAIN_SIGNAL,
    "collaborative_neighbors": NodeType.COLLABORATIVE_NEIGHBORS,
    "candidate_item_card": NodeType.CANDIDATE_ITEM_CARD,
}

EDGE_RULES = [
    (NodeType.LABEL_CONDITION, NodeType.LONG_TERM_PROFILE, EdgeType.CONDITIONS),
    (NodeType.COLLABORATIVE_NEIGHBORS, NodeType.RECENT_INTENT, EdgeType.SUPPORTS),
    (NodeType.CROSS_DOMAIN_SIGNAL, NodeType.LONG_TERM_PROFILE, EdgeType.SUPPORTS),
    (NodeType.LONG_TERM_PROFILE, NodeType.CANDIDATE_ITEM_CARD, EdgeType.GROUNDS),
    (NodeType.RECENT_INTENT, NodeType.CANDIDATE_ITEM_CARD, EdgeType.GROUNDS),
    (NodeType.COLLABORATIVE_NEIGHBORS, NodeType.CANDIDATE_ITEM_CARD, EdgeType.GROUNDS),
]


class GraphCompiler:
    def __init__(self, token_budget: int = 1024, card_style: str = "natural"):
        """
        Args:
            token_budget: Max token cost for the evidence graph.
            card_style: "natural" for NL-style cards, "synthetic" for [TAG] style.
        """
        self.token_budget = token_budget
        self.card_style = card_style

    def compile(
        self,
        tool_results: list[ToolResult],
        candidate_sids: Optional[list[str]] = None,
    ) -> EvidenceGraph:
        graph = EvidenceGraph()

        for result in tool_results:
            node = self._result_to_node(result)
            if node:
                graph.add_node(node)

        if candidate_sids:
            compact = " ".join(candidate_sids[:32])
            node = Node(
                id="candidates",
                node_type=NodeType.CANDIDATE_ITEM_CARD,
                data={"sids": candidate_sids, "compact": compact},
                token_cost=len(candidate_sids) * 5,
                confidence=1.0,
            )
            graph.add_node(node)

        self._infer_edges(graph)
        graph.prune_low_confidence(threshold=0.2)
        pruned = graph.prune_to_token_budget(self.token_budget)
        if pruned > 0:
            logger.info("Pruned %d nodes to fit token budget %d", pruned, self.token_budget)

        return graph

    def serialize(self, graph: EvidenceGraph) -> str:
        if self.card_style == "natural":
            return self.serialize_card_natural(graph)
        return self.serialize_card_synthetic(graph)

    def serialize_card_synthetic(self, graph: EvidenceGraph) -> str:
        """Original [TAG]-based serialization."""
        sections = []

        type_order = [
            NodeType.LONG_TERM_PROFILE,
            NodeType.RECENT_INTENT,
            NodeType.LABEL_CONDITION,
            NodeType.CROSS_DOMAIN_SIGNAL,
            NodeType.COLLABORATIVE_NEIGHBORS,
        ]

        for ntype in type_order:
            nodes = [n for n in graph.nodes.values() if n.node_type == ntype]
            for node in nodes:
                section = self._serialize_node_synthetic(node)
                if section:
                    sections.append(section)

        return "\n".join(sections)

    def serialize_card_natural(self, graph: EvidenceGraph) -> str:
        """Natural language serialization (reduces distribution shift from pretraining)."""
        sections = []

        type_order = [
            NodeType.LONG_TERM_PROFILE,
            NodeType.RECENT_INTENT,
            NodeType.LABEL_CONDITION,
            NodeType.CROSS_DOMAIN_SIGNAL,
            NodeType.COLLABORATIVE_NEIGHBORS,
        ]

        for ntype in type_order:
            nodes = [n for n in graph.nodes.values() if n.node_type == ntype]
            for node in nodes:
                section = self._serialize_node_natural(node)
                if section:
                    sections.append(section)

        return "\n".join(sections)

    def compile_and_serialize(
        self,
        tool_results: list[ToolResult],
        candidate_sids: Optional[list[str]] = None,
    ) -> str:
        graph = self.compile(tool_results, candidate_sids)
        return self.serialize(graph)

    def _result_to_node(self, result: ToolResult) -> Optional[Node]:
        node_type = NODE_TYPE_MAP.get(result.node_type)
        if node_type is None:
            logger.warning("Unknown node type: %s", result.node_type)
            return None

        data = {**result.data}

        return Node(
            id=result.tool_name,
            node_type=node_type,
            data=data,
            token_cost=result.token_cost,
            confidence=result.confidence,
        )

    def _serialize_node_synthetic(self, node: Node) -> str:
        """Synthetic [TAG] style serialization (legacy)."""
        data = node.data
        ntype = node.node_type

        if ntype == NodeType.LONG_TERM_PROFILE:
            parts = [f"[PROFILE] {data.get('summary', '')}"]
            if data.get("positive_sids_str"):
                parts.append(f"preferred: {data['positive_sids_str']}")
            elif data.get("top_sids_str"):
                parts.append(f"frequent: {data['top_sids_str']}")
            return " | ".join(parts)

        if ntype == NodeType.RECENT_INTENT:
            sid_str = data.get("recent_sids_str", "")
            return f"[RECENT] last_{data.get('window_size', 15)}: {sid_str}"

        if ntype == NodeType.LABEL_CONDITION:
            parts = [f"[LABELS] {data.get('summary', '')}"]
            for label in ["like", "follow", "longview"]:
                s = data.get("behavior_strs", {}).get(label, "")
                if s:
                    parts.append(f"{label}: {s}")
            return " | ".join(parts)

        if ntype == NodeType.CROSS_DOMAIN_SIGNAL:
            parts = [f"[CROSS] {data.get('summary', '')}"]
            if data.get("cross_sids_str"):
                parts.append(f"transfer: {data['cross_sids_str']}")
            return " | ".join(parts)

        if ntype == NodeType.COLLABORATIVE_NEIGHBORS:
            parts = [f"[COLLAB] {data.get('summary', '')}"]
            if data.get("merged_sids_str"):
                parts.append(f"neighbors: {data['merged_sids_str']}")
            return " | ".join(parts)

        return f"[{ntype.name}] {data.get('summary', '')}"

    def _serialize_node_natural(self, node: Node) -> str:
        """Natural language style serialization."""
        data = node.data
        ntype = node.node_type

        if ntype == NodeType.LONG_TERM_PROFILE:
            summary = data.get("summary", "")
            sids = data.get("positive_sids_str") or data.get("top_sids_str", "")
            text = f"This user's long-term profile shows: {summary}."
            if sids:
                text += f" Their preferred items include {sids}."
            return text

        if ntype == NodeType.RECENT_INTENT:
            sid_str = data.get("recent_sids_str", "")
            window = data.get("window_size", 15)
            return f"In their most recent {window} interactions, this user engaged with: {sid_str}."

        if ntype == NodeType.LABEL_CONDITION:
            summary = data.get("summary", "")
            text = f"The user's interaction patterns indicate: {summary}."
            behavior_strs = data.get("behavior_strs", {})
            parts = []
            for label in ["like", "follow", "longview"]:
                s = behavior_strs.get(label, "")
                if s:
                    parts.append(f"items they {label}: {s}")
            if parts:
                text += " Specifically, " + "; ".join(parts) + "."
            return text

        if ntype == NodeType.CROSS_DOMAIN_SIGNAL:
            summary = data.get("summary", "")
            sids = data.get("cross_sids_str", "")
            text = f"Cross-domain signals suggest: {summary}."
            if sids:
                text += f" Transferable interests include: {sids}."
            return text

        if ntype == NodeType.COLLABORATIVE_NEIGHBORS:
            summary = data.get("summary", "")
            sids = data.get("merged_sids_str", "")
            text = f"Similar users also engaged with: {summary}."
            if sids:
                text += f" Popular items among neighbors: {sids}."
            return text

        return data.get("summary", "")

    def _infer_edges(self, graph: EvidenceGraph):
        node_by_type: dict[NodeType, list[str]] = {}
        for nid, node in graph.nodes.items():
            node_by_type.setdefault(node.node_type, []).append(nid)

        for src_type, tgt_type, edge_type in EDGE_RULES:
            src_ids = node_by_type.get(src_type, [])
            tgt_ids = node_by_type.get(tgt_type, [])
            for sid in src_ids:
                for tid in tgt_ids:
                    graph.add_edge(Edge(
                        source_id=sid,
                        target_id=tid,
                        edge_type=edge_type,
                        weight=graph.nodes[sid].confidence,
                    ))
