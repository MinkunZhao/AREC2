"""Evidence Graph: typed nodes and edges for dependency-aware context enrichment."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class NodeType(str, Enum):
    LONG_TERM_PROFILE = "long_term_profile"
    RECENT_INTENT = "recent_intent"
    LABEL_CONDITION = "label_condition"
    CROSS_DOMAIN_SIGNAL = "cross_domain_signal"
    COLLABORATIVE_NEIGHBORS = "collaborative_neighbors"
    CANDIDATE_ITEM_CARD = "candidate_item_card"


class EdgeType(str, Enum):
    CONDITIONS = "conditions"      # label_condition → profile (conditions the profile)
    SUPPORTS = "supports"          # collaborative → recent_intent (co-occurrence supports intent)
    CONTRASTS = "contrasts"        # not_interested items contrast with positive signals
    GROUNDS = "grounds"            # profile/intent → candidate (grounds the recommendation)


@dataclass
class Node:
    id: str
    node_type: NodeType
    data: dict[str, Any]
    token_cost: int = 0
    confidence: float = 1.0

    @property
    def compact_str(self) -> str:
        return self.data.get("compact", "")


@dataclass
class Edge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0


@dataclass
class EvidenceGraph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def add_node(self, node: Node):
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge):
        self.edges.append(edge)

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def get_outgoing_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.source_id == node_id]

    def get_incoming_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.target_id == node_id]

    def total_token_cost(self) -> int:
        return sum(n.token_cost for n in self.nodes.values())

    def prune_low_confidence(self, threshold: float = 0.3) -> int:
        to_remove = [nid for nid, n in self.nodes.items() if n.confidence < threshold]
        for nid in to_remove:
            del self.nodes[nid]
        before = len(self.edges)
        self.edges = [
            e for e in self.edges
            if e.source_id in self.nodes and e.target_id in self.nodes
        ]
        return len(to_remove)

    def prune_to_token_budget(self, budget: int) -> int:
        if self.total_token_cost() <= budget:
            return 0

        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda n: n.confidence,
        )
        removed = 0
        while self.total_token_cost() > budget and sorted_nodes:
            node = sorted_nodes.pop(0)
            del self.nodes[node.id]
            removed += 1

        self.edges = [
            e for e in self.edges
            if e.source_id in self.nodes and e.target_id in self.nodes
        ]
        return removed
