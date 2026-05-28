"""ExecutorAgent: executes tool call plans and compiles evidence graphs.

Takes an ordered plan from PlannerAgent, executes each tool, collects results,
and compiles them into a compact context card via the GraphCompiler.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from arec2.agents.planner_agent import ToolCall
from arec2.enrichment.graph_compiler import GraphCompiler
from arec2.retrieval.stores import (
    CollaborativeStore,
    ItemTextStore,
    LabelBehaviorStore,
    ProfileStore,
)
from arec2.tools.collaborative_tool import CollaborativeTool
from arec2.tools.cross_domain_tool import CrossDomainTool
from arec2.tools.label_behavior_tool import LabelBehaviorTool
from arec2.tools.profile_tool import ProfileTool
from arec2.tools.recent_intent_tool import RecentIntentTool
from arec2.tools.tool_base import Tool, ToolResult

logger = logging.getLogger(__name__)


class ExecutorAgent:
    """Executes tool call plans and produces enriched context cards."""

    def __init__(
        self,
        profile_store: Optional[ProfileStore] = None,
        label_store: Optional[LabelBehaviorStore] = None,
        collab_store: Optional[CollaborativeStore] = None,
        text_store: Optional[ItemTextStore] = None,
        token_budget: int = 1024,
        card_style: str = "natural",
    ):
        self.stores = {
            "profile": profile_store,
            "label": label_store,
            "collab": collab_store,
            "text": text_store,
        }
        self.token_budget = token_budget
        self.card_style = card_style
        self.tool_registry = self._build_tool_registry()
        self.compiler = GraphCompiler(token_budget=token_budget, card_style=card_style)

    def _build_tool_registry(self) -> dict[str, Tool]:
        """Initialize all tools with their stores."""
        return {
            "profile": ProfileTool(self.stores["profile"]),
            "recent_intent": RecentIntentTool(),
            "label_behavior": LabelBehaviorTool(self.stores["label"]),
            "cross_domain": CrossDomainTool(self.stores["profile"]),
            "collaborative": CollaborativeTool(self.stores["collab"]),
        }

    def execute(
        self,
        plan: list[ToolCall],
        candidate_sids: list | None = None,
    ) -> str:
        """Execute tool calls, compile graph, return context card.

        Args:
            plan: Ordered list of ToolCall objects from PlannerAgent.
            candidate_sids: Optional candidate SIDs to include in the graph.

        Returns:
            Compact context card string ready for model consumption.
        """
        results = {}
        tool_results = []

        for tool_call in plan:
            tool_name = tool_call.tool_name
            tool = self.tool_registry.get(tool_name)

            if tool is None:
                logger.warning("Unknown tool: %s, skipping", tool_name)
                continue

            # Resolve deferred parameters (e.g., $recent_intent.recent_sids[:3])
            params = self._resolve_params(tool_call.params, results)

            try:
                result = tool.run(**params)
                results[tool_name] = result
                tool_results.append(result)
                logger.debug(
                    "Tool %s executed: confidence=%.2f, tokens=%d",
                    tool_name, result.confidence, result.token_cost,
                )
            except Exception as e:
                logger.error("Tool %s failed: %s", tool_name, e, exc_info=True)
                # Continue with other tools even if one fails

        # Compile evidence graph
        graph = self.compiler.compile(tool_results, candidate_sids)
        logger.info(
            "Graph compiled: %d nodes, %d edges, %d tokens",
            len(graph.nodes), len(graph.edges), graph.total_token_cost(),
        )

        # Serialize to context card
        card = self.compiler.serialize(graph)
        return card

    def _resolve_params(
        self, params: dict[str, Any], results: dict[str, ToolResult]
    ) -> dict[str, Any]:
        """Resolve deferred parameters like $recent_intent.recent_sids[:3].

        Args:
            params: Raw parameters from ToolCall (may contain $-prefixed references).
            results: Dict of tool_name → ToolResult from previously executed tools.

        Returns:
            Resolved parameters with actual values.
        """
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("$"):
                resolved[key] = self._resolve_reference(value, results)
            else:
                resolved[key] = value
        return resolved

    def _resolve_reference(
        self, ref: str, results: dict[str, ToolResult]
    ) -> Any:
        """Resolve a single $tool_name.field[:slice] reference.

        Examples:
            "$recent_intent.recent_sids[:3]" → first 3 recent SIDs
            "$profile.top_sids" → all top SIDs from profile

        Args:
            ref: Reference string starting with $.
            results: Dict of tool_name → ToolResult.

        Returns:
            Resolved value from the referenced tool result.
        """
        # Parse reference: $tool_name.field[slice]
        match = re.match(r"\$(\w+)\.(\w+)(\[.*\])?", ref)
        if not match:
            logger.warning("Invalid reference format: %s", ref)
            return None

        tool_name, field_name, slice_str = match.groups()

        if tool_name not in results:
            logger.warning("Tool %s not yet executed, cannot resolve %s", tool_name, ref)
            return None

        result = results[tool_name]
        value = result.data.get(field_name)

        if value is None:
            logger.warning("Field %s not found in %s result", field_name, tool_name)
            return None

        # Apply slice if present
        if slice_str:
            try:
                value = eval(f"value{slice_str}")
            except Exception as e:
                logger.warning("Failed to apply slice %s: %s", slice_str, e)

        return value
