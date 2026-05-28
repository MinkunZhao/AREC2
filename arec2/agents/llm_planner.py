"""LLM-based planner agent with fallback to rule-based planner."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError, field_validator

from arec2.agents.llm_client import LLMClient
from arec2.agents.planner_agent import PlannerAgent, ToolCall

logger = logging.getLogger(__name__)


class LLMToolCall(BaseModel):
    """Validated tool call from LLM."""
    tool_name: str
    params: dict[str, Any] = {}

    @field_validator("tool_name")
    def validate_tool_name(cls, v):
        valid_tools = {
            "profile",
            "recent_intent",
            "label_behavior",
            "cross_domain",
            "collaborative",
        }
        if v not in valid_tools:
            raise ValueError(f"Invalid tool name: {v}. Must be one of {valid_tools}")
        return v


class LLMPlan(BaseModel):
    """Validated plan from LLM."""
    tool_calls: list[LLMToolCall]
    rationale: str = ""


class LLMPlanner:
    """LLM-driven planner with rule-based fallback.

    Uses an LLM to select tools based on task type and available data.
    Falls back to rule-based planner if LLM produces invalid output.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        instruction_path: str = "prompts/planner_v0.txt",
        fallback: PlannerAgent | None = None,
    ):
        """Initialize LLM planner.

        Args:
            llm_client: LLM client for making API calls.
            instruction_path: Path to planner instruction prompt.
            fallback: Fallback rule-based planner (default: creates new PlannerAgent).
        """
        self.llm = llm_client
        instruction_file = Path(instruction_path)
        if not instruction_file.exists():
            raise FileNotFoundError(f"Planner instruction not found: {instruction_path}")
        self.instruction = instruction_file.read_text(encoding="utf-8")
        self.fallback = fallback or PlannerAgent()
        self.fallback_count = 0
        self.total_count = 0
        logger.info(f"LLMPlanner initialized with instruction from {instruction_path}")

    def plan(
        self,
        *,
        uid: int,
        task_type: str,
        hist_pids: list | None = None,
        hist_labels: dict | None = None,
        pid2sid: dict | None = None,
        pid2sid_video: dict | None = None,
        pid2sid_product: dict | None = None,
        candidate_pool: list | None = None,
        **kwargs,
    ) -> list[ToolCall]:
        """Generate tool plan using LLM.

        Args:
            uid: User ID.
            task_type: Task type (video, ad, product, label_cond, etc.).
            hist_pids: User history PIDs.
            hist_labels: Dict of label arrays.
            pid2sid: PID→SID mapping.
            pid2sid_video: Video-specific PID→SID mapping.
            pid2sid_product: Product-specific PID→SID mapping.
            candidate_pool: Candidate SIDs for scoring.
            **kwargs: Additional parameters.

        Returns:
            List of ToolCall objects.
        """
        self.total_count += 1

        # Build history summary (compact, not raw SID lists)
        history_summary = self._build_history_summary(
            hist_pids, hist_labels, pid2sid or pid2sid_video
        )

        # Build available data flags
        available_data = {
            "has_hist_pids": hist_pids is not None and len(hist_pids) > 0,
            "has_hist_labels": hist_labels is not None and len(hist_labels) > 0,
            "has_candidate_pool": candidate_pool is not None and len(candidate_pool) > 0,
            "hist_length": len(hist_pids) if hist_pids else 0,
        }

        # Construct LLM messages
        user_message = self._render_user_message(task_type, history_summary, available_data)
        messages = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": user_message},
        ]

        # Call LLM
        try:
            resp = self.llm.chat(messages, temperature=0.0, json_mode=True)
            if resp.json_obj is None:
                raise ValueError("LLM did not return valid JSON")
            plan_obj = LLMPlan.model_validate(resp.json_obj)
            logger.debug(f"LLM plan for uid={uid}, task={task_type}: {plan_obj.rationale}")
        except (ValidationError, ValueError, Exception) as e:
            logger.warning(
                f"LLM planner failed for uid={uid}, task={task_type}: {e}. Falling back to rule-based."
            )
            self.fallback_count += 1
            return self.fallback.plan(
                uid=uid,
                task_type=task_type,
                hist_pids=hist_pids,
                hist_labels=hist_labels,
                pid2sid=pid2sid,
                pid2sid_video=pid2sid_video,
                pid2sid_product=pid2sid_product,
                candidate_pool=candidate_pool,
                **kwargs,
            )

        # Convert LLMToolCall to ToolCall
        tool_calls = [
            ToolCall(tool_name=tc.tool_name, params=tc.params)
            for tc in plan_obj.tool_calls
        ]

        # Enrich params with actual data (LLM only specifies high-level params)
        tool_calls = self._enrich_params(
            tool_calls,
            uid=uid,
            task_type=task_type,
            hist_pids=hist_pids,
            hist_labels=hist_labels,
            pid2sid=pid2sid or pid2sid_video,
            pid2sid_video=pid2sid_video,
            pid2sid_product=pid2sid_product,
        )

        return tool_calls

    def _build_history_summary(
        self,
        hist_pids: list | None,
        hist_labels: dict | None,
        pid2sid: dict | None,
    ) -> str:
        """Build compact history summary for LLM (not raw SID lists)."""
        parts = []

        if hist_pids:
            parts.append(f"History length: {len(hist_pids)} items")
            # Last 3 PIDs as example
            last_pids = hist_pids[-3:] if len(hist_pids) >= 3 else hist_pids
            parts.append(f"Last PIDs: {last_pids}")

        if hist_labels:
            label_counts = {k: len(v) for k, v in hist_labels.items() if v}
            if label_counts:
                parts.append(f"Label counts: {label_counts}")
                # Dominant label
                dominant = max(label_counts, key=label_counts.get)
                parts.append(f"Dominant label: {dominant}")

        return " | ".join(parts) if parts else "No history available"

    def _render_user_message(
        self,
        task_type: str,
        history_summary: str,
        available_data: dict,
    ) -> str:
        """Render user message for LLM."""
        return (
            f"Task type: {task_type}\n"
            f"History summary: {history_summary}\n"
            f"Available data: {json.dumps(available_data, indent=2)}\n\n"
            "Return a JSON object with `tool_calls` (list of {tool_name, params}) and a brief `rationale`."
        )

    def _enrich_params(
        self,
        tool_calls: list[ToolCall],
        uid: int,
        task_type: str,
        hist_pids: list | None,
        hist_labels: dict | None,
        pid2sid: dict | None,
        pid2sid_video: dict | None,
        pid2sid_product: dict | None,
    ) -> list[ToolCall]:
        """Enrich tool call params with actual data references.

        LLM specifies high-level params (e.g., top_k), but we need to inject
        actual data like uid, hist_pids, pid2sid, etc.
        """
        enriched = []
        plan_tool_names = {tc.tool_name for tc in tool_calls}
        for tc in tool_calls:
            params = {**tc.params}  # Copy LLM-provided params

            # Inject common params
            params["uid"] = uid
            if hist_pids is not None:
                params["hist_pids"] = hist_pids
            if pid2sid is not None:
                params["pid2sid"] = pid2sid

            # Tool-specific enrichment
            if tc.tool_name == "profile":
                params.setdefault("top_k", 10)
                if hist_labels is not None:
                    params["hist_labels"] = hist_labels

            elif tc.tool_name == "recent_intent":
                params.setdefault("recent_n", 15)

            elif tc.tool_name == "label_behavior":
                params.setdefault("top_k", 5)
                if hist_labels is not None:
                    params["hist_labels"] = hist_labels

            elif tc.tool_name == "cross_domain":
                params.setdefault("top_k", 8)
                params.setdefault("target_domain", task_type if task_type in ["ad", "product"] else "video")
                if pid2sid_video is not None:
                    params["pid2sid_video"] = pid2sid_video
                if pid2sid_product is not None:
                    params["pid2sid_product"] = pid2sid_product

            elif tc.tool_name == "collaborative":
                params.setdefault("top_k", 5)
                params.setdefault("max_queries", 5)
                if "query_sids" not in params:
                    if "recent_intent" in plan_tool_names:
                        params["query_sids"] = "$recent_intent.recent_sids[:3]"
                    elif hist_pids and pid2sid:
                        recent_pids = hist_pids[-3:]
                        query_sids = []
                        for p in recent_pids:
                            t = pid2sid.get(int(p))
                            if t:
                                query_sids.append(
                                    f"<|sid_begin|><s_a_{t[0]}><s_b_{t[1]}><s_c_{t[2]}><|sid_end|>"
                                )
                        if query_sids:
                            params["query_sids"] = query_sids

            enriched.append(ToolCall(tool_name=tc.tool_name, params=params))

        return enriched

    def get_stats(self) -> dict:
        """Get planner statistics."""
        return {
            "total_calls": self.total_count,
            "fallback_calls": self.fallback_count,
            "fallback_rate": self.fallback_count / self.total_count if self.total_count > 0 else 0.0,
        }
