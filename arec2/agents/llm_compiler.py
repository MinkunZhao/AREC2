"""LLM-based compiler agent with fallback to rule-based compiler."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from pydantic import BaseModel, ValidationError

from arec2.agents.llm_client import LLMClient
from arec2.enrichment.evidence_graph import EvidenceGraph
from arec2.enrichment.graph_compiler import GraphCompiler
from arec2.tools.tool_base import ToolResult

logger = logging.getLogger(__name__)


class CompilerOutput(BaseModel):
    """Validated compiler output from LLM."""
    card: str
    kept_signals: list[str] = []
    dropped: list[str] = []


class LLMCompiler:
    """LLM-driven compiler with rule-based fallback.

    Uses an LLM to synthesize tool results into a compact context card.
    Falls back to rule-based compiler if LLM produces invalid output.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        instruction_path: str = "prompts/compiler_v0.txt",
        max_chars: int = 3000,
        fallback: GraphCompiler | None = None,
    ):
        """Initialize LLM compiler.

        Args:
            llm_client: LLM client for making API calls.
            instruction_path: Path to compiler instruction prompt.
            max_chars: Maximum character length for output card.
            fallback: Fallback rule-based compiler (default: creates new GraphCompiler).
        """
        self.llm = llm_client
        instruction_file = Path(instruction_path)
        if not instruction_file.exists():
            raise FileNotFoundError(f"Compiler instruction not found: {instruction_path}")
        self.instruction = instruction_file.read_text(encoding="utf-8")
        self.max_chars = max_chars
        self.fallback = fallback or GraphCompiler(token_budget=1024, card_style="natural")
        self.fallback_count = 0
        self.total_count = 0
        logger.info(f"LLMCompiler initialized with instruction from {instruction_path}")

    def compile(
        self,
        tool_results: list[ToolResult],
        task_type: str = "video",
        candidate_sids: list[str] | None = None,
    ) -> str:
        """Compile tool results into a context card using LLM.

        Args:
            tool_results: List of ToolResult objects from retrieval tools.
            task_type: Task type (for context-aware compilation).
            candidate_sids: Optional candidate SIDs.

        Returns:
            Context card string.
        """
        self.total_count += 1

        # Build structured input for LLM
        tool_data = self._format_tool_results(tool_results)

        # Construct LLM messages
        user_message = self._render_user_message(task_type, tool_data, candidate_sids)
        messages = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": user_message},
        ]

        # Call LLM
        try:
            resp = self.llm.chat(messages, temperature=0.0, json_mode=True)
            if resp.json_obj is None:
                raise ValueError("LLM did not return valid JSON")
            output = CompilerOutput.model_validate(resp.json_obj)

            # Validate card length
            if len(output.card) > self.max_chars:
                logger.warning(
                    f"LLM card too long ({len(output.card)} chars > {self.max_chars}). Truncating."
                )
                output.card = output.card[:self.max_chars]

            # Validate SID tokens are preserved
            if not self._validate_sid_tokens(output.card, tool_results):
                logger.warning("LLM corrupted SID tokens. Falling back to rule-based.")
                raise ValueError("SID token corruption detected")

            logger.debug(f"LLM compiled card: kept={output.kept_signals}, dropped={output.dropped}")
            return output.card

        except (ValidationError, ValueError, Exception) as e:
            logger.warning(f"LLM compiler failed for task={task_type}: {e}. Falling back to rule-based.")
            self.fallback_count += 1
            return self.fallback.compile_and_serialize(tool_results, candidate_sids)

    @staticmethod
    def _numpy_safe(obj):
        """Recursively convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: LLMCompiler._numpy_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [LLMCompiler._numpy_safe(v) for v in obj]
        return obj

    def _format_tool_results(self, tool_results: list[ToolResult]) -> list[dict]:
        """Format tool results for LLM consumption."""
        formatted = []
        for result in tool_results:
            formatted.append({
                "tool_name": result.tool_name,
                "node_type": result.node_type,
                "confidence": result.confidence,
                "token_cost": result.token_cost,
                "data": self._numpy_safe(result.data),
            })
        return formatted

    def _render_user_message(
        self,
        task_type: str,
        tool_data: list[dict],
        candidate_sids: list[str] | None,
    ) -> str:
        """Render user message for LLM."""
        msg = f"Task type: {task_type}\n\n"
        msg += "Tool results:\n"
        msg += json.dumps(tool_data, indent=2, ensure_ascii=False)

        if candidate_sids:
            msg += f"\n\nCandidate pool: {len(candidate_sids)} items"

        msg += "\n\nReturn a JSON object with `card` (the compiled context card), `kept_signals` (list of tool names you included), and `dropped` (list of tool names you excluded with reasons)."
        return msg

    def _validate_sid_tokens(self, card: str, tool_results: list[ToolResult]) -> bool:
        """Validate that SID tokens are preserved correctly.

        Checks that any SID sequences in the card match the pattern and are not corrupted.
        """
        import re

        # Pattern for valid SID tokens
        sid_pattern = re.compile(
            r"<\|sid_begin\|><s_[abc]_\d+><s_[abc]_\d+><s_[abc]_\d+><\|sid_end\|>"
        )

        # Check for partial/corrupted SID tokens
        partial_patterns = [
            r"<\|sid_begin\|>(?!<s_[abc]_\d+>)",  # sid_begin not followed by s_a/b/c
            r"<s_[abc]_\d+>(?!<s_[abc]_\d+>|<\|sid_end\|>)",  # s_a/b/c not followed by next token
            r"<\|sid_end\|>(?=<s_[abc]_\d+>)",  # sid_end followed by s_a/b/c (should be space)
        ]

        for pattern in partial_patterns:
            if re.search(pattern, card):
                logger.warning(f"Detected corrupted SID token pattern: {pattern}")
                return False

        # If card contains sid_begin, validate all sequences are complete
        if "<|sid_begin|>" in card:
            matches = sid_pattern.findall(card)
            begin_count = card.count("<|sid_begin|>")
            if len(matches) != begin_count:
                logger.warning(
                    f"SID token mismatch: {begin_count} begin tokens but only {len(matches)} valid sequences"
                )
                return False

        return True

    def get_stats(self) -> dict:
        """Get compiler statistics."""
        return {
            "total_calls": self.total_count,
            "fallback_calls": self.fallback_count,
            "fallback_rate": self.fallback_count / self.total_count if self.total_count > 0 else 0.0,
        }
