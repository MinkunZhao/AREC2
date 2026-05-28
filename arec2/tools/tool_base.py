"""Base class for AREC² agentic tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    tool_name: str
    node_type: str
    data: dict[str, Any]
    confidence: float = 1.0
    token_cost: int = 0


class Tool(ABC):
    name: str = ""
    description: str = ""
    node_type: str = ""

    @abstractmethod
    def run(self, uid: int, **kwargs) -> ToolResult:
        ...

    def sid_to_str(self, sid: list[int] | tuple) -> str:
        a, b, c = sid[0], sid[1], sid[2]
        return f"<|sid_begin|><s_a_{a}><s_b_{b}><s_c_{c}><|sid_end|>"

    def sids_to_compact(self, sids: list[list[int] | tuple], max_items: int = 10) -> str:
        return " ".join(self.sid_to_str(s) for s in sids[:max_items])
