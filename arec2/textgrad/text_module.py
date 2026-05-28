"""Text module: differentiable text variables for TextGrad."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TextVariable:
    """A 'differentiable' text variable with gradient history.

    In TextGrad, gradients are natural language critiques rather than numeric vectors.
    """

    name: str
    value: str
    role: str = "instruction"  # instruction | output | reward
    requires_grad: bool = False
    gradients: list[str] = field(default_factory=list)  # Accumulated NL critiques
    parents: list[TextVariable] = field(default_factory=list)

    def add_gradient(self, critique: str):
        """Add a natural language gradient (critique)."""
        if self.requires_grad:
            self.gradients.append(critique)

    def reset_grad(self):
        """Clear accumulated gradients."""
        self.gradients.clear()

    def __repr__(self):
        grad_info = f", {len(self.gradients)} grads" if self.requires_grad else ""
        return f"TextVariable(name={self.name}, role={self.role}, len={len(self.value)}{grad_info})"
