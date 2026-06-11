"""Strategy 3: Text-Augmented Itemic Tokens (paper §6.3.1).

Each item is represented as the concatenation of:
    [original 3-layer pretrained-style itemic tokens] + [5 keywords]

Critically (per the paper), the 3-layer itemic tokens are PRESERVED unchanged
(no fourth layer, no structural extension), so hierarchical semantics learned
during pretraining stay intact. The keywords provide semantic disambiguation
(collision rate drops to ~0.47%) and let the model exploit its linguistic core.

The itemic-token surface form matches the project's SID format so the model's
existing itemic vocabulary (<|sid_begin|>, <s_a_*>, <s_b_*>, <s_c_*>,
<|sid_end|>) is reused directly:

    <|sid_begin|><s_a_{c1}><s_b_{c2}><s_c_{c3}><|sid_end|> kw1 kw2 kw3 kw4 kw5
"""

from __future__ import annotations

from typing import Optional

SID_BEGIN = "<|sid_begin|>"
SID_END = "<|sid_end|>"


def itemic_token_str(code: tuple[int, int, int]) -> str:
    """Render a (c1, c2, c3) code into the canonical SID token string."""
    c1, c2, c3 = code
    return f"{SID_BEGIN}<s_a_{c1}><s_b_{c2}><s_c_{c3}>{SID_END}"


class Strategy3Representer:
    """Builds Strategy-3 ``[itemic_tokens] + [keywords]`` strings for items."""

    def __init__(
        self,
        asin2code: dict[str, tuple[int, int, int]],
        asin2keywords: dict[str, list[str]],
        n_keywords: int = 5,
        keyword_sep: str = " ",
    ):
        self.asin2code = asin2code
        self.asin2keywords = asin2keywords
        self.n_keywords = n_keywords
        self.keyword_sep = keyword_sep

    def item_repr(self, asin: str, with_keywords: bool = True) -> Optional[str]:
        """Return the Strategy-3 representation string for one item.

        Returns None if the item has no itemic code (should not happen for
        items kept after 5-core, but guards against stray ASINs).
        """
        code = self.asin2code.get(asin)
        if code is None:
            return None
        tokens = itemic_token_str(code)
        if not with_keywords:
            return tokens
        kws = self.asin2keywords.get(asin, [])[: self.n_keywords]
        if not kws:
            return tokens
        return tokens + " " + self.keyword_sep.join(kws)

    def item_repr_tokens_only(self, asin: str) -> Optional[str]:
        """Itemic tokens only (for the ablation that strips keywords)."""
        return self.item_repr(asin, with_keywords=False)

    def sequence_repr(
        self, asins: list[str], with_keywords: bool = True, joiner: str = ", "
    ) -> str:
        """Render a history sequence into a single context string."""
        parts = []
        for a in asins:
            r = self.item_repr(a, with_keywords=with_keywords)
            if r is not None:
                parts.append(r)
        return joiner.join(parts)

    def target_repr(self, asin: str) -> Optional[str]:
        """Target items are rendered as itemic tokens ONLY.

        The generation target must be the (constrained) itemic code so that
        Recall/NDCG can be measured by exact SID match. Keywords would make the
        target free-form and unrankable, so they are dropped for targets.
        """
        return self.item_repr_tokens_only(asin)
    