"""Rollout / SID-parsing utilities for RecPO preference generation.

This module is the lowest layer (only depends on the model wrapper). It owns
all SID parsing/canonicalization so that preference_pair_gen and the generation
scripts share ONE definition of "what a SID string looks like".

NOTE (bug fix): the previous version substituted the ground truth as the
"best" candidate whenever all K sampled candidates tied at zero overlap. With
RecIF Recall@32 ~3%, that tie happened almost every time, collapsing P1 into a
degenerate "GT-vs-random-sample" signal. That substitution is removed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from arec2.base_model.openonerec_wrapper import OpenOneRecWrapper

logger = logging.getLogger(__name__)

# Matches bare SID sub-tokens, wrapper-agnostic. We group every 3 in order,
# exactly like the official RecIF eval parser, then rebuild canonical wrappers.
_SID_TOKEN_RE = re.compile(r"<s_[abc]_\d+>")

CANONICAL_SID_FMT = "<|sid_begin|>{body}<|sid_end|>"


def parse_sid_list(text: str) -> list[str]:
    """Parse text into an ordered list of CANONICAL SID strings.

    Wrapper-agnostic on input: works whether or not the text contains
    <|sid_begin|>/<|sid_end|>. Output is always the canonical, fully-wrapped
    form with NO separator between tokens (matching the SFT answer format
    produced by _pids_to_sids_str).
    """
    toks = _SID_TOKEN_RE.findall(text)
    out: list[str] = []
    for i in range(0, len(toks) - 2, 3):
        body = toks[i] + toks[i + 1] + toks[i + 2]
        out.append(CANONICAL_SID_FMT.format(body=body))
    return out


def parse_sid_list_unique(text: str) -> list[str]:
    """Same as parse_sid_list but de-duplicated, order preserved."""
    seen: set[str] = set()
    out: list[str] = []
    for s in parse_sid_list(text):
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def canonicalize_sids(text: str) -> str:
    """Rebuild any SID-bearing text into a single canonical string.

    Drops everything that is not a SID triple. chosen and rejected MUST both
    pass through this so the only difference DPO can learn is item identity,
    not formatting (wrappers, separators, stray thinking tokens, etc.).
    """
    return "".join(parse_sid_list(text))


# Backwards-compatible alias used by older callers.
def extract_sids(text: str) -> list[str]:
    return parse_sid_list(text)


@dataclass
class BeamCandidate:
    text: str
    sids: list[str]
    overlap_score: float


def compute_overlap(pred_sids: list[str], gt_sids: list[str], k: int = 10) -> float:
    """Prefix-weighted overlap in [0, 1] (kept for diagnostics/ablations)."""
    if not gt_sids or not pred_sids:
        return 0.0
    gt_set = set(gt_sids[:k])
    score = 0.0
    for rank, sid in enumerate(pred_sids[:k]):
        if sid in gt_set:
            score += 1.0 / (rank + 1)
    max_score = sum(1.0 / (i + 1) for i in range(min(k, len(gt_set))))
    return score / max_score if max_score > 0 else 0.0


def rollout_candidates(
    model: OpenOneRecWrapper,
    messages: list[dict],
    k: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    do_sample: bool = True,
) -> list[BeamCandidate]:
    """Generate K candidates and parse their SIDs (no GT substitution).

    Returned candidates are NOT scored against GT here; the caller decides how
    to use them. Kept available for ablation studies that need sampled rollouts.
    """
    gens = model.generate(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=50,
        top_p=0.95,
        do_sample=do_sample,
        num_return=k,
    )
    return [BeamCandidate(text=g, sids=parse_sid_list(g), overlap_score=0.0) for g in gens]