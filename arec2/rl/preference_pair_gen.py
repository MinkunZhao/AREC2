"""Preference pair generation for RecPO (DPO training).

Design (post bug-fix):
  All pairs share one invariant: `prompt + chosen` reconstructs a sequence that
  is structurally identical to what SFT trained on, and `chosen`/`rejected` are
  CANONICALIZED and LENGTH-MATCHED so DPO can only learn item identity, never a
  formatting shortcut.

Strategies:
  P1 - On-policy hard negative (workhorse):
        chosen   = ground-truth SID list (canonical)
        rejected = the model's OWN greedy top-ranked SIDs that are NOT in GT,
                   truncated/padded to len(GT). These are genuine hard negatives
                   (the model is confident about them) and are on-policy.

  P2 - Score-based hard negative (optional, more expensive):
        rejected = highest-scoring NON-GT candidates from a candidate pool,
                   via candidate-constrained log-prob scoring.

  P3 - Counterfactual (optional, fixed):
        rejected = model generation under a WEAK (card-stripped) prompt,
                   canonicalized + length-matched.

Output schema: {"prompt": str, "chosen": str, "rejected": str}
Compatible with TRL DPOTrainer.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Optional

from arec2.base_model.openonerec_wrapper import OpenOneRecWrapper
from arec2.rl.rollout_beam import canonicalize_sids, parse_sid_list

logger = logging.getLogger(__name__)


@dataclass
class DPOPair:
    prompt: str
    chosen: str
    rejected: str
    strategy: str  # "onpolicy" | "score_hardneg" | "counterfactual"


# ----------------------------------------------------------------------------
# Formatting helpers (single source of truth)
# ----------------------------------------------------------------------------
def build_prompt_text(messages: list[dict], tokenizer) -> str:
    """Render system+user messages up to (and including) the assistant header.

    `messages` must NOT contain the assistant answer. The result ends with the
    chat template's generation prompt (e.g. `<|im_start|>assistant\\n`), so that
    `prompt + completion` is a well-formed, SFT-consistent sequence.
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _turn_end_token(tokenizer) -> str:
    """The token that terminates an assistant turn under the chat template.

    For Qwen-family models this is `<|im_end|>`. We append it to every
    completion so DPO learns to stop, matching what apply_chat_template would
    have emitted for the assistant turn during SFT.
    """
    vocab = getattr(tokenizer, "get_vocab", lambda: {})()
    if "<|im_end|>" in vocab:
        return "<|im_end|>"
    return tokenizer.eos_token or "<|im_end|>"


def _finalize_completion(sid_list: list[str], tokenizer) -> str:
    """Join canonical SIDs (no separator, like SFT) and append the turn-end."""
    return "".join(sid_list) + _turn_end_token(tokenizer)


def _match_length(
    rejected: list[str],
    target_len: int,
    gt_set: set[str],
    filler_pool: Optional[list[str]],
    rng: random.Random,
) -> Optional[list[str]]:
    """Truncate/pad `rejected` to exactly target_len distinct non-GT SIDs."""
    out: list[str] = []
    seen: set[str] = set()
    for s in rejected:
        if s in gt_set or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) == target_len:
            return out
    # Pad from filler pool if available.
    if filler_pool:
        tries = 0
        max_tries = target_len * 50
        while len(out) < target_len and tries < max_tries:
            cand = rng.choice(filler_pool)
            tries += 1
            if cand in gt_set or cand in seen:
                continue
            seen.add(cand)
            out.append(cand)
    return out if len(out) == target_len else None


# ----------------------------------------------------------------------------
# P1: On-policy hard negative (default)
# ----------------------------------------------------------------------------
def generate_onpolicy_pair(
    model: OpenOneRecWrapper,
    messages: list[dict],
    gt_answer: str,
    filler_pool: Optional[list[str]] = None,
    rng: Optional[random.Random] = None,
    max_new_tokens: int = 256,
) -> Optional[DPOPair]:
    """chosen = GT; rejected = model's greedy top-ranked non-GT SIDs.

    Greedy decoding gives the model's deterministic ranking; the first non-GT
    SIDs in that ranking are exactly the items the model wrongly prefers — i.e.
    on-policy hard negatives. This is the right signal when Recall is low.
    """
    rng = rng or random.Random()
    gt_list = parse_sid_list(gt_answer)
    if not gt_list:
        return None
    gt_set = set(gt_list)
    need = len(gt_list)

    gens = model.generate(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,        # greedy => the model's actual ranking
        num_return=1,
        num_beams=1,
    )
    if not gens:
        return None

    pred_list = parse_sid_list(gens[0])
    rejected = _match_length(pred_list, need, gt_set, filler_pool, rng)
    if rejected is None:
        return None
    if rejected == gt_list:  # model fully correct (rare) -> no useful signal
        return None

    return DPOPair(
        prompt=build_prompt_text(messages, model.tokenizer),
        chosen=_finalize_completion(gt_list, model.tokenizer),
        rejected=_finalize_completion(rejected, model.tokenizer),
        strategy="onpolicy",
    )


def generate_onpolicy_pairs_batch(
    model: OpenOneRecWrapper,
    batch: list[tuple[list[dict], str, list[str]]],
    filler_pool: Optional[list[str]] = None,
    rng: Optional[random.Random] = None,
    max_new_tokens: int = 256,
) -> list[Optional[DPOPair]]:
    """Batched P1: one generate_batch call for the whole batch.

    Each entry is (prompt_messages, gt_answer, gt_list) — caller pre-filters.
    Returns one Optional[DPOPair] per entry (None when pair cannot be built).
    """
    if not batch:
        return []
    rng = rng or random.Random()

    messages_list = [msgs for msgs, _, _ in batch]
    gens = model.generate_batch(
        messages_list,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_return=1,
        num_beams=1,
    )

    results: list[Optional[DPOPair]] = []
    for (messages, _gt_answer, gt_list), gen_text in zip(batch, gens):
        gt_set = set(gt_list)
        pred_list = parse_sid_list(gen_text)
        rejected = _match_length(pred_list, len(gt_list), gt_set, filler_pool, rng)
        if rejected is None or rejected == gt_list:
            results.append(None)
            continue
        results.append(DPOPair(
            prompt=build_prompt_text(messages, model.tokenizer),
            chosen=_finalize_completion(gt_list, model.tokenizer),
            rejected=_finalize_completion(rejected, model.tokenizer),
            strategy="onpolicy",
        ))
    return results


# ----------------------------------------------------------------------------
# P2: Score-based hard negative (optional)
# ----------------------------------------------------------------------------
def generate_score_hardneg_pair(
    model: OpenOneRecWrapper,
    messages: list[dict],
    gt_answer: str,
    candidate_pool: list[str],
    rng: Optional[random.Random] = None,
) -> Optional[DPOPair]:
    """rejected = highest log-prob NON-GT candidates (length matched to GT).

    candidate_pool entries MUST be single canonical SIDs (each tokenizes to 5
    tokens). More expensive than P1 (one forward per candidate) but yields the
    hardest negatives.
    """
    rng = rng or random.Random()
    gt_list = parse_sid_list(gt_answer)
    if not gt_list:
        return None
    gt_set = set(gt_list)
    need = len(gt_list)

    scored = model.score_candidates(messages, candidate_pool, enable_thinking=False)
    ranked_non_gt = [s.sid_str for s in scored if s.sid_str not in gt_set]
    rejected = _match_length(ranked_non_gt, need, gt_set, None, rng)
    if rejected is None or rejected == gt_list:
        return None

    return DPOPair(
        prompt=build_prompt_text(messages, model.tokenizer),
        chosen=_finalize_completion(gt_list, model.tokenizer),
        rejected=_finalize_completion(rejected, model.tokenizer),
        strategy="score_hardneg",
    )


# ----------------------------------------------------------------------------
# P3: Counterfactual (optional, fixed)
# ----------------------------------------------------------------------------
def make_weak_messages(strong_messages: list[dict]) -> list[dict]:
    """Strip the appended context card from the user turn (everything after the
    first blank-line separator)."""
    weak = []
    for msg in strong_messages:
        msg_copy = dict(msg)
        if msg_copy.get("role") == "user":
            content = msg_copy.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            parts = content.split("\n\n")
            msg_copy["content"] = parts[0] if len(parts) > 1 else content
        weak.append(msg_copy)
    return weak


def generate_counterfactual_pair(
    model: OpenOneRecWrapper,
    strong_messages: list[dict],
    weak_messages: list[dict],
    gt_answer: str,
    filler_pool: Optional[list[str]] = None,
    rng: Optional[random.Random] = None,
) -> Optional[DPOPair]:
    """prompt = STRONG (card) context; chosen = GT; rejected = model output under
    the WEAK (no-card) prompt, canonicalized and length-matched."""
    rng = rng or random.Random()
    gt_list = parse_sid_list(gt_answer)
    if not gt_list:
        return None
    gt_set = set(gt_list)
    need = len(gt_list)

    weak_gen = model.generate(
        weak_messages,
        max_new_tokens=256,
        do_sample=False,
        num_return=1,
        num_beams=1,
    )
    if not weak_gen:
        return None

    weak_list = parse_sid_list(weak_gen[0])
    rejected = _match_length(weak_list, need, gt_set, filler_pool, rng)
    if rejected is None or rejected == gt_list:
        return None

    return DPOPair(
        prompt=build_prompt_text(strong_messages, model.tokenizer),
        chosen=_finalize_completion(gt_list, model.tokenizer),
        rejected=_finalize_completion(rejected, model.tokenizer),
        strategy="counterfactual",
    )


def pair_to_dict(pair: DPOPair) -> dict:
    return {"prompt": pair.prompt, "chosen": pair.chosen, "rejected": pair.rejected}