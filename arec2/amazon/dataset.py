"""SFT dataset for Amazon sequential recommendation (Strategy 3).

Produces chat-format samples mirroring the project's RecIF SFT format
(system + user + assistant), where:
  - the user message contains an instruction + the Strategy-3 history
    ([itemic_tokens] + [keywords] per item);
  - the assistant message is the target item's itemic tokens ONLY (so the
    answer is a rankable SID).

Two regimes are supported (paper §6.3.2):
  - domain-specific: one domain's training pairs;
  - multi-domain joint: concatenate pairs from several domains, with a small
    domain tag in the instruction so the model can condition on the domain.

History expansion: by default we train on the leave-one-out history -> val
target. Optionally (``augment_subsequences``) we also emit shorter prefixes
(next-item-at-each-step) to densify the signal, which is standard for
sequential rec and helps small domains.
"""

from __future__ import annotations

import logging
from typing import Optional

from torch.utils.data import Dataset

from arec2.amazon.data_amazon import AmazonDomain
from arec2.amazon.item_repr import Strategy3Representer

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "你是一个跨域序列推荐专家，擅长根据用户的历史交互预测其下一个可能感兴趣的商品。"
    "You are a cross-domain sequential recommendation expert. Given a user's "
    "interaction history, predict the next item they will engage with."
)

USER_TEMPLATE = (
    "Domain: {domain}.\n"
    "The user has interacted with the following items in order:\n{history}\n"
    "Recommend the next item the user will interact with."
)


def _build_messages(domain_name: str, history_str: str, target_str: str) -> list[dict]:
    """Build a chat-format sample matching the project's RecIF SFT structure."""
    user_content = USER_TEMPLATE.format(domain=domain_name, history=history_str)
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": user_content}]},
        {"role": "assistant", "content": [{"type": "text", "text": target_str}]},
    ]


class AmazonSFTDataset(Dataset):
    """Sequential-recommendation SFT samples for one or more Amazon domains."""

    def __init__(
        self,
        domains: list[AmazonDomain],
        representers: dict[str, Strategy3Representer],
        with_keywords: bool = True,
        split: str = "train",
        augment_subsequences: bool = True,
        max_subsequences_per_user: int = 5,
        min_history: int = 1,
    ):
        """
        Args:
            domains: list of AmazonDomain (1 = domain-specific, >1 = joint).
            representers: category -> Strategy3Representer.
            with_keywords: True = Strategy 3 ([itemic]+[keywords]); False = the
                "itemic tokens only" ablation.
            split: "train" uses history->val_target; "val"/"test" are produced
                by the evaluator, not here.
            augment_subsequences: emit shorter prefixes as extra training pairs.
            max_subsequences_per_user: cap augmented pairs per user.
            min_history: minimum history length to keep a pair.
        """
        if split != "train":
            raise ValueError("AmazonSFTDataset only builds the 'train' split.")

        self.with_keywords = with_keywords
        self.samples: list[dict] = []

        for domain in domains:
            rep = representers[domain.category]
            n_before = len(self.samples)
            for uid, hist in domain.train_histories.items():
                target = domain.val_targets.get(uid)
                if target is None or len(hist) < min_history:
                    continue

                # Primary pair: full history -> validation target.
                self._add_pair(domain, rep, hist, target)

                # Augmented next-item-at-each-step pairs over the history.
                if augment_subsequences and len(hist) >= 2:
                    added = 0
                    # Walk from the end backwards so recent prefixes are kept.
                    for cut in range(len(hist) - 1, 0, -1):
                        sub_hist = hist[:cut]
                        sub_target = hist[cut]
                        if len(sub_hist) < min_history:
                            break
                        self._add_pair(domain, rep, sub_hist, sub_target)
                        added += 1
                        if added >= max_subsequences_per_user:
                            break

            logger.info(
                "  Domain '%s': built %d training pairs",
                domain.category, len(self.samples) - n_before,
            )

        logger.info(
            "AmazonSFTDataset ready: %d samples (with_keywords=%s, joint=%s)",
            len(self.samples), with_keywords, len(domains) > 1,
        )

    def _add_pair(self, domain, rep, hist, target):
        history_str = rep.sequence_repr(hist, with_keywords=self.with_keywords)
        target_str = rep.target_repr(target)
        if not history_str or not target_str:
            return
        messages = _build_messages(domain.short_name, history_str, target_str)
        self.samples.append(
            {
                "messages": messages,
                "task_type": "amazon",
                "category": domain.category,
                "metadata": {"answer": target_str},
            }
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]
