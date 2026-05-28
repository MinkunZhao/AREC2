"""Wrapper around OpenOneRec-1.7B for generation and candidate-constrained scoring."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

SID_BEGIN_TOKEN = "<|sid_begin|>"
SID_END_TOKEN = "<|sid_end|>"
SID_PATTERN = re.compile(
    r"<\|sid_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)><\|sid_end\|>"
)
TOKENS_PER_SID = 5  # sid_begin, s_a, s_b, s_c, sid_end


@dataclass
class ScoredCandidate:
    sid_str: str
    log_prob: float
    pid: Optional[int] = None


class OpenOneRecWrapper:
    def __init__(
        self,
        model_path: str = "OpenOneRec/OneRec-1.7B",
        device: str = "auto",
        torch_dtype: str = "bfloat16",
    ):
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": "auto",
        }
        _p = Path(model_path)
        if _p.exists():
            model_path = str(_p.resolve())
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype_map.get(torch_dtype, "auto"),
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

        self._sid_begin_id = self.tokenizer.encode(
            SID_BEGIN_TOKEN, add_special_tokens=False
        )[0]
        self._sid_end_id = self.tokenizer.encode(
            SID_END_TOKEN, add_special_tokens=False
        )[0]

        self._s_a_base = self.tokenizer.encode("<s_a_0>", add_special_tokens=False)[0]
        self._s_b_base = self.tokenizer.encode("<s_b_0>", add_special_tokens=False)[0]
        self._s_c_base = self.tokenizer.encode("<s_c_0>", add_special_tokens=False)[0]

        logger.info(
            "Loaded OpenOneRec from %s | device=%s | dtype=%s",
            model_path, device, torch_dtype,
        )

    # ------------------------------------------------------------------
    # Public: free-form generation
    # ------------------------------------------------------------------
    def generate(
            self,
            messages: list[dict],
            max_new_tokens: int = 512,
            enable_thinking: bool = False,
            temperature: float = 0.6,
            top_k: int = 20,
            top_p: float = 0.95,
            do_sample: bool = True,
            num_return: int = 1,
            num_beams: int = 1,  # <--- 新增
    ) -> list[str]:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = self.tokenizer(
            [text], return_tensors="pt"
        ).to(self.model.device)

        # 修复：当不采样且需要多个返回值时，必须使用 beam search
        if not do_sample and num_return > 1:
            num_beams = max(num_beams, num_return)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_k=top_k if do_sample else None,
                top_p=top_p if do_sample else None,
                do_sample=do_sample,
                num_return_sequences=num_return,
                num_beams=num_beams,  # <--- 新增
            )

        prompt_len = inputs.input_ids.shape[1]
        results = []
        for seq in out:
            gen_ids = seq[prompt_len:].tolist()
            content = self._strip_thinking(gen_ids)
            results.append(content)
        return results

    # ------------------------------------------------------------------
    # Public: batched free-form generation
    # ------------------------------------------------------------------
    def generate_batch(
            self,
            messages_list: list[list[dict]],
            max_new_tokens: int = 512,
            enable_thinking: bool = False,
            temperature: float = 0.6,
            top_k: int = 20,
            top_p: float = 0.95,
            do_sample: bool = True,
            num_return: int = 1,
            num_beams: int = 1,  # <--- 新增
    ) -> list[str] | list[list[str]]:
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        texts = [
            self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            for msgs in messages_list
        ]

        orig_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
        ).to(self.model.device)
        self.tokenizer.padding_side = orig_side

        prompt_len = inputs.input_ids.shape[1]

        if not do_sample and num_return > 1:
            num_beams = max(num_beams, num_return)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_k=top_k if do_sample else None,
                top_p=top_p if do_sample else None,
                do_sample=do_sample,
                num_return_sequences=num_return,
                num_beams=num_beams,  # <--- 新增
            )

        B = len(messages_list)
        if num_return == 1:
            results = []
            for seq in out:
                gen_ids = seq[prompt_len:].tolist()
                results.append(self._strip_thinking(gen_ids))
            return results
        else:
            results = []
            for i in range(B):
                group = []
                for j in range(num_return):
                    seq = out[i * num_return + j]
                    gen_ids = seq[prompt_len:].tolist()
                    group.append(self._strip_thinking(gen_ids))
                results.append(group)
            return results

    # ------------------------------------------------------------------
    # Public: candidate-constrained scoring via log-likelihood
    # ------------------------------------------------------------------
    def score_candidates(
        self,
        messages: list[dict],
        candidate_sids: list[str],
        enable_thinking: bool = False,
    ) -> list[ScoredCandidate]:
        """Score each candidate SID string by its log-probability given the context.

        Uses KV-cache: encode the prompt once, then for each candidate, run a single
        forward pass on [sid_begin, s_a, s_b, s_c, sid_end] reusing the cached prefix.
        """
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        prompt_ids = self.tokenizer.encode(text, add_special_tokens=False)

        all_cand_token_ids = []
        for sid_str in candidate_sids:
            toks = self.tokenizer.encode(sid_str, add_special_tokens=False)
            assert len(toks) == TOKENS_PER_SID, (
                f"SID '{sid_str}' tokenized to {len(toks)} tokens, expected {TOKENS_PER_SID}"
            )
            all_cand_token_ids.append(toks)

        input_ids = torch.tensor([prompt_ids], device=self.model.device)
        with torch.no_grad():
            prefix_out = self.model(input_ids=input_ids, use_cache=True)

        prefix_cache = prefix_out.past_key_values
        last_prompt_logits = prefix_out.logits[0, -1, :]

        scored = []
        for sid_str, cand_toks in zip(candidate_sids, all_cand_token_ids):
            cand_input = torch.tensor([cand_toks[:-1]], device=self.model.device)
            with torch.no_grad():
                cand_out = self.model(
                    input_ids=cand_input,
                    past_key_values=prefix_cache,
                    use_cache=False,
                )
            all_logits = torch.cat(
                [last_prompt_logits.unsqueeze(0), cand_out.logits[0]], dim=0
            )

            log_prob = 0.0
            for t_idx, tok_id in enumerate(cand_toks):
                token_log_probs = F.log_softmax(all_logits[t_idx], dim=-1)
                log_prob += token_log_probs[tok_id].item()

            scored.append(ScoredCandidate(sid_str=sid_str, log_prob=log_prob))

        scored.sort(key=lambda x: x.log_prob, reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def sid_triple_to_str(self, a: int, b: int, c: int) -> str:
        return f"<|sid_begin|><s_a_{a}><s_b_{b}><s_c_{c}><|sid_end|>"

    def sid_triple_to_token_ids(self, a: int, b: int, c: int) -> list[int]:
        return [
            self._sid_begin_id,
            self._s_a_base + a,
            self._s_b_base + b,
            self._s_c_base + c,
            self._sid_end_id,
        ]

    def parse_sids_from_text(self, text: str) -> list[tuple[int, int, int]]:
        return [(int(a), int(b), int(c)) for a, b, c in SID_PATTERN.findall(text)]

    def _strip_thinking(self, token_ids: list[int]) -> str:
        THINK_END = 151668
        try:
            idx = len(token_ids) - token_ids[::-1].index(THINK_END)
        except ValueError:
            idx = 0
        # skip_special_tokens=False so SID tokens (<s_a_*> etc.) survive decoding;
        # only strip chat-template control tokens.
        text = self.tokenizer.decode(token_ids[idx:], skip_special_tokens=False)
        for ctrl in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
            text = text.replace(ctrl, "")
        return text.strip()

    def format_messages_from_benchmark(self, raw_messages: str | list) -> list[dict]:
        """Convert benchmark `messages` field to the format expected by apply_chat_template.

        Benchmark messages have content as a list of dicts with 'type' and 'text' keys.
        We flatten to plain string content.
        """
        if isinstance(raw_messages, str):
            raw_messages = json.loads(raw_messages)

        formatted = []
        for msg in raw_messages:
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, list):
                text_parts = [
                    part["text"] for part in content if part.get("type") == "text"
                ]
                content = "".join(text_parts)
            formatted.append({"role": role, "content": content})
        return formatted
