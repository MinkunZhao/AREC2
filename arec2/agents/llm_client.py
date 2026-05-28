"""LLM client wrapper for yunwu.ai with caching, retry, and batch support."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import diskcache as dc
from openai import AsyncOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (USD)
PRICES = {
    "gpt-4o-mini": {"in": 0.150, "out": 0.600},
    "gpt-4o": {"in": 2.50, "out": 10.00},
}


@dataclass
class LLMResponse:
    """Response from LLM with usage tracking."""
    text: str
    json_obj: dict | None
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    cached: bool


class LLMClient:
    """OpenAI-compatible client for yunwu.ai with disk caching and retry logic."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str = "https://yunwu.ai/v1",
        api_key: str = "sk-hJEO1ngxEgAb0wAAhTml9CR8Skw52sH8PUw36VLpV0sYHB3n",
        cache_dir: str = "./caches/llm_cache",
        max_concurrency: int = 8,
    ):
        """Initialize LLM client.

        Args:
            model: Default model name (e.g., "gpt-4o-mini").
            base_url: API base URL.
            api_key: API key for authentication.
            cache_dir: Directory for disk cache.
            max_concurrency: Max concurrent async requests.
        """
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.aclient = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.cache = dc.Cache(cache_dir, size_limit=int(2e9))  # 2GB cache
        self.sem = asyncio.Semaphore(max_concurrency)
        self.total_cost = 0.0
        logger.info(
            f"LLMClient initialized | model={model} | base_url={base_url} | cache_dir={cache_dir}"
        )

    def _cache_key(
        self,
        messages: list[dict],
        temperature: float,
        json_mode: bool,
        model: str,
    ) -> str:
        """Generate cache key from request parameters."""
        payload = json.dumps(
            {"m": model, "msgs": messages, "t": temperature, "j": json_mode},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=30))
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        json_mode: bool = False,
        model: str | None = None,
    ) -> LLMResponse:
        """Synchronous chat completion with caching and retry.

        Args:
            messages: List of message dicts with "role" and "content".
            temperature: Sampling temperature.
            json_mode: Enable JSON output mode.
            model: Override default model.

        Returns:
            LLMResponse with text, json_obj, usage, and cost.
        """
        model = model or self.model
        key = self._cache_key(messages, temperature, json_mode, model)

        # Check cache
        if key in self.cache:
            cached_data = self.cache[key]
            logger.debug(f"Cache hit for key {key[:8]}...")
            return LLMResponse(**cached_data, cached=True)

        # Make API call
        kwargs = dict(model=model, messages=messages, temperature=temperature)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self.client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content
        usage = resp.usage

        # Calculate cost
        cost = (
            usage.prompt_tokens * PRICES.get(model, PRICES["gpt-4o-mini"])["in"] / 1e6
            + usage.completion_tokens * PRICES.get(model, PRICES["gpt-4o-mini"])["out"] / 1e6
        )
        self.total_cost += cost

        # Parse JSON if requested
        json_obj = None
        if json_mode:
            try:
                json_obj = json.loads(text)
            except json.JSONDecodeError:
                logger.warning(f"LLM produced invalid JSON despite json_mode: {text[:200]}")

        # Cache result
        out = dict(
            text=text,
            json_obj=json_obj,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=cost,
        )
        self.cache[key] = out

        logger.debug(
            f"API call | model={model} | tokens={usage.prompt_tokens}+{usage.completion_tokens} | cost=${cost:.4f}"
        )
        return LLMResponse(**out, cached=False)

    async def _achat_uncached(
        self,
        messages: list[dict],
        temperature: float,
        json_mode: bool,
        model: str,
    ) -> LLMResponse:
        """Async chat completion without cache check (internal)."""
        kwargs = dict(model=model, messages=messages, temperature=temperature)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = await self.aclient.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content
        usage = resp.usage

        cost = (
            usage.prompt_tokens * PRICES.get(model, PRICES["gpt-4o-mini"])["in"] / 1e6
            + usage.completion_tokens * PRICES.get(model, PRICES["gpt-4o-mini"])["out"] / 1e6
        )
        self.total_cost += cost

        json_obj = None
        if json_mode:
            try:
                json_obj = json.loads(text)
            except json.JSONDecodeError:
                logger.warning(f"LLM produced invalid JSON: {text[:200]}")

        return LLMResponse(
            text=text,
            json_obj=json_obj,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=cost,
            cached=False,
        )

    async def _achat_one(
        self,
        messages: list[dict],
        temperature: float,
        json_mode: bool,
        model: str,
    ) -> LLMResponse:
        """Async chat with semaphore and cache check."""
        key = self._cache_key(messages, temperature, json_mode, model)

        # Check cache
        if key in self.cache:
            cached_data = self.cache[key]
            return LLMResponse(**cached_data, cached=True)

        # Acquire semaphore and make call
        async with self.sem:
            resp = await self._achat_uncached(messages, temperature, json_mode, model)

        # Cache result
        self.cache[key] = dict(
            text=resp.text,
            json_obj=resp.json_obj,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            cost_usd=resp.cost_usd,
        )
        return resp

    async def batch_chat_async(
        self,
        messages_list: list[list[dict]],
        temperature: float = 0.0,
        json_mode: bool = False,
        model: str | None = None,
    ) -> list[LLMResponse]:
        """Async batch chat completion.

        Args:
            messages_list: List of message lists.
            temperature: Sampling temperature.
            json_mode: Enable JSON output mode.
            model: Override default model.

        Returns:
            List of LLMResponse objects.
        """
        model = model or self.model
        tasks = [
            self._achat_one(msgs, temperature, json_mode, model)
            for msgs in messages_list
        ]
        return await asyncio.gather(*tasks)

    def batch_chat(
        self,
        messages_list: list[list[dict]],
        temperature: float = 0.0,
        json_mode: bool = False,
        model: str | None = None,
    ) -> list[LLMResponse]:
        """Synchronous wrapper for batch chat.

        Args:
            messages_list: List of message lists.
            temperature: Sampling temperature.
            json_mode: Enable JSON output mode.
            model: Override default model.

        Returns:
            List of LLMResponse objects.
        """
        return asyncio.run(
            self.batch_chat_async(messages_list, temperature, json_mode, model)
        )

    def get_stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_cost_usd": self.total_cost,
            "cache_size": len(self.cache),
        }
