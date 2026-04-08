"""OpenRouter API client with tool-call support and cost tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from src.config import API_CALL_TIMEOUT, OPENROUTER_BASE_URL


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    elapsed_seconds: float = 0.0


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    reasoning: str | None = None
    reasoning_details: list[dict[str, Any]] | None = None
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class OpenRouterClient:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client = httpx.Client(timeout=API_CALL_TIMEOUT)
        self.total_cost: float = 0.0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
        provider: dict[str, Any] | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        if provider:
            payload["provider"] = provider
        if reasoning:
            payload["reasoning"] = reasoning

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/mikhailsal/memcomp-bench",
            "X-Title": "memcomp-bench",
        }

        max_retries = 5
        retryable_codes = {429, 500, 502, 503, 504}
        last_error = None

        for attempt in range(max_retries):
            start = time.monotonic()
            try:
                resp = self._client.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                )
            except httpx.RequestError as e:
                if attempt < max_retries - 1:
                    wait = min(2 ** attempt * 3, 30)
                    print(f"[retry] Network error: {e}, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Network error after {max_retries} retries: {e}")

            elapsed = time.monotonic() - start

            if resp.status_code in retryable_codes and attempt < max_retries - 1:
                wait = min(2 ** attempt * 3, 30)
                error_body = resp.text[:200]
                print(f"[retry] API error {resp.status_code}: {error_body}, retrying in {wait}s...")
                time.sleep(wait)
                last_error = f"OpenRouter API error {resp.status_code}: {error_body}"
                continue

            if resp.status_code != 200:
                error_body = resp.text[:500]
                raise RuntimeError(
                    f"OpenRouter API error {resp.status_code}: {error_body}"
                )
            break
        else:
            raise RuntimeError(last_error or "Max retries exceeded")

        data = resp.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage_raw = data.get("usage", {})

        cost = float(usage_raw.get("cost", 0.0))

        usage = Usage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            cost_usd=cost,
            elapsed_seconds=round(elapsed, 2),
        )

        self.total_cost += usage.cost_usd
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens

        return LLMResponse(
            content=message.get("content"),
            tool_calls=message.get("tool_calls"),
            reasoning=message.get("reasoning"),
            reasoning_details=message.get("reasoning_details") or None,
            usage=usage,
            finish_reason=choice.get("finish_reason", ""),
            raw=data,
        )

    def close(self) -> None:
        self._client.close()
