"""OpenRouter API client with tool-call support and cost tracking."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import httpx

from memcomp_bench.config import API_CALL_TIMEOUT, OPENROUTER_BASE_URL


def validate_human_context_messages(messages: list[dict[str, Any]]) -> None:
    """Reject malformed human-side chat histories before sending them upstream."""
    if not messages:
        raise ValueError("Human context cannot be empty")
    if messages[0].get("role") != "system":
        raise ValueError("Human context must start with a system message")
    previous_role: str | None = None
    for index, message in enumerate(messages):
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Human context has invalid role at index {index}: {role!r}")
        if index and role == "system":
            raise ValueError("Human context cannot contain non-initial system messages")
        if previous_role == role:
            raise ValueError(f"Human context has consecutive {role!r} messages at index {index}")
        previous_role = role


def _should_validate_human_context(messages: list[dict[str, Any]], request_role: str | None) -> bool:
    if request_role != "human" or not messages:
        return False
    return len(messages) > 1 or messages[0].get("role") == "system"


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


@dataclass
class _RPMWindow:
    limit: int
    timestamps: deque[float] = field(default_factory=deque)


class OpenRouterClient:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client = httpx.Client(timeout=API_CALL_TIMEOUT)
        self.total_cost: float = 0.0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self._rpm_windows: dict[str, _RPMWindow] = {}

    _RETRYABLE_CODES = frozenset({429, 500, 502, 503, 504})
    _MAX_RETRIES = 5

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
        request_role: str | None = None,
        rpm_limit: int | None = None,
    ) -> LLMResponse:
        if _should_validate_human_context(messages, request_role):
            validate_human_context_messages(messages)
        payload = self._build_payload(model, messages, max_tokens, temperature, tools, tool_choice, provider, reasoning)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/mikhailsal/memcomp-bench",
            "X-Title": "memcomp-bench",
        }

        resp, elapsed = self._send_with_retries(payload, headers, request_role=request_role, rpm_limit=rpm_limit)

        data = resp.json()
        return self._parse_response(data, elapsed)

    def _build_payload(
        self,
        model: str,
        messages: list,
        max_tokens: int,
        temperature: float,
        tools: list | None,
        tool_choice: str | dict | None,
        provider: dict | None,
        reasoning: dict | None,
    ) -> dict[str, Any]:
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
        return payload

    def _ensure_rpm_state(self) -> None:
        if not hasattr(self, "_rpm_windows"):
            self._rpm_windows = {}

    def _prune_rpm_window(self, window: _RPMWindow, now: float) -> None:
        cutoff = now - 60.0
        while window.timestamps and window.timestamps[0] <= cutoff:
            window.timestamps.popleft()

    def _apply_rpm_limit(self, request_role: str | None, rpm_limit: int | None) -> None:
        if request_role is None or rpm_limit is None:
            return
        if rpm_limit <= 0:
            raise ValueError("rpm_limit must be a positive integer")

        self._ensure_rpm_state()
        window = self._rpm_windows.get(request_role)
        if window is None or window.limit != rpm_limit:
            window = _RPMWindow(limit=rpm_limit)
            self._rpm_windows[request_role] = window

        now = time.monotonic()
        self._prune_rpm_window(window, now)
        while len(window.timestamps) >= rpm_limit:
            wait_seconds = 60.0 - (now - window.timestamps[0])
            if wait_seconds > 0:
                print(f"[rate-limit] {request_role} reached {rpm_limit} RPM, waiting {wait_seconds:.1f}s...")
                time.sleep(wait_seconds)
            now = time.monotonic()
            self._prune_rpm_window(window, now)
        window.timestamps.append(now)

    def _send_with_retries(
        self,
        payload: dict,
        headers: dict,
        *,
        request_role: str | None,
        rpm_limit: int | None,
    ) -> tuple[Any, float]:
        last_error = None
        for attempt in range(self._MAX_RETRIES):
            self._apply_rpm_limit(request_role, rpm_limit)
            start = time.monotonic()
            try:
                resp = self._client.post(f"{OPENROUTER_BASE_URL}/chat/completions", json=payload, headers=headers)
            except httpx.RequestError as e:
                if attempt < self._MAX_RETRIES - 1:
                    wait = min(2**attempt * 3, 30)
                    print(f"[retry] Network error: {e}, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Network error after {self._MAX_RETRIES} retries: {e}")

            elapsed = time.monotonic() - start

            if resp.status_code in self._RETRYABLE_CODES and attempt < self._MAX_RETRIES - 1:
                wait = min(2**attempt * 3, 30)
                error_body = resp.text[:200]
                print(f"[retry] API error {resp.status_code}: {error_body}, retrying in {wait}s...")
                time.sleep(wait)
                last_error = f"OpenRouter API error {resp.status_code}: {error_body}"
                continue

            if resp.status_code != 200:
                raise RuntimeError(f"OpenRouter API error {resp.status_code}: {resp.text[:500]}")
            return resp, elapsed
        raise RuntimeError(last_error or "Max retries exceeded")

    def _parse_response(self, data: dict, elapsed: float) -> LLMResponse:
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage_raw = data.get("usage", {})

        cost_raw = float(usage_raw.get("cost") or 0.0)
        cost_details = usage_raw.get("cost_details") or {}
        if usage_raw.get("is_byok") and cost_details.get("upstream_inference_cost"):
            cost = cost_raw + float(cost_details["upstream_inference_cost"])
        else:
            cost = cost_raw or float(usage_raw.get("market_cost") or 0.0)

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
