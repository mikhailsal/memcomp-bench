"""Shared fixtures for memcomp-bench tests."""

from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.openrouter_client import LLMResponse, OpenRouterClient, Usage


# ---------------------------------------------------------------------------
# FakeChatClient — scripted OpenRouterClient for offline tests
# ---------------------------------------------------------------------------

class FakeChatClient(OpenRouterClient):
    """Drop-in replacement that returns pre-queued LLMResponse objects."""

    def __init__(self, responses: list[LLMResponse] | None = None) -> None:
        self._api_key = "fake-key"
        self._client = httpx.Client(timeout=5)
        self.total_cost: float = 0.0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self._queue: deque[LLMResponse] = deque(responses or [])
        self.call_log: list[dict[str, Any]] = []

    def enqueue(self, *responses: LLMResponse) -> None:
        self._queue.extend(responses)

    def chat(self, **kwargs: Any) -> LLMResponse:
        self.call_log.append(kwargs)
        if not self._queue:
            raise RuntimeError("FakeChatClient: no more queued responses")
        resp = self._queue.popleft()
        self.total_cost += resp.usage.cost_usd
        self.total_prompt_tokens += resp.usage.prompt_tokens
        self.total_completion_tokens += resp.usage.completion_tokens
        return resp

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------

def make_tool_call_response(
    text: str,
    tool_call_id: str = "tc_001",
    *,
    reasoning: str | None = None,
    finish_reason: str = "tool_calls",
    prompt_tokens: int = 50,
    completion_tokens: int = 30,
    cost: float = 0.0,
) -> LLMResponse:
    """Build an LLMResponse that looks like a write_message_to_human tool call."""
    args: dict[str, Any] = {"text": text}
    if reasoning:
        args["reasoning"] = reasoning
    return LLMResponse(
        content=None,
        tool_calls=[{
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": "write_message_to_human",
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        }],
        reasoning=None,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
        ),
        finish_reason=finish_reason,
        raw={},
    )


def make_plain_response(
    text: str,
    *,
    reasoning: str | None = None,
    prompt_tokens: int = 50,
    completion_tokens: int = 30,
    cost: float = 0.0,
) -> LLMResponse:
    """Build an LLMResponse with plain content (no tool calls)."""
    return LLMResponse(
        content=text,
        tool_calls=None,
        reasoning=reasoning,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
        ),
        finish_reason="stop",
        raw={},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_client() -> FakeChatClient:
    return FakeChatClient()


@pytest.fixture()
def tmp_output_dir(tmp_path: Path) -> Path:
    d = tmp_path / "output"
    d.mkdir()
    return d


@pytest.fixture()
def live_proxy_or_skip():
    """Skip unless MEMCOMP_BENCH_LIVE=1 and the local proxy responds."""
    if os.environ.get("MEMCOMP_BENCH_LIVE") != "1":
        pytest.skip("MEMCOMP_BENCH_LIVE not set")

    from src.config import OPENROUTER_BASE_URL

    try:
        key = os.environ.get("OPENROUTER_KEY", "")
        r = httpx.get(
            f"{OPENROUTER_BASE_URL}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=5,
        )
        r.raise_for_status()
    except Exception as exc:
        pytest.skip(f"Proxy unreachable: {exc}")


@pytest.fixture()
def network_or_skip():
    """Skip unless MEMCOMP_BENCH_NETWORK=1."""
    if os.environ.get("MEMCOMP_BENCH_NETWORK") != "1":
        pytest.skip("MEMCOMP_BENCH_NETWORK not set")
