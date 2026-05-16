"""Tests for memcomp_bench.openrouter_client using httpx.MockTransport."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
import pytest

from memcomp_bench.openrouter_client import OpenRouterClient
from memcomp_bench.prompts import AI_TOOLS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_response(
    content: str = "Hello",
    *,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    cost: float = 0.001,
    is_byok: bool = False,
    upstream_cost: float | None = None,
    market_cost: float | None = None,
) -> dict[str, Any]:
    """Build a valid chat/completions JSON response body."""
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost": cost,
        "is_byok": is_byok,
    }
    if upstream_cost is not None:
        usage["cost_details"] = {"upstream_inference_cost": upstream_cost}
    if market_cost is not None:
        usage["market_cost"] = market_cost
    return {
        "choices": [
            {
                "message": {"content": content, "role": "assistant"},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


def _make_client(handler) -> OpenRouterClient:
    """Create an OpenRouterClient whose internal httpx.Client uses a mock transport."""
    transport = httpx.MockTransport(handler)
    client = OpenRouterClient.__new__(OpenRouterClient)
    client._api_key = "test-key"
    client._client = httpx.Client(transport=transport, timeout=5)
    client.total_cost = 0.0
    client.total_prompt_tokens = 0
    client.total_completion_tokens = 0
    return client


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_200_returns_content(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_ok_response("World"))

        client = _make_client(handler)
        resp = client.chat(model="test/m", messages=[{"role": "user", "content": "Hi"}])
        assert resp.content == "World"
        assert resp.finish_reason == "stop"
        assert resp.usage.prompt_tokens == 10
        client.close()

    def test_cost_accumulated(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_ok_response(cost=0.005))

        client = _make_client(handler)
        client.chat(model="m", messages=[])
        client.chat(model="m", messages=[])
        assert abs(client.total_cost - 0.01) < 1e-9
        client.close()

    def test_headers_contain_auth(self):
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=_ok_response())

        client = _make_client(handler)
        client.chat(model="m", messages=[])
        assert captured[0].headers["Authorization"] == "Bearer test-key"
        assert captured[0].headers["Content-Type"] == "application/json"
        client.close()

    def test_payload_includes_optional_fields(self):
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=_ok_response())

        client = _make_client(handler)
        client.chat(
            model="m",
            messages=[],
            tools=[{"type": "function"}],
            tool_choice={"type": "function", "function": {"name": "fn"}},
            provider={"only": ["prov"]},
            reasoning={"effort": "minimal"},
        )
        body = json.loads(captured[0].content)
        assert body["tools"] == [{"type": "function"}]
        assert body["tool_choice"]["function"]["name"] == "fn"
        assert body["provider"] == {"only": ["prov"]}
        assert body["reasoning"] == {"effort": "minimal"}
        client.close()

    def test_payload_preserves_write_message_schema_contract(self):
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=_ok_response())

        client = _make_client(handler)
        client.chat(model="m", messages=[], tools=AI_TOOLS)
        body = json.loads(captured[0].content)
        params = body["tools"][0]["function"]["parameters"]
        assert list(params["properties"]) == ["reasoning", "text"]
        assert params["required"] == ["reasoning", "text"]
        client.close()


class TestGeneratorToolChoiceConfig:
    def test_hy3_model_omits_tool_choice(self, monkeypatch):
        captured: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return httpx.Response(200, json=_ok_response())

        monkeypatch.setattr(
            "memcomp_bench.generator.generate_seed", lambda _: ["alpha", "beta", "gamma", "delta", "epsilon"]
        )

        from memcomp_bench.generator import ConversationGenerator

        client = _make_client(handler)
        generator = ConversationGenerator(
            client,
            {"name": "Test", "backstory": "Tester."},
            ai_model="tencent/hy3-preview",
        )

        generator._get_ai_response()

        assert captured, "expected an AI request"
        assert "tools" in captured[0]
        assert "tool_choice" not in captured[0]
        client.close()

    def test_default_model_still_sends_tool_choice(self, monkeypatch):
        captured: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return httpx.Response(200, json=_ok_response())

        monkeypatch.setattr(
            "memcomp_bench.generator.generate_seed", lambda _: ["alpha", "beta", "gamma", "delta", "epsilon"]
        )

        from memcomp_bench.generator import ConversationGenerator

        client = _make_client(handler)
        generator = ConversationGenerator(
            client,
            {"name": "Test", "backstory": "Tester."},
            ai_model="minimax/minimax-m2.7",
        )

        generator._get_ai_response()

        assert captured, "expected an AI request"
        assert captured[0]["tool_choice"]["function"]["name"] == "write_message_to_human"
        client.close()


# ---------------------------------------------------------------------------
# Retry on 429 / 5xx
# ---------------------------------------------------------------------------


class TestRetry:
    def test_429_then_200(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda _: None)
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(429, text="rate limited")
            return httpx.Response(200, json=_ok_response("Recovered"))

        client = _make_client(handler)
        resp = client.chat(model="m", messages=[])
        assert resp.content == "Recovered"
        assert call_count == 2
        client.close()

    def test_5xx_exhaustion(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda _: None)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(502, text="bad gateway")

        client = _make_client(handler)
        with pytest.raises(RuntimeError, match="502"):
            client.chat(model="m", messages=[])
        client.close()

    def test_non_retryable_error_raises_immediately(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text="unauthorized")

        client = _make_client(handler)
        with pytest.raises(RuntimeError, match="401"):
            client.chat(model="m", messages=[])
        client.close()


class TestRequestRateLimits:
    def test_waits_until_same_role_window_expires(self, monkeypatch):
        now = {"value": 0.0}
        sleeps: list[float] = []
        call_times: list[float] = []

        def fake_monotonic() -> float:
            return now["value"]

        def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)
            now["value"] += seconds

        def handler(request: httpx.Request) -> httpx.Response:
            call_times.append(now["value"])
            return httpx.Response(200, json=_ok_response())

        monkeypatch.setattr("memcomp_bench.openrouter_client.time.monotonic", fake_monotonic)
        monkeypatch.setattr("memcomp_bench.openrouter_client.time.sleep", fake_sleep)

        client = _make_client(handler)
        client.chat(model="m", messages=[], request_role="ai", rpm_limit=2)
        client.chat(model="m", messages=[], request_role="ai", rpm_limit=2)
        client.chat(model="m", messages=[], request_role="ai", rpm_limit=2)

        assert sleeps == [60.0]
        assert call_times == [0.0, 0.0, 60.0]
        client.close()

    def test_ai_and_human_use_separate_windows(self, monkeypatch):
        now = {"value": 0.0}
        sleeps: list[float] = []
        seen_roles: list[str] = []

        def fake_monotonic() -> float:
            return now["value"]

        def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)
            now["value"] += seconds

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            seen_roles.append(body["messages"][0]["content"])
            return httpx.Response(200, json=_ok_response())

        monkeypatch.setattr("memcomp_bench.openrouter_client.time.monotonic", fake_monotonic)
        monkeypatch.setattr("memcomp_bench.openrouter_client.time.sleep", fake_sleep)

        client = _make_client(handler)
        client.chat(model="m", messages=[{"role": "user", "content": "ai-1"}], request_role="ai", rpm_limit=1)
        client.chat(
            model="m",
            messages=[{"role": "user", "content": "human-1"}],
            request_role="human",
            rpm_limit=1,
        )
        client.chat(model="m", messages=[{"role": "user", "content": "ai-2"}], request_role="ai", rpm_limit=1)

        assert seen_roles == ["ai-1", "human-1", "ai-2"]
        assert sleeps == [60.0]
        client.close()


# ---------------------------------------------------------------------------
# Network errors
# ---------------------------------------------------------------------------


class TestNetworkError:
    def test_retries_on_connection_error(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda _: None)
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise httpx.ConnectError("Connection refused")
            return httpx.Response(200, json=_ok_response("Back"))

        client = _make_client(handler)
        resp = client.chat(model="m", messages=[])
        assert resp.content == "Back"
        assert call_count == 3
        client.close()

    def test_exhausted_network_errors(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda _: None)

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        client = _make_client(handler)
        with pytest.raises(RuntimeError, match="Network error"):
            client.chat(model="m", messages=[])
        client.close()


# ---------------------------------------------------------------------------
# BYOK cost accounting
# ---------------------------------------------------------------------------


class TestCostAccounting:
    def test_byok_adds_upstream_cost(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = _ok_response(cost=0.001, is_byok=True, upstream_cost=0.01)
            return httpx.Response(200, json=body)

        client = _make_client(handler)
        resp = client.chat(model="m", messages=[])
        assert abs(resp.usage.cost_usd - 0.011) < 1e-9
        client.close()

    def test_non_byok_uses_cost_field(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_ok_response(cost=0.005))

        client = _make_client(handler)
        resp = client.chat(model="m", messages=[])
        assert abs(resp.usage.cost_usd - 0.005) < 1e-9
        client.close()

    def test_fallback_to_market_cost(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = _ok_response(cost=0.0, market_cost=0.003)
            return httpx.Response(200, json=body)

        client = _make_client(handler)
        resp = client.chat(model="m", messages=[])
        assert abs(resp.usage.cost_usd - 0.003) < 1e-9
        client.close()


# ---------------------------------------------------------------------------
# Tool calls in response
# ---------------------------------------------------------------------------


class TestToolCallResponse:
    def test_tool_calls_parsed(self):
        body = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "tc_1",
                                "type": "function",
                                "function": {"name": "write_message_to_human", "arguments": '{"text":"hi"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0},
        }

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=body)

        client = _make_client(handler)
        resp = client.chat(model="m", messages=[])
        assert resp.tool_calls is not None
        assert resp.tool_calls[0]["function"]["name"] == "write_message_to_human"
        client.close()

    def test_reasoning_fields_parsed(self):
        body = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "reasoning": "deep thought",
                        "reasoning_details": [{"type": "reasoning.text", "text": "hmm"}],
                        "role": "assistant",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0},
        }

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=body)

        client = _make_client(handler)
        resp = client.chat(model="m", messages=[])
        assert resp.reasoning == "deep thought"
        assert resp.reasoning_details[0]["text"] == "hmm"
        client.close()


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_is_idempotent(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_ok_response())

        client = _make_client(handler)
        client.close()
        client.close()  # no error
