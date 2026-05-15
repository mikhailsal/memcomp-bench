"""Regression tests for mandatory write_message_to_human reasoning."""

from __future__ import annotations

import json
import time

from memcomp_bench.generator import ConversationGenerator
from memcomp_bench.prompts import HUMAN_PROFILES
from tests.conftest import FakeChatClient, make_plain_response, make_tool_call_response


class TestGeneratorReasoningContract:
    def test_generate_retries_missing_reasoning_parameter(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda _: None)
        client = FakeChatClient()
        client.enqueue(make_plain_response("Plan: Keep it casual."))
        client.enqueue(
            make_tool_call_response(
                "Hello there!",
                tool_call_id="wmth_bad_001",
                include_reasoning=False,
            )
        )
        client.enqueue(
            make_tool_call_response(
                "Hello there!",
                tool_call_id="wmth_good_001",
                reasoning="I should open with a simple greeting before asking anything more.",
            )
        )
        client.enqueue(make_plain_response("Hey, nice to meet you."))
        client.enqueue(
            make_tool_call_response(
                "Nice to meet you too.",
                tool_call_id="wmth_good_002",
                reasoning="They sound friendly, so I can respond warmly and keep things moving.",
            )
        )
        client.enqueue(make_plain_response("What have you been thinking about?"))

        gen = ConversationGenerator(
            client,
            HUMAN_PROFILES[0],
            target_tokens=200,
            max_turns=3,
        )

        record = gen.generate()

        assert len(record.turns) == 3
        ai_calls = [call for call in client.call_log if call.get("request_role") == "ai"]
        assert len(ai_calls) == 3
        greeting_msg = next(msg for msg in gen._ai_messages if msg.get("tool_calls"))
        greeting_args = json.loads(greeting_msg["tool_calls"][0]["function"]["arguments"])
        assert greeting_args["reasoning"] == "I should open with a simple greeting before asking anything more."
