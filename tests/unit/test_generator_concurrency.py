from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from memcomp_bench.generator import ConversationGenerator
from memcomp_bench.prompts import HUMAN_PROFILES
from tests.conftest import FakeChatClient, make_plain_response, make_tool_call_response


def test_initial_ids_are_unique_with_fixed_timestamp(monkeypatch):
    fixed_now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    class FrozenDateTime:
        @staticmethod
        def now(tz=None):
            return fixed_now

    monkeypatch.setattr("memcomp_bench.generator_runtime.datetime", FrozenDateTime)

    ids = {ConversationGenerator(FakeChatClient(), HUMAN_PROFILES[0])._record.id for _ in range(5)}

    assert len(ids) == 5
    assert all(value.startswith("20260101_000000_") for value in ids)


def test_cost_tracking_is_local_to_each_generator(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    shared_client = FakeChatClient()

    shared_client.enqueue(make_plain_response("Plan 1", cost=0.01))
    shared_client.enqueue(make_tool_call_response("Hello 1", tool_call_id="wmth00001", cost=0.02))
    shared_client.enqueue(make_plain_response("Human 1", cost=0.03))

    first = ConversationGenerator(shared_client, HUMAN_PROFILES[0], target_tokens=10, max_turns=1)
    first_record = first.generate()

    shared_client.enqueue(make_plain_response("Plan 2", cost=0.04))
    shared_client.enqueue(make_tool_call_response("Hello 2", tool_call_id="wmth00001", cost=0.05))
    shared_client.enqueue(make_plain_response("Human 2", cost=0.06))

    second = ConversationGenerator(shared_client, HUMAN_PROFILES[0], target_tokens=10, max_turns=1)
    second_record = second.generate()

    assert first_record.total_cost_usd == pytest.approx(0.06)
    assert second_record.total_cost_usd == pytest.approx(0.15)
