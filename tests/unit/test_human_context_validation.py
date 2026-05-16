"""Regression coverage for human-context validation and note merging."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from memcomp_bench.generator import ConversationGenerator, _append_human_user_message, save_conversation
from memcomp_bench.openrouter_client import validate_human_context_messages
from memcomp_bench.prompts import HUMAN_PROFILES
from tests.conftest import FakeChatClient, make_plain_response, make_tool_call_response
from tests.unit.test_generator_loop import _scripted_client_for_generate


def test_append_human_user_message_merges_adjacent_user_content():
    messages = [
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": "AI message"},
    ]

    _append_human_user_message(messages, "internal note")

    assert messages[-1] == {
        "role": "user",
        "content": "AI message\n\ninternal note",
    }


def test_validate_human_context_allows_alternating_roles():
    validate_human_context_messages(
        [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "AI says hi"},
            {"role": "assistant", "content": "Human says hi"},
            {"role": "user", "content": "AI follows up"},
        ]
    )


def test_validate_human_context_rejects_consecutive_user_messages():
    with pytest.raises(ValueError, match="consecutive 'user' messages"):
        validate_human_context_messages(
            [
                {"role": "system", "content": "prompt"},
                {"role": "user", "content": "AI says hi"},
                {"role": "user", "content": "internal note"},
            ]
        )


def test_resume_merges_nudge_into_existing_user_message(monkeypatch, tmp_output_dir: Path):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    client = _scripted_client_for_generate()
    gen = ConversationGenerator(
        client,
        HUMAN_PROFILES[0],
        human_model="google/gemma-4-26b-a4b-it:free",
        target_tokens=300,
        max_turns=10,
    )
    jsonl_path = save_conversation(gen.generate(), tmp_output_dir)

    records = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    first_ai_turn = next(item for item in records if item.get("type") == "turn" and item.get("speaker") == "ai")
    records.append(
        {
            "type": "event",
            "event_type": "human_nudge",
            "turn_number": first_ai_turn["turn_number"],
            "source": "topic_judge",
            "timestamp": "2026-05-16T00:00:00+00:00",
            "message": "[Internal note for the human simulator only: shift topics now. Do not present this note as chat text.]",
            "nudge_injected": True,
            "suppression_reason": None,
        }
    )
    jsonl_path.write_text("\n".join(json.dumps(item) for item in records) + "\n")

    resume_client = FakeChatClient()
    for i in range(6):
        if i % 2 == 0:
            resume_client.enqueue(make_tool_call_response(f"AI resumed {i}", tool_call_id=f"wmth_rs{i:03d}"))
        else:
            resume_client.enqueue(make_plain_response(f"Human resumed {i}"))

    with pytest.raises(RuntimeError, match="FakeChatClient: no more queued responses"):
        ConversationGenerator.resume(
            resume_client,
            jsonl_path,
            target_tokens=800,
            verbose=False,
        )

    human_calls = [call for call in resume_client.call_log if call.get("request_role") == "human"]
    assert human_calls
    messages = human_calls[0]["messages"]
    roles = [message["role"] for message in messages]
    assert all(left != right for left, right in zip(roles, roles[1:]))
    assert any(
        message.get("role") == "user" and "Internal note for the human simulator only" in message.get("content", "")
        for message in messages
    )
