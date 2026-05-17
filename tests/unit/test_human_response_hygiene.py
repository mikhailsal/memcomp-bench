"""Tests for human-response sanitization and retry behavior."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from memcomp_bench.generator import ConversationGenerator, save_conversation
from memcomp_bench.openrouter_client import LLMResponse, Usage
from memcomp_bench.prompts import HUMAN_PROFILES
from tests.conftest import FakeChatClient, make_plain_response, make_tool_call_response
from tests.unit.test_generator_loop import _scripted_client_for_generate


def test_generate_strips_human_thinking_tags_before_recording_and_forwarding(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    client = FakeChatClient(
        [
            make_plain_response("Plan: keep it casual."),
            make_tool_call_response("Hey there!", tool_call_id="wmth00001"),
            make_plain_response("<thought>private notes</thought>Visible reply"),
            make_tool_call_response("Nice to meet you.", tool_call_id="wmth00002"),
        ]
    )

    gen = ConversationGenerator(
        client,
        HUMAN_PROFILES[0],
        target_tokens=80,
        max_turns=2,
    )
    record = gen.generate()

    assert record.turns[0].visible_text == "Visible reply"
    assert record.turns[0].human_reasoning == "private notes"
    tool_messages = [msg for msg in gen._ai_messages if msg.get("role") == "tool"]
    assert tool_messages[0]["content"] == "Visible reply"
    assert "private notes" not in json.dumps(gen._ai_messages)


def test_generate_merges_native_and_tagged_human_reasoning_into_history_and_markdown(monkeypatch, tmp_output_dir: Path):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    client = FakeChatClient(
        [
            make_plain_response("Plan: keep it casual."),
            make_tool_call_response("Hey there!", tool_call_id="wmth00001"),
            make_plain_response(
                "<think>tagged reasoning</think>Visible reply",
                reasoning="native reasoning",
            ),
            make_tool_call_response("Nice to meet you.", tool_call_id="wmth00002"),
        ]
    )

    gen = ConversationGenerator(
        client,
        HUMAN_PROFILES[0],
        target_tokens=80,
        max_turns=2,
    )
    record = gen.generate()
    jsonl_path = save_conversation(record, tmp_output_dir)
    md_path = tmp_output_dir / f"{jsonl_path.stem}.md"
    markdown = md_path.read_text()

    assert record.turns[0].human_reasoning == "native reasoning\n\ntagged reasoning"
    assert record.turns[0].visible_text == "Visible reply"
    assert "native reasoning" in markdown
    assert "tagged reasoning" in markdown
    assert "<think>" not in markdown


def test_generate_retries_human_non_stop_finish_reason(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    client = FakeChatClient(
        [
            make_plain_response("Plan: keep it casual."),
            make_tool_call_response("Hey there!", tool_call_id="wmth00001"),
            LLMResponse(
                content="Cut off reply",
                tool_calls=None,
                reasoning="partial",
                usage=Usage(prompt_tokens=40, completion_tokens=10),
                finish_reason="length",
                raw={},
            ),
            make_plain_response("Recovered reply", reasoning="complete"),
            make_tool_call_response("Thanks for replying.", tool_call_id="wmth00002"),
        ]
    )

    gen = ConversationGenerator(
        client,
        HUMAN_PROFILES[0],
        target_tokens=80,
        max_turns=2,
    )
    record = gen.generate()

    assert record.turns[0].visible_text == "Recovered reply"
    human_calls = [call for call in client.call_log if call.get("request_role") == "human"]
    assert len(human_calls) == 3
    assert "Cut off reply" not in json.dumps(gen._ai_messages)


def test_generate_retries_when_human_returns_dormant_placeholder(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    client = FakeChatClient(
        [
            make_plain_response("Plan: keep it casual."),
            make_tool_call_response("Hey there!", tool_call_id="wmth00001"),
            make_plain_response("[No message — the conversation is dormant]"),
            make_plain_response("Hey. Sorry, got pulled into work for a couple days."),
            make_tool_call_response("Glad you're back.", tool_call_id="wmth00002"),
        ]
    )

    gen = ConversationGenerator(
        client,
        HUMAN_PROFILES[0],
        target_tokens=80,
        max_turns=2,
    )
    record = gen.generate()

    assert record.turns[0].visible_text == "Hey. Sorry, got pulled into work for a couple days."
    assert all("No message" not in turn.visible_text for turn in record.turns)
    human_calls = [call for call in client.call_log if call.get("request_role") == "human"]
    assert len(human_calls) == 3


def test_resume_sanitizes_existing_raw_human_tool_messages(monkeypatch, tmp_output_dir: Path):
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

    lines = jsonl_path.read_text().splitlines()
    metadata = json.loads(lines[0])
    metadata["max_turns"] = 7
    lines[0] = json.dumps(metadata)
    jsonl_path.write_text("\n".join(lines) + "\n")

    raw_path = tmp_output_dir / f"{jsonl_path.stem}_raw_ai_context.json"
    raw_messages = json.loads(raw_path.read_text())
    for msg in raw_messages:
        if msg.get("role") == "tool":
            msg["content"] = "<think>hidden</think>Visible again"
            break
    raw_path.write_text(json.dumps(raw_messages))

    resume_client = FakeChatClient(
        [
            make_tool_call_response("AI resumed", tool_call_id="wmth_rs001"),
            make_plain_response("Human resumed"),
            make_tool_call_response("AI follow-up", tool_call_id="wmth_rs002"),
            make_plain_response("Human follow-up"),
            make_tool_call_response("AI wrap-up", tool_call_id="wmth_rs003"),
            make_plain_response("Human closer"),
        ]
    )

    with pytest.raises(RuntimeError, match="FakeChatClient: no more queued responses"):
        ConversationGenerator.resume(
            resume_client,
            jsonl_path,
            target_tokens=441,
            verbose=False,
        )

    ai_calls = [call for call in resume_client.call_log if call.get("request_role") == "ai"]
    assert ai_calls
    tool_messages = [msg for msg in ai_calls[0]["messages"] if msg.get("role") == "tool"]
    assert any(msg.get("content") == "Visible again" for msg in tool_messages)
    assert "<think>" not in json.dumps(ai_calls[0]["messages"])


def test_empty_human_retry_merges_internal_note_into_existing_user_prompt(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    client = FakeChatClient(
        [
            make_plain_response("Plan: keep it casual."),
            make_tool_call_response("Hey there!", tool_call_id="wmth00001"),
            make_plain_response("Visible first reply", reasoning="complete first try"),
            make_tool_call_response("Nice to meet you.", tool_call_id="wmth00002"),
            make_plain_response("", reasoning="empty second try"),
            make_plain_response("Recovered second reply", reasoning="complete retry"),
            make_tool_call_response("Thanks for clarifying.", tool_call_id="wmth00003"),
        ]
    )

    gen = ConversationGenerator(
        client,
        HUMAN_PROFILES[0],
        target_tokens=150,
        max_turns=4,
    )

    with pytest.raises(RuntimeError, match="FakeChatClient: no more queued responses"):
        gen.generate()

    human_calls = [call for call in client.call_log if call.get("request_role") == "human"]
    assert len(human_calls) >= 4
    retry_messages = next(
        call["messages"]
        for call in human_calls
        if any(
            "Internal note for the human simulator only" in message.get("content", "")
            for message in call["messages"]
            if message.get("role") == "user"
        )
    )
    roles = [message["role"] for message in retry_messages]
    assert all(left != right for left, right in zip(roles, roles[1:]))
    assert any(
        "Internal note for the human simulator only" in message.get("content", "")
        for message in retry_messages
        if message.get("role") == "user"
    )
