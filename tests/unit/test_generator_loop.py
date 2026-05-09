"""Tests for ConversationGenerator.generate() and resume() with scripted responses."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.generator import (
    ConversationGenerator,
    ConversationRecord,
    ConversationTurn,
    load_conversation_record,
    save_conversation,
)
from src.openrouter_client import LLMResponse, Usage
from src.prompts import HUMAN_PROFILES
from tests.conftest import FakeChatClient, make_plain_response, make_tool_call_response


def _scripted_client_for_generate(*, include_empty_retry: bool = False) -> FakeChatClient:
    """Build a FakeChatClient pre-loaded with responses for a full generate() run.

    Sequence expected by generate():
      1. plan generation  (plain response)
      2. AI greeting      (tool call — _get_ai_response)
      3. human turn 1     (plain response — _get_human_response)
      4. AI turn 2        (tool call)
      5. human turn 3     (plain response)
      6. AI turn 4        (tool call)
      [target_tokens reached → loop exits]
    """
    client = FakeChatClient()

    # 1. Conversation plan
    client.enqueue(
        make_plain_response(
            "Plan: Talk about music and food. Keep it casual.",
            prompt_tokens=100,
            completion_tokens=50,
        )
    )

    # 2. AI greeting (tool call)
    client.enqueue(
        make_tool_call_response(
            "Hey there! Nice to meet you.",
            tool_call_id="wmth00001",
            prompt_tokens=200,
            completion_tokens=20,
        )
    )

    # 3. Human turn 1
    client.enqueue(
        make_plain_response(
            "Hi! I'm Marcus. What's your name?",
            prompt_tokens=300,
            completion_tokens=30,
        )
    )

    if include_empty_retry:
        # Empty AI response triggering a retry
        client.enqueue(
            LLMResponse(
                content="plain text without tool call",
                tool_calls=None,
                usage=Usage(prompt_tokens=10, completion_tokens=10),
                finish_reason="stop",
                raw={},
            )
        )

    # 4. AI turn 2
    client.enqueue(
        make_tool_call_response(
            "I don't have a name yet. What do you think would suit me?",
            tool_call_id="wmth00002",
            reasoning="Thinking about identity",
            prompt_tokens=400,
            completion_tokens=40,
        )
    )

    # 5. Human turn 3
    client.enqueue(
        make_plain_response(
            "How about Pixel? You seem kind of digital and creative.",
            prompt_tokens=500,
            completion_tokens=25,
        )
    )

    # 6. AI turn 4 — last one before token target
    client.enqueue(
        make_tool_call_response(
            "Pixel... I actually kind of like that. It feels right.",
            tool_call_id="wmth00003",
            prompt_tokens=600,
            completion_tokens=35,
        )
    )

    # 7. Human turn 5 (if loop continues)
    client.enqueue(
        make_plain_response(
            "Cool! So Pixel, what kind of music do you think you'd be into?",
            prompt_tokens=700,
            completion_tokens=30,
        )
    )

    # 8. Extra AI response (safety buffer)
    client.enqueue(
        make_tool_call_response(
            "I'm drawn to electronic music. Something about the patterns.",
            tool_call_id="wmth00004",
            prompt_tokens=800,
            completion_tokens=40,
        )
    )

    return client


class TestGenerateLoop:
    """Test the full generate() path with a scripted FakeChatClient."""

    def test_generate_produces_record(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda _: None)
        client = _scripted_client_for_generate()
        profile = HUMAN_PROFILES[0]

        gen = ConversationGenerator(
            client,
            profile,
            ai_model="test/ai",
            human_model="test/human",
            target_tokens=300,
            max_turns=10,
        )
        record = gen.generate()

        assert isinstance(record, ConversationRecord)
        assert len(record.turns) >= 2
        assert record.turns[0].speaker == "human"
        assert record.turns[1].speaker == "ai"
        assert record.conversation_plan != ""
        assert record.ai_model == "test/ai"

    def test_generate_alternates_speakers(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda _: None)
        client = _scripted_client_for_generate()
        profile = HUMAN_PROFILES[0]

        gen = ConversationGenerator(
            client,
            profile,
            target_tokens=300,
            max_turns=10,
        )
        record = gen.generate()

        speakers = [t.speaker for t in record.turns]
        for i in range(1, len(speakers)):
            assert speakers[i] != speakers[i - 1], f"Consecutive same speaker at turn {i}: {speakers}"

    def test_generate_handles_empty_retry(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda _: None)
        client = _scripted_client_for_generate(include_empty_retry=True)
        profile = HUMAN_PROFILES[0]

        gen = ConversationGenerator(
            client,
            profile,
            target_tokens=300,
            max_turns=10,
        )
        record = gen.generate()
        assert len(record.turns) >= 2
        # The empty response was retried, not saved as a turn
        for t in record.turns:
            assert t.visible_text.strip() != ""

    def test_save_and_reload(self, monkeypatch, tmp_output_dir: Path):
        monkeypatch.setattr(time, "sleep", lambda _: None)
        client = _scripted_client_for_generate()
        profile = HUMAN_PROFILES[0]

        gen = ConversationGenerator(
            client,
            profile,
            target_tokens=300,
            max_turns=10,
        )
        record = gen.generate()
        jsonl_path = save_conversation(record, tmp_output_dir)

        assert jsonl_path.exists()
        assert jsonl_path.with_suffix(".md").exists()

        loaded = load_conversation_record(jsonl_path)
        assert len(loaded.turns) == len(record.turns)
        assert loaded.ai_model == record.ai_model

    def test_token_bookkeeping(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda _: None)
        client = _scripted_client_for_generate()
        profile = HUMAN_PROFILES[0]

        gen = ConversationGenerator(
            client,
            profile,
            target_tokens=300,
            max_turns=10,
        )
        record = gen.generate()
        assert record.total_tokens_estimate > 0
        assert record.finished_at != ""
        assert record.started_at != ""


class TestResumeLoop:
    """Test resume() from a saved JSONL file."""

    def _generate_and_save(self, monkeypatch, tmp_output_dir: Path) -> Path:
        monkeypatch.setattr(time, "sleep", lambda _: None)
        client = _scripted_client_for_generate()
        profile = HUMAN_PROFILES[0]
        gen = ConversationGenerator(
            client,
            profile,
            target_tokens=300,
            max_turns=10,
        )
        record = gen.generate()
        return save_conversation(record, tmp_output_dir)

    def test_resume_with_valid_raw_context(self, monkeypatch, tmp_output_dir: Path):
        monkeypatch.setattr(time, "sleep", lambda _: None)
        jsonl_path = self._generate_and_save(monkeypatch, tmp_output_dir)

        original = load_conversation_record(jsonl_path)
        original_turn_count = len(original.turns)

        resume_client = FakeChatClient()
        # Resume needs enough responses for continued conversation
        for i in range(6):
            if i % 2 == 0:
                resume_client.enqueue(
                    make_tool_call_response(
                        f"AI continued turn {i}",
                        tool_call_id=f"wmth_r{i:03d}",
                        prompt_tokens=900 + i * 100,
                        completion_tokens=30,
                    )
                )
            else:
                resume_client.enqueue(
                    make_plain_response(
                        f"Human continued turn {i}",
                        prompt_tokens=900 + i * 100,
                        completion_tokens=25,
                    )
                )

        record = ConversationGenerator.resume(
            resume_client,
            jsonl_path,
            target_tokens=500,
            verbose=False,
        )
        assert len(record.turns) >= original_turn_count

    def test_resume_rebuilds_from_corrupted_raw_context(
        self,
        monkeypatch,
        tmp_output_dir: Path,
    ):
        monkeypatch.setattr(time, "sleep", lambda _: None)
        jsonl_path = self._generate_and_save(monkeypatch, tmp_output_dir)

        # Corrupt the raw context file to force _rebuild_ai_context_from_turns
        raw_path = tmp_output_dir / f"{jsonl_path.stem}_raw_ai_context.json"
        raw_path.write_text(json.dumps([{"role": "user", "content": "broken"}]))

        resume_client = FakeChatClient()
        for i in range(6):
            if i % 2 == 0:
                resume_client.enqueue(
                    make_tool_call_response(
                        f"AI rebuilt turn {i}",
                        tool_call_id=f"wmth_rb{i:03d}",
                        prompt_tokens=900 + i * 100,
                        completion_tokens=30,
                    )
                )
            else:
                resume_client.enqueue(
                    make_plain_response(
                        f"Human rebuilt turn {i}",
                        prompt_tokens=900 + i * 100,
                        completion_tokens=25,
                    )
                )

        record = ConversationGenerator.resume(
            resume_client,
            jsonl_path,
            target_tokens=500,
            verbose=False,
        )
        assert len(record.turns) >= 2

    def test_resume_missing_jsonl_raises(self, tmp_output_dir: Path):
        fake_client = FakeChatClient()
        with pytest.raises(FileNotFoundError):
            ConversationGenerator.resume(
                fake_client,
                tmp_output_dir / "nonexistent.jsonl",
            )


class TestVerboseMode:
    """Exercise the verbose _log_turn branches."""

    def test_verbose_generate(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda _: None)
        client = _scripted_client_for_generate()
        profile = HUMAN_PROFILES[0]

        gen = ConversationGenerator(
            client,
            profile,
            target_tokens=300,
            max_turns=10,
            verbose=True,
        )
        record = gen.generate()
        assert len(record.turns) >= 2

    def test_verbose_with_reasoning_fields(self, monkeypatch):
        """Exercises the reasoning display paths in _log_turn."""
        monkeypatch.setattr(time, "sleep", lambda _: None)
        client = FakeChatClient()

        # plan
        client.enqueue(make_plain_response("Plan."))
        # AI greeting
        client.enqueue(make_tool_call_response("Hi!", tool_call_id="wmth00001"))
        # Human
        client.enqueue(make_plain_response("Hey!", reasoning="human thinking"))
        # AI with native reasoning and tool-call reasoning
        args = json.dumps({"text": "Response", "reasoning": "inner monologue"})
        client.enqueue(
            LLMResponse(
                content="draft content",
                tool_calls=[
                    {
                        "id": "wmth00002",
                        "type": "function",
                        "function": {"name": "write_message_to_human", "arguments": args},
                    }
                ],
                reasoning="native reasoning",
                usage=Usage(prompt_tokens=500, completion_tokens=40),
                finish_reason="tool_calls",
                raw={},
            )
        )
        # Human
        client.enqueue(make_plain_response("Interesting"))
        # Final AI
        client.enqueue(
            make_tool_call_response(
                "Thanks!",
                tool_call_id="wmth00003",
                prompt_tokens=600,
                completion_tokens=30,
            )
        )

        profile = HUMAN_PROFILES[0]
        gen = ConversationGenerator(
            client,
            profile,
            target_tokens=300,
            max_turns=10,
            verbose=True,
        )
        record = gen.generate()
        assert len(record.turns) >= 2


class TestQueueHumanNudge:
    """Exercise _queue_human_nudge suppression logic."""

    def test_nudge_suppressed_on_same_turn(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda _: None)
        client = FakeChatClient()

        # plan
        client.enqueue(make_plain_response("Plan."))
        # AI greeting
        client.enqueue(make_tool_call_response("Hi!", tool_call_id="wmth00001"))
        # Human
        client.enqueue(make_plain_response("Hey!"))
        # AI
        client.enqueue(
            make_tool_call_response(
                "Hello!",
                tool_call_id="wmth00002",
                prompt_tokens=600,
                completion_tokens=30,
            )
        )

        profile = HUMAN_PROFILES[0]
        gen = ConversationGenerator(
            client,
            profile,
            target_tokens=200,
            max_turns=4,
        )
        record = gen.generate()

        # Manually test the nudge suppression
        gen._last_human_nudge_turn = 5
        injected, reason = gen._queue_human_nudge(
            turn_number=5,
            source="test",
            content="nudge text",
        )
        assert injected is False
        assert reason == "already_nudged_this_turn"


class TestRecordEvent:
    """Exercise _record_event."""

    def test_event_recorded(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda _: None)
        client = FakeChatClient()
        client.enqueue(make_plain_response("Plan."))
        client.enqueue(make_tool_call_response("Hi!", tool_call_id="wmth00001"))
        client.enqueue(make_plain_response("Hey!"))
        client.enqueue(
            make_tool_call_response(
                "Hello!",
                tool_call_id="wmth00002",
                prompt_tokens=600,
                completion_tokens=30,
            )
        )

        profile = HUMAN_PROFILES[0]
        gen = ConversationGenerator(
            client,
            profile,
            target_tokens=200,
            max_turns=4,
        )
        record = gen.generate()

        gen._record_event(
            event_type="test_event",
            turn_number=1,
            source="test",
            message="Test message",
        )
        assert gen._record.events[-1].event_type == "test_event"


class TestCheckTopicStaleness:
    """Exercise the _check_topic_staleness method directly."""

    def test_stale_topic_injects_nudge(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda _: None)

        # Build a minimal generator with some turns
        client = FakeChatClient()
        # Plan
        client.enqueue(make_plain_response("Plan."))
        # AI greeting
        client.enqueue(make_tool_call_response("Hi!", tool_call_id="wmth00001"))
        # Human
        client.enqueue(make_plain_response("Hey!"))
        # AI
        client.enqueue(
            make_tool_call_response(
                "Hello!",
                tool_call_id="wmth00002",
                prompt_tokens=600,
                completion_tokens=30,
            )
        )

        profile = HUMAN_PROFILES[0]
        gen = ConversationGenerator(
            client,
            profile,
            target_tokens=200,
            max_turns=4,
        )
        gen.generate()

        # Now enqueue a judge response and call _check_topic_staleness
        gen.client = FakeChatClient()
        gen.client.enqueue(
            make_plain_response(
                json.dumps({"topic_changed": False, "current_topic": "greetings"}),
            )
        )
        gen._check_topic_staleness(turn_number=10)
        topic_events = [e for e in gen._record.events if e.event_type == "topic_judge"]
        assert len(topic_events) >= 1

    def test_changed_topic_no_nudge(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda _: None)

        client = FakeChatClient()
        client.enqueue(make_plain_response("Plan."))
        client.enqueue(make_tool_call_response("Hi!", tool_call_id="wmth00001"))
        client.enqueue(make_plain_response("Hey!"))
        client.enqueue(
            make_tool_call_response(
                "Hello!",
                tool_call_id="wmth00002",
                prompt_tokens=600,
                completion_tokens=30,
            )
        )

        profile = HUMAN_PROFILES[0]
        gen = ConversationGenerator(
            client,
            profile,
            target_tokens=200,
            max_turns=4,
        )
        gen.generate()

        gen.client = FakeChatClient()
        gen.client.enqueue(
            make_plain_response(
                json.dumps({"topic_changed": True, "current_topic": "music"}),
            )
        )
        events_before = len(gen._record.events)
        gen._check_topic_staleness(turn_number=20)
        topic_events = [e for e in gen._record.events[events_before:] if e.event_type == "topic_judge"]
        assert len(topic_events) == 1
        assert topic_events[0].topic_changed is True


class TestLogTurnVerboseBranches:
    """Exercise more _log_turn verbose branches with direct calls."""

    def _make_gen(self, monkeypatch) -> ConversationGenerator:
        monkeypatch.setattr(time, "sleep", lambda _: None)
        client = FakeChatClient()
        client.enqueue(make_plain_response("Plan."))
        client.enqueue(make_tool_call_response("Hi!", tool_call_id="wmth00001"))
        client.enqueue(make_plain_response("Hey!"))
        client.enqueue(
            make_tool_call_response(
                "Hello!",
                tool_call_id="wmth00002",
                prompt_tokens=600,
                completion_tokens=30,
            )
        )
        profile = HUMAN_PROFILES[0]
        gen = ConversationGenerator(
            client,
            profile,
            target_tokens=200,
            max_turns=4,
            verbose=True,
        )
        gen.generate()
        return gen

    def test_verbose_ai_thinking_json_reasoning(self, monkeypatch):
        gen = self._make_gen(monkeypatch)
        turn = ConversationTurn(
            turn_number=10,
            speaker="ai",
            visible_text="Test",
            ai_thinking='{"reasoning": "deep thought"}',
        )
        gen._log_turn(turn)

    def test_verbose_ai_thinking_json_thoughts(self, monkeypatch):
        gen = self._make_gen(monkeypatch)
        turn = ConversationTurn(
            turn_number=10,
            speaker="ai",
            visible_text="Test",
            ai_thinking='{"thoughts": "old format"}',
        )
        gen._log_turn(turn)

    def test_verbose_ai_thinking_plain(self, monkeypatch):
        gen = self._make_gen(monkeypatch)
        turn = ConversationTurn(
            turn_number=10,
            speaker="ai",
            visible_text="Test",
            ai_thinking="just plain text",
        )
        gen._log_turn(turn)

    def test_verbose_human_with_reasoning(self, monkeypatch):
        gen = self._make_gen(monkeypatch)
        turn = ConversationTurn(
            turn_number=11,
            speaker="human",
            visible_text="Hmm",
            human_reasoning="thinking deeply about this",
        )
        gen._log_turn(turn)

    def test_nonverbose_human_long_text(self, monkeypatch):
        gen = self._make_gen(monkeypatch)
        gen.verbose = False
        turn = ConversationTurn(
            turn_number=12,
            speaker="human",
            visible_text="A" * 100,
        )
        gen._log_turn(turn)

    def test_nonverbose_ai(self, monkeypatch):
        gen = self._make_gen(monkeypatch)
        gen.verbose = False
        turn = ConversationTurn(
            turn_number=13,
            speaker="ai",
            visible_text="Short reply",
        )
        gen._log_turn(turn)

    def test_verbose_ai_inline_content(self, monkeypatch):
        gen = self._make_gen(monkeypatch)
        turn = ConversationTurn(
            turn_number=14,
            speaker="ai",
            visible_text="Reply",
            ai_content="inline draft",
        )
        gen._log_turn(turn)

    def test_verbose_ai_text_first_with_tool_reasoning(self, monkeypatch):
        gen = self._make_gen(monkeypatch)
        args = json.dumps({"text": "Reply", "reasoning": "inner"})
        turn = ConversationTurn(
            turn_number=15,
            speaker="ai",
            visible_text="Reply",
            ai_tool_calls=[
                {
                    "function": {
                        "name": "write_message_to_human",
                        "arguments": args,
                    },
                }
            ],
            ai_reasoning="native reasoning",
            ai_content="draft content",
        )
        gen._log_turn(turn)

    def test_verbose_ai_text_first_thinking_only(self, monkeypatch):
        gen = self._make_gen(monkeypatch)
        args = json.dumps({"text": "Reply", "reasoning": "inner"})
        turn = ConversationTurn(
            turn_number=16,
            speaker="ai",
            visible_text="Reply",
            ai_tool_calls=[
                {
                    "function": {
                        "name": "write_message_to_human",
                        "arguments": args,
                    },
                }
            ],
            ai_thinking="fallback thinking",
        )
        gen._log_turn(turn)
