"""Extended tests for ConversationGenerator: nudge logic, events, topic checking, verbose branches."""

from __future__ import annotations

import json
import time

from memcomp_bench.generator import ConversationGenerator, ConversationTurn
from memcomp_bench.prompts import HUMAN_PROFILES
from tests.conftest import FakeChatClient, make_plain_response, make_tool_call_response


class TestQueueHumanNudge:
    """Exercise _queue_human_nudge suppression logic."""

    def test_nudge_suppressed_on_same_turn(self, monkeypatch):
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
        gen.generate()

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
                json.dumps({"topic_changed": False, "current_topic": "greetings"}),
            )
        )
        gen._check_topic_staleness(turn_number=10)
        topic_events = [e for e in gen._record.events if e.event_type == "topic_judge"]
        assert len(topic_events) >= 1
        assert topic_events[-1].nudge_injected is True

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
        assert topic_events[0].nudge_injected is None


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

    def test_verbose_ai_thinking_json_reasoning(self, monkeypatch, capsys):
        gen = self._make_gen(monkeypatch)
        turn = ConversationTurn(
            turn_number=10,
            speaker="ai",
            visible_text="Test",
            ai_thinking='{"reasoning": "deep thought"}',
        )
        gen._log_turn(turn)
        captured = capsys.readouterr()
        assert "deep thought" in captured.out

    def test_verbose_ai_thinking_json_thoughts(self, monkeypatch, capsys):
        gen = self._make_gen(monkeypatch)
        turn = ConversationTurn(
            turn_number=10,
            speaker="ai",
            visible_text="Test",
            ai_thinking='{"thoughts": "old format"}',
        )
        gen._log_turn(turn)
        captured = capsys.readouterr()
        assert "old format" in captured.out
        assert '{"thoughts":' not in captured.out

    def test_verbose_ai_thinking_plain(self, monkeypatch, capsys):
        gen = self._make_gen(monkeypatch)
        turn = ConversationTurn(
            turn_number=10,
            speaker="ai",
            visible_text="Test",
            ai_thinking="just plain text",
        )
        gen._log_turn(turn)
        captured = capsys.readouterr()
        assert "just plain text" in captured.out

    def test_verbose_human_with_reasoning(self, monkeypatch, capsys):
        gen = self._make_gen(monkeypatch)
        turn = ConversationTurn(
            turn_number=11,
            speaker="human",
            visible_text="Hmm",
            human_reasoning="thinking deeply about this",
        )
        gen._log_turn(turn)
        captured = capsys.readouterr()
        assert "thinking deeply about this" in captured.out

    def test_nonverbose_human_long_text(self, monkeypatch, capsys):
        gen = self._make_gen(monkeypatch)
        gen.verbose = False
        turn = ConversationTurn(
            turn_number=12,
            speaker="human",
            visible_text="A" * 100,
        )
        gen._log_turn(turn)
        captured = capsys.readouterr()
        assert "A" * 80 in captured.out
        assert "…" in captured.out

    def test_nonverbose_ai(self, monkeypatch, capsys):
        gen = self._make_gen(monkeypatch)
        gen.verbose = False
        turn = ConversationTurn(
            turn_number=13,
            speaker="ai",
            visible_text="Short reply",
        )
        gen._log_turn(turn)
        captured = capsys.readouterr()
        assert "Short reply" in captured.out

    def test_verbose_ai_inline_content(self, monkeypatch, capsys):
        gen = self._make_gen(monkeypatch)
        turn = ConversationTurn(
            turn_number=14,
            speaker="ai",
            visible_text="Reply",
            ai_content="inline draft",
        )
        gen._log_turn(turn)
        captured = capsys.readouterr()
        assert "inline draft" in captured.out

    def test_verbose_ai_text_first_with_tool_reasoning(self, monkeypatch, capsys):
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
        captured = capsys.readouterr()
        assert "native reasoning" in captured.out
        assert "inner" in captured.out
        assert "draft content" in captured.out

    def test_verbose_ai_text_first_thinking_only(self, monkeypatch, capsys):
        gen = self._make_gen(monkeypatch)
        args = '{"text": "Reply", "reasoning": ""}'
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
        captured = capsys.readouterr()
        assert "fallback thinking" in captured.out
