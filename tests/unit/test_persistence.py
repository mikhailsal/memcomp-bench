"""Tests for save/load/reformat cycle in memcomp_bench.generator."""

from __future__ import annotations

import json
from pathlib import Path

from memcomp_bench.generator import (
    ConversationEvent,
    ConversationRecord,
    ConversationTurn,
    _write_conversation_markdown,
    load_conversation_record,
    reformat_markdown,
    save_conversation,
)

_TOOL_CALL_TC01 = {
    "id": "tc_01",
    "type": "function",
    "function": {"name": "write_message_to_human", "arguments": json.dumps({"text": "Hello!"})},
}


def _default_turns() -> list[ConversationTurn]:
    return [
        ConversationTurn(
            turn_number=1,
            speaker="human",
            visible_text="Hey there",
            token_estimate=5,
            cost_usd=0.0,
            timestamp="2026-01-01T00:00:00Z",
        ),
        ConversationTurn(
            turn_number=2,
            speaker="ai",
            visible_text="Hello!",
            ai_thinking='{"thoughts": "greeting"}',
            ai_tool_calls=[_TOOL_CALL_TC01],
            token_estimate=8,
            cost_usd=0.001,
            timestamp="2026-01-01T00:00:01Z",
        ),
    ]


def _default_raw_ai_messages() -> list[dict]:
    return [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "[start]"},
        {"role": "assistant", "content": None, "tool_calls": [_TOOL_CALL_TC01]},
        {"role": "tool", "content": "Hey there", "tool_call_id": "tc_01"},
    ]


def _make_record(
    *,
    turns: list[ConversationTurn] | None = None,
    events: list[ConversationEvent] | None = None,
    ai_messages_raw: list | None = None,
) -> ConversationRecord:
    """Build a minimal but valid ConversationRecord for testing."""
    used_turns = turns or _default_turns()
    record = ConversationRecord(
        id="20260101_000000",
        human_profile={"name": "TestUser", "backstory": "A tester."},
        ai_model="test/model-a",
        human_model="test/model-b",
        seed_words=["ocean", "ember"],
        conversation_plan="Talk about stuff",
        language="english",
        companion_mode="honest",
        ai_provider={"only": ["ai-prov"], "allow_fallbacks": False},
        ai_reasoning={"effort": "minimal"},
        ai_rpm_limit=12,
        human_provider={"only": ["human-prov"], "allow_fallbacks": False},
        human_reasoning={"effort": "low"},
        human_rpm_limit=8,
    )
    record.turns = used_turns
    record.events = events or []
    record.total_tokens_estimate = sum(t.token_estimate for t in used_turns)
    record.total_cost_usd = sum(t.cost_usd for t in used_turns)
    record.started_at = "2026-01-01T00:00:00Z"
    record.finished_at = "2026-01-01T00:01:00Z"
    record.ai_messages_raw = ai_messages_raw or _default_raw_ai_messages()
    return record


# ---------------------------------------------------------------------------
# save_conversation
# ---------------------------------------------------------------------------


class TestSaveConversation:
    def test_creates_three_files(self, tmp_output_dir: Path):
        record = _make_record()
        jsonl_path = save_conversation(record, tmp_output_dir)
        assert jsonl_path.exists()
        assert jsonl_path.with_suffix(".md").exists()
        raw_ctx = tmp_output_dir / f"{jsonl_path.stem}_raw_ai_context.json"
        assert raw_ctx.exists()

    def test_jsonl_first_line_is_metadata(self, tmp_output_dir: Path):
        record = _make_record()
        jsonl_path = save_conversation(record, tmp_output_dir)
        with open(jsonl_path) as f:
            meta = json.loads(f.readline())
        assert meta["type"] == "metadata"
        assert meta["ai_model"] == "test/model-a"
        assert meta["conversation_id"] == "20260101_000000"
        assert meta["ai_rpm_limit"] == 12
        assert meta["human_rpm_limit"] == 8
        assert meta["human_provider"] == {"only": ["human-prov"], "allow_fallbacks": False}

    def test_jsonl_contains_all_turns(self, tmp_output_dir: Path):
        record = _make_record()
        jsonl_path = save_conversation(record, tmp_output_dir)
        with open(jsonl_path) as f:
            lines = [json.loads(l) for l in f]
        turn_lines = [l for l in lines if l.get("type") == "turn"]
        assert len(turn_lines) == 2

    def test_events_persisted(self, tmp_output_dir: Path):
        event = ConversationEvent(
            event_type="topic_judge",
            turn_number=2,
            source="topic_judge",
            current_topic="greetings",
            topic_changed=False,
            nudge_injected=True,
        )
        record = _make_record(events=[event])
        jsonl_path = save_conversation(record, tmp_output_dir)
        with open(jsonl_path) as f:
            lines = [json.loads(l) for l in f]
        event_lines = [l for l in lines if l.get("type") == "event"]
        assert len(event_lines) == 1
        assert event_lines[0]["current_topic"] == "greetings"

    def test_raw_context_valid_json(self, tmp_output_dir: Path):
        record = _make_record()
        jsonl_path = save_conversation(record, tmp_output_dir)
        raw_path = tmp_output_dir / f"{jsonl_path.stem}_raw_ai_context.json"
        with open(raw_path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert data[0]["role"] == "system"


# ---------------------------------------------------------------------------
# load_conversation_record
# ---------------------------------------------------------------------------


class TestLoadConversationRecord:
    def test_roundtrip(self, tmp_output_dir: Path):
        original = _make_record()
        jsonl_path = save_conversation(original, tmp_output_dir)
        loaded = load_conversation_record(jsonl_path)
        assert loaded.id == original.id
        assert loaded.ai_model == original.ai_model
        assert loaded.language == original.language
        assert loaded.ai_rpm_limit == 12
        assert loaded.human_rpm_limit == 8
        assert loaded.human_provider == {"only": ["human-prov"], "allow_fallbacks": False}
        assert len(loaded.turns) == len(original.turns)
        assert loaded.turns[0].speaker == "human"
        assert loaded.turns[1].speaker == "ai"

    def test_turns_preserve_tool_calls(self, tmp_output_dir: Path):
        record = _make_record()
        jsonl_path = save_conversation(record, tmp_output_dir)
        loaded = load_conversation_record(jsonl_path)
        assert loaded.turns[1].ai_tool_calls is not None
        assert loaded.turns[1].ai_tool_calls[0]["function"]["name"] == "write_message_to_human"

    def test_events_loaded(self, tmp_output_dir: Path):
        event = ConversationEvent(
            event_type="human_nudge",
            turn_number=4,
            source="b3_refresh",
            nudge_injected=True,
            message="Time to switch topics.",
        )
        record = _make_record(events=[event])
        jsonl_path = save_conversation(record, tmp_output_dir)
        loaded = load_conversation_record(jsonl_path)
        assert len(loaded.events) == 1
        assert loaded.events[0].event_type == "human_nudge"


# ---------------------------------------------------------------------------
# reformat_markdown
# ---------------------------------------------------------------------------


class TestReformatMarkdown:
    def test_regenerates_md_only(self, tmp_output_dir: Path):
        record = _make_record()
        jsonl_path = save_conversation(record, tmp_output_dir)
        md_path = jsonl_path.with_suffix(".md")

        original_md = md_path.read_text()
        md_path.write_text("CORRUPTED")
        reformat_markdown(jsonl_path)
        assert md_path.read_text() != "CORRUPTED"
        assert md_path.read_text() == original_md


# ---------------------------------------------------------------------------
# _write_conversation_markdown — branch coverage
# ---------------------------------------------------------------------------


class TestWriteConversationMarkdown:
    def _render(self, record: ConversationRecord) -> str:
        from io import StringIO

        f = StringIO()
        _write_conversation_markdown(f, record)
        return f.getvalue()

    def test_basic_header(self):
        record = _make_record()
        md = self._render(record)
        assert "# Conversation: TestUser & AI" in md
        assert "test/model-a" in md

    def test_ai_thinking_rendered(self):
        turns = [
            ConversationTurn(
                turn_number=1,
                speaker="human",
                visible_text="Hi",
            ),
            ConversationTurn(
                turn_number=2,
                speaker="ai",
                visible_text="Hello!",
                ai_thinking='{"thoughts": "greeting"}',
            ),
        ]
        record = _make_record(turns=turns)
        md = self._render(record)
        assert "Thinking" in md or "thinking" in md

    def test_native_reasoning_rendered(self):
        turns = [
            ConversationTurn(
                turn_number=1,
                speaker="ai",
                visible_text="Hi",
                ai_reasoning="Native reasoning text",
            ),
        ]
        record = _make_record(turns=turns)
        md = self._render(record)
        assert "Native reasoning" in md

    def test_text_first_branch(self):
        args = json.dumps({"text": "Hi", "reasoning": "after"})
        turns = [
            ConversationTurn(
                turn_number=1,
                speaker="ai",
                visible_text="Hi",
                ai_tool_calls=[
                    {
                        "function": {
                            "name": "write_message_to_human",
                            "arguments": args,
                        },
                    }
                ],
            ),
        ]
        record = _make_record(turns=turns)
        md = self._render(record)
        assert "Hi" in md

    def test_events_section(self):
        event = ConversationEvent(
            event_type="topic_judge",
            turn_number=2,
            source="topic_judge",
            current_topic="music",
            topic_changed=True,
        )
        record = _make_record(events=[event])
        md = self._render(record)
        assert "System Events" in md
        assert "music" in md

    def test_conversation_plan_section(self):
        record = _make_record()
        record.conversation_plan = "Plan: talk about cats"
        md = self._render(record)
        assert "Conversation Plan" in md
        assert "cats" in md

    def test_human_reasoning_rendered(self):
        turns = [
            ConversationTurn(
                turn_number=1,
                speaker="human",
                visible_text="Hey",
                human_reasoning="thinking about what to say",
            ),
        ]
        record = _make_record(turns=turns)
        md = self._render(record)
        assert "thinking about what to say" in md

    def test_context_tokens_displayed(self):
        turns = [
            ConversationTurn(
                turn_number=1,
                speaker="human",
                visible_text="Hey",
                ai_context_tokens=100,
                human_context_tokens=80,
            ),
        ]
        record = _make_record(turns=turns)
        md = self._render(record)
        assert "100" in md
        assert "80" in md

    def test_no_plan_no_section(self):
        record = _make_record()
        record.conversation_plan = ""
        md = self._render(record)
        assert "Conversation Plan" not in md

    def test_no_events_no_section(self):
        record = _make_record(events=[])
        md = self._render(record)
        assert "System Events" not in md

    def test_event_with_message_note(self):
        event = ConversationEvent(
            event_type="human_nudge",
            turn_number=3,
            source="b3_refresh",
            nudge_injected=True,
            message="Life event happened.",
        )
        record = _make_record(events=[event])
        md = self._render(record)
        assert "Life event happened." in md

    def test_event_with_suppression_reason(self):
        event = ConversationEvent(
            event_type="human_nudge",
            turn_number=4,
            source="topic_judge",
            nudge_injected=False,
            suppression_reason="already_nudged_this_turn",
        )
        record = _make_record(events=[event])
        md = self._render(record)
        assert "already_nudged_this_turn" in md


class TestSaveConversationFallbackRebuild:
    """Test the fallback path when ai_messages_raw is not restorable."""

    def test_rebuild_from_turns_on_empty_raw(self, tmp_output_dir: Path):
        record = _make_record(ai_messages_raw=[])
        jsonl_path = save_conversation(record, tmp_output_dir)
        raw_path = tmp_output_dir / f"{jsonl_path.stem}_raw_ai_context.json"
        with open(raw_path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) >= 2
        assert data[0]["role"] == "system"
