"""Unit tests for interactive mode utilities: truncation, timestamps, sorting, display."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from rich.console import Console

from memcomp_bench import interactive as interactive_module
from memcomp_bench._interactive_display import (
    format_run_line,
    render_summary_header,
)
from memcomp_bench._interactive_prompts import (
    language_abbrev,
    relative_time,
    truncate_model_name,
)
from memcomp_bench.generator import ConversationRecord, ConversationTurn, save_conversation
from memcomp_bench.interactive import sort_summaries


def _console():
    stream = StringIO()
    return Console(file=stream, force_terminal=False, color_system=None, width=120), stream


def _make_record():
    tool_call = {
        "id": "wmth00001",
        "type": "function",
        "function": {"name": "write_message_to_human", "arguments": json.dumps({"text": "Hello"})},
    }
    record = ConversationRecord(
        id="20260101_000000",
        human_profile={"name": "Marcus", "backstory": "A tester."},
        ai_model="test/ai",
        human_model="test/human",
        language="english",
        ai_reasoning={"effort": "high"},
        human_reasoning={"effort": "minimal"},
        ai_rpm_limit=7,
        human_rpm_limit=5,
    )
    record.turns = [
        ConversationTurn(turn_number=1, speaker="human", visible_text="Hi"),
        ConversationTurn(
            turn_number=2,
            speaker="ai",
            visible_text="Hello",
            ai_tool_calls=[tool_call],
            ai_context_tokens=120,
        ),
    ]
    record.total_tokens_estimate = 120
    record.started_at = "2026-01-01T00:00:00Z"
    record.finished_at = "2026-01-01T00:01:00Z"
    record.ai_messages_raw = [
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": "[start]"},
        {"role": "assistant", "content": None, "tool_calls": [tool_call]},
        {"role": "tool", "content": "Hi", "tool_call_id": "wmth00001"},
    ]
    return record


def test_relative_time_formatting():
    assert relative_time("") == "-"
    assert relative_time("interrupted") == "-"
    assert relative_time("not-a-date") == "-"
    assert "ago" in relative_time("2020-01-01T00:00:00Z")


def test_truncate_model_name():
    assert truncate_model_name("openai/gpt-4.1", 30) == "openai/gpt-4.1"
    assert truncate_model_name("openai/gpt-4.1", 10) == "gpt-4.1"
    assert truncate_model_name("anthropic/claude-sonnet-4-20260514", 15) == "...t-4-20260514"
    assert truncate_model_name("anthropic/claude-sonnet-4", 20) == "claude-sonnet-4"
    assert truncate_model_name("-", 10) == "-"
    assert truncate_model_name("", 10) == "-"
    assert truncate_model_name("anthropic/very-long-model-name", 3) == "ame"


def test_language_abbrev():
    assert language_abbrev("english") == "en"
    assert language_abbrev("russian") == "ru"
    assert language_abbrev("japanese") == "ja"
    assert language_abbrev("hebrew") == "he"
    assert language_abbrev(None) == "--"
    assert language_abbrev("") == "--"


def test_sort_summaries(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    rec1 = _make_record()
    rec1.total_tokens_estimate = 500
    rec1.started_at = "2026-01-02T00:00:00Z"
    save_conversation(rec1, output_dir)

    rec2 = _make_record()
    rec2.id = "20260103_000000"
    rec2.total_tokens_estimate = 200
    rec2.started_at = "2026-01-03T00:00:00Z"
    save_conversation(rec2, output_dir)

    summaries = interactive_module.scan_saved_conversations(output_dir)
    assert len(summaries) == 2

    by_most_tokens = sort_summaries(summaries, "Most tokens")
    assert by_most_tokens[0].total_tokens_estimate >= by_most_tokens[1].total_tokens_estimate

    by_fewest = sort_summaries(summaries, "Fewest tokens")
    assert by_fewest[0].total_tokens_estimate <= by_fewest[1].total_tokens_estimate

    by_oldest = sort_summaries(summaries, "Oldest first")
    assert by_oldest[0].started_at <= by_oldest[1].started_at


def test_format_run_line_adapts_to_width(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    save_conversation(_make_record(), output_dir)
    summaries = interactive_module.scan_saved_conversations(output_dir)
    summary = summaries[0]

    wide = format_run_line(1, 1, summary, width=140)
    medium = format_run_line(1, 1, summary, width=100)
    narrow = format_run_line(1, 1, summary, width=85)
    minimal = format_run_line(1, 1, summary, width=70)

    assert "ai" in wide.lower() or "test" in wide.lower()
    assert "Marcus" in wide
    assert "Marcus" in medium
    assert "Marcus" in narrow
    assert "Marcus" in minimal


def test_render_summary_header(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    save_conversation(_make_record(), output_dir)
    summaries = interactive_module.scan_saved_conversations(output_dir)

    console, stream = _console()
    render_summary_header(console, summaries, "Newest first")
    output = stream.getvalue()
    assert "1" in output
    assert "saved runs" in output
    assert "Newest first" in output
