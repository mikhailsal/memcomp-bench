"""Unit tests for interactive mode utilities: truncation, timestamps, sorting, display, prompts."""

from __future__ import annotations

import json
from argparse import Namespace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

from memcomp_bench import _interactive_prompts as prompt_module
from memcomp_bench import interactive as interactive_module
from memcomp_bench._interactive_display import (
    format_run_line,
    render_run_detail_lines,
    render_summary_header,
    render_summary_title,
)
from memcomp_bench._interactive_prompts import (
    language_abbrev,
    relative_time,
    truncate_model_name,
)
from memcomp_bench.generator import ConversationRecord, ConversationTurn, save_conversation
from memcomp_bench.interactive import sort_summaries
from memcomp_bench.model_registry import MISSING


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


def test_render_summary_title(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    save_conversation(_make_record(), output_dir)
    summaries = interactive_module.scan_saved_conversations(output_dir)

    title = render_summary_title(summaries, "Newest first")
    assert "1 runs" in title
    assert "Newest first" in title
    assert "resumable" in title


def test_render_run_detail_lines(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    save_conversation(_make_record(), output_dir)
    summaries = interactive_module.scan_saved_conversations(output_dir)

    lines = render_run_detail_lines(summaries[0])
    text = "\n".join(lines)
    assert "--- General ---" in text
    assert "--- Conversation ---" in text
    assert "--- AI Model ---" in text
    assert "--- Human Simulator ---" in text
    assert "Marcus" in text
    assert "test/ai" in text
    assert "120" in text
    assert "english" in text


def test_render_run_detail_lines_shows_resume_defaults_diff(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    save_conversation(_make_record(), output_dir)
    summaries = interactive_module.scan_saved_conversations(output_dir)

    summary = summaries[0]
    summary.saved_defaults = dict(summary.saved_defaults)
    summary.saved_defaults["ai_model"] = "different/ai"

    lines = render_run_detail_lines(summary)
    text = "\n".join(lines)
    assert "Resume Defaults" in text
    assert "different/ai" in text


# ---------------------------------------------------------------------------
# Prompter implementation tests
# ---------------------------------------------------------------------------


class _ScriptedPrompter:
    def __init__(self, answers: list[str], confirms: list[bool] | None = None) -> None:
        self.answers = list(answers)
        self.confirms = list(confirms or [])

    def ask(self, prompt: str, *, default: str | None = None) -> str:
        if not self.answers:
            return "" if default is None else default
        answer = self.answers.pop(0)
        if answer == "" and default is not None:
            return default
        return answer

    def confirm(self, prompt: str, *, default: bool = False) -> bool:
        if not self.confirms:
            return default
        return self.confirms.pop(0)

    def select(self, prompt: str, choices: list[str], *, default: str | None = None) -> str:
        if not self.answers:
            return default if default is not None else choices[0]
        answer = self.answers.pop(0)
        if answer == "":
            return default if default is not None else choices[0]
        return answer


def _preset(**overrides):
    values = {
        "provider": MISSING,
        "reasoning": MISSING,
        "temperature": MISSING,
        "max_tokens": MISSING,
        "rpm_limit": MISSING,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_rich_prompter_uses_defaults_and_retries_confirm(monkeypatch):
    console, stream = _console()
    responses = iter(["", "maybe", "yes", "", "2"])
    monkeypatch.setattr(console, "input", lambda prompt: next(responses))

    prompter = prompt_module.RichPrompter(console)

    assert prompter.ask("Question", default="fallback") == "fallback"
    assert prompter.confirm("Continue", default=False) is True
    assert "Please answer y or n" in stream.getvalue()
    assert prompter.select("Pick", ["a", "b", "c"], default="a") == "a"
    assert prompter.select("Pick", ["x", "y", "z"]) == "y"


def test_prompt_generate_args_uses_resolved_defaults(monkeypatch):
    monkeypatch.setattr(prompt_module, "default_model_for", lambda role: f"default/{role}")
    monkeypatch.setattr(
        prompt_module,
        "resolve_model_preset",
        lambda model, role: _preset(provider="minimax" if role == "ai" else MISSING, temperature=1.2, max_tokens=512),
    )

    console, _ = _console()
    prompter = _ScriptedPrompter(
        ["", "", "honest", "", "", "", "", "", "", "", "", "", ""],
        confirms=[True],
    )

    args = prompt_module.prompt_generate_args(console, prompter)

    assert args == Namespace(
        profile="0",
        target_tokens=70_000,
        language="english",
        companion_mode="honest",
        verbose=True,
        ai_model="default/ai",
        human_model="default/human",
        ai_provider="minimax",
        human_provider=None,
        ai_temperature=1.2,
        human_temperature=1.2,
        ai_max_tokens=512,
        human_max_tokens=512,
        ai_rpm_limit=None,
        human_rpm_limit=None,
    )


def test_prompt_resume_overrides_parses_special_values_and_retries():
    console, stream = _console()
    prompter = _ScriptedPrompter(
        [
            "",
            "override/ai",
            "",
            "none",
            "",
            "oops",
            "1.25",
            "",
            "bad",
            "4096",
            "",
            "0",
            "12",
            "",
        ]
    )

    overrides = prompt_module.prompt_resume_overrides(
        console,
        prompter,
        {
            "language": "english",
            "ai_model": "test/ai",
            "human_model": "test/human",
            "ai_provider": {"only": ["minimax"], "allow_fallbacks": False},
            "human_provider": None,
            "ai_temperature": 1.1,
            "human_temperature": 0.9,
            "ai_max_tokens": 2048,
            "human_max_tokens": 800,
            "ai_rpm_limit": 7,
            "human_rpm_limit": 5,
        },
    )

    assert overrides["language"] is None
    assert overrides["ai_model"] == "override/ai"
    assert overrides["ai_provider"] == ""
    assert overrides["ai_temperature"] == 1.25
    assert overrides["ai_max_tokens"] == 4096
    assert overrides["ai_rpm_limit"] == 12
    assert overrides["human_rpm_limit"] is None
    output = stream.getvalue()
    assert "Enter a number or leave it blank" in output
    assert "Enter an integer value" in output
    assert "Enter a value greater than zero" in output
