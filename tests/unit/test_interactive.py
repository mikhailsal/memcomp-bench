"""Unit tests for the interactive benchmark CLI helpers."""

from __future__ import annotations

import json
from argparse import Namespace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

from memcomp_bench import _interactive_prompts as prompt_module
from memcomp_bench import interactive as interactive_module
from memcomp_bench._interactive_display import format_run_line
from memcomp_bench.generator import ConversationRecord, ConversationTurn, save_conversation
from memcomp_bench.interactive import (
    MODE_NEW,
    MODE_QUIT,
    MODE_RESUME,
    MODE_VIEW,
)
from memcomp_bench.model_registry import MISSING


class ScriptedPrompter:
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


def _console() -> tuple[Console, StringIO]:
    stream = StringIO()
    return Console(file=stream, force_terminal=False, color_system=None, width=120), stream


def _make_record(*, resume_defaults: dict | None = None) -> ConversationRecord:
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
        resume_defaults=resume_defaults,
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


# ---------------------------------------------------------------------------
# Tests: Prompter implementations
# ---------------------------------------------------------------------------


def test_rich_prompter_uses_defaults_and_retries_confirm(monkeypatch):
    console, stream = _console()
    responses = iter(["", "maybe", "yes", "", "2"])
    monkeypatch.setattr(console, "input", lambda prompt: next(responses))

    prompter = prompt_module.RichPrompter(console)

    assert prompter.ask("Question", default="fallback") == "fallback"
    assert prompter.confirm("Continue", default=False) is True
    assert "Please answer y or n" in stream.getvalue()
    assert prompter.select("Pick", ["a", "b", "c"], default="a") == "a"  # blank → default
    assert prompter.select("Pick", ["x", "y", "z"]) == "y"  # "2" → index 2 → "y"


def test_prompt_generate_args_uses_resolved_defaults(monkeypatch):
    monkeypatch.setattr(prompt_module, "default_model_for", lambda role: f"default/{role}")
    monkeypatch.setattr(
        prompt_module,
        "resolve_model_preset",
        lambda model, role: _preset(provider="minimax" if role == "ai" else MISSING, temperature=1.2, max_tokens=512),
    )

    console, _ = _console()
    prompter = ScriptedPrompter(
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
    prompter = ScriptedPrompter(
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


def test_scan_saved_conversations_reads_effective_and_saved_defaults(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    resume_defaults = {
        "language": "english",
        "ai_model": "saved/ai",
        "human_model": "saved/human",
        "ai_provider": None,
        "human_provider": None,
        "ai_reasoning": {"effort": "high"},
        "human_reasoning": {"effort": "minimal"},
        "ai_temperature": 1.1,
        "human_temperature": 0.9,
        "ai_max_tokens": 2048,
        "human_max_tokens": 800,
        "ai_rpm_limit": None,
        "human_rpm_limit": None,
    }
    save_conversation(_make_record(resume_defaults=resume_defaults), output_dir)
    (output_dir / "conv_broken.jsonl").write_text("not json\n")

    summaries = interactive_module.scan_saved_conversations(output_dir)

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.profile_name == "Marcus"
    assert summary.resumable is True
    assert summary.effective_config["ai_model"] == "test/ai"
    assert summary.saved_defaults["ai_model"] == "saved/ai"
    assert summary.saved_defaults["ai_reasoning"] == {"effort": "high"}


def test_run_interactive_resume_flow_calls_handler_with_overrides(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("memcomp_bench._interactive_prompts.shutil.get_terminal_size", lambda: (120, 40))
    monkeypatch.setattr("memcomp_bench._interactive_display.terminal_width", lambda: 120)
    monkeypatch.setattr("memcomp_bench.interactive.terminal_width", lambda: 120)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    save_conversation(_make_record(), output_dir)
    summaries = interactive_module.scan_saved_conversations(output_dir)

    # Build a run choice line matching the format_run_line output
    run_line = format_run_line(1, 1, summaries[0], width=120)

    called = {}
    console, stream = _console()
    prompter = ScriptedPrompter(
        [
            MODE_RESUME,  # select: main action
            "Newest first",  # select: sort order
            run_line,  # select: pick the run
            "Edit defaults before continuing",  # select: resume mode
            "",  # language override
            "override/ai",  # ai_model override
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "100",  # too-low target
            "500",  # valid target
        ],
        confirms=[False],
    )

    interactive_module.run_interactive(
        lambda args: called.setdefault("generate", args),
        lambda args: called.setdefault("resume", args),
        output_dir=output_dir,
        console=console,
        prompter=prompter,
    )

    args = called["resume"]
    assert args.ai_model == "override/ai"
    assert args.persist_resume_defaults is False
    assert args.target_tokens == 500
    output = stream.getvalue()
    assert "Target must be greater than 120" in output


def test_run_interactive_generate_flow_calls_handler(tmp_path: Path):
    called = {}
    console, stream = _console()
    prompter = ScriptedPrompter(
        [
            MODE_NEW,  # select: main action (no saved runs)
            "2  James",  # select: profile
            "900",  # target tokens
            "spanish",  # language
            "honest",  # select: companion_mode
            "custom/ai",
            "custom/human",
            "openai",
            "",
            "0.8",
            "1.0",
            "1024",
            "512",
            "20",
            "",
        ],
        confirms=[True],
    )

    interactive_module.run_interactive(
        lambda args: called.setdefault("generate", args),
        lambda args: called.setdefault("resume", args),
        output_dir=tmp_path / "missing-output",
        console=console,
        prompter=prompter,
    )

    args = called["generate"]
    assert args.profile == "2"
    assert args.language == "spanish"
    assert args.verbose is True
    assert args.ai_provider == "openai"
    assert args.human_provider is None


def test_run_interactive_non_resumable_run_does_not_call_handler(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("memcomp_bench.interactive.terminal_width", lambda: 120)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    jsonl_path = save_conversation(_make_record(), output_dir)
    raw_path = output_dir / f"{jsonl_path.stem}_raw_ai_context.json"
    raw_path.unlink()
    summaries = interactive_module.scan_saved_conversations(output_dir)
    run_line = format_run_line(1, 1, summaries[0], width=120)

    called = {}
    console, stream = _console()
    prompter = ScriptedPrompter(
        [
            MODE_RESUME,  # main action
            "Newest first",  # sort order
            run_line,  # pick run
        ]
    )

    interactive_module.run_interactive(
        lambda args: called.setdefault("generate", args),
        lambda args: called.setdefault("resume", args),
        output_dir=output_dir,
        console=console,
        prompter=prompter,
    )

    assert called == {}
    assert "Cannot resume" in stream.getvalue()


def test_default_target_tokens_and_format_value_helpers():
    assert prompt_module.default_target_tokens(120) == 70_000
    assert prompt_module.default_target_tokens(80_000) == 85_000
    assert prompt_module.format_value(None) == "-"
    assert prompt_module.format_value("") == "auto"
    assert prompt_module.format_value({"a": 1}) == '{"a": 1}'


def test_view_flow_shows_detail_and_returns(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("memcomp_bench.interactive.terminal_width", lambda: 120)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    save_conversation(_make_record(), output_dir)
    summaries = interactive_module.scan_saved_conversations(output_dir)
    run_line = format_run_line(1, 1, summaries[0], width=120)

    console, stream = _console()
    prompter = ScriptedPrompter(
        [
            MODE_VIEW,  # main action
            "Newest first",  # sort order
            run_line,  # pick run
            "",  # press Enter to return
            "\u2190 Back",  # back from list
        ]
    )

    interactive_module.run_interactive(
        lambda args: None,
        lambda args: None,
        output_dir=output_dir,
        console=console,
        prompter=prompter,
    )

    output = stream.getvalue()
    assert "Run Details" in output
    assert "Marcus" in output
    assert "AI Model" in output
    assert "Human Simulator" in output


def test_main_actions_with_no_saved_runs(tmp_path: Path):
    console, stream = _console()
    prompter = ScriptedPrompter([MODE_QUIT])

    interactive_module.run_interactive(
        lambda args: None,
        lambda args: None,
        output_dir=tmp_path / "empty",
        console=console,
        prompter=prompter,
    )
