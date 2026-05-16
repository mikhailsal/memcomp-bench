"""Unit tests for the interactive benchmark CLI helpers."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from rich.console import Console

from memcomp_bench import _interactive_prompts as prompt_module
from memcomp_bench import interactive as interactive_module
from memcomp_bench._interactive_display import format_run_line
from memcomp_bench._interactive_prompts import CANCEL, SORT_ACTION
from memcomp_bench.generator import ConversationRecord, ConversationTurn, save_conversation
from memcomp_bench.interactive import (
    DETAIL_BACK,
    MODE_NEW,
    MODE_QUIT,
    MODE_RESUME,
    MODE_VIEW,
)


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

    run_line = format_run_line(1, 1, summaries[0], width=120)

    called = {}
    console, stream = _console()
    prompter = ScriptedPrompter(
        [
            MODE_RESUME,  # main action
            run_line,  # pick the run
            DETAIL_BACK,  # detail view → back
            "Edit defaults before continuing",  # resume mode
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
    defaults = prompt_module._default_generate_values()
    called = {}
    console, stream = _console()
    prompter = ScriptedPrompter(
        [
            MODE_NEW,  # main action (no saved runs)
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
    assert args.human_provider == defaults["human_provider"]


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
            run_line,  # pick run
            CANCEL,  # Esc from run picker after non-resumable error
            CANCEL,  # Esc from main menu → exit
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
    assert prompt_module.default_target_tokens(120) == 5_000
    assert prompt_module.default_target_tokens(80_000) == 85_000
    assert prompt_module.default_target_tokens(21_463) == 25_000
    assert prompt_module.default_target_tokens(22_500) == 30_000
    assert prompt_module.format_value(None) == "-"
    assert prompt_module.format_value("") == "auto"
    assert prompt_module.format_value({"a": 1}) == '{"a": 1}'


def test_prompt_target_tokens_uses_rounded_default(monkeypatch):
    monkeypatch.setattr("memcomp_bench.interactive.default_target_tokens", lambda current_tokens: 25_000)
    console, _ = _console()

    class DefaultOnlyPrompter:
        def __init__(self) -> None:
            self.default: str | None = None

        def ask(self, prompt: str, *, default: str | None = None) -> str:
            self.default = default
            return default or ""

    prompter = DefaultOnlyPrompter()

    target_tokens = interactive_module._prompt_target_tokens(console, prompter, 21_463)

    assert prompter.default == "25000"
    assert target_tokens == 25_000


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
            run_line,  # pick run
            DETAIL_BACK,  # detail view → back to list
            CANCEL,  # Esc from run list → back to main
            CANCEL,  # Esc from main menu → exit
        ]
    )

    interactive_module.run_interactive(
        lambda args: None,
        lambda args: None,
        output_dir=output_dir,
        console=console,
        prompter=prompter,
    )


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


def test_cancel_from_main_menu_exits(tmp_path: Path):
    """CANCEL sentinel from main menu should exit (Esc at top = exit)."""
    called = {}
    console, _ = _console()
    prompter = ScriptedPrompter([CANCEL])

    interactive_module.run_interactive(
        lambda args: called.setdefault("generate", args),
        lambda args: called.setdefault("resume", args),
        output_dir=tmp_path / "empty",
        console=console,
        prompter=prompter,
    )

    assert called == {}


def test_cancel_from_generate_profile_picker_returns_to_main(tmp_path: Path):
    """CANCEL from the profile picker should return to main menu, not exit."""
    called = {}
    console, _ = _console()
    prompter = ScriptedPrompter(
        [
            MODE_NEW,  # main action
            CANCEL,  # Esc from profile picker → back to main
            CANCEL,  # Esc from main menu → exit
        ]
    )

    interactive_module.run_interactive(
        lambda args: called.setdefault("generate", args),
        lambda args: called.setdefault("resume", args),
        output_dir=tmp_path / "empty",
        console=console,
        prompter=prompter,
    )

    assert called == {}


def test_cancel_from_run_picker_returns_to_main(tmp_path: Path, monkeypatch):
    """CANCEL from the run picker should return to main menu, not exit."""
    monkeypatch.setattr("memcomp_bench.interactive.terminal_width", lambda: 120)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    save_conversation(_make_record(), output_dir)

    called = {}
    console, _ = _console()
    prompter = ScriptedPrompter(
        [
            MODE_RESUME,  # main action
            CANCEL,  # Esc from run picker → back to main
            CANCEL,  # Esc from main menu → exit
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


def test_inline_sort_in_view_flow(tmp_path: Path, monkeypatch):
    """SORT_ACTION from the run picker triggers sort and re-displays."""
    monkeypatch.setattr("memcomp_bench.interactive.terminal_width", lambda: 120)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    save_conversation(_make_record(), output_dir)
    summaries = interactive_module.scan_saved_conversations(output_dir)
    run_line = format_run_line(1, 1, summaries[0], width=120)

    console, stream = _console()
    prompter = ScriptedPrompter(
        [
            MODE_VIEW,
            SORT_ACTION,  # 's' key → sort
            "[3] Most tokens",  # pick new sort order
            run_line,  # pick run after re-sort
            DETAIL_BACK,  # detail view → back to list
            CANCEL,  # Esc from run list → back to main
            CANCEL,  # Esc from main menu → exit
        ]
    )

    interactive_module.run_interactive(
        lambda args: None,
        lambda args: None,
        output_dir=output_dir,
        console=console,
        prompter=prompter,
    )


def test_main_loop_returns_to_menu_after_view(tmp_path: Path, monkeypatch):
    """After viewing a run, Esc from the run list should return to the main menu."""
    monkeypatch.setattr("memcomp_bench.interactive.terminal_width", lambda: 120)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    save_conversation(_make_record(), output_dir)

    console, _ = _console()
    prompter = ScriptedPrompter(
        [
            MODE_VIEW,  # main action → view flow
            CANCEL,  # Esc from run list → back to main menu
            MODE_QUIT,  # quit from main menu
        ]
    )

    interactive_module.run_interactive(
        lambda args: None,
        lambda args: None,
        output_dir=output_dir,
        console=console,
        prompter=prompter,
    )


def test_keyboard_interrupt_from_nested_prompt_exits_cleanly(tmp_path: Path):
    class InterruptingPrompter:
        def __init__(self) -> None:
            self.step = 0

        def select(self, prompt: str, choices: list[str], *, default: str | None = None) -> str:
            del prompt, choices, default
            self.step += 1
            if self.step == 1:
                return MODE_NEW
            return "0  Marcus"

        def ask(self, prompt: str, *, default: str | None = None) -> str:
            del prompt, default
            raise KeyboardInterrupt

        def confirm(self, prompt: str, *, default: bool = False) -> bool:
            del prompt, default
            raise AssertionError("confirm should not be reached")

    called = {}
    console, _ = _console()

    interactive_module.run_interactive(
        lambda args: called.setdefault("generate", args),
        lambda args: called.setdefault("resume", args),
        output_dir=tmp_path / "empty",
        console=console,
        prompter=InterruptingPrompter(),
    )

    assert called == {}
