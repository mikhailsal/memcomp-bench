"""Regression coverage for resume detail ESC handling."""

from __future__ import annotations

from pathlib import Path

from memcomp_bench._interactive_display import format_run_line
from memcomp_bench._interactive_prompts import CANCEL, TerminalMenuPrompter
from memcomp_bench.generator import save_conversation
from memcomp_bench.interactive import MODE_RESUME, scan_saved_conversations
from tests.functional.test_interactive_cli import (
    _make_console,
    _make_resume_record,
)


class RecordingTerminalMenuPrompter(TerminalMenuPrompter):
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.select_calls: list[tuple[str, list[str], str | None]] = []

    def ask(self, prompt: str, *, default: str | None = None) -> str:
        raise AssertionError(f"Unexpected ask: {prompt}")

    def confirm(self, prompt: str, *, default: bool = False) -> bool:
        raise AssertionError(f"Unexpected confirm: {prompt}")

    def select(self, prompt: str, choices: list[str], *, default: str | None = None) -> str:
        self.select_calls.append((prompt, list(choices), default))
        if not self.answers:
            raise AssertionError(f"Unexpected select: {prompt}")
        answer = self.answers.pop(0)
        if answer == "":
            return default if default is not None else choices[0]
        return answer

    def menu(
        self,
        prompt: str,
        choices: list[str],
        *,
        default: str | None = None,
        extra_accept_keys: tuple[str, ...] = (),
        status: str | None = None,
        skip_indices: list[int] | None = None,
    ) -> tuple[str, str | None]:
        if not self.answers:
            raise AssertionError(f"Unexpected menu: {prompt}")
        answer = self.answers.pop(0)
        if answer == CANCEL:
            return CANCEL, None
        if answer == "":
            default_val = default if default in choices else choices[0]
            return default_val, "enter"
        if answer in choices:
            return answer, "enter"
        raise AssertionError(f"Invalid menu answer: {answer}, valid choices: {choices}")


def test_interactive_resume_detail_esc_returns_to_run_picker(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("memcomp_bench.interactive.terminal_width", lambda: 120)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    save_conversation(_make_resume_record(), output_dir)
    summaries = scan_saved_conversations(output_dir)
    run_line = format_run_line(1, 1, summaries[0], width=120)

    console, _ = _make_console()
    prompter = RecordingTerminalMenuPrompter(
        answers=[
            MODE_RESUME,
            run_line,
            CANCEL,
            CANCEL,
            CANCEL,
        ],
    )

    from memcomp_bench import interactive as interactive_module

    interactive_module.run_interactive(
        lambda args: None,
        lambda args: None,
        output_dir=output_dir,
        console=console,
        prompter=prompter,
    )

    resume_mode_prompts = [call for call in prompter.select_calls if call[0] == "How would you like to continue?"]
    run_picker_prompts = [call for call in prompter.select_calls if "saved runs" in call[0] or "Sort:" in call[0]]

    assert not resume_mode_prompts
    assert len(run_picker_prompts) == 2
