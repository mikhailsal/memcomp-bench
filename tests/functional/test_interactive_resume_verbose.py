"""Regression coverage for interactive resume verbose defaults."""

from __future__ import annotations

from pathlib import Path

from memcomp_bench._interactive_display import format_run_line
from memcomp_bench.generator import save_conversation
from memcomp_bench.interactive import DETAIL_BACK, MODE_RESUME, scan_saved_conversations
from tests.functional.test_interactive_cli import ScriptedPrompter, _make_console, _make_resume_record


def test_interactive_resume_defaults_verbose_on(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("memcomp_bench.interactive.terminal_width", lambda: 120)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    save_conversation(_make_resume_record(), output_dir)
    summaries = scan_saved_conversations(output_dir)
    run_line = format_run_line(1, 1, summaries[0], width=120)

    called = {}
    console, _ = _make_console()
    prompter = ScriptedPrompter(
        answers=[
            MODE_RESUME,
            run_line,
            DETAIL_BACK,
            "Continue with saved defaults",
            "",
        ],
    )

    from memcomp_bench import interactive as interactive_module

    interactive_module.run_interactive(
        lambda args: called.setdefault("generate", args),
        lambda args: called.setdefault("resume", args),
        output_dir=output_dir,
        console=console,
        prompter=prompter,
    )

    assert called["resume"].verbose is True
