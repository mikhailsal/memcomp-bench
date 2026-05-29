"""Regression tests for interactive resume target suggestions."""

from __future__ import annotations

from pathlib import Path

from memcomp_bench import interactive as interactive_module
from memcomp_bench._interactive_display import format_run_line
from memcomp_bench.generator import save_conversation
from memcomp_bench.interactive import DETAIL_BACK, MODE_RESUME
from tests.unit.test_interactive import ScriptedPrompter, _console, _make_record


def test_scan_saved_conversations_prefers_latest_ai_context_tokens(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    record = _make_record()
    record.total_tokens_estimate = 21_463
    record.turns[-1].ai_context_tokens = 23_647
    save_conversation(record, output_dir)

    summaries = interactive_module.scan_saved_conversations(output_dir)

    assert summaries[0].total_tokens_estimate == 23_647


def test_prompt_target_tokens_uses_latest_scanned_context_when_metadata_is_stale(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("memcomp_bench.interactive.terminal_width", lambda: 120)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    record = _make_record()
    record.total_tokens_estimate = 21_463
    record.turns[-1].ai_context_tokens = 23_647
    save_conversation(record, output_dir)
    summaries = interactive_module.scan_saved_conversations(output_dir)
    run_line = format_run_line(1, 1, summaries[0], width=120)

    called = {}
    console, _ = _console()
    prompter = ScriptedPrompter(
        [
            MODE_RESUME,
            run_line,
            DETAIL_BACK,
            "Continue with saved defaults",
            "",
        ]
    )

    interactive_module.run_interactive(
        lambda args: called.setdefault("generate", args),
        lambda args: called.setdefault("resume", args),
        output_dir=output_dir,
        console=console,
        prompter=prompter,
    )

    assert called["resume"].target_tokens == 25_000
