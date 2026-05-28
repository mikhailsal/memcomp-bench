from __future__ import annotations

from pathlib import Path

from memcomp_bench import interactive as interactive_module
from memcomp_bench.generator import save_conversation
from tests.unit.test_interactive import _make_record


def test_scan_saved_conversations_keeps_legacy_runs_visible_without_ready_marker(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    jsonl_path = save_conversation(_make_record(), output_dir)
    jsonl_path.with_suffix(".ready").unlink()

    summaries = interactive_module.scan_saved_conversations(output_dir)

    assert len(summaries) == 1


def test_scan_saved_conversations_skips_locked_unpublished_run(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    jsonl_path = save_conversation(_make_record(), output_dir)
    jsonl_path.with_suffix(".ready").unlink()
    jsonl_path.with_suffix(".lock").write_text("123", encoding="utf-8")

    summaries = interactive_module.scan_saved_conversations(output_dir)

    assert summaries == []
