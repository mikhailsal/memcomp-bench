"""Functional coverage for the interactive benchmark mode."""

from __future__ import annotations

import json
import time
from argparse import Namespace
from io import StringIO
from pathlib import Path

from rich.console import Console

from memcomp_bench._interactive_display import format_run_line
from memcomp_bench._interactive_prompts import CANCEL, SORT_ACTION
from memcomp_bench.generator import ConversationRecord, ConversationTurn, save_conversation
from memcomp_bench.interactive import (
    DETAIL_BACK,
    MODE_NEW,
    MODE_RESUME,
    MODE_RESUME_LAST,
    MODE_VIEW,
    scan_saved_conversations,
)
from memcomp_bench.persistence import load_conversation_metadata
from tests.conftest import FakeChatClient, make_plain_response, make_tool_call_response


class ScriptedPrompter:
    """Scripted prompt driver for interactive CLI tests."""

    def __init__(self, answers: list[str], confirms: list[bool] | None = None) -> None:
        self.answers = list(answers)
        self.confirms = list(confirms or [])

    def ask(self, prompt: str, *, default: str | None = None) -> str:
        if not self.answers:
            raise AssertionError(f"Unexpected prompt: {prompt}")
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
            raise AssertionError(f"Unexpected select: {prompt}")
        answer = self.answers.pop(0)
        if answer == "":
            return default if default is not None else choices[0]
        return answer


class RecordingPrompter(ScriptedPrompter):
    """Scripted prompter that records select menus for assertions."""

    def __init__(self, answers: list[str], confirms: list[bool] | None = None) -> None:
        super().__init__(answers, confirms)
        self.select_calls: list[tuple[str, list[str], str | None]] = []

    def select(self, prompt: str, choices: list[str], *, default: str | None = None) -> str:
        self.select_calls.append((prompt, list(choices), default))
        return super().select(prompt, choices, default=default)


def _make_console() -> tuple[Console, StringIO]:
    stream = StringIO()
    return Console(file=stream, force_terminal=False, color_system=None, width=120), stream


def _make_resume_record(*, ai_model: str = "test/ai", human_model: str = "test/human") -> ConversationRecord:
    profile = {"name": "Marcus", "backstory": "A tester."}
    tool_call = {
        "id": "wmth00001",
        "type": "function",
        "function": {"name": "write_message_to_human", "arguments": json.dumps({"text": "Hello"})},
    }
    record = ConversationRecord(
        id="20260101_000000",
        human_profile=profile,
        ai_model=ai_model,
        human_model=human_model,
        seed_words=["ocean", "ember"],
        language="english",
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


def _make_resume_client() -> FakeChatClient:
    fake = FakeChatClient()
    for index in range(6):
        if index % 2 == 0:
            fake.enqueue(
                make_tool_call_response(
                    f"AI {index}",
                    tool_call_id=f"wmth_r{index:03d}",
                    prompt_tokens=700 + index * 50,
                    completion_tokens=30,
                )
            )
        else:
            fake.enqueue(make_plain_response(f"Human {index}", prompt_tokens=700 + index * 50, completion_tokens=25))
    return fake


def _make_generate_client() -> FakeChatClient:
    fake = FakeChatClient()
    fake.enqueue(make_plain_response("Plan."))
    fake.enqueue(make_tool_call_response("Hi!", tool_call_id="wmth00001"))
    fake.enqueue(make_plain_response("Hello"))
    fake.enqueue(make_tool_call_response("Done!", tool_call_id="wmth00002", prompt_tokens=600, completion_tokens=30))
    fake.enqueue(make_plain_response("Bye"))
    fake.enqueue(make_tool_call_response("Wrap", tool_call_id="wmth00003", prompt_tokens=650, completion_tokens=30))
    return fake


def _patch_terminal_width(monkeypatch):
    monkeypatch.setattr("memcomp_bench.interactive.terminal_width", lambda: 120)


def test_interactive_resume_can_apply_temporary_override(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    _patch_terminal_width(monkeypatch)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.setattr("memcomp_bench.cli.OUTPUT_DIR", output_dir)
    monkeypatch.setattr("memcomp_bench.config.ENV_PATH", tmp_path / ".env")
    monkeypatch.setenv("OPENROUTER_KEY", "interactive-test-key")

    jsonl_path = save_conversation(_make_resume_record(), output_dir)
    summaries = scan_saved_conversations(output_dir)
    run_line = format_run_line(1, 1, summaries[0], width=120)
    fake = _make_resume_client()
    monkeypatch.setattr("memcomp_bench.cli.OpenRouterClient", lambda key: fake)

    from memcomp_bench.cli import cmd_interactive

    console, stream = _make_console()
    prompter = ScriptedPrompter(
        answers=[
            MODE_RESUME,  # main action
            run_line,  # pick run
            DETAIL_BACK,  # detail view → back
            "Edit defaults before continuing",  # resume mode
            "",
            "override/ai",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "400",
        ],
        confirms=[False],
    )

    cmd_interactive(Namespace(), prompter=prompter, console_override=console)

    metadata = load_conversation_metadata(jsonl_path)
    assert metadata["ai_model"] == "override/ai"
    assert metadata["resume_defaults"]["ai_model"] == "test/ai"


def test_interactive_resume_can_persist_override(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    _patch_terminal_width(monkeypatch)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.setattr("memcomp_bench.cli.OUTPUT_DIR", output_dir)
    monkeypatch.setattr("memcomp_bench.config.ENV_PATH", tmp_path / ".env")
    monkeypatch.setenv("OPENROUTER_KEY", "interactive-test-key")

    jsonl_path = save_conversation(_make_resume_record(), output_dir)
    summaries = scan_saved_conversations(output_dir)
    run_line = format_run_line(1, 1, summaries[0], width=120)
    fake = _make_resume_client()
    monkeypatch.setattr("memcomp_bench.cli.OpenRouterClient", lambda key: fake)

    from memcomp_bench.cli import cmd_interactive

    console, _ = _make_console()
    prompter = ScriptedPrompter(
        answers=[
            MODE_RESUME,  # main action
            run_line,  # pick run
            DETAIL_BACK,  # detail view → back
            "Edit defaults before continuing",  # resume mode
            "",
            "permanent/ai",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "450",
        ],
        confirms=[True],
    )

    cmd_interactive(Namespace(), prompter=prompter, console_override=console)

    metadata = load_conversation_metadata(jsonl_path)
    assert metadata["ai_model"] == "permanent/ai"
    assert metadata["resume_defaults"]["ai_model"] == "permanent/ai"


def test_interactive_resume_uses_rounded_default_target(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("memcomp_bench.interactive.terminal_width", lambda: 120)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    record = _make_resume_record()
    record.total_tokens_estimate = 21_463
    save_conversation(record, output_dir)
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

    assert called["resume"].target_tokens == 25_000


def test_interactive_resume_last_uses_latest_resumable_run(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("memcomp_bench.interactive.terminal_width", lambda: 120)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    newest = _make_resume_record()
    newest.id = "20260102_000000"
    newest.total_tokens_estimate = 18_200
    newest_jsonl = save_conversation(newest, output_dir)
    (output_dir / f"{newest_jsonl.stem}_raw_ai_context.json").unlink()

    older = _make_resume_record()
    older.id = "20260101_000000"
    older.total_tokens_estimate = 12_300
    older_jsonl = save_conversation(older, output_dir)

    called = {}
    console, _ = _make_console()
    prompter = ScriptedPrompter(
        answers=[
            MODE_RESUME_LAST,
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

    assert called["resume"].file == str(older_jsonl)
    assert called["resume"].target_tokens == 15_000


def test_interactive_main_menu_hides_resume_last_when_no_resumable_runs(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("memcomp_bench.interactive.terminal_width", lambda: 120)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    jsonl_path = save_conversation(_make_resume_record(), output_dir)
    (output_dir / f"{jsonl_path.stem}_raw_ai_context.json").unlink()

    called = {}
    console, _ = _make_console()
    prompter = RecordingPrompter([CANCEL])

    from memcomp_bench import interactive as interactive_module

    interactive_module.run_interactive(
        lambda args: called.setdefault("generate", args),
        lambda args: called.setdefault("resume", args),
        output_dir=output_dir,
        console=console,
        prompter=prompter,
    )

    assert MODE_RESUME_LAST not in prompter.select_calls[0][1]
    assert called == {}


def test_interactive_can_start_new_generation(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    output_dir = tmp_path / "output"
    monkeypatch.setattr("memcomp_bench.cli.OUTPUT_DIR", output_dir)
    monkeypatch.setattr("memcomp_bench.config.ENV_PATH", tmp_path / ".env")
    monkeypatch.setenv("OPENROUTER_KEY", "interactive-test-key")

    fake = _make_generate_client()
    monkeypatch.setattr("memcomp_bench.cli.OpenRouterClient", lambda key: fake)

    from memcomp_bench.cli import cmd_interactive

    console, _ = _make_console()
    prompter = ScriptedPrompter(
        answers=[
            MODE_NEW,  # main action (no saved runs)
            "0  Marcus",  # select: profile
            "200",
            "",
            "",
            "",
            "google/gemma-4-26b-a4b-it:free",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        confirms=[False],
    )

    cmd_interactive(Namespace(), prompter=prompter, console_override=console)

    jsonl_files = list(output_dir.glob("*.jsonl"))
    assert jsonl_files


def test_interactive_generate_shows_disabled_model_error(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    output_dir = tmp_path / "output"
    monkeypatch.setattr("memcomp_bench.cli.OUTPUT_DIR", output_dir)
    monkeypatch.setattr("memcomp_bench.config.ENV_PATH", tmp_path / ".env")
    monkeypatch.setenv("OPENROUTER_KEY", "interactive-test-key")

    called = False

    def fake_init(self, client, human_profile, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("memcomp_bench.cli.OpenRouterClient", lambda key: _make_generate_client())
    monkeypatch.setattr("memcomp_bench.cli.ConversationGenerator.__init__", fake_init)

    from memcomp_bench.cli import cmd_interactive

    console, stream = _make_console()
    prompter = ScriptedPrompter(
        answers=[
            MODE_NEW,
            "0  Marcus",
            "200",
            "",
            "",
            "",
            "x-ai/grok-4.1-fast",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        confirms=[False],
    )

    cmd_interactive(Namespace(), prompter=prompter, console_override=console)

    assert called is False
    out = capsys.readouterr().out
    assert "disabled in models.toml" in out
    assert "new generations" in out


def test_interactive_view_mode_shows_details(monkeypatch, tmp_path: Path):
    _patch_terminal_width(monkeypatch)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.setattr("memcomp_bench.cli.OUTPUT_DIR", output_dir)
    monkeypatch.setattr("memcomp_bench.config.ENV_PATH", tmp_path / ".env")
    monkeypatch.setenv("OPENROUTER_KEY", "interactive-test-key")

    save_conversation(_make_resume_record(), output_dir)
    summaries = scan_saved_conversations(output_dir)
    run_line = format_run_line(1, 1, summaries[0], width=120)

    from memcomp_bench.cli import cmd_interactive

    console, stream = _make_console()
    prompter = ScriptedPrompter(
        answers=[
            MODE_VIEW,  # main action
            run_line,  # pick run
            DETAIL_BACK,  # detail view → back to list
            CANCEL,  # Esc from run list → back to main
            CANCEL,  # Esc from main menu → exit
        ],
    )

    cmd_interactive(Namespace(), prompter=prompter, console_override=console)


def test_interactive_sort_by_tokens(monkeypatch, tmp_path: Path):
    _patch_terminal_width(monkeypatch)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.setattr("memcomp_bench.cli.OUTPUT_DIR", output_dir)
    monkeypatch.setattr("memcomp_bench.config.ENV_PATH", tmp_path / ".env")
    monkeypatch.setenv("OPENROUTER_KEY", "interactive-test-key")

    rec1 = _make_resume_record()
    rec1.total_tokens_estimate = 500
    save_conversation(rec1, output_dir)

    rec2 = _make_resume_record()
    rec2.id = "20260102_000000"
    rec2.total_tokens_estimate = 200
    save_conversation(rec2, output_dir)

    from memcomp_bench.cli import cmd_interactive

    console, stream = _make_console()
    prompter = ScriptedPrompter(
        answers=[
            MODE_VIEW,  # main action
            SORT_ACTION,  # 's' key triggers sort
            "[3] Most tokens",  # pick new sort order
            CANCEL,  # Esc from run list → back to main
            CANCEL,  # Esc from main menu → exit
        ],
    )

    cmd_interactive(Namespace(), prompter=prompter, console_override=console)
