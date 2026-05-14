"""Functional coverage for the interactive benchmark mode."""

from __future__ import annotations

import json
import time
from argparse import Namespace
from io import StringIO
from pathlib import Path

from rich.console import Console

from memcomp_bench.generator import ConversationRecord, ConversationTurn, save_conversation
from memcomp_bench.interactive import format_run_choice, scan_saved_conversations
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


def test_interactive_resume_can_apply_temporary_override(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.setattr("memcomp_bench.cli.OUTPUT_DIR", output_dir)
    monkeypatch.setattr("memcomp_bench.config.ENV_PATH", tmp_path / ".env")
    monkeypatch.setenv("OPENROUTER_KEY", "interactive-test-key")

    jsonl_path = save_conversation(_make_resume_record(), output_dir)
    run_choice = format_run_choice(scan_saved_conversations(output_dir)[0])
    fake = _make_resume_client()
    monkeypatch.setattr("memcomp_bench.cli.OpenRouterClient", lambda key: fake)

    from memcomp_bench.cli import cmd_interactive

    console, stream = _make_console()
    prompter = ScriptedPrompter(
        answers=[
            "Continue a saved generation",  # select: main action
            run_choice,  # select: pick run
            "Edit defaults before continuing",  # select: resume mode
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
    assert "Saved Generations" in stream.getvalue()
    assert "120" in stream.getvalue()


def test_interactive_resume_can_persist_override(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.setattr("memcomp_bench.cli.OUTPUT_DIR", output_dir)
    monkeypatch.setattr("memcomp_bench.config.ENV_PATH", tmp_path / ".env")
    monkeypatch.setenv("OPENROUTER_KEY", "interactive-test-key")

    jsonl_path = save_conversation(_make_resume_record(), output_dir)
    run_choice = format_run_choice(scan_saved_conversations(output_dir)[0])
    fake = _make_resume_client()
    monkeypatch.setattr("memcomp_bench.cli.OpenRouterClient", lambda key: fake)

    from memcomp_bench.cli import cmd_interactive

    console, _ = _make_console()
    prompter = ScriptedPrompter(
        answers=[
            "Continue a saved generation",  # select: main action
            run_choice,  # select: pick run
            "Edit defaults before continuing",  # select: resume mode
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
            "Start a new generation",  # select: main action (no saved runs)
            "0  Marcus",  # select: profile
            "200",
            "",
            "",
            "",
            "",
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
