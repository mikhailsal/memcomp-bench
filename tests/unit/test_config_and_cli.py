"""Tests for memcomp_bench.config and memcomp_bench.cli."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# config — ensure_dirs / load_api_key
# ---------------------------------------------------------------------------


class TestEnsureDirs:
    def test_creates_output_dir(self, tmp_path: Path, monkeypatch):
        out = tmp_path / "new_output"
        monkeypatch.setattr("memcomp_bench.config.OUTPUT_DIR", out)
        from memcomp_bench.config import ensure_dirs

        ensure_dirs()
        assert out.is_dir()

    def test_idempotent(self, tmp_path: Path, monkeypatch):
        out = tmp_path / "new_output"
        out.mkdir()
        monkeypatch.setattr("memcomp_bench.config.OUTPUT_DIR", out)
        from memcomp_bench.config import ensure_dirs

        ensure_dirs()
        assert out.is_dir()


class TestLoadApiKey:
    def test_returns_key_when_set(self, monkeypatch, tmp_path: Path):
        env_file = tmp_path / ".env"
        env_file.write_text("OPENROUTER_KEY=test-key-123\n")
        monkeypatch.setattr("memcomp_bench.config.ENV_PATH", env_file)
        monkeypatch.setenv("OPENROUTER_KEY", "test-key-123")
        from memcomp_bench.config import load_api_key

        assert load_api_key() == "test-key-123"

    def test_exits_when_missing(self, monkeypatch, tmp_path: Path):
        env_file = tmp_path / ".env"
        env_file.write_text("")
        monkeypatch.setattr("memcomp_bench.config.ENV_PATH", env_file)
        monkeypatch.delenv("OPENROUTER_KEY", raising=False)
        from memcomp_bench.config import load_api_key

        with pytest.raises(SystemExit):
            load_api_key()

    def test_exits_when_placeholder(self, monkeypatch, tmp_path: Path):
        env_file = tmp_path / ".env"
        env_file.write_text("OPENROUTER_KEY=your-key-here\n")
        monkeypatch.setattr("memcomp_bench.config.ENV_PATH", env_file)
        monkeypatch.setenv("OPENROUTER_KEY", "your-key-here")
        from memcomp_bench.config import load_api_key

        with pytest.raises(SystemExit):
            load_api_key()


# ---------------------------------------------------------------------------
# cli — _resolve_profile
# ---------------------------------------------------------------------------


class TestResolveProfile:
    def test_by_index(self):
        from memcomp_bench.cli import _resolve_profile
        from memcomp_bench.prompts import HUMAN_PROFILES

        p = _resolve_profile("0")
        assert p["name"] == HUMAN_PROFILES[0]["name"]

    def test_by_name_case_insensitive(self):
        from memcomp_bench.cli import _resolve_profile
        from memcomp_bench.prompts import HUMAN_PROFILES

        name = HUMAN_PROFILES[0]["name"]
        p = _resolve_profile(name.upper())
        assert p["name"] == name

    def test_unknown_exits(self):
        from memcomp_bench.cli import _resolve_profile

        with pytest.raises(SystemExit):
            _resolve_profile("nonexistent_profile_xyz")


# ---------------------------------------------------------------------------
# cli — main() dispatch
# ---------------------------------------------------------------------------


class TestMainDispatch:
    def test_no_args_prints_help(self, capsys):
        from memcomp_bench.cli import main

        with pytest.raises(SystemExit) as exc:
            sys.argv = ["memcomp"]
            main()
        assert exc.value.code == 1

    def test_profiles_command(self, capsys):
        from memcomp_bench.cli import main

        sys.argv = ["memcomp", "profiles"]
        main()
        out = capsys.readouterr().out
        assert "Marcus" in out or "Anya" in out or len(out) > 0

    def test_generate_calls_handler(self, monkeypatch):
        from memcomp_bench.cli import main

        called = {}

        def fake_generate(args):
            called["generate"] = True

        monkeypatch.setattr("memcomp_bench.cli.cmd_generate", fake_generate)
        sys.argv = ["memcomp", "generate", "--profile", "0"]
        main()
        assert called.get("generate") is True

    def test_resume_calls_handler(self, monkeypatch, tmp_path):
        from memcomp_bench.cli import main

        called = {}

        def fake_resume(args):
            called["resume"] = True

        monkeypatch.setattr("memcomp_bench.cli.cmd_resume", fake_resume)
        fake_file = tmp_path / "conv.jsonl"
        fake_file.touch()
        sys.argv = ["memcomp", "resume", str(fake_file)]
        main()
        assert called.get("resume") is True

    def test_reformat_calls_handler(self, monkeypatch, tmp_path):
        from memcomp_bench.cli import main

        called = {}

        def fake_reformat(args):
            called["reformat"] = True

        monkeypatch.setattr("memcomp_bench.cli.cmd_reformat", fake_reformat)
        fake_file = tmp_path / "conv.jsonl"
        fake_file.touch()
        sys.argv = ["memcomp", "reformat", str(fake_file)]
        main()
        assert called.get("reformat") is True

    def test_generate_prefers_ai_provider_flag(self, monkeypatch):
        from memcomp_bench.cli import main

        seen = {}

        def fake_generate(args):
            seen["ai_provider"] = args.ai_provider

        monkeypatch.setattr("memcomp_bench.cli.cmd_generate", fake_generate)
        sys.argv = ["memcomp", "generate", "--profile", "0", "--ai-provider", "minimax"]
        main()
        assert seen["ai_provider"] == "minimax"

    def test_generate_accepts_provider_alias(self, monkeypatch):
        from memcomp_bench.cli import main

        seen = {}

        def fake_generate(args):
            seen["ai_provider"] = args.ai_provider

        monkeypatch.setattr("memcomp_bench.cli.cmd_generate", fake_generate)
        sys.argv = ["memcomp", "generate", "--profile", "0", "--provider", "minimax"]
        main()
        assert seen["ai_provider"] == "minimax"


class TestCmdReformatDirect:
    """Test cmd_reformat with an actual JSONL file."""

    def test_reformat_produces_md(self, monkeypatch, tmp_path):
        import json

        from memcomp_bench.generator import ConversationRecord, ConversationTurn, save_conversation

        profile = {"name": "TestUser", "backstory": "A tester."}
        record = ConversationRecord(
            id="20260101_000000",
            human_profile=profile,
            ai_model="test/model",
            human_model="test/model",
        )
        record.turns = [
            ConversationTurn(turn_number=1, speaker="human", visible_text="Hi"),
            ConversationTurn(turn_number=2, speaker="ai", visible_text="Hello"),
        ]
        record.total_tokens_estimate = 10
        record.started_at = "2026-01-01T00:00:00Z"
        record.finished_at = "2026-01-01T00:01:00Z"
        record.ai_messages_raw = [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "[start]"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "write_message_to_human", "arguments": json.dumps({"text": "Hello"})},
                    }
                ],
            },
        ]

        jsonl_path = save_conversation(record, tmp_path)
        md_path = jsonl_path.with_suffix(".md")
        md_path.write_text("CORRUPTED")

        from argparse import Namespace

        from memcomp_bench.cli import cmd_reformat

        cmd_reformat(Namespace(file=str(jsonl_path)))
        assert md_path.read_text() != "CORRUPTED"
        assert "TestUser" in md_path.read_text()


def _make_generate_args(**overrides):
    """Build a Namespace for cmd_generate with sensible defaults."""
    from argparse import Namespace

    defaults = dict(
        profile="0",
        ai_model="test/ai",
        human_model="test/human",
        target_tokens=200,
        language="english",
        companion_mode="honest",
        verbose=False,
        ai_provider=None,
        human_provider=None,
        ai_temperature=None,
        human_temperature=None,
        ai_max_tokens=None,
        human_max_tokens=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _make_generate_client():
    """Build a FakeChatClient pre-loaded for a cmd_generate run."""
    from tests.conftest import FakeChatClient, make_plain_response, make_tool_call_response

    fake = FakeChatClient()
    fake.enqueue(make_plain_response("Plan."))
    fake.enqueue(make_tool_call_response("Hi!", tool_call_id="wmth00001"))
    fake.enqueue(make_plain_response("Hey!"))
    fake.enqueue(make_tool_call_response("Hello!", tool_call_id="wmth00002", prompt_tokens=600, completion_tokens=30))
    fake.enqueue(make_plain_response("More."))
    fake.enqueue(make_tool_call_response("Done!", tool_call_id="wmth00003", prompt_tokens=700, completion_tokens=30))
    return fake


class TestCmdGenerateIntegration:
    """Test cmd_generate with a FakeChatClient to exercise the handler body."""

    def test_cmd_generate_runs(self, monkeypatch, tmp_path):
        monkeypatch.setattr(time, "sleep", lambda _: None)
        output_dir = tmp_path / "output"
        monkeypatch.setattr("memcomp_bench.cli.OUTPUT_DIR", output_dir)
        monkeypatch.setattr("memcomp_bench.config.ENV_PATH", tmp_path / ".env")
        monkeypatch.setenv("OPENROUTER_KEY", "test-key-for-generate")

        fake = _make_generate_client()
        monkeypatch.setattr("memcomp_bench.cli.OpenRouterClient", lambda key: fake)

        from memcomp_bench.cli import cmd_generate

        cmd_generate(_make_generate_args())

        assert output_dir.exists()
        jsonl_files = list(output_dir.glob("*.jsonl"))
        assert len(jsonl_files) >= 1


def _make_resume_record():
    """Build a minimal saved conversation record for resume tests."""
    from memcomp_bench.generator import ConversationRecord, ConversationTurn

    profile = {"name": "Marcus", "backstory": "A tester."}
    record = ConversationRecord(
        id="20260101_000000",
        human_profile=profile,
        ai_model="test/ai",
        human_model="test/human",
        seed_words=["ocean", "ember"],
    )
    tc = {
        "id": "wmth00001",
        "type": "function",
        "function": {"name": "write_message_to_human", "arguments": json.dumps({"text": "Hello"})},
    }
    record.turns = [
        ConversationTurn(turn_number=1, speaker="human", visible_text="Hi"),
        ConversationTurn(turn_number=2, speaker="ai", visible_text="Hello", ai_tool_calls=[tc]),
    ]
    record.total_tokens_estimate = 10
    record.started_at = "2026-01-01T00:00:00Z"
    record.finished_at = "2026-01-01T00:01:00Z"
    record.ai_messages_raw = [
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": "[start]"},
        {"role": "assistant", "content": None, "tool_calls": [tc]},
        {"role": "tool", "content": "Hi", "tool_call_id": "wmth00001"},
    ]
    return record


def _make_resume_client():
    """Build a FakeChatClient pre-loaded for a cmd_resume run."""
    from tests.conftest import FakeChatClient, make_plain_response, make_tool_call_response

    fake = FakeChatClient()
    for i in range(6):
        if i % 2 == 0:
            fake.enqueue(
                make_tool_call_response(
                    f"Continued AI {i}", tool_call_id=f"wmth_r{i:03d}", prompt_tokens=700 + i * 50, completion_tokens=30
                )
            )
        else:
            fake.enqueue(make_plain_response(f"Continued Human {i}", prompt_tokens=700 + i * 50, completion_tokens=25))
    return fake


def _make_resume_args(jsonl_path, **overrides):
    """Build a Namespace for cmd_resume with sensible defaults."""
    from argparse import Namespace

    defaults = dict(
        file=str(jsonl_path),
        target_tokens=400,
        language=None,
        ai_model=None,
        human_model=None,
        ai_provider=None,
        human_provider=None,
        ai_temperature=None,
        human_temperature=None,
        ai_max_tokens=None,
        human_max_tokens=None,
        verbose=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


class TestCmdResumeIntegration:
    """Test cmd_resume with a saved JSONL and FakeChatClient."""

    def test_cmd_resume_runs(self, monkeypatch, tmp_path):
        monkeypatch.setattr(time, "sleep", lambda _: None)
        monkeypatch.setattr("memcomp_bench.config.ENV_PATH", tmp_path / ".env")
        monkeypatch.setattr("memcomp_bench.cli.OUTPUT_DIR", tmp_path / "output_resume")
        monkeypatch.setenv("OPENROUTER_KEY", "test-key-for-resume")

        from memcomp_bench.generator import save_conversation

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        monkeypatch.setattr("memcomp_bench.config.OUTPUT_DIR", output_dir)
        jsonl_path = save_conversation(_make_resume_record(), output_dir)

        fake = _make_resume_client()
        monkeypatch.setattr("memcomp_bench.cli.OpenRouterClient", lambda key: fake)

        from memcomp_bench.cli import cmd_resume

        cmd_resume(_make_resume_args(jsonl_path))

        assert len(fake.call_log) > 0
