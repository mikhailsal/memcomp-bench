"""PTY-based functional tests for the interactive TUI mode.

These tests spawn the interactive CLI inside a real pseudo-terminal via
*pexpect*, so they verify actual terminal rendering and keyboard handling
(Esc, Ctrl-C, navigation, hotkeys).

Uses ``j``/``k`` keys for menu navigation instead of arrow keys since
arrow escape sequences can be split across reads in a PTY, causing
simple-term-menu to mis-parse them.

Requirements:
  - A real TTY (CI containers usually provide one, but headless Docker may not)
  - ``pexpect`` installed (included in ``[project.optional-dependencies].test``)

Run with::

    pytest -m pty tests/functional/test_interactive_pty.py -v
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from pathlib import Path

import pexpect
import pytest

from memcomp_bench.generator import ConversationRecord, ConversationTurn, save_conversation

pytestmark = pytest.mark.pty

TIMEOUT = 10
KEY_DELAY = 0.15
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])


def _seed_output_dir(output_dir: Path) -> None:
    """Create a minimal conversation JSONL so the run list is populated."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tool_call = {
        "id": "wmth00001",
        "type": "function",
        "function": {"name": "write_message_to_human", "arguments": json.dumps({"text": "Hello"})},
    }
    record = ConversationRecord(
        id="20260101_000000",
        human_profile={"name": "Marcus", "backstory": "A tester."},
        ai_model="test/ai-model",
        human_model="test/human-model",
        language="english",
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
    save_conversation(record, output_dir)


def _write_launcher(tmp_path: Path, output_dir: Path) -> Path:
    """Write a small Python script that patches OUTPUT_DIR then runs the CLI."""
    launcher = tmp_path / "_pty_launcher.py"
    launcher.write_text(
        textwrap.dedent(f"""\
        import sys
        sys.path.insert(0, {PROJECT_ROOT!r})

        import memcomp_bench.config as _cfg
        import memcomp_bench.cli as _cli
        from pathlib import Path

        _cfg.OUTPUT_DIR = Path({str(output_dir)!r})
        _cli.OUTPUT_DIR = _cfg.OUTPUT_DIR
        _cli.main()
        """)
    )
    return launcher


def _spawn(launcher: Path) -> pexpect.spawn:
    """Spawn the interactive CLI via the launcher script."""
    env = os.environ.copy()
    env["TERM"] = "xterm-256color"
    child = pexpect.spawn(
        f"{sys.executable} {launcher} interactive",
        encoding="utf-8",
        timeout=TIMEOUT,
        env=env,
        dimensions=(40, 120),
    )
    child.delaybeforesend = 0
    return child


def _send_key(child: pexpect.spawn, key: str) -> None:
    """Send a key with a short delay to avoid mis-parsing."""
    child.send(key)
    time.sleep(KEY_DELAY)


@pytest.fixture()
def pty_env(tmp_path: Path):
    """Provide a launcher pointing at a temporary output dir with one conversation."""
    out = tmp_path / "output"
    _seed_output_dir(out)
    return _write_launcher(tmp_path, out)


def test_esc_exits_from_main_menu(pty_env: Path):
    """Pressing Escape on the main menu should exit (top of stack)."""
    child = _spawn(pty_env)
    child.expect("What would you like to do", timeout=TIMEOUT)
    time.sleep(0.3)
    child.send("\x1b")
    child.expect(pexpect.EOF, timeout=TIMEOUT)
    child.close()
    assert child.exitstatus == 0


def test_ctrl_c_exits_cleanly(pty_env: Path):
    """Ctrl-C at any point should exit without a traceback."""
    child = _spawn(pty_env)
    child.expect("What would you like to do", timeout=TIMEOUT)
    time.sleep(0.3)
    child.sendcontrol("c")
    child.expect(pexpect.EOF, timeout=TIMEOUT)
    child.close()
    assert child.exitstatus == 0
    assert "Traceback" not in (child.before or "")


def test_q_hotkey_exits_from_main_menu(pty_env: Path):
    """Pressing 'q' on the main menu should select [q] Quit and exit."""
    child = _spawn(pty_env)
    child.expect("What would you like to do", timeout=TIMEOUT)
    time.sleep(0.3)
    _send_key(child, "q")
    child.expect(pexpect.EOF, timeout=TIMEOUT)
    child.close()
    assert child.exitstatus == 0


def test_v_hotkey_navigates_to_view(pty_env: Path):
    """Pressing 'v' should open the view-runs screen; Esc returns to main menu."""
    child = _spawn(pty_env)
    child.expect("What would you like to do", timeout=TIMEOUT)
    time.sleep(0.3)
    _send_key(child, "v")
    child.expect("sorted:", timeout=TIMEOUT)
    time.sleep(0.3)
    child.send("\x1b")
    time.sleep(0.5)
    child.expect("What would you like to do", timeout=TIMEOUT)
    time.sleep(0.3)
    child.send("\x1b")
    child.expect(pexpect.EOF, timeout=TIMEOUT)
    child.close()
    assert child.exitstatus == 0


def test_esc_from_sub_menu_returns_to_main(pty_env: Path):
    """Navigate to View, then Esc should return to main menu (not exit)."""
    child = _spawn(pty_env)
    child.expect("What would you like to do", timeout=TIMEOUT)
    time.sleep(0.3)
    _send_key(child, "v")
    child.expect("sorted:", timeout=TIMEOUT)
    time.sleep(0.3)
    child.send("\x1b")
    time.sleep(0.5)
    child.expect("What would you like to do", timeout=TIMEOUT)
    time.sleep(0.3)
    _send_key(child, "q")
    child.expect(pexpect.EOF, timeout=TIMEOUT)
    child.close()
    assert child.exitstatus == 0
