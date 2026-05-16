"""Unit tests for raw interactive prompt helpers."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from io import StringIO

import pytest

from memcomp_bench import _interactive_questionary as questionary_module


def test_prompt_prefix_uses_questionary_style_prefix():
    assert questionary_module._prompt_prefix("Language") == "? Language "


def test_confirm_hint_matches_default_direction():
    assert questionary_module._confirm_hint(True) == "Y/n"
    assert questionary_module._confirm_hint(False) == "y/N"


def test_status_lines_describe_available_hotkeys():
    assert "Esc back" in questionary_module._TEXT_STATUS
    assert "Enter default" in questionary_module._CONFIRM_STATUS


def test_drain_escape_sequence_reads_all_immediately_available_bytes(monkeypatch):
    remaining = deque([b"[", b"A"])

    def fake_select(_read, _write, _error, _timeout):
        return ([1], [], []) if remaining else ([], [], [])

    def fake_read(_fd, _size):
        return remaining.popleft()

    monkeypatch.setattr(questionary_module.select, "select", fake_select)
    monkeypatch.setattr(questionary_module.os, "read", fake_read)

    assert questionary_module._drain_escape_sequence(1) == b"[A"


def test_supports_raw_tty_requires_tty_stdin_and_stdout(monkeypatch):
    class TtyStream:
        def __init__(self, is_tty: bool) -> None:
            self._is_tty = is_tty

        def isatty(self) -> bool:
            return self._is_tty

    monkeypatch.setattr(questionary_module.sys, "stdin", TtyStream(True))
    monkeypatch.setattr(questionary_module.sys, "stdout", TtyStream(False))

    assert questionary_module._supports_raw_tty() is False


def test_ask_text_uses_fallback_when_no_raw_tty(monkeypatch):
    monkeypatch.setattr(questionary_module, "_supports_raw_tty", lambda: False)
    monkeypatch.setattr(questionary_module, "_fallback_text", lambda prompt, default: f"{prompt}:{default}")

    assert questionary_module.ask_text("Language", "english") == "Language:english"


def test_ask_confirm_uses_raw_prompt_when_tty(monkeypatch):
    monkeypatch.setattr(questionary_module, "_supports_raw_tty", lambda: True)
    monkeypatch.setattr(questionary_module, "_read_confirm_prompt", lambda prompt, default: not default)

    assert questionary_module.ask_confirm("Continue?", default=False) is True


def test_fallback_text_returns_default_on_blank(monkeypatch):
    monkeypatch.setattr(questionary_module.builtins, "input", lambda _prompt: "")

    assert questionary_module._fallback_text("Language", "english") == "english"


def test_fallback_confirm_retries_until_valid(monkeypatch):
    answers = iter(["maybe", "y"])
    monkeypatch.setattr(questionary_module.builtins, "input", lambda _prompt: next(answers))

    assert questionary_module._fallback_confirm("Continue?", default=False) is True


def test_read_text_prompt_handles_typing_backspace_and_enter(monkeypatch):
    keys = iter(["x", "backspace", "y", "enter"])
    renders: list[tuple[str, str, str | None]] = []
    finished: list[str] = []

    @contextmanager
    def fake_raw_mode():
        yield 7

    monkeypatch.setattr(questionary_module, "_raw_mode", fake_raw_mode)
    monkeypatch.setattr(questionary_module, "_read_key", lambda _fd: next(keys))
    monkeypatch.setattr(
        questionary_module,
        "_render_line",
        lambda prefix, value="", status=None: renders.append((prefix, value, status)),
    )
    monkeypatch.setattr(questionary_module, "_finish_prompt", lambda value="": finished.append(value))

    result = questionary_module._read_text_prompt("Language", "en")

    assert result == "eny"
    assert renders[0] == ("? Language ", "en", questionary_module._TEXT_STATUS)
    assert renders[-1] == ("? Language ", "eny", questionary_module._TEXT_STATUS)
    assert finished == [""]


def test_read_text_prompt_ignores_escape_sequences(monkeypatch):
    keys = iter(["escape-sequence", "z", "enter"])

    @contextmanager
    def fake_raw_mode():
        yield 9

    monkeypatch.setattr(questionary_module, "_raw_mode", fake_raw_mode)
    monkeypatch.setattr(questionary_module, "_read_key", lambda _fd: next(keys))
    monkeypatch.setattr(questionary_module, "_render_line", lambda *args, **kwargs: None)
    monkeypatch.setattr(questionary_module, "_finish_prompt", lambda value="": None)

    assert questionary_module._read_text_prompt("Language", None) == "z"


@pytest.mark.parametrize(
    ("keys", "default", "expected_finish", "expected_result"),
    [(["enter"], True, "", True), (["y"], False, "y", True), (["n"], True, "n", False)],
)
def test_read_confirm_prompt_handles_default_yes_and_no(monkeypatch, keys, default, expected_finish, expected_result):
    key_iter = iter(keys)
    finished: list[str] = []

    @contextmanager
    def fake_raw_mode():
        yield 11

    monkeypatch.setattr(questionary_module, "_raw_mode", fake_raw_mode)
    monkeypatch.setattr(questionary_module, "_read_key", lambda _fd: next(key_iter))
    monkeypatch.setattr(questionary_module, "_render_line", lambda *args, **kwargs: None)
    monkeypatch.setattr(questionary_module, "_finish_prompt", lambda value="": finished.append(value))

    assert questionary_module._read_confirm_prompt("Continue?", default) is expected_result
    assert finished == [expected_finish]


def test_read_key_maps_special_inputs(monkeypatch):
    monkeypatch.setattr(questionary_module.os, "read", lambda _fd, _size: b"\r")
    assert questionary_module._read_key(1) == "enter"

    monkeypatch.setattr(questionary_module.os, "read", lambda _fd, _size: b"\x7f")
    assert questionary_module._read_key(1) == "backspace"


def test_read_key_raises_keyboard_interrupt_for_ctrl_c(monkeypatch):
    monkeypatch.setattr(questionary_module.os, "read", lambda _fd, _size: b"\x03")

    with pytest.raises(KeyboardInterrupt):
        questionary_module._read_key(1)


def test_read_key_handles_escape_sequence_and_prompt_back(monkeypatch):
    monkeypatch.setattr(questionary_module.os, "read", lambda _fd, _size: b"\x1b")
    monkeypatch.setattr(questionary_module, "_drain_escape_sequence", lambda _fd: b"[A")
    assert questionary_module._read_key(1) == "escape-sequence"

    monkeypatch.setattr(questionary_module, "_drain_escape_sequence", lambda _fd: b"")
    with pytest.raises(questionary_module.PromptBack):
        questionary_module._read_key(1)


def test_render_line_and_finish_prompt_emit_status_sequences(monkeypatch):
    stream = StringIO()
    monkeypatch.setattr(questionary_module.sys, "stdout", stream)

    questionary_module._render_line("? Language ", "english", questionary_module._TEXT_STATUS)
    questionary_module._finish_prompt("x")

    output = stream.getvalue()
    assert questionary_module._TEXT_STATUS in output
    assert "\x1b[2m" in output
    assert output.endswith("\r\n")
