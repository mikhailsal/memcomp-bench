"""Interactive prompt helpers with immediate Escape-to-back support."""

from __future__ import annotations

import builtins
import os
import select
import sys
import termios
import tty
from contextlib import contextmanager


class PromptBack(Exception):
    """Raised when the user presses Esc to back out of a nested prompt."""


_ESCAPE_SEQUENCE_POLL = 0.03
_TEXT_STATUS = "  Type value | Enter accept | Esc back"
_CONFIRM_STATUS = "  Y yes | N no | Enter default | Esc back"


def ask_text(prompt: str, default: str | None = None, *, unsafe: bool = False) -> str:  # pragma: no cover
    del unsafe
    if not _supports_raw_tty():
        return _fallback_text(prompt, default)
    return _read_text_prompt(prompt, default)


def ask_confirm(prompt: str, default: bool = False, *, unsafe: bool = False) -> bool:  # pragma: no cover
    del unsafe
    if not _supports_raw_tty():
        return _fallback_confirm(prompt, default)
    return _read_confirm_prompt(prompt, default)


def _supports_raw_tty() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _fallback_text(prompt: str, default: str | None) -> str:
    suffix = f" [{default}]" if default else ""
    response = builtins.input(f"? {prompt}{suffix} ")
    return response if response else (default or "")


def _fallback_confirm(prompt: str, default: bool) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        response = builtins.input(f"? {prompt} [{hint}] ").strip().lower()
        if not response:
            return default
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False


def _read_text_prompt(prompt: str, default: str | None) -> str:
    prefix = _prompt_prefix(prompt)
    buffer = list(default or "")
    _render_line(prefix, "".join(buffer), _TEXT_STATUS)

    with _raw_mode() as fd:
        while True:
            key = _read_key(fd)
            if key == "enter":
                _finish_prompt()
                return "".join(buffer)
            if key == "backspace":
                if buffer:
                    buffer.pop()
                    _render_line(prefix, "".join(buffer), _TEXT_STATUS)
                continue
            if key == "escape-sequence":
                continue
            if len(key) == 1 and key.isprintable():
                buffer.append(key)
                _render_line(prefix, "".join(buffer), _TEXT_STATUS)


def _read_confirm_prompt(prompt: str, default: bool) -> bool:
    prefix = _prompt_prefix(f"{prompt} [{_confirm_hint(default)}]")
    _render_line(prefix, status=_CONFIRM_STATUS)

    with _raw_mode() as fd:
        while True:
            key = _read_key(fd)
            if key == "enter":
                _finish_prompt()
                return default
            if key == "escape-sequence":
                continue
            if key in {"y", "Y"}:
                _finish_prompt("y")
                return True
            if key in {"n", "N"}:
                _finish_prompt("n")
                return False


def _prompt_prefix(prompt: str) -> str:
    return f"? {prompt} "


def _confirm_hint(default: bool) -> str:
    return "Y/n" if default else "y/N"


def _render_line(prefix: str, value: str = "", status: str | None = None) -> None:
    sys.stdout.write(f"\r\x1b[2K{prefix}{value}")
    if status:
        sys.stdout.write(f"\x1b7\r\n\x1b[2K\x1b[2m{status}\x1b[0m\x1b8")
    sys.stdout.flush()


def _finish_prompt(value: str = "") -> None:
    if value:
        sys.stdout.write(value)
    sys.stdout.write("\x1b7\r\n\x1b[2K\x1b8\r\n")
    sys.stdout.flush()


@contextmanager
def _raw_mode():
    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        yield fd
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original)


def _read_key(fd: int) -> str:
    data = os.read(fd, 1)
    if data in {b"\x03", b"\x11"}:
        raise KeyboardInterrupt
    if data in {b"\r", b"\n"}:
        return "enter"
    if data in {b"\x7f", b"\b"}:
        return "backspace"
    if data == b"\x1b":
        if _drain_escape_sequence(fd):
            return "escape-sequence"
        raise PromptBack
    return data.decode("utf-8", errors="ignore")


def _drain_escape_sequence(fd: int) -> bytes:
    suffix = bytearray()
    while True:
        ready, _, _ = select.select([fd], [], [], _ESCAPE_SEQUENCE_POLL)
        if not ready:
            return bytes(suffix)
        suffix.extend(os.read(fd, 1))
