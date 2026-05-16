"""Questionary helpers for interactive prompts with Escape-to-back support."""

from __future__ import annotations

from typing import Any

import questionary


class PromptBack(Exception):
    """Raised when the user presses Esc to back out of a nested prompt."""


_ESCAPE_RESULT = object()


def ask_text(prompt: str, default: str | None = None, *, unsafe: bool = False) -> str:  # pragma: no cover
    question = questionary.text(prompt, default=default or "", key_bindings=_escape_key_bindings())
    result = question.unsafe_ask() if unsafe else question.ask()
    return _unwrap_prompt_result(result)


def ask_confirm(prompt: str, default: bool = False, *, unsafe: bool = False) -> bool:  # pragma: no cover
    question = _build_confirm_question(prompt, default)
    result = question.unsafe_ask() if unsafe else question.ask()
    return _unwrap_prompt_result(result)


def _unwrap_prompt_result(result: Any) -> Any:  # pragma: no cover
    if result is _ESCAPE_RESULT:
        raise PromptBack
    return result


def _build_confirm_question(prompt: str, default: bool) -> questionary.Question:  # pragma: no cover
    from prompt_toolkit import PromptSession
    from questionary.styles import merge_styles_default

    merged_style = merge_styles_default([None])
    status = {"answer": None, "complete": False}
    session = PromptSession(
        _confirm_prompt_tokens(prompt, default, status),
        key_bindings=_confirm_key_bindings(default, status),
        style=merged_style,
    )
    return questionary.Question(session.app)


def _confirm_prompt_tokens(prompt: str, default: bool, status: dict[str, Any]):  # pragma: no cover
    from prompt_toolkit.formatted_text import to_formatted_text
    from questionary.constants import DEFAULT_QUESTION_PREFIX, NO, NO_OR_YES, YES, YES_OR_NO

    def get_prompt_tokens():
        tokens: list[tuple[str, str]] = [
            ("class:qmark", DEFAULT_QUESTION_PREFIX),
            ("class:question", f" {prompt} "),
        ]
        if not status["complete"]:
            instruction = YES_OR_NO if default else NO_OR_YES
            tokens.append(("class:instruction", f"{instruction} "))
        if status["answer"] is not None:
            tokens.append(("class:answer", YES if status["answer"] else NO))
        return to_formatted_text(tokens)

    return get_prompt_tokens


def _confirm_key_bindings(default: bool, status: dict[str, Any]):  # pragma: no cover
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys

    bindings = KeyBindings()

    def exit_with_result(event) -> None:
        status["complete"] = True
        event.app.exit(result=status["answer"])

    @bindings.add(Keys.ControlQ, eager=True)
    @bindings.add(Keys.ControlC, eager=True)
    def _interrupt(event):
        event.app.exit(exception=KeyboardInterrupt, style="class:aborting")

    @bindings.add(Keys.Escape, eager=True)
    def _escape(event):
        event.app.exit(result=_ESCAPE_RESULT)

    @bindings.add("n")
    @bindings.add("N")
    def _no(event):
        status["answer"] = False
        exit_with_result(event)

    @bindings.add("y")
    @bindings.add("Y")
    def _yes(event):
        status["answer"] = True
        exit_with_result(event)

    @bindings.add(Keys.ControlH)
    def _backspace(event):
        del event
        status["answer"] = None

    @bindings.add(Keys.ControlM, eager=True)
    def _enter(event):
        if status["answer"] is None:
            status["answer"] = default
        exit_with_result(event)

    @bindings.add(Keys.Any)
    def _other(event):
        del event

    return bindings


def _escape_key_bindings():  # pragma: no cover
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys

    bindings = KeyBindings()

    @bindings.add(Keys.Escape, eager=True)
    def _escape(event):
        event.app.exit(result=_ESCAPE_RESULT)

    return bindings
