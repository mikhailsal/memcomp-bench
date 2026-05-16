"""Prompt helpers for the interactive benchmark CLI."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from typing import Any, Protocol

import questionary
from rich.console import Console

from memcomp_bench._interactive_questionary import PromptBack, ask_confirm, ask_text
from memcomp_bench.config import (
    AI_MAX_TOKENS,
    AI_MODEL,
    AI_PROVIDER,
    AI_TEMPERATURE,
    COMPANION_MODE,
    HUMAN_MAX_TOKENS,
    HUMAN_MODEL,
    HUMAN_PROVIDER,
    HUMAN_TEMPERATURE,
    TARGET_TOKENS,
)
from memcomp_bench.model_registry import MISSING, default_model_for, resolve_model_preset

CANCEL = "\x00__CANCEL__"
"""Sentinel returned by ``select()`` when the user presses Esc."""

SORT_ACTION = "\x00__SORT__"
"""Sentinel returned by ``select()`` when the user presses ``s`` (sort)."""


class Prompter(Protocol):
    """Typed prompt interface so tests can script interactive flows."""

    def ask(self, prompt: str, *, default: str | None = None) -> str: ...

    def confirm(self, prompt: str, *, default: bool = False) -> bool: ...

    def select(self, prompt: str, choices: list[str], *, default: str | None = None) -> str: ...


class RichPrompter:
    """Console-backed prompt implementation."""

    def __init__(self, console: Console) -> None:
        self.console = console

    def ask(self, prompt: str, *, default: str | None = None) -> str:
        suffix = f" [{default}]" if default is not None else ""
        response = self.console.input(f"{prompt}{suffix}: ").strip()
        if response:
            return response
        return "" if default is None else default

    def confirm(self, prompt: str, *, default: bool = False) -> bool:
        default_text = "Y/n" if default else "y/N"
        while True:
            response = self.console.input(f"{prompt} [{default_text}]: ").strip().lower()
            if not response:
                return default
            if response in {"y", "yes"}:
                return True
            if response in {"n", "no"}:
                return False
            self.console.print("[yellow]Please answer y or n.[/yellow]")

    def select(self, prompt: str, choices: list[str], *, default: str | None = None) -> str:
        default_val = default if default in choices else choices[0]
        default_idx = choices.index(default_val) + 1
        for i, choice in enumerate(choices, start=1):
            self.console.print(f"  {i}. {choice}")
        while True:
            raw = self.console.input(f"{prompt} [{default_idx}]: ").strip()
            if not raw:
                return default_val
            try:
                idx = int(raw)
            except ValueError:
                self.console.print("[yellow]Enter a number.[/yellow]")
                continue
            if 1 <= idx <= len(choices):
                return choices[idx - 1]
            self.console.print("[yellow]Enter a valid number.[/yellow]")


class QuestionaryPrompter:
    """Arrow-key-navigable prompter backed by questionary."""

    def ask(self, prompt: str, *, default: str | None = None) -> str:
        return ask_text(prompt, default)

    def confirm(self, prompt: str, *, default: bool = False) -> bool:
        return ask_confirm(prompt, default)

    def select(self, prompt: str, choices: list[str], *, default: str | None = None) -> str:
        result = questionary.select(prompt, choices=choices, default=default).ask()
        return result if result is not None else (default or choices[0])


class TerminalMenuPrompter:
    """TerminalMenu-backed prompter with search, scroll, and status bar."""

    def ask(self, prompt: str, *, default: str | None = None) -> str:
        return ask_text(prompt, default, unsafe=True)

    def confirm(self, prompt: str, *, default: bool = False) -> bool:
        return ask_confirm(prompt, default, unsafe=True)

    def select(self, prompt: str, choices: list[str], *, default: str | None = None) -> str:
        return self.menu(prompt, choices, default=default)[0]

    def menu(
        self,
        prompt: str,
        choices: list[str],
        *,
        default: str | None = None,
        extra_accept_keys: tuple[str, ...] = (),
        status: str | None = None,
        skip_indices: list[int] | None = None,
    ) -> tuple[str, str | None]:
        """Show a TerminalMenu. Returns ``(chosen_entry_or_CANCEL, accept_key)``."""
        try:
            from simple_term_menu import TerminalMenu  # type: ignore[import-untyped]
        except (ImportError, OSError):
            return QuestionaryPrompter().select(prompt, choices, default=default), "enter"

        cursor_idx = 0
        if default and default in choices:
            cursor_idx = choices.index(default)

        searchable = len(choices) > 8
        accept = ("enter",) + extra_accept_keys
        if status is None:
            parts = []
            if searchable:
                parts.append("/ search")
            for key in extra_accept_keys:
                if key == "s":
                    parts.append("s sort")
            parts.extend(["Enter select", "Esc back"])
            status = "  " + " | ".join(parts)

        try:
            tm = TerminalMenu(
                choices,
                title=f"\n  {prompt}\n",
                cursor_index=cursor_idx,
                accept_keys=accept,
                quit_keys=("escape",),
                search_key="/",
                show_search_hint=False,
                status_bar=status,
                clear_screen=True,
                menu_cursor="  ",
                menu_highlight_style=("fg_cyan", "bold"),
                skip_empty_entries=True,
            )
            if skip_indices:
                tm._skip_indices = skip_indices
            chosen = tm.show()
        except OSError:
            return QuestionaryPrompter().select(prompt, choices, default=default), "enter"

        if chosen is None:
            return CANCEL, None
        return choices[chosen], tm.chosen_accept_key


# ---------------------------------------------------------------------------
# Utilities: relative time, model truncation, language abbreviation, sorting
# ---------------------------------------------------------------------------


def relative_time(iso_str: str) -> str:
    """Convert an ISO timestamp string to a human-friendly relative time."""
    if not iso_str or iso_str == "interrupted":
        return "-"
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return "-"
    now = datetime.now(timezone.utc)
    delta = now - dt
    seconds = int(delta.total_seconds())
    if seconds < 0:
        return "just now"
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days < 30:
        return f"{days}d ago"
    months = days // 30
    return f"{months}mo ago"


def truncate_model_name(model_id: str, max_width: int) -> str:
    """Smart-truncate a model ID: strip provider first, then trim from beginning."""
    if not model_id or model_id == "-":
        return "-"
    if len(model_id) <= max_width:
        return model_id
    # Strip provider prefix
    if "/" in model_id:
        short = model_id.split("/", 1)[1]
    else:
        short = model_id
    if len(short) <= max_width:
        return short
    # Truncate from the beginning
    if max_width <= 3:
        return short[-max_width:]
    return "..." + short[-(max_width - 3) :]


def language_abbrev(language: str | None) -> str:
    """Convert a language name to a 2-letter abbreviation."""
    if not language:
        return "--"
    return language[:2].lower()


def terminal_width() -> int:
    """Get current terminal width."""
    return shutil.get_terminal_size().columns


# ---------------------------------------------------------------------------
# Prompt helpers (generate / resume flows)
# ---------------------------------------------------------------------------


def prompt_resume_overrides(console: Console, prompter: Prompter, saved_defaults: dict[str, Any]) -> dict[str, Any]:
    """Collect optional resume overrides from interactive prompts."""
    console.print("[dim]Press Enter to keep the saved value. For providers, type 'none' to clear it.[/dim]")
    get = saved_defaults.get
    return {
        "language": _prompt_text(prompter, "Language", get("language"), return_none_on_blank=True),
        "ai_model": _prompt_text(prompter, "AI model", get("ai_model"), return_none_on_blank=True),
        "human_model": _prompt_text(prompter, "Human model", get("human_model"), return_none_on_blank=True),
        "ai_provider": _prompt_provider(prompter, "AI provider", get("ai_provider"), return_none_on_blank=True),
        "human_provider": _prompt_provider(
            prompter, "Human provider", get("human_provider"), return_none_on_blank=True
        ),
        "ai_temperature": _prompt_float(
            console, prompter, "AI temperature", get("ai_temperature"), return_none_on_blank=True
        ),
        "human_temperature": _prompt_float(
            console, prompter, "Human temperature", get("human_temperature"), return_none_on_blank=True
        ),
        "ai_max_tokens": _prompt_int(
            console, prompter, "AI max tokens", get("ai_max_tokens"), return_none_on_blank=True
        ),
        "human_max_tokens": _prompt_int(
            console, prompter, "Human max tokens", get("human_max_tokens"), return_none_on_blank=True
        ),
        "ai_rpm_limit": _prompt_optional_positive_int(
            console, prompter, "AI RPM limit", get("ai_rpm_limit"), return_none_on_blank=True
        ),
        "human_rpm_limit": _prompt_optional_positive_int(
            console, prompter, "Human RPM limit", get("human_rpm_limit"), return_none_on_blank=True
        ),
    }


def prompt_generate_args(console: Console, prompter: Prompter, *, profile: str = "0") -> argparse.Namespace:
    """Collect arguments for a fresh generation flow."""
    defaults = _default_generate_values()
    console.print("[dim]Enter to keep default, or type a model ID / value.[/dim]")
    return argparse.Namespace(
        profile=profile,
        target_tokens=_prompt_int(console, prompter, "Target tokens", defaults["target_tokens"]),
        language=_prompt_text(prompter, "Language", defaults["language"], keep_blank=False),
        companion_mode=_prompt_choice(console, prompter, "Companion mode", ["honest"], defaults["companion_mode"]),
        verbose=prompter.confirm("Verbose output?", default=True),
        ai_model=_prompt_text(prompter, "AI model", defaults["ai_model"], keep_blank=False),
        human_model=_prompt_text(prompter, "Human model", defaults["human_model"], keep_blank=False),
        ai_provider=_prompt_provider(prompter, "AI provider", defaults["ai_provider"], blank_means_none=True),
        human_provider=_prompt_provider(prompter, "Human provider", defaults["human_provider"], blank_means_none=True),
        ai_temperature=_prompt_float(console, prompter, "AI temperature", defaults["ai_temperature"]),
        human_temperature=_prompt_float(console, prompter, "Human temperature", defaults["human_temperature"]),
        ai_max_tokens=_prompt_int(console, prompter, "AI max tokens", defaults["ai_max_tokens"]),
        human_max_tokens=_prompt_int(console, prompter, "Human max tokens", defaults["human_max_tokens"]),
        ai_rpm_limit=_prompt_optional_positive_int(console, prompter, "AI RPM limit", defaults["ai_rpm_limit"]),
        human_rpm_limit=_prompt_optional_positive_int(
            console, prompter, "Human RPM limit", defaults["human_rpm_limit"]
        ),
    )


def default_target_tokens(current_tokens: int) -> int:
    """Return the suggested continuation target token count."""
    suggested = current_tokens + 5_000
    return ((suggested + 2_500) // 5_000) * 5_000


def format_value(value: Any) -> str:
    """Format a config value for display in the interactive UI."""
    if value is None:
        return "-"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value == "":
        return "auto"
    return str(value)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _default_generate_values() -> dict[str, Any]:
    ai_model = default_model_for("ai") or AI_MODEL
    human_model = default_model_for("human") or HUMAN_MODEL
    ai_preset = resolve_model_preset(ai_model, "ai")
    human_preset = resolve_model_preset(human_model, "human")
    return {
        "target_tokens": TARGET_TOKENS,
        "language": "english",
        "companion_mode": COMPANION_MODE,
        "ai_model": ai_model,
        "human_model": human_model,
        "ai_provider": _provider_default(ai_preset.provider, AI_PROVIDER),
        "human_provider": _provider_default(human_preset.provider, HUMAN_PROVIDER),
        "ai_temperature": _preset_default(ai_preset.temperature, AI_TEMPERATURE),
        "human_temperature": _preset_default(human_preset.temperature, HUMAN_TEMPERATURE),
        "ai_max_tokens": _preset_default(ai_preset.max_tokens, AI_MAX_TOKENS),
        "human_max_tokens": _preset_default(human_preset.max_tokens, HUMAN_MAX_TOKENS),
        "ai_rpm_limit": _preset_default(ai_preset.rpm_limit, None),
        "human_rpm_limit": _preset_default(human_preset.rpm_limit, None),
    }


def _preset_default(value: Any, fallback: Any) -> Any:
    return fallback if value is MISSING else value


def _provider_default(value: Any, fallback: Any) -> str | None:
    provider = fallback if value is MISSING else value
    if isinstance(provider, dict):
        only = provider.get("only") or []
        return only[0] if only else None
    return provider


def _prompt_choice(console: Console, prompter: Prompter, label: str, choices: list[str], default: str) -> str:
    del console
    result = prompter.select(label, choices, default=default)
    if result == CANCEL:
        raise PromptBack
    return result


def _prompt_text(
    prompter: Prompter,
    label: str,
    default: Any,
    *,
    keep_blank: bool = True,
    return_none_on_blank: bool = False,
) -> str | None:
    if return_none_on_blank:
        raw = prompter.ask(_current_value_label(label, default)).strip()
        return raw or None
    raw = prompter.ask(label, default=_default_text(default)).strip()
    if raw:
        return raw
    return None if keep_blank else _default_text(default)


def _prompt_provider(
    prompter: Prompter,
    label: str,
    default: Any,
    *,
    blank_means_none: bool = False,
    return_none_on_blank: bool = False,
) -> str | None:
    provider_default = _default_text(_provider_default(default, default))
    if return_none_on_blank:
        raw = prompter.ask(_current_value_label(label, provider_default)).strip()
    else:
        raw = prompter.ask(label, default=provider_default).strip()
    if not raw:
        return None if blank_means_none else None
    if raw.lower() == "none":
        return ""
    return raw


def _prompt_float(
    console: Console,
    prompter: Prompter,
    label: str,
    default: Any,
    *,
    return_none_on_blank: bool = False,
) -> float | None:
    while True:
        raw = _prompt_numeric_value(prompter, label, default, return_none_on_blank=return_none_on_blank)
        if raw is None:
            return None
        try:
            return float(raw)
        except ValueError:
            console.print("[yellow]Enter a number or leave it blank.[/yellow]")


def _prompt_int(
    console: Console,
    prompter: Prompter,
    label: str,
    default: Any,
    *,
    return_none_on_blank: bool = False,
) -> int | None:
    while True:
        raw = _prompt_numeric_value(prompter, label, default, return_none_on_blank=return_none_on_blank)
        if raw is None:
            return None
        try:
            return int(raw)
        except ValueError:
            console.print("[yellow]Enter an integer value.[/yellow]")


def _prompt_optional_positive_int(
    console: Console,
    prompter: Prompter,
    label: str,
    default: Any,
    *,
    return_none_on_blank: bool = False,
) -> int | None:
    while True:
        raw = _prompt_numeric_value(prompter, label, default, return_none_on_blank=return_none_on_blank)
        if raw is None:
            return None
        if raw == "":
            return None
        try:
            value = int(raw)
        except ValueError:
            console.print("[yellow]Enter a positive integer or leave it blank.[/yellow]")
            continue
        if value <= 0:
            console.print("[yellow]Enter a value greater than zero.[/yellow]")
            continue
        return value


def _prompt_numeric_value(
    prompter: Prompter,
    label: str,
    default: Any,
    *,
    return_none_on_blank: bool,
) -> str | None:
    if return_none_on_blank:
        raw = prompter.ask(_current_value_label(label, default)).strip()
        return raw or None
    return prompter.ask(label, default=_default_text(default)).strip()


def _default_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return _provider_default(value, value)
    return str(value)


def _current_value_label(label: str, value: Any) -> str:
    text = _default_text(value)
    if text is None:
        return f"{label} [current: unset]"
    return f"{label} [current: {text}]"
