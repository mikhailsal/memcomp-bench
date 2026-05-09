"""Verbose and compact turn-logging helpers for ConversationGenerator."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.panel import Panel

from memcomp_bench.generator_helpers import (
    ConversationTurn,
    _extract_tool_call_reasoning,
    _tool_call_text_before_reasoning,
)

console = Console()


def log_turn(gen: Any, turn: ConversationTurn) -> None:
    """Display turn info — compact by default, full panels in verbose mode."""
    if not gen.verbose:
        if turn.speaker == "human":
            label = f"[bold blue]\U0001f464 {gen.human_profile['name']}[/bold blue]"
        else:
            label = "[bold green]\U0001f916 AI[/bold green]"
        preview = turn.visible_text[:80].replace("\n", " ")
        if len(turn.visible_text) > 80:
            preview += "\u2026"
        console.print(f"  {label} t{turn.turn_number}: {preview}")
        return

    name = gen.human_profile["name"]
    if turn.speaker == "human":
        if turn.human_reasoning:
            console.print()
            console.print(
                Panel(
                    turn.human_reasoning,
                    title=f"\U0001f4ad {name} thinking  [dim]turn {turn.turn_number}[/dim]",
                    border_style="dim cyan",
                    padding=(0, 1),
                )
            )
        console.print()
        console.print(
            Panel(
                turn.visible_text,
                title=f"\U0001f464 {name}  [dim]turn {turn.turn_number}[/dim]",
                border_style="blue",
                padding=(0, 1),
            )
        )
    else:
        _log_ai_turn_verbose(turn)


def _print_reasoning_panel(content: str, title: str, style: str) -> None:
    """Print a single reasoning panel."""
    console.print()
    console.print(Panel(content, title=title, border_style=style, padding=(0, 1)))


def _log_ai_text_first(turn: ConversationTurn, native: str | None, tool_inner: str | None, inline: str | None) -> None:
    """Verbose AI display when text was transmitted before reasoning."""
    console.print(
        Panel(
            turn.visible_text,
            title=f"\U0001f916 AI  [dim]turn {turn.turn_number}[/dim]",
            border_style="green",
            padding=(0, 1),
        )
    )
    if tool_inner and tool_inner != native:
        _print_reasoning_panel(
            tool_inner, f"\U0001f4ad Inner monologue (after reply)  [dim]turn {turn.turn_number}[/dim]", "dim magenta"
        )
    if inline and inline not in (native, tool_inner):
        _print_reasoning_panel(
            inline, f"\U0001f4cb Response draft (after reply)  [dim]turn {turn.turn_number}[/dim]", "dim cyan"
        )
    if not native and not tool_inner and not inline and turn.ai_thinking:
        _print_reasoning_panel(
            turn.ai_thinking, f"\U0001f9e0 AI thinking  [dim]turn {turn.turn_number}[/dim]", "dim yellow"
        )


def _log_ai_reasoning_first(
    turn: ConversationTurn, native: str | None, tool_inner: str | None, inline: str | None
) -> None:
    """Verbose AI display when reasoning was transmitted before text."""
    shown_any = bool(native)
    if tool_inner and tool_inner != native:
        _print_reasoning_panel(
            tool_inner, f"\U0001f4ad Inner monologue  [dim]turn {turn.turn_number}[/dim]", "dim magenta"
        )
        shown_any = True
    if inline and inline not in (native, tool_inner):
        _print_reasoning_panel(inline, f"\U0001f4cb Response draft  [dim]turn {turn.turn_number}[/dim]", "dim cyan")
        shown_any = True
    if not shown_any and turn.ai_thinking:
        thinking_display = _parse_thinking_display(turn.ai_thinking)
        _print_reasoning_panel(
            thinking_display, f"\U0001f9e0 AI thinking  [dim]turn {turn.turn_number}[/dim]", "dim yellow"
        )
    console.print(
        Panel(
            turn.visible_text,
            title=f"\U0001f916 AI  [dim]turn {turn.turn_number}[/dim]",
            border_style="green",
            padding=(0, 1),
        )
    )


def _parse_thinking_display(ai_thinking: str) -> str:
    """Try to extract a structured reasoning/thoughts label from JSON thinking."""
    try:
        parsed = json.loads(ai_thinking)
        if parsed.get("reasoning"):
            return f"\U0001f9e0 {parsed['reasoning']}"
        if parsed.get("thoughts"):
            return f"\U0001f4ad {parsed['thoughts']}"
    except (json.JSONDecodeError, AttributeError):
        pass
    return ai_thinking


def _log_ai_turn_verbose(turn: ConversationTurn) -> None:
    """Verbose logging for AI turns \u2014 handles all reasoning display paths."""
    native = turn.ai_reasoning
    tool_inner = _extract_tool_call_reasoning(turn)
    text_first = _tool_call_text_before_reasoning(turn)
    inline = turn.ai_content

    if native:
        _print_reasoning_panel(native, f"\U0001f9e0 Native reasoning  [dim]turn {turn.turn_number}[/dim]", "dim yellow")

    if text_first:
        _log_ai_text_first(turn, native, tool_inner, inline)
    else:
        _log_ai_reasoning_first(turn, native, tool_inner, inline)
