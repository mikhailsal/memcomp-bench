"""Verbose and compact turn-logging helpers for ConversationGenerator."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.panel import Panel

from src.generator_helpers import (
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


def _log_ai_turn_verbose(turn: ConversationTurn) -> None:
    """Verbose logging for AI turns — handles all reasoning display paths."""
    native = turn.ai_reasoning
    tool_inner = _extract_tool_call_reasoning(turn)
    text_first = _tool_call_text_before_reasoning(turn)
    inline = turn.ai_content

    if native:
        console.print()
        console.print(
            Panel(
                native,
                title=f"\U0001f9e0 Native reasoning  [dim]turn {turn.turn_number}[/dim]",
                border_style="dim yellow",
                padding=(0, 1),
            )
        )

    if text_first:
        console.print(
            Panel(
                turn.visible_text,
                title=f"\U0001f916 AI  [dim]turn {turn.turn_number}[/dim]",
                border_style="green",
                padding=(0, 1),
            )
        )
        if tool_inner and tool_inner != native:
            console.print()
            console.print(
                Panel(
                    tool_inner,
                    title=f"\U0001f4ad Inner monologue (after reply)  [dim]turn {turn.turn_number}[/dim]",
                    border_style="dim magenta",
                    padding=(0, 1),
                )
            )
        if inline and inline not in (native, tool_inner):
            console.print()
            console.print(
                Panel(
                    inline,
                    title=f"\U0001f4cb Response draft (after reply)  [dim]turn {turn.turn_number}[/dim]",
                    border_style="dim cyan",
                    padding=(0, 1),
                )
            )
        if not native and not tool_inner and not inline and turn.ai_thinking:
            console.print()
            console.print(
                Panel(
                    turn.ai_thinking,
                    title=f"\U0001f9e0 AI thinking  [dim]turn {turn.turn_number}[/dim]",
                    border_style="dim yellow",
                    padding=(0, 1),
                )
            )
    else:
        shown_any = bool(native)
        if tool_inner and tool_inner != native:
            console.print()
            console.print(
                Panel(
                    tool_inner,
                    title=f"\U0001f4ad Inner monologue  [dim]turn {turn.turn_number}[/dim]",
                    border_style="dim magenta",
                    padding=(0, 1),
                )
            )
            shown_any = True
        if inline and inline not in (native, tool_inner):
            console.print()
            console.print(
                Panel(
                    inline,
                    title=f"\U0001f4cb Response draft  [dim]turn {turn.turn_number}[/dim]",
                    border_style="dim cyan",
                    padding=(0, 1),
                )
            )
            shown_any = True
        if not shown_any and turn.ai_thinking:
            thinking_display = turn.ai_thinking
            try:
                parsed = json.loads(turn.ai_thinking)
                if parsed.get("reasoning"):
                    thinking_display = f"\U0001f9e0 {parsed['reasoning']}"
                elif parsed.get("thoughts"):
                    thinking_display = f"\U0001f4ad {parsed['thoughts']}"
            except (json.JSONDecodeError, AttributeError):
                pass
            console.print()
            console.print(
                Panel(
                    thinking_display,
                    title=f"\U0001f9e0 AI thinking  [dim]turn {turn.turn_number}[/dim]",
                    border_style="dim yellow",
                    padding=(0, 1),
                )
            )
        console.print(
            Panel(
                turn.visible_text,
                title=f"\U0001f916 AI  [dim]turn {turn.turn_number}[/dim]",
                border_style="green",
                padding=(0, 1),
            )
        )
