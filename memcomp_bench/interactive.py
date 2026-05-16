"""Interactive prompt-driven workflows for generating and resuming conversations."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

from memcomp_bench._interactive_display import (
    format_run_line,
    render_run_detail_lines,
    render_summary_title,
)
from memcomp_bench._interactive_prompts import (
    CANCEL,
    SORT_ACTION,
    Prompter,
    TerminalMenuPrompter,
    default_target_tokens,
    prompt_generate_args,
    prompt_resume_overrides,
    terminal_width,
)
from memcomp_bench.persistence import (
    build_resume_defaults_payload,
    get_saved_resume_defaults,
    load_conversation_metadata,
)
from memcomp_bench.prompts import HUMAN_PROFILES

SORT_ORDERS = [
    "[1] Newest first",
    "[2] Oldest first",
    "[3] Most tokens",
    "[4] Fewest tokens",
    "[5] Most turns",
    "[6] By profile (A-Z)",
]

MODE_NEW = "[n] New generation"
MODE_RESUME_LAST = "[l] Resume last generation"
MODE_RESUME = "[r] Resume a run"
MODE_VIEW = "[v] View saved runs"
MODE_QUIT = "[q] Quit"
MAIN_ACTIONS = [MODE_NEW, MODE_RESUME_LAST, MODE_RESUME, MODE_VIEW, MODE_QUIT]

DETAIL_BACK = "[b] Back to list"  # kept for ScriptedPrompter fallback in tests


@dataclass
class SavedConversationSummary:
    """Compact metadata used by the interactive resume picker."""

    jsonl_path: Path
    raw_context_path: Path
    profile_name: str
    started_at: str
    finished_at: str
    total_tokens_estimate: int
    total_turns: int
    effective_config: dict[str, Any]
    saved_defaults: dict[str, Any]

    @property
    def resumable(self) -> bool:
        return self.raw_context_path.exists()


def run_interactive(
    generate_handler: Any,
    resume_handler: Any,
    *,
    output_dir: Path,
    console: Console | None = None,
    prompter: Prompter | None = None,
) -> None:
    """Main-menu loop: sub-flows return here, only Esc/q from main exits."""
    active_console = console or Console()
    active_prompter = prompter or TerminalMenuPrompter()

    try:
        while True:
            summaries = scan_saved_conversations(output_dir)
            latest_resumable = _latest_resumable_summary(summaries)
            action = _prompt_main_action(active_console, active_prompter, summaries, latest_resumable is not None)
            if action == "exit":
                return
            if action == "resume_last":
                if latest_resumable and _continue_resume_flow(
                    active_console, active_prompter, latest_resumable, resume_handler
                ):
                    return
            elif action == "resume":
                ran = _run_resume_flow(active_console, active_prompter, summaries, resume_handler)
                if ran:
                    return
            elif action == "generate":
                ran = _run_generate_flow(active_console, active_prompter, generate_handler)
                if ran:
                    return
            elif action == "view":
                _run_view_flow(active_console, active_prompter, summaries)
    except KeyboardInterrupt:
        return


def scan_saved_conversations(output_dir: Path) -> list[SavedConversationSummary]:
    """Read saved JSONL files from the output directory and build summaries."""
    summaries: list[SavedConversationSummary] = []
    if not output_dir.exists():
        return summaries
    for jsonl_path in sorted(output_dir.glob("conv_*.jsonl"), reverse=True):
        try:
            metadata = load_conversation_metadata(jsonl_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        raw_context_path = jsonl_path.parent / f"{jsonl_path.stem}_raw_ai_context.json"
        summaries.append(
            SavedConversationSummary(
                jsonl_path=jsonl_path,
                raw_context_path=raw_context_path,
                profile_name=metadata.get("human_profile", {}).get("name", "unknown"),
                started_at=metadata.get("started_at", ""),
                finished_at=metadata.get("finished_at", ""),
                total_tokens_estimate=metadata.get("total_tokens_estimate", 0),
                total_turns=metadata.get("total_turns", 0),
                effective_config=build_resume_defaults_payload(metadata),
                saved_defaults=get_saved_resume_defaults(metadata),
            )
        )
    return summaries


def sort_summaries(summaries: list[SavedConversationSummary], order: str) -> list[SavedConversationSummary]:
    """Sort summaries according to the selected sort order."""
    if "Oldest first" in order:
        return sorted(summaries, key=lambda s: s.started_at or "")
    if "Most tokens" in order:
        return sorted(summaries, key=lambda s: s.total_tokens_estimate, reverse=True)
    if "Fewest tokens" in order:
        return sorted(summaries, key=lambda s: s.total_tokens_estimate)
    if "Most turns" in order:
        return sorted(summaries, key=lambda s: s.total_turns, reverse=True)
    if "By profile" in order:
        return sorted(summaries, key=lambda s: s.profile_name.lower())
    return sorted(summaries, key=lambda s: s.started_at or "", reverse=True)


# ---------------------------------------------------------------------------
# Flow: main action
# ---------------------------------------------------------------------------


def _prompt_main_action(
    console: Console,
    prompter: Prompter,
    summaries: list[SavedConversationSummary],
    has_resumable: bool,
) -> str:
    choices = [MODE_NEW, MODE_QUIT]
    if summaries:
        choices = [MODE_NEW, MODE_RESUME, MODE_VIEW, MODE_QUIT]
        if has_resumable:
            choices.insert(1, MODE_RESUME_LAST)
    title = "What would you like to do?"
    if summaries:
        total_tokens = sum(s.total_tokens_estimate for s in summaries)
        resumable = sum(1 for s in summaries if s.resumable)
        title = (
            f"{len(summaries)} saved runs | {total_tokens:,} tokens"
            f" | {resumable} resumable\n\n  What would you like to do?"
        )
    result = prompter.select(title, choices)
    if result == CANCEL or result == MODE_QUIT:
        return "exit"
    if result == MODE_RESUME_LAST:
        return "resume_last"
    if result == MODE_RESUME:
        return "resume"
    if result == MODE_NEW:
        return "generate"
    if result == MODE_VIEW:
        return "view"
    return "exit"


# ---------------------------------------------------------------------------
# Flow: view saved runs (read-only browse)
# ---------------------------------------------------------------------------


def _run_view_flow(
    console: Console,
    prompter: Prompter,
    summaries: list[SavedConversationSummary],
) -> None:
    if not summaries:
        console.print("[yellow]No saved generations found.[/yellow]")
        return
    sort_order = "Newest first"
    sorted_runs = sort_summaries(summaries, sort_order)
    while True:
        result = _prompt_run_picker(prompter, sorted_runs, sort_order)
        if result == CANCEL:
            return
        if result == SORT_ACTION:
            new_order = _prompt_sort_order(prompter)
            if new_order != CANCEL:
                sort_order = new_order
                sorted_runs = sort_summaries(summaries, sort_order)
            continue
        summary = result
        action = _show_run_detail(prompter, summary)
        if action == "back":
            continue


# ---------------------------------------------------------------------------
# Flow: resume
# ---------------------------------------------------------------------------


def _run_resume_flow(
    console: Console,
    prompter: Prompter,
    summaries: list[SavedConversationSummary],
    resume_handler: Any,
) -> bool:
    """Returns True if the handler was actually invoked."""
    if not summaries:
        console.print("[yellow]No saved generations were found.[/yellow]")
        return False
    sort_order = "Newest first"
    sorted_runs = sort_summaries(summaries, sort_order)
    while True:
        result = _prompt_run_picker(prompter, sorted_runs, sort_order)
        if result == CANCEL:
            return False
        if result == SORT_ACTION:
            new_order = _prompt_sort_order(prompter)
            if new_order != CANCEL:
                sort_order = new_order
                sorted_runs = sort_summaries(summaries, sort_order)
            continue
        action = _continue_resume_flow(console, prompter, result, resume_handler)
        if action == "resumed":
            return True
        if action == "back":
            continue
        return False


def _continue_resume_flow(
    console: Console,
    prompter: Prompter,
    summary: SavedConversationSummary,
    resume_handler: Any,
) -> str:
    if not summary.resumable:
        console.print("[bold red]Cannot resume: raw AI context file is missing.[/bold red]")
        return "back"
    detail_action = _show_run_detail(prompter, summary)
    if detail_action == "cancel":
        return "back"
    mode = _prompt_resume_mode(console, prompter)
    if mode == "cancel":
        return "cancel"
    overrides = {} if mode == "saved" else prompt_resume_overrides(console, prompter, summary.saved_defaults)
    persist_defaults = mode == "edited" and prompter.confirm(
        "Persist the edited values as future resume defaults?",
        default=False,
    )
    target_tokens = _prompt_target_tokens(console, prompter, summary.total_tokens_estimate)
    resume_handler(_build_resume_args(summary.jsonl_path, target_tokens, overrides, persist_defaults))
    return "resumed"


def _latest_resumable_summary(summaries: list[SavedConversationSummary]) -> SavedConversationSummary | None:
    for summary in sort_summaries(summaries, "Newest first"):
        if summary.resumable:
            return summary
    return None


# ---------------------------------------------------------------------------
# Flow: generate
# ---------------------------------------------------------------------------


def _run_generate_flow(console: Console, prompter: Prompter, generate_handler: Any) -> bool:
    """Returns True if the handler was actually invoked."""
    profile_choices = [f"{i}  {p['name']}" for i, p in enumerate(HUMAN_PROFILES)]
    selected = prompter.select("Select a human profile", profile_choices)
    if selected == CANCEL:
        return False
    profile_idx = selected.split()[0]
    generate_handler(prompt_generate_args(console, prompter, profile=profile_idx))
    return True


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _prompt_sort_order(prompter: Prompter) -> str:
    return prompter.select("Sort order", SORT_ORDERS, default="Newest first")


def _prompt_run_picker(
    prompter: Prompter,
    summaries: list[SavedConversationSummary],
    sort_label: str,
) -> SavedConversationSummary | str:
    """Pick a run. Returns a summary, SORT_ACTION, or CANCEL."""
    tw = terminal_width()
    choices = [format_run_line(i + 1, len(summaries), s, tw) for i, s in enumerate(summaries)]
    title = render_summary_title(summaries, sort_label)

    if isinstance(prompter, TerminalMenuPrompter):
        result, accept_key = prompter.menu(
            title,
            choices,
            extra_accept_keys=("s",),
        )
        if result == CANCEL:
            return CANCEL
        if accept_key == "s":
            return SORT_ACTION
        idx = choices.index(result)
        return summaries[idx]

    result = prompter.select(title, choices)
    if result == CANCEL:
        return CANCEL
    if result == SORT_ACTION:
        return SORT_ACTION
    idx = choices.index(result)
    return summaries[idx]


def _show_run_detail(prompter: Prompter, summary: SavedConversationSummary) -> str:
    """Render run details and report whether the user went back or canceled."""
    detail_lines = render_run_detail_lines(summary)
    menu = getattr(prompter, "menu", None)

    if callable(menu):
        result, _ = menu(
            f"Run Details: {summary.profile_name}",
            detail_lines,
            status="  j/k scroll | Esc back",
        )
        if result == CANCEL:
            return "cancel"
        return "back"

    for line in detail_lines:
        print(f"  {line}")
    result = prompter.select(f"Run Details: {summary.profile_name}", [DETAIL_BACK])
    if result == CANCEL:
        return "cancel"
    return "back"


def _prompt_resume_mode(console: Console, prompter: Prompter) -> str:
    choices = ["Continue with saved defaults", "Edit defaults before continuing", "Cancel"]
    result = prompter.select("How would you like to continue?", choices)
    if result == CANCEL:
        return "cancel"
    if result == "Continue with saved defaults":
        return "saved"
    if result == "Edit defaults before continuing":
        return "edited"
    return "cancel"


def _prompt_target_tokens(console: Console, prompter: Prompter, current_tokens: int) -> int:
    suggested = default_target_tokens(current_tokens)
    while True:
        raw = prompter.ask("New target token count", default=str(suggested))
        try:
            target_tokens = int(raw)
        except ValueError:
            console.print("[yellow]Enter a positive integer token target.[/yellow]")
            continue
        if target_tokens <= current_tokens:
            console.print(f"[yellow]Target must be greater than {current_tokens:,}.[/yellow]")
            continue
        return target_tokens


def _build_resume_args(
    jsonl_path: Path,
    target_tokens: int,
    overrides: dict[str, Any],
    persist_defaults: bool,
) -> argparse.Namespace:
    return argparse.Namespace(
        file=str(jsonl_path),
        target_tokens=target_tokens,
        language=overrides.get("language"),
        ai_model=overrides.get("ai_model"),
        human_model=overrides.get("human_model"),
        ai_provider=overrides.get("ai_provider"),
        human_provider=overrides.get("human_provider"),
        ai_temperature=overrides.get("ai_temperature"),
        human_temperature=overrides.get("human_temperature"),
        ai_max_tokens=overrides.get("ai_max_tokens"),
        human_max_tokens=overrides.get("human_max_tokens"),
        ai_rpm_limit=overrides.get("ai_rpm_limit"),
        human_rpm_limit=overrides.get("human_rpm_limit"),
        verbose=False,
        persist_resume_defaults=persist_defaults,
    )
