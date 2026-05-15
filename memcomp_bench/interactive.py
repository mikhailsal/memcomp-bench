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
    render_run_detail,
    render_summary_header,
)
from memcomp_bench._interactive_prompts import (
    SORT_ORDERS,
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

MODE_NEW = "New generation"
MODE_RESUME = "Resume a run"
MODE_VIEW = "View saved runs"
MODE_QUIT = "Quit"
MAIN_ACTIONS = [MODE_NEW, MODE_RESUME, MODE_VIEW, MODE_QUIT]
BACK_LABEL = "\u2190 Back"


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
    """Run a single interactive CLI session with three-way mode selection."""
    active_console = console or Console()
    active_prompter = prompter or TerminalMenuPrompter()
    summaries = scan_saved_conversations(output_dir)

    action = _prompt_main_action(active_console, active_prompter, summaries)
    if action == "resume":
        _run_resume_flow(active_console, active_prompter, summaries, resume_handler)
    elif action == "generate":
        _run_generate_flow(active_console, active_prompter, generate_handler)
    elif action == "view":
        _run_view_flow(active_console, active_prompter, summaries)


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


def format_run_choice(summary: SavedConversationSummary) -> str:
    """Format a saved run as a display string for the selection menu."""
    ai_model = summary.effective_config.get("ai_model") or "-"
    human_model = summary.effective_config.get("human_model") or "-"
    from memcomp_bench._interactive_prompts import truncate_model_name

    ai_short = truncate_model_name(ai_model, 20)
    hu_short = truncate_model_name(human_model, 20)
    return (
        f"{summary.profile_name}  \u2014  {summary.total_tokens_estimate:,} tokens"
        f"  \u2014  {summary.total_turns} turns  \u2014  {ai_short} / {hu_short}"
    )


def sort_summaries(summaries: list[SavedConversationSummary], order: str) -> list[SavedConversationSummary]:
    """Sort summaries according to the selected sort order."""
    if order == "Oldest first":
        return sorted(summaries, key=lambda s: s.started_at or "")
    if order == "Most tokens":
        return sorted(summaries, key=lambda s: s.total_tokens_estimate, reverse=True)
    if order == "Fewest tokens":
        return sorted(summaries, key=lambda s: s.total_tokens_estimate)
    if order == "Most turns":
        return sorted(summaries, key=lambda s: s.total_turns, reverse=True)
    if order == "By profile (A-Z)":
        return sorted(summaries, key=lambda s: s.profile_name.lower())
    # Default: newest first
    return sorted(summaries, key=lambda s: s.started_at or "", reverse=True)


# ---------------------------------------------------------------------------
# Flow: main action
# ---------------------------------------------------------------------------


def _prompt_main_action(console: Console, prompter: Prompter, summaries: list[SavedConversationSummary]) -> str:
    if summaries:
        total_tokens = sum(s.total_tokens_estimate for s in summaries)
        resumable = sum(1 for s in summaries if s.resumable)
        console.print(
            f"\n  [bold]{len(summaries)}[/bold] saved runs "
            f"| [bold]{total_tokens:,}[/bold] tokens "
            f"| [bold]{resumable}[/bold] resumable\n"
        )
    choices = MAIN_ACTIONS if summaries else [MODE_NEW, MODE_QUIT]
    result = prompter.select("What would you like to do?", choices)
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
    sort_order = _prompt_sort_order(prompter)
    sorted_runs = sort_summaries(summaries, sort_order)
    while True:
        render_summary_header(console, sorted_runs, sort_order)
        summary = _prompt_run_picker(console, prompter, sorted_runs)
        if summary is None:
            return
        render_run_detail(console, summary)
        prompter.ask("Press Enter to return to list", default="")


# ---------------------------------------------------------------------------
# Flow: resume
# ---------------------------------------------------------------------------


def _run_resume_flow(
    console: Console,
    prompter: Prompter,
    summaries: list[SavedConversationSummary],
    resume_handler: Any,
) -> None:
    if not summaries:
        console.print("[yellow]No saved generations were found.[/yellow]")
        return
    sort_order = _prompt_sort_order(prompter)
    sorted_runs = sort_summaries(summaries, sort_order)
    render_summary_header(console, sorted_runs, sort_order)
    summary = _prompt_run_picker(console, prompter, sorted_runs)
    if summary is None:
        return
    if not summary.resumable:
        console.print("[bold red]Cannot resume: raw AI context file is missing.[/bold red]")
        return
    render_run_detail(console, summary)
    mode = _prompt_resume_mode(console, prompter)
    if mode == "cancel":
        return
    overrides = {} if mode == "saved" else prompt_resume_overrides(console, prompter, summary.saved_defaults)
    persist_defaults = mode == "edited" and prompter.confirm(
        "Persist the edited values as future resume defaults?",
        default=False,
    )
    target_tokens = _prompt_target_tokens(console, prompter, summary.total_tokens_estimate)
    resume_handler(_build_resume_args(summary.jsonl_path, target_tokens, overrides, persist_defaults))


# ---------------------------------------------------------------------------
# Flow: generate
# ---------------------------------------------------------------------------


def _run_generate_flow(console: Console, prompter: Prompter, generate_handler: Any) -> None:
    profile_choices = [f"{i}  {p['name']}" for i, p in enumerate(HUMAN_PROFILES)]
    selected = prompter.select("Select a human profile", profile_choices)
    profile_idx = selected.split()[0]
    generate_handler(prompt_generate_args(console, prompter, profile=profile_idx))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _prompt_sort_order(prompter: Prompter) -> str:
    return prompter.select("Sort order", SORT_ORDERS, default="Newest first")


def _prompt_run_picker(
    console: Console,
    prompter: Prompter,
    summaries: list[SavedConversationSummary],
) -> SavedConversationSummary | None:
    tw = terminal_width()
    choices = [format_run_line(i + 1, len(summaries), s, tw) for i, s in enumerate(summaries)]
    choices.append(BACK_LABEL)
    result = prompter.select("Select a run", choices)
    if result == BACK_LABEL:
        return None
    idx = choices.index(result)
    return summaries[idx]


def _prompt_resume_mode(console: Console, prompter: Prompter) -> str:
    choices = ["Continue with saved defaults", "Edit defaults before continuing", "Cancel"]
    result = prompter.select("How would you like to continue?", choices)
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
