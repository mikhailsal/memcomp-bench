"""Interactive prompt-driven workflows for generating and resuming conversations."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from memcomp_bench._interactive_prompts import (
    Prompter,
    QuestionaryPrompter,
    default_target_tokens,
    format_value,
    prompt_generate_args,
    prompt_resume_overrides,
)
from memcomp_bench.persistence import (
    build_resume_defaults_payload,
    get_saved_resume_defaults,
    load_conversation_metadata,
)
from memcomp_bench.prompts import HUMAN_PROFILES


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
    """Run a single interactive CLI session."""
    active_console = console or Console()
    active_prompter = prompter or QuestionaryPrompter()
    summaries = scan_saved_conversations(output_dir)
    if summaries:
        _render_saved_runs(active_console, summaries)
    action = _prompt_main_action(active_console, active_prompter, has_saved_runs=bool(summaries))
    if action == "resume":
        _run_resume_flow(active_console, active_prompter, summaries, resume_handler)
        return
    if action == "generate":
        _run_generate_flow(active_console, active_prompter, generate_handler)


def scan_saved_conversations(output_dir: Path) -> list[SavedConversationSummary]:
    """Read saved JSONL files from the output directory and build summaries."""
    summaries: list[SavedConversationSummary] = []
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
    status = "ready" if summary.resumable else "no context"
    ai_model = summary.effective_config.get("ai_model") or "-"
    return (
        f"{summary.profile_name}  \u2014  {summary.total_tokens_estimate:,} tokens"
        f"  \u2014  {summary.total_turns} turns  \u2014  {ai_model}  [{status}]"
    )


def _render_saved_runs(console: Console, summaries: list[SavedConversationSummary]) -> None:
    table = Table(title="Saved Generations")
    table.add_column("#", justify="right")
    table.add_column("Profile")
    table.add_column("Tokens", justify="right")
    table.add_column("Turns", justify="right")
    table.add_column("AI model")
    table.add_column("Human model")
    table.add_column("Status")
    for index, summary in enumerate(summaries, start=1):
        status = "ready" if summary.resumable else "missing raw context"
        table.add_row(
            str(index),
            summary.profile_name,
            f"{summary.total_tokens_estimate:,}",
            str(summary.total_turns),
            str(summary.effective_config.get("ai_model", "-")),
            str(summary.effective_config.get("human_model", "-")),
            status,
        )
    console.print(table)


def _prompt_main_action(console: Console, prompter: Prompter, *, has_saved_runs: bool) -> str:
    choices = ["Start a new generation", "Exit"]
    if has_saved_runs:
        choices = ["Continue a saved generation", "Start a new generation", "Exit"]
    result = prompter.select("What would you like to do?", choices)
    if result == "Continue a saved generation":
        return "resume"
    if result == "Start a new generation":
        return "generate"
    return "exit"


def _run_resume_flow(
    console: Console,
    prompter: Prompter,
    summaries: list[SavedConversationSummary],
    resume_handler: Any,
) -> None:
    if not summaries:
        console.print("[yellow]No saved generations were found.[/yellow]")
        return
    summary = _prompt_saved_run(console, prompter, summaries)
    if summary is None:
        return
    if not summary.resumable:
        console.print("[bold red]Cannot resume: raw AI context file is missing.[/bold red]")
        return
    _render_run_details(console, summary)
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


def _prompt_saved_run(
    console: Console,
    prompter: Prompter,
    summaries: list[SavedConversationSummary],
) -> SavedConversationSummary | None:
    choices = [format_run_choice(s) for s in summaries]
    choices.append("\u2190 Back")
    result = prompter.select("Select a generation to resume", choices)
    if result == "\u2190 Back":
        return None
    idx = choices.index(result)
    return summaries[idx]


def _render_run_details(console: Console, summary: SavedConversationSummary) -> None:
    lines = [
        f"Path: {summary.jsonl_path}",
        f"Started: {summary.started_at or '-'}",
        f"Finished: {summary.finished_at or '-'}",
        f"Tokens generated: {summary.total_tokens_estimate:,}",
        f"Total turns: {summary.total_turns}",
        f"AI model used: {format_value(summary.effective_config.get('ai_model'))}",
        f"Human model used: {format_value(summary.effective_config.get('human_model'))}",
        f"Language: {format_value(summary.effective_config.get('language'))}",
        f"AI provider: {format_value(summary.effective_config.get('ai_provider'))}",
        f"Human provider: {format_value(summary.effective_config.get('human_provider'))}",
        f"AI reasoning: {format_value(summary.effective_config.get('ai_reasoning'))}",
        f"Human reasoning: {format_value(summary.effective_config.get('human_reasoning'))}",
        f"AI temperature: {format_value(summary.effective_config.get('ai_temperature'))}",
        f"Human temperature: {format_value(summary.effective_config.get('human_temperature'))}",
        f"AI max tokens: {format_value(summary.effective_config.get('ai_max_tokens'))}",
        f"Human max tokens: {format_value(summary.effective_config.get('human_max_tokens'))}",
        f"AI RPM limit: {format_value(summary.effective_config.get('ai_rpm_limit'))}",
        f"Human RPM limit: {format_value(summary.effective_config.get('human_rpm_limit'))}",
    ]
    if summary.saved_defaults != summary.effective_config:
        lines.append("")
        lines.append("Future resume defaults:")
        lines.extend(_render_saved_defaults(summary.saved_defaults))
    console.print(Panel("\n".join(lines), title=f"Saved Run: {summary.profile_name}"))


def _render_saved_defaults(saved_defaults: dict[str, Any]) -> list[str]:
    return [
        f"  AI model: {format_value(saved_defaults.get('ai_model'))}",
        f"  Human model: {format_value(saved_defaults.get('human_model'))}",
        f"  Language: {format_value(saved_defaults.get('language'))}",
        f"  AI provider: {format_value(saved_defaults.get('ai_provider'))}",
        f"  Human provider: {format_value(saved_defaults.get('human_provider'))}",
        f"  AI reasoning: {format_value(saved_defaults.get('ai_reasoning'))}",
        f"  Human reasoning: {format_value(saved_defaults.get('human_reasoning'))}",
        f"  AI temperature: {format_value(saved_defaults.get('ai_temperature'))}",
        f"  Human temperature: {format_value(saved_defaults.get('human_temperature'))}",
        f"  AI max tokens: {format_value(saved_defaults.get('ai_max_tokens'))}",
        f"  Human max tokens: {format_value(saved_defaults.get('human_max_tokens'))}",
        f"  AI RPM limit: {format_value(saved_defaults.get('ai_rpm_limit'))}",
        f"  Human RPM limit: {format_value(saved_defaults.get('human_rpm_limit'))}",
    ]


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


def _run_generate_flow(console: Console, prompter: Prompter, generate_handler: Any) -> None:
    profile_choices = [f"{i}  {p['name']}" for i, p in enumerate(HUMAN_PROFILES)]
    selected = prompter.select("Select a human profile", profile_choices)
    profile_idx = selected.split()[0]
    generate_handler(prompt_generate_args(console, prompter, profile=profile_idx))
