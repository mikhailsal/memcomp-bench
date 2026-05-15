"""Display helpers for the interactive benchmark CLI: run list formatting and detail panels."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from memcomp_bench._interactive_prompts import (
    format_value,
    language_abbrev,
    relative_time,
    terminal_width,
    truncate_model_name,
)


def format_run_line(index: int, total: int, summary: Any, width: int | None = None) -> str:
    """Format a single run as a fixed-width line for the TerminalMenu picker."""
    tw = width or terminal_width()
    idx_w = len(str(total))
    idx_str = str(index).rjust(idx_w)

    profile = summary.profile_name[:10].ljust(10)
    tokens = f"{summary.total_tokens_estimate:,}".rjust(9)
    turns = str(summary.total_turns).rjust(5)
    lang = language_abbrev(summary.effective_config.get("language"))

    ai_model = summary.effective_config.get("ai_model") or "-"
    human_model = summary.effective_config.get("human_model") or "-"
    age = relative_time(summary.finished_at or summary.started_at)

    if tw >= 120:
        ai_col = truncate_model_name(ai_model, 22)
        hu_col = truncate_model_name(human_model, 22)
        return f"{idx_str}. {profile} {tokens} {turns}  {lang}  {ai_col:<22s}  {hu_col:<22s}  {age}"
    if tw >= 100:
        ai_col = truncate_model_name(ai_model, 18)
        hu_col = truncate_model_name(human_model, 18)
        return f"{idx_str}. {profile} {tokens} {turns}  {lang}  {ai_col:<18s}  {hu_col:<18s}  {age}"
    if tw >= 80:
        ai_col = truncate_model_name(ai_model, 20)
        return f"{idx_str}. {profile} {tokens} {turns}  {lang}  {ai_col:<20s}  {age}"
    # Minimal
    ai_col = truncate_model_name(ai_model, 14)
    return f"{idx_str}. {profile} {tokens} {turns}  {ai_col}"


def render_summary_header(console: Console, summaries: list[Any], sort_label: str) -> None:
    """Print a compact summary line above the run list."""
    total_tokens = sum(s.total_tokens_estimate for s in summaries)
    resumable = sum(1 for s in summaries if s.resumable)
    tokens_display = f"{total_tokens:,}"
    console.print(
        f"  [bold]{len(summaries)}[/bold] saved runs "
        f"| [bold]{tokens_display}[/bold] total tokens "
        f"| [bold]{resumable}[/bold] resumable "
        f"| sorted: [dim]{sort_label}[/dim]"
    )


def render_run_detail(console: Console, summary: Any) -> None:
    """Render a comprehensive detail panel for a single run."""
    tw = terminal_width()
    panel_width = min(tw - 4, 100)

    sections: list[str] = []
    sections.append(_section_general(summary))
    sections.append(_section_conversation(summary))
    sections.append(_section_ai_model(summary))
    sections.append(_section_human_model(summary))

    if summary.saved_defaults != summary.effective_config:
        sections.append(_section_resume_defaults(summary))

    body = "\n".join(sections)
    panel = Panel(
        body,
        title=f"Run Details: {summary.profile_name}",
        width=panel_width,
        padding=(1, 2),
    )
    console.print(panel)


def _section_general(summary: Any) -> str:
    """Build the General section of the detail panel."""
    file_size = _file_size_str(summary.jsonl_path)
    started_rel = relative_time(summary.started_at) if summary.started_at else "-"
    finished_rel = relative_time(summary.finished_at) if summary.finished_at else "-"
    duration = _duration_str(summary.started_at, summary.finished_at)
    status = "Ready to resume" if summary.resumable else "Missing raw context"
    lines = [
        "[bold]General[/bold]",
        f"  File:      {summary.jsonl_path.name}  ({file_size})",
        f"  Started:   {summary.started_at or '-'}  ({started_rel})",
        f"  Finished:  {summary.finished_at or '-'}  (duration: {duration})",
        f"  Status:    {status}",
    ]
    return "\n".join(lines)


def _section_conversation(summary: Any) -> str:
    """Build the Conversation section of the detail panel."""
    cost = summary.effective_config.get("total_cost_usd")
    cost_str = f"${cost:.4f}" if cost else "-"
    lang = summary.effective_config.get("language", "-")
    lines = [
        "",
        "[bold]Conversation[/bold]",
        f"  Profile:   {summary.profile_name}",
        f"  Language:  {lang}",
        f"  Tokens:    {summary.total_tokens_estimate:,}",
        f"  Turns:     {summary.total_turns}",
        f"  Cost:      {cost_str}",
    ]
    return "\n".join(lines)


def _section_ai_model(summary: Any) -> str:
    """Build the AI Model section of the detail panel."""
    cfg = summary.effective_config
    lines = [
        "",
        "[bold]AI Model[/bold]",
        f"  Model:       {format_value(cfg.get('ai_model'))}",
        f"  Provider:    {format_value(cfg.get('ai_provider'))}",
        f"  Temperature: {format_value(cfg.get('ai_temperature'))}",
        f"  Max tokens:  {format_value(cfg.get('ai_max_tokens'))}",
        f"  RPM limit:   {format_value(cfg.get('ai_rpm_limit'))}",
        f"  Reasoning:   {format_value(cfg.get('ai_reasoning'))}",
    ]
    return "\n".join(lines)


def _section_human_model(summary: Any) -> str:
    """Build the Human Simulator section of the detail panel."""
    cfg = summary.effective_config
    lines = [
        "",
        "[bold]Human Simulator[/bold]",
        f"  Model:       {format_value(cfg.get('human_model'))}",
        f"  Provider:    {format_value(cfg.get('human_provider'))}",
        f"  Temperature: {format_value(cfg.get('human_temperature'))}",
        f"  Max tokens:  {format_value(cfg.get('human_max_tokens'))}",
        f"  RPM limit:   {format_value(cfg.get('human_rpm_limit'))}",
        f"  Reasoning:   {format_value(cfg.get('human_reasoning'))}",
    ]
    return "\n".join(lines)


def _section_resume_defaults(summary: Any) -> str:
    """Build the Resume Defaults section (only shown when different from effective)."""
    sd = summary.saved_defaults
    ec = summary.effective_config
    lines = ["", "[bold]Resume Defaults[/bold] [dim](differs from last run)[/dim]"]
    for key in (
        "ai_model",
        "human_model",
        "language",
        "ai_provider",
        "human_provider",
        "ai_temperature",
        "human_temperature",
        "ai_max_tokens",
        "human_max_tokens",
        "ai_rpm_limit",
        "human_rpm_limit",
        "ai_reasoning",
        "human_reasoning",
    ):
        saved_val = sd.get(key)
        effective_val = ec.get(key)
        if saved_val != effective_val:
            label = key.replace("_", " ").title()
            lines.append(f"  {label}: {format_value(saved_val)}  [dim](was: {format_value(effective_val)})[/dim]")
    return "\n".join(lines)


def _file_size_str(path: Path) -> str:
    """Return human-readable file size."""
    try:
        size = path.stat().st_size
    except OSError:
        return "?"
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _duration_str(started: str | None, finished: str | None) -> str:
    """Compute duration between two ISO timestamps."""
    if not started or not finished or finished == "interrupted":
        return "-"
    try:
        start_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(finished.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return "-"
    delta = end_dt - start_dt
    total_seconds = int(delta.total_seconds())
    if total_seconds < 0:
        return "-"
    if total_seconds < 60:
        return f"{total_seconds}s"
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m"
