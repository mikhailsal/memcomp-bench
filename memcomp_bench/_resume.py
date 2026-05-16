"""Resume logic for ConversationGenerator — extracted to stay under 500-line limit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console

from memcomp_bench._resume_config import _extract_resume_config
from memcomp_bench.config import (
    TARGET_TOKENS,
)
from memcomp_bench.context_hygiene import _is_restorable_ai_context, sanitize_human_tool_messages
from memcomp_bench.generator_helpers import (
    ConversationEvent,
    ConversationRecord,
    ConversationTurn,
    _append_human_user_message,
    _enforce_reasoning_before_text,
    _estimate_context_tokens,
    _migrate_assistant_reasoning_fields,
    _normalize_tool_arguments,
    _rebuild_ai_context_from_turns,
    _uses_native_reasoning_field,
)
from memcomp_bench.openrouter_client import OpenRouterClient
from memcomp_bench.persistence import build_resume_defaults_payload
from memcomp_bench.prompts import build_ai_system_prompt, set_tool_call_counter

console = Console()


def _do_resume(
    cls: type,
    client: OpenRouterClient,
    jsonl_path: str | Path,
    *,
    target_tokens: int = TARGET_TOKENS,
    verbose: bool = False,
    language_override: str | None = None,
    ai_model_override: str | None = None,
    human_model_override: str | None = None,
    ai_provider_override: object,
    human_provider_override: object,
    ai_temperature_override: float | None = None,
    human_temperature_override: float | None = None,
    ai_max_tokens_override: int | None = None,
    human_max_tokens_override: int | None = None,
    ai_rpm_limit_override: int | None = None,
    human_rpm_limit_override: int | None = None,
    persist_resume_defaults: bool = False,
) -> ConversationRecord:
    """Implementation of ConversationGenerator.resume()."""
    from memcomp_bench.generator import _UNSET

    jsonl_path = Path(jsonl_path)
    metadata, turns, events = _load_resume_files(jsonl_path)

    cfg = _extract_resume_config(
        metadata,
        ai_model_override=ai_model_override,
        human_model_override=human_model_override,
        ai_provider_override=ai_provider_override,
        human_provider_override=human_provider_override,
        ai_temperature_override=ai_temperature_override,
        human_temperature_override=human_temperature_override,
        ai_max_tokens_override=ai_max_tokens_override,
        human_max_tokens_override=human_max_tokens_override,
        ai_rpm_limit_override=ai_rpm_limit_override,
        human_rpm_limit_override=human_rpm_limit_override,
        _UNSET=_UNSET,
        language_override=language_override,
    )

    ai_messages = _restore_ai_context(jsonl_path, turns, cfg)

    _print_resume_header(cfg, jsonl_path, turns, target_tokens, _UNSET=_UNSET)

    max_tc, last_tc_id = _scan_tool_call_ids(ai_messages)
    set_tool_call_counter(max_tc)

    gen = _build_resumed_generator(
        cls,
        client,
        cfg,
        turns,
        events,
        metadata,
        ai_messages,
        last_tc_id,
        target_tokens=target_tokens,
        verbose=verbose,
    )
    gen._record.resume_defaults = _resume_defaults_for_save(cfg, persist_resume_defaults)
    _restore_events_and_turns(gen, events, turns)

    return _start_resumed_loop(gen, turns, ai_messages)


def _resume_defaults_for_save(cfg: dict[str, Any], persist_resume_defaults: bool) -> dict[str, Any]:
    """Choose which resume defaults should be written after a continuation."""
    if persist_resume_defaults:
        return build_resume_defaults_payload(cfg)
    return cfg["saved_resume_defaults"]


def _start_resumed_loop(gen: Any, turns: list, ai_messages: list) -> ConversationRecord:
    """Compute starting position and launch the conversation loop."""
    last_turn = turns[-1]["turn_number"] if turns else 0
    last_ai_turn = next((t for t in reversed(turns) if t.get("speaker") == "ai"), None)
    stored_ctx = last_ai_turn.get("ai_context_tokens", 0) if last_ai_turn else 0
    existing_tokens = stored_ctx or _estimate_context_tokens(ai_messages)

    console.print(f"  Existing tokens: ~{existing_tokens:,}")
    console.print()

    return gen._run_loop(start_turn=last_turn, start_tokens=existing_tokens)


def _load_resume_files(jsonl_path: Path) -> tuple[dict, list, list]:
    """Load and validate the JSONL + raw context files for resume."""
    raw_json_path = jsonl_path.parent / f"{jsonl_path.stem}_raw_ai_context.json"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
    if not raw_json_path.exists():
        raise FileNotFoundError(f"Raw AI context not found: {raw_json_path}")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    metadata = lines[0]
    turns = [l for l in lines[1:] if l.get("type") == "turn"]
    events = [l for l in lines[1:] if l.get("type") == "event"]
    return metadata, turns, events


def _resolve_resume_value(metadata: dict, key: str, default: Any, override: Any, *, unset: object | None = None) -> Any:
    """Return the saved config value unless an explicit override was provided."""
    if unset is not None and override is unset:
        override = None
    if override is not None:
        return override
    if key in metadata:
        return metadata[key]
    return default


def _restore_ai_context(jsonl_path: Path, turns: list, cfg: dict) -> list[dict[str, Any]]:
    """Load raw AI context from disk, rebuilding/normalizing as needed."""
    raw_json_path = jsonl_path.parent / f"{jsonl_path.stem}_raw_ai_context.json"
    with open(raw_json_path, "r", encoding="utf-8") as f:
        ai_messages = json.load(f)

    if not _is_restorable_ai_context(ai_messages):
        ai_messages = _rebuild_ai_context_from_turns(
            build_ai_system_prompt(cfg["seed_words"], companion_mode=cfg["companion_mode"]),
            turns,
            use_reasoning_field=_uses_native_reasoning_field(cfg["ai_reasoning"]),
        )
        console.print("  [dim yellow]Raw AI context missing or incomplete; rebuilt from saved turns.[/dim yellow]")

    migrated = _migrate_assistant_reasoning_fields(
        ai_messages,
        use_reasoning_field=_uses_native_reasoning_field(cfg["ai_reasoning"]),
    )
    if migrated > 0:
        console.print(f"  [dim]Migrated {migrated} assistant messages from content to reasoning.[/dim]")

    chars_saved = _normalize_tool_arguments(ai_messages)
    if chars_saved > 0:
        console.print(
            f"  [dim]Normalized Unicode escapes in context (saved ~{chars_saved:,} chars / ~{chars_saved // 4:,} tokens)[/dim]"
        )

    reordered = _enforce_reasoning_before_text(ai_messages)
    if reordered > 0:
        console.print(f"  [dim]Fixed reasoning/text order in {reordered} tool call(s)[/dim]")

    sanitized = sanitize_human_tool_messages(ai_messages)
    if sanitized > 0:
        console.print(f"  [dim]Sanitized {sanitized} human tool message(s) in AI context.[/dim]")

    return ai_messages


def _scan_tool_call_ids(ai_messages: list[dict[str, Any]]) -> tuple[int, str | None]:
    """Scan AI context for the highest tool call counter and last tool call id."""
    max_tc = 0
    last_tc_id = None
    for msg in ai_messages:
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id", "")
                last_tc_id = tc_id
                if tc_id.startswith("wmth"):
                    try:
                        max_tc = max(max_tc, int(tc_id[4:]))
                    except ValueError:
                        pass
        if msg.get("tool_call_id"):
            last_tc_id = msg["tool_call_id"]
    return max_tc, last_tc_id


def _build_resumed_generator(
    cls: type,
    client: OpenRouterClient,
    cfg: dict,
    turns: list,
    events: list,
    metadata: dict,
    ai_messages: list[dict[str, Any]],
    last_tc_id: str | None,
    *,
    target_tokens: int,
    verbose: bool,
) -> Any:
    """Instantiate a ConversationGenerator and restore its internal state."""
    gen = cls(
        client,
        cfg["profile"],
        ai_model=cfg["ai_model"],
        human_model=cfg["human_model"],
        target_tokens=target_tokens,
        language=cfg["language"],
        companion_mode=cfg["companion_mode"],
        verbose=verbose,
        ai_provider=cfg["ai_provider"],
        ai_reasoning=cfg["ai_reasoning"],
        ai_temperature=cfg["ai_temperature"],
        ai_max_tokens=cfg["ai_max_tokens"],
        ai_rpm_limit=cfg["ai_rpm_limit"],
        human_provider=cfg["human_provider"],
        human_reasoning=cfg["human_reasoning"],
        human_temperature=cfg["human_temperature"],
        human_max_tokens=cfg["human_max_tokens"],
        human_rpm_limit=cfg["human_rpm_limit"],
    )

    client.total_cost = cfg["previous_cost"]
    gen._seed_words = cfg["seed_words"]
    gen._conversation_plan = cfg["conversation_plan"]
    gen._ai_messages = ai_messages
    gen._last_tool_call_id = last_tc_id

    topic_events = [e for e in events if e.get("event_type") == "topic_judge"]
    if topic_events:
        gen._current_topic = topic_events[-1].get("current_topic")
    nudge_events = [e for e in events if e.get("event_type") == "human_nudge" and e.get("nudge_injected")]
    if nudge_events:
        gen._last_human_nudge_turn = max(e.get("turn_number", 0) for e in nudge_events)

    gen._init_human_context()
    _rebuild_human_context(gen, turns, events)

    gen._record.id = metadata["conversation_id"]
    gen._record.seed_words = cfg["seed_words"]
    gen._record.conversation_plan = cfg["conversation_plan"]
    gen._record.language = cfg["language"]
    gen._record.companion_mode = cfg["companion_mode"]
    gen._record.started_at = metadata["started_at"]
    return gen


def _print_resume_header(
    cfg: dict,
    jsonl_path: Path,
    turns: list,
    target_tokens: int,
    *,
    _UNSET: object,
) -> None:
    """Print the resume status header to console."""
    _ov = " [yellow](overridden)[/yellow]"

    console.print(f"\n[bold]Resuming conversation with {cfg['profile']['name']}[/bold]")
    console.print(f"  From: {jsonl_path.name}")
    console.print(f"  Existing turns: {len(turns)}")
    console.print(f"  AI model: {cfg['ai_model']}" + (_ov if cfg.get("ai_model_override") else ""))
    console.print(f"  Human model: {cfg['human_model']}" + (_ov if cfg.get("human_model_override") else ""))
    if cfg["ai_provider"]:
        console.print(
            f"  AI provider: {cfg['ai_provider']}" + (_ov if cfg.get("ai_provider_override") is not _UNSET else "")
        )
    if cfg["ai_reasoning"]:
        console.print(f"  AI reasoning: {cfg['ai_reasoning']}")
    if cfg["human_provider"]:
        console.print(
            f"  Human provider: {cfg['human_provider']}"
            + (_ov if cfg.get("human_provider_override") is not _UNSET else "")
        )
    if cfg["human_reasoning"]:
        console.print(f"  Human reasoning: {cfg['human_reasoning']}")
    temp_line = f"  AI temperature: {cfg['ai_temperature']}" + (
        _ov if cfg.get("ai_temperature_override") is not None else ""
    )
    temp_line += f" / Human temperature: {cfg['human_temperature']}" + (
        _ov if cfg.get("human_temperature_override") is not None else ""
    )
    console.print(temp_line)
    tok_line = f"  AI max tokens: {cfg['ai_max_tokens']}" + (
        _ov if cfg.get("ai_max_tokens_override") is not None else ""
    )
    tok_line += f" / Human max tokens: {cfg['human_max_tokens']}" + (
        _ov if cfg.get("human_max_tokens_override") is not None else ""
    )
    console.print(tok_line)
    rpm_line = f"  AI RPM limit: {cfg['ai_rpm_limit']}" + (_ov if cfg.get("ai_rpm_limit_override") is not None else "")
    rpm_line += f" / Human RPM limit: {cfg['human_rpm_limit']}" + (
        _ov if cfg.get("human_rpm_limit_override") is not None else ""
    )
    console.print(rpm_line)
    console.print(f"  Previous cost: ${cfg['previous_cost']:.4f}")
    console.print(f"  New target: ~{target_tokens:,} tokens")
    console.print(f"  Seed: {', '.join(cfg['seed_words'])}")
    console.print(f"  Language: {cfg['language']}")


def _rebuild_human_context(gen: Any, turns: list, events: list) -> None:
    """Rebuild the human-side context from saved turns and events."""
    nudges_by_turn: dict[int, list[str]] = {}
    for event in events:
        if event.get("event_type") != "human_nudge":
            continue
        if not event.get("nudge_injected"):
            continue
        message = event.get("message")
        turn_number = event.get("turn_number")
        if not message or not isinstance(turn_number, int):
            continue
        nudges_by_turn.setdefault(turn_number, []).append(message)

    for turn_data in turns:
        speaker = turn_data["speaker"]
        text = turn_data["visible_text"]
        if speaker == "human":
            human_msg: dict[str, Any] = {"role": "assistant", "content": text}
            h_reasoning = turn_data.get("human_reasoning")
            if h_reasoning:
                human_msg["reasoning"] = h_reasoning
            h_reasoning_details = turn_data.get("human_reasoning_details")
            if h_reasoning_details:
                human_msg["reasoning_details"] = h_reasoning_details
            gen._human_messages.append(human_msg)
        else:
            _append_human_user_message(gen._human_messages, text)
            for note in nudges_by_turn.get(turn_data["turn_number"], []):
                _append_human_user_message(gen._human_messages, note)


def _restore_events_and_turns(gen: Any, events: list, turns: list) -> None:
    """Restore events and turns from saved data into the generator record."""
    for event_data in events:
        gen._record.events.append(
            ConversationEvent(
                event_type=event_data.get("event_type", "unknown"),
                turn_number=event_data.get("turn_number", 0),
                source=event_data.get("source", "unknown"),
                timestamp=event_data.get("timestamp", ""),
                message=event_data.get("message"),
                previous_topic=event_data.get("previous_topic"),
                current_topic=event_data.get("current_topic"),
                topic_changed=event_data.get("topic_changed"),
                nudge_injected=event_data.get("nudge_injected"),
                suppression_reason=event_data.get("suppression_reason"),
            )
        )
    for turn_data in turns:
        gen._record.turns.append(
            ConversationTurn(
                turn_number=turn_data["turn_number"],
                speaker=turn_data["speaker"],
                visible_text=turn_data["visible_text"],
                ai_thinking=turn_data.get("ai_thinking"),
                ai_content=turn_data.get("ai_content"),
                ai_reasoning=turn_data.get("ai_reasoning"),
                ai_tool_calls=turn_data.get("ai_tool_calls"),
                ai_reasoning_details=turn_data.get("ai_reasoning_details"),
                human_reasoning=turn_data.get("human_reasoning"),
                human_reasoning_details=turn_data.get("human_reasoning_details"),
                token_estimate=turn_data.get("token_estimate", 0),
                cost_usd=turn_data.get("cost_usd", 0.0),
                timestamp=turn_data.get("timestamp", ""),
                ai_context_tokens=turn_data.get("ai_context_tokens", 0),
                human_context_tokens=turn_data.get("human_context_tokens", 0),
            )
        )
