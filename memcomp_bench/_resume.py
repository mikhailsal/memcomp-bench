"""Resume logic for ConversationGenerator — extracted to stay under 500-line limit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console

from memcomp_bench.config import (
    AI_MAX_TOKENS,
    AI_PROVIDER,
    AI_REASONING,
    AI_TEMPERATURE,
    HUMAN_MAX_TOKENS,
    HUMAN_PROVIDER,
    HUMAN_REASONING,
    HUMAN_TEMPERATURE,
    TARGET_TOKENS,
)
from memcomp_bench.generator_helpers import (
    ConversationEvent,
    ConversationRecord,
    ConversationTurn,
    _estimate_context_tokens,
    _is_restorable_ai_context,
    _migrate_assistant_reasoning_fields,
    _normalize_tool_arguments,
    _rebuild_ai_context_from_turns,
    _uses_native_reasoning_field,
)
from memcomp_bench.openrouter_client import OpenRouterClient
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
) -> ConversationRecord:
    """Implementation of ConversationGenerator.resume()."""
    from memcomp_bench.generator import _UNSET

    jsonl_path = Path(jsonl_path)
    base = jsonl_path.stem
    raw_json_path = jsonl_path.parent / f"{base}_raw_ai_context.json"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
    if not raw_json_path.exists():
        raise FileNotFoundError(f"Raw AI context not found: {raw_json_path}")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    metadata = lines[0]
    turns = [l for l in lines[1:] if l.get("type") == "turn"]
    events = [l for l in lines[1:] if l.get("type") == "event"]

    profile = metadata["human_profile"]
    ai_model = ai_model_override or metadata["ai_model"]
    human_model = human_model_override or metadata["human_model"]
    seed_words = metadata.get("seed_words", [])
    conversation_plan = metadata.get("conversation_plan", "")
    language = language_override or metadata.get("language", "english")
    companion_mode = metadata.get("companion_mode", "supportive")

    saved_ai_provider = metadata.get("ai_provider", AI_PROVIDER)
    ai_provider = ai_provider_override if ai_provider_override is not _UNSET else saved_ai_provider
    saved_human_provider = metadata.get("human_provider", HUMAN_PROVIDER)
    human_provider = human_provider_override if human_provider_override is not _UNSET else saved_human_provider

    ai_reasoning = metadata.get("ai_reasoning", AI_REASONING)
    human_reasoning = metadata.get("human_reasoning", HUMAN_REASONING)
    ai_temperature = metadata.get("ai_temperature", AI_TEMPERATURE)
    if ai_temperature_override is not None:
        ai_temperature = ai_temperature_override
    ai_max_tokens = metadata.get("ai_max_tokens", AI_MAX_TOKENS)
    if ai_max_tokens_override is not None:
        ai_max_tokens = ai_max_tokens_override
    human_temperature = metadata.get("human_temperature", HUMAN_TEMPERATURE)
    if human_temperature_override is not None:
        human_temperature = human_temperature_override
    human_max_tokens = metadata.get("human_max_tokens", HUMAN_MAX_TOKENS)
    if human_max_tokens_override is not None:
        human_max_tokens = human_max_tokens_override
    previous_cost = metadata.get("total_cost_usd", 0.0)

    with open(raw_json_path, "r", encoding="utf-8") as f:
        ai_messages = json.load(f)

    if not _is_restorable_ai_context(ai_messages):
        ai_messages = _rebuild_ai_context_from_turns(
            build_ai_system_prompt(seed_words, companion_mode=companion_mode),
            turns,
            use_reasoning_field=_uses_native_reasoning_field(ai_reasoning),
        )
        console.print("  [dim yellow]Raw AI context missing or incomplete; rebuilt from saved turns.[/dim yellow]")

    migrated_reasoning = _migrate_assistant_reasoning_fields(
        ai_messages,
        use_reasoning_field=_uses_native_reasoning_field(ai_reasoning),
    )
    if migrated_reasoning > 0:
        console.print(f"  [dim]Migrated {migrated_reasoning} assistant messages from content to reasoning.[/dim]")

    chars_saved = _normalize_tool_arguments(ai_messages)
    if chars_saved > 0:
        console.print(
            f"  [dim]Normalized Unicode escapes in context (saved ~{chars_saved:,} chars / ~{chars_saved // 4:,} tokens)[/dim]"
        )

    _print_resume_header(
        profile,
        jsonl_path,
        turns,
        ai_model,
        human_model,
        ai_provider,
        human_provider,
        ai_reasoning,
        human_reasoning,
        ai_temperature,
        human_temperature,
        ai_max_tokens,
        human_max_tokens,
        previous_cost,
        target_tokens,
        seed_words,
        language,
        ai_model_override=ai_model_override,
        human_model_override=human_model_override,
        ai_provider_override=ai_provider_override,
        human_provider_override=human_provider_override,
        ai_temperature_override=ai_temperature_override,
        human_temperature_override=human_temperature_override,
        ai_max_tokens_override=ai_max_tokens_override,
        human_max_tokens_override=human_max_tokens_override,
        _UNSET=_UNSET,
    )

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
    set_tool_call_counter(max_tc)

    gen = cls(
        client,
        profile,
        ai_model=ai_model,
        human_model=human_model,
        target_tokens=target_tokens,
        language=language,
        companion_mode=companion_mode,
        verbose=verbose,
        ai_provider=ai_provider,
        ai_reasoning=ai_reasoning,
        ai_temperature=ai_temperature,
        ai_max_tokens=ai_max_tokens,
        human_provider=human_provider,
        human_reasoning=human_reasoning,
        human_temperature=human_temperature,
        human_max_tokens=human_max_tokens,
    )

    client.total_cost = previous_cost
    gen._seed_words = seed_words
    gen._conversation_plan = conversation_plan
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
    gen._record.seed_words = seed_words
    gen._record.conversation_plan = conversation_plan
    gen._record.language = language
    gen._record.companion_mode = companion_mode
    gen._record.started_at = metadata["started_at"]
    _restore_events_and_turns(gen, events, turns)

    last_turn = turns[-1]["turn_number"] if turns else 0
    last_ai_turn = next((t for t in reversed(turns) if t.get("speaker") == "ai"), None)
    stored_ctx = last_ai_turn.get("ai_context_tokens", 0) if last_ai_turn else 0
    existing_tokens = stored_ctx or _estimate_context_tokens(ai_messages)

    console.print(f"  Existing tokens: ~{existing_tokens:,}")
    console.print()

    return gen._run_loop(start_turn=last_turn, start_tokens=existing_tokens)


def _print_resume_header(
    profile: dict,
    jsonl_path: Path,
    turns: list,
    ai_model: str,
    human_model: str,
    ai_provider: Any,
    human_provider: Any,
    ai_reasoning: Any,
    human_reasoning: Any,
    ai_temperature: float,
    human_temperature: float,
    ai_max_tokens: int,
    human_max_tokens: int,
    previous_cost: float,
    target_tokens: int,
    seed_words: list,
    language: str,
    **kwargs: Any,
) -> None:
    """Print the resume status header to console."""
    ai_model_override = kwargs.get("ai_model_override")
    human_model_override = kwargs.get("human_model_override")
    ai_provider_override = kwargs.get("ai_provider_override")
    human_provider_override = kwargs.get("human_provider_override")
    ai_temperature_override = kwargs.get("ai_temperature_override")
    human_temperature_override = kwargs.get("human_temperature_override")
    ai_max_tokens_override = kwargs.get("ai_max_tokens_override")
    human_max_tokens_override = kwargs.get("human_max_tokens_override")
    _UNSET = kwargs.get("_UNSET")

    console.print(f"\n[bold]Resuming conversation with {profile['name']}[/bold]")
    console.print(f"  From: {jsonl_path.name}")
    console.print(f"  Existing turns: {len(turns)}")
    console.print(f"  AI model: {ai_model}" + (" [yellow](overridden)[/yellow]" if ai_model_override else ""))
    console.print(f"  Human model: {human_model}" + (" [yellow](overridden)[/yellow]" if human_model_override else ""))
    if ai_provider:
        overridden = ai_provider_override is not _UNSET
        console.print(f"  AI provider: {ai_provider}" + (" [yellow](overridden)[/yellow]" if overridden else ""))
    if ai_reasoning:
        console.print(f"  AI reasoning: {ai_reasoning}")
    if human_provider:
        overridden_hp = human_provider_override is not _UNSET
        console.print(
            f"  Human provider: {human_provider}" + (" [yellow](overridden)[/yellow]" if overridden_hp else "")
        )
    if human_reasoning:
        console.print(f"  Human reasoning: {human_reasoning}")
    temp_line = f"  AI temperature: {ai_temperature}" + (
        " [yellow](overridden)[/yellow]" if ai_temperature_override is not None else ""
    )
    temp_line += f" / Human temperature: {human_temperature}" + (
        " [yellow](overridden)[/yellow]" if human_temperature_override is not None else ""
    )
    console.print(temp_line)
    tokens_line = f"  AI max tokens: {ai_max_tokens}" + (
        " [yellow](overridden)[/yellow]" if ai_max_tokens_override is not None else ""
    )
    tokens_line += f" / Human max tokens: {human_max_tokens}" + (
        " [yellow](overridden)[/yellow]" if human_max_tokens_override is not None else ""
    )
    console.print(tokens_line)
    console.print(f"  Previous cost: ${previous_cost:.4f}")
    console.print(f"  New target: ~{target_tokens:,} tokens")
    console.print(f"  Seed: {', '.join(seed_words)}")
    console.print(f"  Language: {language}")


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
            gen._human_messages.append({"role": "user", "content": text})
            for note in nudges_by_turn.get(turn_data["turn_number"], []):
                gen._human_messages.append({"role": "user", "content": note})


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
                human_reasoning=turn_data.get("human_reasoning"),
                human_reasoning_details=turn_data.get("human_reasoning_details"),
                token_estimate=turn_data.get("token_estimate", 0),
                cost_usd=turn_data.get("cost_usd", 0.0),
                timestamp=turn_data.get("timestamp", ""),
                ai_context_tokens=turn_data.get("ai_context_tokens", 0),
                human_context_tokens=turn_data.get("human_context_tokens", 0),
            )
        )
