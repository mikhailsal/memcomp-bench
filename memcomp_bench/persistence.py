"""Persistence layer for conversation records: save, load, and reformat.

Handles serialization to JSONL, raw AI context JSON, and human-readable markdown.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console

from memcomp_bench.config import AI_REASONING
from memcomp_bench.generator_helpers import (
    ConversationEvent,
    ConversationRecord,
    ConversationTurn,
    _estimate_context_tokens,
    _extract_tool_call_reasoning,
    _format_thinking_markdown,
    _is_restorable_ai_context,
    _rebuild_ai_context_from_turns,
    _tool_call_text_before_reasoning,
    _turns_to_context_rows,
    _uses_native_reasoning_field,
)
from memcomp_bench.prompts import build_ai_system_prompt

console = Console()

_PERSON = "👤"
_ROBOT = "🤖"
_BRAIN = "🧠"
_THOUGHT = "💭"
_CLIPBOARD = "📋"
_DOT = "·"


def _write_md_header(f: Any, record: ConversationRecord) -> None:
    """Write the metadata header section of the markdown file."""
    f.write(f"# Conversation: {record.human_profile['name']} & AI\n\n")
    f.write(f"- **AI model**: {record.ai_model}\n")
    f.write(f"- **Human model**: {record.human_model}\n")
    f.write(f"- **Turns**: {len(record.turns)}\n")
    f.write(f"- **Tokens (est.)**: {record.total_tokens_estimate:,}\n")
    f.write(f"- **Cost**: ${record.total_cost_usd:.4f}\n")
    f.write(f"- **Seed**: {', '.join(record.seed_words)}\n")
    f.write(f"- **Started**: {record.started_at}\n")
    f.write(f"- **Finished**: {record.finished_at}\n\n")
    f.write("## Human Profile\n\n")
    f.write(f"**{record.human_profile['name']}**: {record.human_profile['backstory']}\n\n")
    if record.conversation_plan:
        f.write("## Conversation Plan\n\n")
        f.write(f"{record.conversation_plan}\n\n")
    if record.events:
        _write_md_events(f, record.events)
    f.write("---\n\n")


def _write_md_events(f: Any, events: list[ConversationEvent]) -> None:
    """Write the system events section of the markdown file."""
    f.write("## System Events\n\n")
    for event in events:
        parts = [f"turn {event.turn_number}", f"{event.event_type} ({event.source})"]
        if event.current_topic:
            parts.append(f"topic={event.current_topic}")
        if event.topic_changed is not None:
            parts.append(f"changed={event.topic_changed}")
        if event.nudge_injected is not None:
            parts.append(f"nudge_injected={event.nudge_injected}")
        if event.suppression_reason:
            parts.append(f"suppressed={event.suppression_reason}")
        joined = " \u00b7 ".join(parts)
        f.write(f"- {joined}\n")
        if event.message:
            f.write(f"  - note: {event.message}\n")
    f.write("\n")


def _write_md_human_turn(f: Any, turn: ConversationTurn, profile_name: str) -> None:
    """Write a single human turn in markdown."""
    f.write(f"### {_PERSON} {profile_name} (turn {turn.turn_number})\n\n")
    if turn.human_context_tokens or turn.ai_context_tokens:
        f.write(
            f"*{_PERSON} human ctx: {turn.human_context_tokens:,} tok {_DOT} {_BRAIN} AI ctx: {turn.ai_context_tokens:,} tok*\n\n"
        )
    if turn.human_reasoning:
        f.write(f"{_format_thinking_markdown(turn.human_reasoning)}\n\n")
    f.write(f"{turn.visible_text}\n\n")


def _write_md_ai_turn(f: Any, turn: ConversationTurn) -> None:
    """Write a single AI turn in markdown."""
    f.write(f"### {_ROBOT} AI (turn {turn.turn_number})\n\n")
    if turn.ai_context_tokens or turn.human_context_tokens:
        f.write(
            f"*{_BRAIN} AI ctx: {turn.ai_context_tokens:,} tok {_DOT} {_PERSON} human ctx: {turn.human_context_tokens:,} tok*\n\n"
        )
    native = turn.ai_reasoning
    tool_inner = _extract_tool_call_reasoning(turn)
    text_first = _tool_call_text_before_reasoning(turn)
    inline = turn.ai_content

    if native:
        f.write(f"{_format_thinking_markdown(native, f'{_BRAIN} Native reasoning:')}\n\n")
    if text_first:
        f.write(f"{turn.visible_text}\n\n")
        if tool_inner and tool_inner != native:
            f.write(f"{_format_thinking_markdown(tool_inner, f'{_THOUGHT} Inner monologue (after reply):')}\n\n")
        if inline and inline not in (native, tool_inner):
            f.write(f"{_format_thinking_markdown(inline, f'{_CLIPBOARD} Response draft (after reply):')}\n\n")
        if not native and not tool_inner and not inline and turn.ai_thinking:
            f.write(f"{_format_thinking_markdown(turn.ai_thinking)}\n\n")
    else:
        shown_any = bool(native)
        if tool_inner and tool_inner != native:
            f.write(f"{_format_thinking_markdown(tool_inner, f'{_THOUGHT} Inner monologue:')}\n\n")
            shown_any = True
        if inline and inline not in (native, tool_inner):
            f.write(f"{_format_thinking_markdown(inline, f'{_CLIPBOARD} Response draft:')}\n\n")
            shown_any = True
        if not shown_any and turn.ai_thinking:
            f.write(f"{_format_thinking_markdown(turn.ai_thinking)}\n\n")
        f.write(f"{turn.visible_text}\n\n")


def _write_conversation_markdown(f: Any, record: ConversationRecord) -> None:
    """Write the human-readable markdown for a conversation to an open file handle."""
    _write_md_header(f, record)
    for turn in record.turns:
        if turn.speaker == "human":
            _write_md_human_turn(f, turn, record.human_profile["name"])
        else:
            _write_md_ai_turn(f, turn)


def _parse_turn(obj: dict[str, Any]) -> ConversationTurn:
    return ConversationTurn(
        turn_number=obj["turn_number"],
        speaker=obj["speaker"],
        visible_text=obj["visible_text"],
        ai_thinking=obj.get("ai_thinking"),
        ai_content=obj.get("ai_content"),
        ai_reasoning=obj.get("ai_reasoning"),
        ai_tool_calls=obj.get("ai_tool_calls"),
        ai_reasoning_details=obj.get("ai_reasoning_details"),
        human_reasoning=obj.get("human_reasoning"),
        human_reasoning_details=obj.get("human_reasoning_details"),
        token_estimate=obj.get("token_estimate", 0),
        cost_usd=obj.get("cost_usd", 0.0),
        timestamp=obj.get("timestamp", ""),
        ai_context_tokens=obj.get("ai_context_tokens", 0),
        human_context_tokens=obj.get("human_context_tokens", 0),
    )


def _parse_event(obj: dict[str, Any]) -> ConversationEvent:
    return ConversationEvent(
        event_type=obj["event_type"],
        turn_number=obj["turn_number"],
        source=obj["source"],
        timestamp=obj.get("timestamp", ""),
        message=obj.get("message"),
        previous_topic=obj.get("previous_topic"),
        current_topic=obj.get("current_topic"),
        topic_changed=obj.get("topic_changed"),
        nudge_injected=obj.get("nudge_injected"),
        suppression_reason=obj.get("suppression_reason"),
    )


def load_conversation_record(jsonl_path: Path) -> ConversationRecord:
    """Load a ConversationRecord from a saved JSONL file."""
    from memcomp_bench.config import AI_MAX_TOKENS, AI_TEMPERATURE, HUMAN_MAX_TOKENS, HUMAN_TEMPERATURE

    turns: list[ConversationTurn] = []
    events: list[ConversationEvent] = []
    meta: dict[str, Any] = {}

    with open(jsonl_path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            t = obj.get("type")
            if t == "metadata":
                meta = obj
            elif t == "turn":
                turns.append(_parse_turn(obj))
            elif t == "event":
                events.append(_parse_event(obj))

    record = ConversationRecord(
        id=meta.get("conversation_id", ""),
        human_profile=meta.get("human_profile", {}),
        ai_model=meta.get("ai_model", ""),
        human_model=meta.get("human_model", ""),
        seed_words=meta.get("seed_words", []),
        conversation_plan=meta.get("conversation_plan", ""),
        language=meta.get("language", "english"),
        companion_mode=meta.get("companion_mode", "honest"),
        ai_provider=meta.get("ai_provider"),
        ai_reasoning=meta.get("ai_reasoning"),
        ai_temperature=meta.get("ai_temperature", AI_TEMPERATURE),
        ai_max_tokens=meta.get("ai_max_tokens", AI_MAX_TOKENS),
        human_temperature=meta.get("human_temperature", HUMAN_TEMPERATURE),
        human_max_tokens=meta.get("human_max_tokens", HUMAN_MAX_TOKENS),
        total_tokens_estimate=meta.get("total_tokens_estimate", 0),
        total_cost_usd=meta.get("total_cost_usd", 0.0),
        started_at=meta.get("started_at", ""),
        finished_at=meta.get("finished_at", ""),
    )
    record.turns = turns
    record.events = events
    return record


def reformat_markdown(jsonl_path: Path) -> Path:
    """Rewrite the .md file for an existing conversation using the current render format.

    Loads the record from *jsonl_path* and rewrites the companion ``.md`` file
    in-place, applying updated rendering (e.g. multi-source reasoning labels).
    Returns the path of the updated markdown file.
    """
    record = load_conversation_record(jsonl_path)
    md_path = jsonl_path.with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as f:
        _write_conversation_markdown(f, record)
    return md_path


def _write_jsonl(jsonl_path: Path, record: ConversationRecord) -> None:
    """Write the JSONL file with metadata, turns, and events."""
    meta = {
        "type": "metadata",
        "conversation_id": record.id,
        "human_profile": record.human_profile,
        "ai_model": record.ai_model,
        "human_model": record.human_model,
        "seed_words": record.seed_words,
        "conversation_plan": record.conversation_plan,
        "language": record.language,
        "companion_mode": record.companion_mode,
        "ai_provider": record.ai_provider,
        "ai_reasoning": record.ai_reasoning,
        "ai_temperature": record.ai_temperature,
        "ai_max_tokens": record.ai_max_tokens,
        "human_temperature": record.human_temperature,
        "human_max_tokens": record.human_max_tokens,
        "total_turns": len(record.turns),
        "total_tokens_estimate": record.total_tokens_estimate,
        "total_cost_usd": record.total_cost_usd,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
    }
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for turn in record.turns:
            line = {
                "type": "turn",
                "turn_number": turn.turn_number,
                "speaker": turn.speaker,
                "visible_text": turn.visible_text,
                "ai_thinking": turn.ai_thinking,
                "ai_content": turn.ai_content,
                "ai_reasoning": turn.ai_reasoning,
                "ai_tool_calls": turn.ai_tool_calls,
                "ai_reasoning_details": turn.ai_reasoning_details,
                "human_reasoning": turn.human_reasoning,
                "human_reasoning_details": turn.human_reasoning_details,
                "token_estimate": turn.token_estimate,
                "cost_usd": turn.cost_usd,
                "timestamp": turn.timestamp,
                "ai_context_tokens": turn.ai_context_tokens,
                "human_context_tokens": turn.human_context_tokens,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        for event in record.events:
            line = {
                "type": "event",
                "event_type": event.event_type,
                "turn_number": event.turn_number,
                "source": event.source,
                "timestamp": event.timestamp,
                "message": event.message,
                "previous_topic": event.previous_topic,
                "current_topic": event.current_topic,
                "topic_changed": event.topic_changed,
                "nudge_injected": event.nudge_injected,
                "suppression_reason": event.suppression_reason,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def _ensure_raw_ai_context(record: ConversationRecord) -> list[dict[str, Any]]:
    """Ensure ai_messages_raw is usable, rebuilding from turns if needed."""
    raw_messages = record.ai_messages_raw
    if not _is_restorable_ai_context(raw_messages) and record.turns:
        raw_messages = _rebuild_ai_context_from_turns(
            build_ai_system_prompt(record.seed_words, companion_mode=record.companion_mode),
            _turns_to_context_rows(record.turns),
            use_reasoning_field=_uses_native_reasoning_field(AI_REASONING),
        )
        record.ai_messages_raw = raw_messages
    if record.turns and record.total_tokens_estimate <= 0:
        record.total_tokens_estimate = _estimate_context_tokens(raw_messages)
    return raw_messages


def save_conversation(record: ConversationRecord, output_dir: Path) -> Path:
    """Save a conversation record to JSONL and a readable markdown file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    base = f"conv_{record.id}_{record.human_profile['name'].lower()}"

    jsonl_path = output_dir / f"{base}.jsonl"
    _write_jsonl(jsonl_path, record)

    md_path = output_dir / f"{base}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        _write_conversation_markdown(f, record)

    raw_messages = _ensure_raw_ai_context(record)
    raw_path = output_dir / f"{base}_raw_ai_context.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_messages, f, ensure_ascii=False, indent=2)

    console.print("\n[bold]Files saved:[/bold]")
    console.print(f"  \U0001f4c4 {jsonl_path}")
    console.print(f"  \U0001f4d6 {md_path}")
    console.print(f"  \U0001f527 {raw_path}")

    return jsonl_path
