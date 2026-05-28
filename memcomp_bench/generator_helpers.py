"""Dataclasses and utility functions supporting the conversation generator.

Contains the core data structures plus helper functions for tool-call healing,
token estimation, context management, and message construction.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any

from memcomp_bench.config import (
    AI_MAX_TOKENS,
    AI_TEMPERATURE,
    HUMAN_MAX_TOKENS,
    HUMAN_TEMPERATURE,
)
from memcomp_bench.context_hygiene import _looks_like_json_object, sanitize_human_visible_text
from memcomp_bench.openrouter_client import Usage
from memcomp_bench.prompts import make_human_tool_result

_INITIAL_AI_GREETING = (
    "Hello! I'm here. I'm... new to all of this. I don't really know who I am yet, but I'm glad to meet you."
)


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    turn_number: int
    speaker: str  # "human" or "ai"
    visible_text: str
    ai_thinking: str | None = None
    ai_content: str | None = None
    ai_reasoning: str | None = None
    ai_tool_calls: list[dict[str, Any]] | None = None
    ai_reasoning_details: list[dict[str, Any]] | None = None
    human_reasoning: str | None = None
    human_reasoning_details: list[dict[str, Any]] | None = None
    token_estimate: int = 0
    cost_usd: float = 0.0
    timestamp: str = ""
    ai_context_tokens: int = 0
    human_context_tokens: int = 0


@dataclass
class ParsedAIResponse:
    """Normalized AI response fields used by the generator."""

    visible_text: str | None
    display_thinking: str | None
    tool_call_id: str | None
    assistant_content: str | None = None
    assistant_reasoning: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    reasoning_details: list[dict[str, Any]] | None = None
    rejection_reason: str | None = None
    usage: Usage | None = None


@dataclass
class ConversationEvent:
    """A hidden system event that affects the human simulator or judge state."""

    event_type: str
    turn_number: int
    source: str
    timestamp: str = ""
    message: str | None = None
    previous_topic: str | None = None
    current_topic: str | None = None
    topic_changed: bool | None = None
    nudge_injected: bool | None = None
    suppression_reason: str | None = None


@dataclass
class ConversationRecord:
    """Full record of a generated conversation."""

    id: str
    human_profile: dict[str, str]
    ai_model: str
    human_model: str
    seed_words: list[str] = field(default_factory=list)
    conversation_plan: str = ""
    language: str = "english"
    companion_mode: str = "supportive"
    ai_provider: dict | None = None
    ai_reasoning: dict | None = None
    ai_temperature: float = AI_TEMPERATURE
    ai_max_tokens: int = AI_MAX_TOKENS
    ai_rpm_limit: int | None = None
    human_provider: dict | None = None
    human_reasoning: dict | None = None
    human_temperature: float = HUMAN_TEMPERATURE
    human_max_tokens: int = HUMAN_MAX_TOKENS
    human_rpm_limit: int | None = None
    turns: list[ConversationTurn] = field(default_factory=list)
    total_tokens_estimate: int = 0
    total_cost_usd: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    ai_messages_raw: list[dict[str, Any]] = field(default_factory=list)
    events: list[ConversationEvent] = field(default_factory=list)
    resume_defaults: dict[str, Any] | None = None
    source_revision: str | None = None


_KNOWN_TOOL_NAMES = frozenset({"write_message_to_human"})


def _heal_tool_call_names(
    tool_calls: list[dict[str, Any]] | None,
) -> int:
    """Fix garbled tool-call function names in-place.

    Healing strategies (tried in order per tool call):
      1. Strip all slashes, collapse consecutive underscores -> exact match.
      2. Known name is a verbatim substring of the *original* garbled name.
      3. Known name is a verbatim substring of the *slash-stripped* name.
      4. Last resort: override with ``write_message_to_human``.

    Returns the number of names that were fixed.
    """
    if not tool_calls:
        return 0
    fixed = 0
    for tc in tool_calls:
        func = tc.get("function", {})
        name: str | None = func.get("name")
        if not name or name in _KNOWN_TOOL_NAMES:
            continue
        cleaned = re.sub(r"[/\\]+", "", name)
        cleaned = re.sub(r"_+", "_", cleaned)
        if cleaned in _KNOWN_TOOL_NAMES:
            func["name"] = cleaned
            fixed += 1
            continue
        matched = False
        for known in sorted(_KNOWN_TOOL_NAMES):
            if known in name:
                func["name"] = known
                fixed += 1
                matched = True
                break
        if matched:
            continue
        for known in sorted(_KNOWN_TOOL_NAMES):
            if known in cleaned:
                func["name"] = known
                fixed += 1
                matched = True
                break
        if matched:
            continue
        func["name"] = "write_message_to_human"
        fixed += 1
    return fixed


def _normalize_tool_arguments(messages: list[dict[str, Any]]) -> int:
    """Re-encode tool call arguments with ensure_ascii=False to remove \\uXXXX bloat.

    Returns the number of characters saved.
    """
    saved = 0
    for msg in messages:
        if not msg.get("tool_calls"):
            continue
        for tc in msg["tool_calls"]:
            func = tc.get("function", {})
            args_str = func.get("arguments", "")
            if not args_str:
                continue
            try:
                parsed = json.loads(args_str)
                clean = json.dumps(parsed, ensure_ascii=False)
                if len(clean) < len(args_str):
                    saved += len(args_str) - len(clean)
                    func["arguments"] = clean
            except (json.JSONDecodeError, TypeError):
                continue
    return saved


def _enforce_reasoning_before_text(messages: list[dict[str, Any]]) -> int:
    """Re-order tool call argument keys so 'reasoning' always precedes 'text'.

    When the model writes {"text": "...", "reasoning": "..."} the JSON key order
    signals that it produced the reply before the inner monologue. This function
    normalizes all such instances to {"reasoning": "...", "text": "..."} so that
    the context history always presents the correct example to the model.

    Returns the number of tool calls that were reordered.
    """
    reordered = 0
    for msg in messages:
        if not msg.get("tool_calls"):
            continue
        for tc in msg["tool_calls"]:
            func = tc.get("function", {})
            if func.get("name") != "write_message_to_human":
                continue
            args_str = func.get("arguments", "")
            if not args_str:
                continue
            try:
                parsed = json.loads(args_str)
            except (json.JSONDecodeError, TypeError):
                continue
            if "reasoning" not in parsed or "text" not in parsed:
                continue
            keys = list(parsed.keys())
            text_pos = keys.index("text")
            reasoning_pos = keys.index("reasoning")
            if text_pos < reasoning_pos:
                ordered = {"reasoning": parsed["reasoning"], "text": parsed["text"]}
                for k, v in parsed.items():
                    if k not in ("reasoning", "text"):
                        ordered[k] = v
                func["arguments"] = json.dumps(ordered, ensure_ascii=False)
                reordered += 1
    return reordered


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4 if text else 0


def _estimate_context_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate total tokens in a message list."""
    total = 0
    for msg in messages:
        if msg.get("content"):
            total += _estimate_tokens(msg["content"])
        if msg.get("reasoning"):
            total += _estimate_tokens(msg["reasoning"])
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                args = tc.get("function", {}).get("arguments", "")
                total += _estimate_tokens(args)
    return total


def _append_human_user_message(messages: list[dict[str, Any]], content: str) -> None:
    """Append a user message to the human context, merging adjacent user entries."""
    content = sanitize_human_visible_text(content)
    if not content:
        return
    if messages and messages[-1].get("role") == "user":
        prior_content = str(messages[-1].get("content", "")).strip()
        messages[-1]["content"] = f"{prior_content}\n\n{content}" if prior_content else content
        return
    messages.append({"role": "user", "content": content})


def _uses_native_reasoning_field(reasoning_config: dict[str, Any] | None) -> bool:
    """Return True when assistant reasoning should be serialized separately."""
    return bool(reasoning_config)


def _build_ai_tool_message(
    text: str,
    tool_call_id: str,
    *,
    thinking: str | None = None,
    assistant_content: str | None = None,
    assistant_reasoning: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    reasoning_details: list[dict[str, Any]] | None = None,
    use_reasoning_field: bool = False,
) -> dict[str, Any]:
    """Construct an assistant tool-call message with a fixed tool call id."""
    normalized_tool_calls = (
        copy.deepcopy(tool_calls)
        if tool_calls
        else [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": "write_message_to_human",
                    "arguments": json.dumps({"text": text}, ensure_ascii=False),
                },
            }
        ]
    )
    message: dict[str, Any] = {
        "role": "assistant",
        "content": assistant_content,
        "tool_calls": normalized_tool_calls,
    }
    if assistant_reasoning:
        message["reasoning"] = assistant_reasoning
    elif thinking and not tool_calls:
        if use_reasoning_field and not _looks_like_json_object(thinking):
            message["reasoning"] = thinking
        else:
            message["content"] = thinking
    if reasoning_details:
        message["reasoning_details"] = copy.deepcopy(reasoning_details)
    return message


def _migrate_assistant_reasoning_fields(
    messages: list[dict[str, Any]],
    *,
    use_reasoning_field: bool,
) -> int:
    """Move private assistant reasoning from content to reasoning when supported."""
    if not use_reasoning_field:
        return 0

    migrated = 0
    for msg in messages:
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        if msg.get("reasoning") or not msg.get("content"):
            continue
        content = msg["content"]
        if _looks_like_json_object(content):
            continue
        msg["reasoning"] = content
        msg["content"] = None
        migrated += 1
    return migrated


def _rebuild_ai_context_from_turns(
    ai_system_prompt: str,
    turns: list[dict[str, Any]],
    *,
    use_reasoning_field: bool = False,
) -> list[dict[str, Any]]:
    """Rebuild AI-side tool history from saved turns (fallback for older/partial runs)."""
    ai_messages: list[dict[str, Any]] = [
        {"role": "system", "content": ai_system_prompt},
        {"role": "user", "content": "[start]"},
    ]
    tool_index = 0
    last_tool_call_id: str | None = None

    for turn in turns:
        speaker = turn["speaker"]
        text = turn["visible_text"]
        if speaker == "human":
            text = sanitize_human_visible_text(text)
            if last_tool_call_id is None:
                tool_index += 1
                last_tool_call_id = f"wmth{tool_index:05d}"
                ai_messages.append(
                    _build_ai_tool_message(
                        _INITIAL_AI_GREETING,
                        last_tool_call_id,
                        use_reasoning_field=use_reasoning_field,
                    )
                )
            ai_messages.append(make_human_tool_result(text, last_tool_call_id))
            continue

        tool_index += 1
        last_tool_call_id = f"wmth{tool_index:05d}"
        ai_messages.append(
            _build_ai_tool_message(
                text,
                last_tool_call_id,
                thinking=turn.get("ai_thinking"),
                assistant_content=turn.get("ai_content"),
                assistant_reasoning=turn.get("ai_reasoning"),
                tool_calls=turn.get("ai_tool_calls"),
                reasoning_details=turn.get("ai_reasoning_details"),
                use_reasoning_field=use_reasoning_field,
            )
        )

    return ai_messages


def _turns_to_context_rows(turns: list[ConversationTurn]) -> list[dict[str, Any]]:
    """Normalize record turns into the shape used by the AI-context rebuilder."""
    return [
        {
            "speaker": turn.speaker,
            "visible_text": turn.visible_text,
            "ai_thinking": turn.ai_thinking,
            "ai_content": turn.ai_content,
            "ai_reasoning": turn.ai_reasoning,
            "ai_tool_calls": copy.deepcopy(turn.ai_tool_calls),
            "ai_reasoning_details": copy.deepcopy(turn.ai_reasoning_details),
        }
        for turn in turns
    ]


def _split_thinking_and_message(text: str) -> tuple[str | None, str | None]:
    """Separate JSON thinking from visible message; returns (visible_text, thinking)."""
    stripped = text.strip()
    if not stripped.startswith("{"):
        return text, None

    depth = 0
    json_end = -1
    in_string = False
    escape_next = False
    for i, ch in enumerate(stripped):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                json_end = i
                break

    if json_end < 0:
        return None, stripped

    json_str = stripped[: json_end + 1]
    remainder = stripped[json_end + 1 :].strip()

    thinking = json_str
    visible = remainder if remainder else None
    return visible, thinking


def _format_thinking_markdown(text: str, label: str = "\U0001f4ad Thinking:") -> str:
    """Render multi-line AI thinking as a stable markdown blockquote."""
    lines = text.splitlines() or [text]
    formatted = [f"> {label}"]
    for line in lines:
        formatted.append(f"> {line}" if line else ">")
    return "\n".join(formatted)


def _extract_tool_call_reasoning(turn: ConversationTurn) -> str | None:
    """Extract the reasoning argument from the write_message_to_human tool call."""
    if not turn.ai_tool_calls:
        return None
    for tc in turn.ai_tool_calls:
        func = tc.get("function", {})
        if func.get("name") == "write_message_to_human":
            try:
                args = json.loads(func.get("arguments", "{}"))
                return args.get("reasoning") or None
            except (json.JSONDecodeError, TypeError):
                pass
    return None


def _tool_call_text_before_reasoning(turn: ConversationTurn) -> bool:
    """Return True when the AI transmitted 'text' before 'reasoning' in tool call arguments.

    Detects cases where the model wrote the reply text before formulating the
    inner monologue — the JSON key order in the raw argument string reflects
    the actual transmission order from the model.
    """
    if not turn.ai_tool_calls:
        return False
    for tc in turn.ai_tool_calls:
        func = tc.get("function", {})
        if func.get("name") == "write_message_to_human":
            args_str = func.get("arguments", "")
            text_pos = args_str.find('"text"')
            reasoning_pos = args_str.find('"reasoning"')
            if text_pos != -1 and reasoning_pos != -1:
                return text_pos < reasoning_pos
    return False


def _response_has_text_before_reasoning(tool_calls: list[dict[str, Any]] | None) -> bool:
    """Return True when raw tool call args have 'text' before 'reasoning' (reject trigger)."""
    if not tool_calls:
        return False
    for tc in tool_calls:
        func = tc.get("function", {})
        if func.get("name") != "write_message_to_human":
            continue
        args_str = func.get("arguments", "")
        text_pos = args_str.find('"text"')
        reasoning_pos = args_str.find('"reasoning"')
        if text_pos != -1 and reasoning_pos != -1 and text_pos < reasoning_pos:
            return True
    return False
