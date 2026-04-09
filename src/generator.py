"""Conversation generator: orchestrates turn-by-turn dialogue between two models.

The AI companion uses tool-based communication (write_message_to_human) matching
the MAI Companion protocol. The human simulator uses standard user/assistant format.
"""

from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from src.config import (
    AI_MAX_TOKENS,
    AI_MODEL,
    AI_PROVIDER,
    AI_REASONING,
    AI_TEMPERATURE,
    COMPANION_MODE,
    HUMAN_MAX_TOKENS,
    HUMAN_MODEL,
    HUMAN_PROVIDER,
    HUMAN_REASONING,
    HUMAN_TEMPERATURE,
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    MAX_TURNS,
    TARGET_TOKENS,
    TOPIC_CHECK_INTERVAL,
)
from src.openrouter_client import OpenRouterClient, Usage
from src.prompts import (
    AI_TOOLS,
    CONVERSATION_PLAN_PROMPT,
    build_ai_system_prompt,
    build_human_system_prompt,
    extract_tool_call_text,
    generate_seed,
    get_human_profile,
    make_ai_greeting_turn,
    make_ai_tool_call,
    make_human_tool_result,
    reset_tool_call_counter,
    set_tool_call_counter,
)

console = Console()

_UNSET = object()  # sentinel for "parameter not provided" in resume()

_INITIAL_AI_GREETING = (
    "Hello! I'm here. I'm... new to all of this. I don't really know who I am yet, "
    "but I'm glad to meet you."
)

_TOPIC_STALE_NOTE = (
    "[System note: The conversation has been on the same topic for a while. "
    "Time to shift gears — bring up something new from your life or interests. "
    "Check your conversation plan for topics you haven't covered yet.]"
)

_B3_REFRESH_NOTE = (
    "[System note: Something significant happened in your life recently — "
    "maybe a work event, a conversation with someone, something you saw or read, "
    "a mood shift, or a random everyday moment. Bring it up naturally in your "
    "next message. It should be specific, emotionally charged, and unrelated "
    "to what you've been discussing lately. Time to change the topic.]"
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
    human_reasoning: str | None = None
    human_reasoning_details: list[dict[str, Any]] | None = None
    token_estimate: int = 0
    cost_usd: float = 0.0
    timestamp: str = ""
    ai_context_tokens: int = 0    # cumulative AI context size after this turn
    human_context_tokens: int = 0  # cumulative human-emulator context size after this turn


@dataclass
class ParsedAIResponse:
    """Normalized AI response fields used by the generator."""
    visible_text: str | None
    display_thinking: str | None
    tool_call_id: str | None
    assistant_content: str | None = None
    assistant_reasoning: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
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
    human_provider: dict | None = None
    human_reasoning: dict | None = None
    human_temperature: float = HUMAN_TEMPERATURE
    human_max_tokens: int = HUMAN_MAX_TOKENS
    turns: list[ConversationTurn] = field(default_factory=list)
    total_tokens_estimate: int = 0
    total_cost_usd: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    ai_messages_raw: list[dict[str, Any]] = field(default_factory=list)
    events: list[ConversationEvent] = field(default_factory=list)


# Valid tool function names defined in AI_TOOLS.
_KNOWN_TOOL_NAMES = frozenset({"write_message_to_human", "stop"})


def _heal_tool_call_names(
    tool_calls: list[dict[str, Any]] | None,
) -> int:
    """Fix garbled tool-call function names in-place.

    Some models inject spurious slashes, extra segments, or random ID suffixes
    into function names when context grows long or temperature is high.  This
    normalises those artefacts so history always contains valid calls.

    Healing strategies (tried in order per tool call):
      1. Strip all slashes, collapse consecutive underscores → exact match.
      2. Known name is a verbatim substring of the *original* garbled name.
      3. Known name is a verbatim substring of the *slash-stripped* name
         (catches ``write_//message_to_human_<ID>`` patterns).
      4. Last resort: if tool_calls exist but none of the above matched,
         override with ``write_message_to_human``.  Safe because every AI
         chat call forces ``tool_choice=write_message_to_human``.

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
        # Strategy 1: strip slashes, collapse double-underscores.
        cleaned = re.sub(r"[/\\]+", "", name)
        cleaned = re.sub(r"_+", "_", cleaned)
        if cleaned in _KNOWN_TOOL_NAMES:
            func["name"] = cleaned
            fixed += 1
            continue
        # Strategy 2: known name embedded verbatim in the *original* name.
        matched = False
        for known in sorted(_KNOWN_TOOL_NAMES):  # sorted for determinism
            if known in name:
                func["name"] = known
                fixed += 1
                matched = True
                break
        if matched:
            continue
        # Strategy 3: known name embedded verbatim in the *slash-stripped* name.
        for known in sorted(_KNOWN_TOOL_NAMES):
            if known in cleaned:
                func["name"] = known
                fixed += 1
                matched = True
                break
        if matched:
            continue
        # Strategy 4 — last resort: we force tool_choice=write_message_to_human
        # on every AI call, so any remaining unrecognised name must be that.
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


def _uses_native_reasoning_field(reasoning_config: dict[str, Any] | None) -> bool:
    """Return True when assistant reasoning should be serialized separately."""
    return bool(reasoning_config)


def _looks_like_json_object(text: str | None) -> bool:
    """Return True when text appears to be a JSON object payload."""
    return bool(text and text.lstrip().startswith("{"))


def _build_ai_tool_message(
    text: str,
    tool_call_id: str,
    *,
    thinking: str | None = None,
    assistant_content: str | None = None,
    assistant_reasoning: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    use_reasoning_field: bool = False,
) -> dict[str, Any]:
    """Construct an assistant tool-call message with a fixed tool call id."""
    normalized_tool_calls = copy.deepcopy(tool_calls) if tool_calls else [{
        "id": tool_call_id,
        "type": "function",
        "function": {
            "name": "write_message_to_human",
            "arguments": json.dumps({"text": text}, ensure_ascii=False),
        },
    }]
    message: dict[str, Any] = {
        "role": "assistant",
        "content": assistant_content,
        "tool_calls": normalized_tool_calls,
    }
    if assistant_reasoning:
        message["reasoning"] = assistant_reasoning
    elif thinking:
        if use_reasoning_field and not _looks_like_json_object(thinking):
            message["reasoning"] = thinking
        else:
            message["content"] = thinking
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


def _is_restorable_ai_context(ai_messages: Any) -> bool:
    """Return True when saved raw AI context looks usable for resume."""
    if not isinstance(ai_messages, list) or len(ai_messages) < 3:
        return False
    if ai_messages[0].get("role") != "system":
        return False
    return any(msg.get("tool_calls") for msg in ai_messages)


def _rebuild_ai_context_from_turns(
    ai_system_prompt: str,
    turns: list[dict[str, Any]],
    *,
    use_reasoning_field: bool = False,
) -> list[dict[str, Any]]:
    """Rebuild AI-side tool history from saved turns.

    This is a fallback for older or partial runs where the raw AI context file was
    not saved correctly.
    """
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
        }
        for turn in turns
    ]


def _split_thinking_and_message(text: str) -> tuple[str | None, str | None]:
    """Separate JSON thinking from visible message.

    Returns (visible_text, thinking). If no JSON thinking found, all text is visible.
    Handles both valid and truncated JSON objects.
    """
    stripped = text.strip()
    if not stripped.startswith("{"):
        return text, None

    # Find the end of the JSON object
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
        # Truncated JSON — the entire text is thinking with no message
        return None, stripped

    json_str = stripped[: json_end + 1]
    remainder = stripped[json_end + 1 :].strip()

    thinking = json_str
    visible = remainder if remainder else None
    return visible, thinking


def _format_thinking_markdown(text: str, label: str = "💭 Thinking:") -> str:
    """Render multi-line AI thinking as a stable markdown blockquote."""
    lines = text.splitlines() or [text]
    formatted = [f"> {label}"]
    for line in lines:
        formatted.append(f"> {line}" if line else ">")
    return "\n".join(formatted)


def _extract_tool_call_reasoning(turn: ConversationTurn) -> str | None:
    """Extract the reasoning argument from the write_message_to_human tool call, if present."""
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


def _write_conversation_markdown(f: Any, record: ConversationRecord) -> None:
    """Write the human-readable markdown for a conversation to an open file handle."""
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
        f.write("## System Events\n\n")
        for event in record.events:
            parts = [f"turn {event.turn_number}", f"{event.event_type} ({event.source})"]
            if event.current_topic:
                parts.append(f"topic={event.current_topic}")
            if event.topic_changed is not None:
                parts.append(f"changed={event.topic_changed}")
            if event.nudge_injected is not None:
                parts.append(f"nudge_injected={event.nudge_injected}")
            if event.suppression_reason:
                parts.append(f"suppressed={event.suppression_reason}")
            f.write(f"- {' · '.join(parts)}\n")
            if event.message:
                f.write(f"  - note: {event.message}\n")
        f.write("\n")
    f.write("---\n\n")

    for turn in record.turns:
        if turn.speaker == "human":
            f.write(f"### 👤 {record.human_profile['name']} (turn {turn.turn_number})\n\n")
            if turn.human_context_tokens or turn.ai_context_tokens:
                f.write(
                    f"*👤 human ctx: {turn.human_context_tokens:,} tok"
                    f" · 🧠 AI ctx: {turn.ai_context_tokens:,} tok*\n\n"
                )
            if turn.human_reasoning:
                f.write(f"{_format_thinking_markdown(turn.human_reasoning)}\n\n")
            f.write(f"{turn.visible_text}\n\n")
        else:
            f.write(f"### 🤖 AI (turn {turn.turn_number})\n\n")
            if turn.ai_context_tokens or turn.human_context_tokens:
                f.write(
                    f"*🧠 AI ctx: {turn.ai_context_tokens:,} tok"
                    f" · 👤 human ctx: {turn.human_context_tokens:,} tok*\n\n"
                )
            native = turn.ai_reasoning
            tool_inner = _extract_tool_call_reasoning(turn)
            text_first = _tool_call_text_before_reasoning(turn)
            inline = turn.ai_content
            if native:
                f.write(f"{_format_thinking_markdown(native, '🧠 Native reasoning:')}\n\n")
            if text_first:
                # The AI wrote the reply text before the inner monologue —
                # render in actual transmission order so the log is honest.
                f.write(f"{turn.visible_text}\n\n")
                if tool_inner and tool_inner != native:
                    f.write(f"{_format_thinking_markdown(tool_inner, '💭 Inner monologue (after reply):')}\n\n")
                if inline and inline not in (native, tool_inner):
                    f.write(f"{_format_thinking_markdown(inline, '📋 Response draft (after reply):')}\n\n")
                if not native and not tool_inner and not inline and turn.ai_thinking:
                    f.write(f"{_format_thinking_markdown(turn.ai_thinking)}\n\n")
            else:
                shown_any = bool(native)
                if tool_inner and tool_inner != native:
                    f.write(f"{_format_thinking_markdown(tool_inner, '💭 Inner monologue:')}\n\n")
                    shown_any = True
                if inline and inline not in (native, tool_inner):
                    f.write(f"{_format_thinking_markdown(inline, '📋 Response draft:')}\n\n")
                    shown_any = True
                if not shown_any and turn.ai_thinking:
                    f.write(f"{_format_thinking_markdown(turn.ai_thinking)}\n\n")
                f.write(f"{turn.visible_text}\n\n")


class ConversationGenerator:
    """Generates a single conversation between a human simulator and an AI companion."""

    def __init__(
        self,
        client: OpenRouterClient,
        human_profile: dict[str, str],
        *,
        ai_model: str = AI_MODEL,
        human_model: str = HUMAN_MODEL,
        target_tokens: int = TARGET_TOKENS,
        max_turns: int = MAX_TURNS,
        language: str = "english",
        companion_mode: str = COMPANION_MODE,
        verbose: bool = False,
        ai_provider: dict | None = AI_PROVIDER,
        ai_reasoning: dict | None = AI_REASONING,
        ai_temperature: float = AI_TEMPERATURE,
        ai_max_tokens: int = AI_MAX_TOKENS,
        human_provider: dict | None = HUMAN_PROVIDER,
        human_reasoning: dict | None = HUMAN_REASONING,
        human_temperature: float = HUMAN_TEMPERATURE,
        human_max_tokens: int = HUMAN_MAX_TOKENS,
    ) -> None:
        self.client = client
        self.human_profile = human_profile
        self.ai_model = ai_model
        self.human_model = human_model
        self.target_tokens = target_tokens
        self.max_turns = max_turns
        self.language = language.lower()
        self.companion_mode = companion_mode
        self.verbose = verbose
        self.ai_provider = ai_provider
        self.ai_reasoning = ai_reasoning
        self.ai_temperature = ai_temperature
        self.ai_max_tokens = ai_max_tokens
        self.human_provider = human_provider
        self.human_reasoning = human_reasoning
        self.human_temperature = human_temperature
        self.human_max_tokens = human_max_tokens

        self._seed_words = generate_seed(5)
        self._ai_system_prompt = build_ai_system_prompt(
            self._seed_words,
            companion_mode=self.companion_mode,
        )
        self._conversation_plan: str = ""

        # AI context: system + conversation history (tool-role format)
        self._ai_messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._ai_system_prompt},
            {"role": "user", "content": (
                "Say hello to the human. Use write_message_to_human "
                "with a single brief greeting — one or two words."
            )},
        ]

        # Human messages will be initialized after plan generation
        self._human_messages: list[dict[str, Any]] = []

        self._last_tool_call_id: str | None = None
        self._current_topic: str | None = None
        self._last_human_nudge_turn: int | None = None
        self._record = ConversationRecord(
            id=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
            human_profile=human_profile,
            ai_model=ai_model,
            human_model=human_model,
            seed_words=self._seed_words,
            language=self.language,
            companion_mode=self.companion_mode,
            ai_provider=ai_provider,
            ai_reasoning=ai_reasoning,
            ai_temperature=ai_temperature,
            ai_max_tokens=ai_max_tokens,
            human_provider=human_provider,
            human_reasoning=human_reasoning,
            human_temperature=human_temperature,
            human_max_tokens=human_max_tokens,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

    def _record_event(
        self,
        *,
        event_type: str,
        turn_number: int,
        source: str,
        message: str | None = None,
        previous_topic: str | None = None,
        current_topic: str | None = None,
        topic_changed: bool | None = None,
        nudge_injected: bool | None = None,
        suppression_reason: str | None = None,
    ) -> None:
        self._record.events.append(
            ConversationEvent(
                event_type=event_type,
                turn_number=turn_number,
                source=source,
                timestamp=datetime.now(timezone.utc).isoformat(),
                message=message,
                previous_topic=previous_topic,
                current_topic=current_topic,
                topic_changed=topic_changed,
                nudge_injected=nudge_injected,
                suppression_reason=suppression_reason,
            )
        )

    def _queue_human_nudge(
        self,
        *,
        turn_number: int,
        source: str,
        content: str,
    ) -> tuple[bool, str | None]:
        suppression_reason = None
        if self._last_human_nudge_turn == turn_number:
            suppression_reason = "already_nudged_this_turn"
        else:
            self._human_messages.append({
                "role": "user",
                "content": content,
            })
            self._last_human_nudge_turn = turn_number

        injected = suppression_reason is None
        self._record_event(
            event_type="human_nudge",
            turn_number=turn_number,
            source=source,
            message=content if injected else None,
            nudge_injected=injected,
            suppression_reason=suppression_reason,
        )
        return injected, suppression_reason

    def _check_topic_staleness(self, turn_number: int) -> None:
        """Use a cheap judge model to check if the conversation topic has changed.
        If the topic is stale, inject a nudge to the human to change it."""
        recent_turns = self._record.turns[-20:]
        if not recent_turns:
            return

        lines: list[str] = []
        for t in recent_turns:
            text = (t.visible_text or "").strip()
            if not text:
                continue
            lines.append(f"{t.speaker.upper()}: {text}")
        if not lines:
            return
        formatted = "\n".join(lines)
        prompt = (
            "You are a conversation topic analyzer. Below are the last messages from a conversation.\n\n"
            f"Previous main topic: {self._current_topic or 'unknown (conversation just started)'}\n\n"
            f"Messages:\n{formatted}\n\n"
            'Answer in JSON:\n'
            '{"topic_changed": true/false, "current_topic": "brief 3-5 word topic description"}\n\n'
            "If the conversation has been on the same topic for these messages, set topic_changed to false."
        )
        messages = [
            {"role": "system", "content": "You are a conversation topic analyzer. Respond only in valid JSON."},
            {"role": "user", "content": prompt},
        ]
        try:
            response = self.client.chat(
                model=JUDGE_MODEL,
                messages=messages,
                max_tokens=JUDGE_MAX_TOKENS,
                temperature=0.0,
            )
            raw = (response.content or "").strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0]
            result = json.loads(raw.strip())
            topic_changed = result.get("topic_changed", False)
            current_topic = result.get("current_topic", "unknown")
        except Exception as exc:
            console.print(f"  [dim yellow]Topic judge error: {exc}[/dim yellow]")
            return

        previous_topic = self._current_topic
        self._current_topic = current_topic

        nudge_injected: bool | None = None
        suppression_reason: str | None = None
        if not topic_changed:
            nudge_injected, suppression_reason = self._queue_human_nudge(
                turn_number=turn_number,
                source="topic_judge",
                content=_TOPIC_STALE_NOTE,
            )

        self._record_event(
            event_type="topic_judge",
            turn_number=turn_number,
            source="topic_judge",
            previous_topic=previous_topic,
            current_topic=current_topic,
            topic_changed=topic_changed,
            nudge_injected=nudge_injected,
            suppression_reason=suppression_reason,
        )

        if topic_changed:
            status = "changed"
        elif nudge_injected:
            status = "STALE -> nudge injected"
        else:
            status = f"STALE -> nudge suppressed ({suppression_reason})"
        console.print(f"  [dim]Topic judge (turn {turn_number}): {current_topic} — {status}[/dim]")

    _VALID_FINISH_REASONS = {"stop", "tool_calls", "end_turn"}

    def _get_ai_response(self) -> ParsedAIResponse:
        """Call the AI model and normalize visible text, private fields, and tool calls."""
        response = self.client.chat(
            model=self.ai_model,
            messages=self._ai_messages,
            max_tokens=self.ai_max_tokens,
            temperature=self.ai_temperature,
            tools=AI_TOOLS,
            tool_choice={"type": "function", "function": {"name": "write_message_to_human"}},
            provider=self.ai_provider,
            reasoning=self.ai_reasoning,
        )

        healed = _heal_tool_call_names(response.tool_calls)
        if healed:
            console.print(f"[dim yellow]Healed {healed} garbled tool-call name(s)[/dim yellow]")

        fr = (response.finish_reason or "").strip()
        if fr and fr not in self._VALID_FINISH_REASONS:
            console.print(f"[yellow]AI finish_reason: {fr} — retrying[/yellow]")
            return ParsedAIResponse(None, None, None)

        assistant_content = response.content
        assistant_reasoning = response.reasoning
        visible_text, tc_id, tool_reasoning = extract_tool_call_text(response)

        if visible_text is not None:
            # Tool call was used — but the model may have put JSON thinking
            # inside the tool call text instead of in the content field
            if visible_text.strip().startswith("{"):
                msg_part, json_part = _split_thinking_and_message(visible_text)
                if json_part:
                    if not tool_reasoning and not assistant_reasoning and not assistant_content:
                        assistant_content = json_part
                    visible_text = msg_part
        elif assistant_content:
            # Model returned plain content without using the tool — reject and signal a retry
            # so the history is never contaminated with synthetic tool-call entries.
            return ParsedAIResponse(None, None, None, rejection_reason="no tool call")

        display_thinking = assistant_reasoning or tool_reasoning or assistant_content

        return ParsedAIResponse(
            visible_text=visible_text,
            display_thinking=display_thinking,
            tool_call_id=tc_id,
            assistant_content=assistant_content,
            assistant_reasoning=assistant_reasoning,
            tool_calls=copy.deepcopy(response.tool_calls),
            usage=response.usage,
        )

    def _get_human_response(self) -> tuple[str, str | None, list[dict[str, Any]] | None, Usage]:
        """Call the human simulator model. Returns (content, reasoning, reasoning_details, usage)."""
        response = self.client.chat(
            model=self.human_model,
            messages=self._human_messages,
            max_tokens=self.human_max_tokens,
            temperature=self.human_temperature,
            provider=self.human_provider,
            reasoning=self.human_reasoning,
        )
        return response.content or "", response.reasoning, response.reasoning_details, response.usage

    def _add_ai_turn_to_contexts(
        self,
        response: ParsedAIResponse,
    ) -> str:
        """Add an AI turn to both context histories. Returns the tool_call_id used."""
        if not response.tool_call_id:
            raise ValueError(
                "_add_ai_turn_to_contexts called without a tool_call_id — "
                "content-only responses must be rejected in _get_ai_response before reaching here"
            )
        ai_msg = _build_ai_tool_message(
            response.visible_text or "",
            response.tool_call_id,
            thinking=response.display_thinking,
            assistant_content=response.assistant_content,
            assistant_reasoning=response.assistant_reasoning,
            tool_calls=response.tool_calls,
            use_reasoning_field=_uses_native_reasoning_field(self.ai_reasoning),
        )

        self._ai_messages.append(ai_msg)

        # For human context: AI messages appear as "user" messages
        # (the human model sees AI messages as incoming user messages it needs to reply to)
        self._human_messages.append({
            "role": "user",
            "content": response.visible_text,
        })

        self._last_tool_call_id = response.tool_call_id
        return response.tool_call_id or ""

    def _add_human_turn_to_contexts(
        self,
        text: str,
        *,
        is_first: bool = False,
        reasoning: str | None = None,
        reasoning_details: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add a human turn to both context histories."""
        # For AI context: add as tool result
        if self._last_tool_call_id:
            self._ai_messages.append(
                make_human_tool_result(text, self._last_tool_call_id)
            )
        else:
            # First message — need a greeting from AI first
            greeting_msg, tc_id = make_ai_greeting_turn()
            self._ai_messages.append(greeting_msg)
            self._ai_messages.append(make_human_tool_result(text, tc_id))
            self._last_tool_call_id = tc_id

        # For human context: the human's own messages appear as "assistant"
        # (since in the human model's context, the human IS the assistant).
        # Include reasoning if present so the cache prefix matches exactly.
        human_msg: dict[str, Any] = {
            "role": "assistant",
            "content": text,
        }
        if reasoning:
            human_msg["reasoning"] = reasoning
        if reasoning_details:
            human_msg["reasoning_details"] = reasoning_details
        self._human_messages.append(human_msg)

    def _log_turn(self, turn: ConversationTurn) -> None:
        """Display turn info — compact by default, full panels in verbose mode."""
        if not self.verbose:
            if turn.speaker == "human":
                label = f"[bold blue]👤 {self.human_profile['name']}[/bold blue]"
            else:
                label = "[bold green]🤖 AI[/bold green]"

            preview = turn.visible_text[:80].replace("\n", " ")
            if len(turn.visible_text) > 80:
                preview += "…"
            console.print(f"  {label} t{turn.turn_number}: {preview}")
            return

        name = self.human_profile["name"]

        if turn.speaker == "human":
            if turn.human_reasoning:
                console.print()
                console.print(
                    Panel(
                        turn.human_reasoning,
                        title=f"💭 {name} thinking  [dim]turn {turn.turn_number}[/dim]",
                        border_style="dim cyan",
                        padding=(0, 1),
                    )
                )
            console.print()
            console.print(
                Panel(
                    turn.visible_text,
                    title=f"👤 {name}  [dim]turn {turn.turn_number}[/dim]",
                    border_style="blue",
                    padding=(0, 1),
                )
            )
        else:
            native = turn.ai_reasoning
            tool_inner = _extract_tool_call_reasoning(turn)
            text_first = _tool_call_text_before_reasoning(turn)
            inline = turn.ai_content
            if native:
                console.print()
                console.print(Panel(
                    native,
                    title=f"🧠 Native reasoning  [dim]turn {turn.turn_number}[/dim]",
                    border_style="dim yellow",
                    padding=(0, 1),
                ))
            if text_first:
                # AI answered before reasoning — show in actual order.
                console.print(Panel(
                    turn.visible_text,
                    title=f"🤖 AI  [dim]turn {turn.turn_number}[/dim]",
                    border_style="green",
                    padding=(0, 1),
                ))
                if tool_inner and tool_inner != native:
                    console.print()
                    console.print(Panel(
                        tool_inner,
                        title=f"💭 Inner monologue (after reply)  [dim]turn {turn.turn_number}[/dim]",
                        border_style="dim magenta",
                        padding=(0, 1),
                    ))
                if inline and inline not in (native, tool_inner):
                    console.print()
                    console.print(Panel(
                        inline,
                        title=f"📋 Response draft (after reply)  [dim]turn {turn.turn_number}[/dim]",
                        border_style="dim cyan",
                        padding=(0, 1),
                    ))
                if not native and not tool_inner and not inline and turn.ai_thinking:
                    console.print()
                    console.print(Panel(
                        turn.ai_thinking,
                        title=f"🧠 AI thinking  [dim]turn {turn.turn_number}[/dim]",
                        border_style="dim yellow",
                        padding=(0, 1),
                    ))
            else:
                shown_any = bool(native)
                if tool_inner and tool_inner != native:
                    console.print()
                    console.print(Panel(
                        tool_inner,
                        title=f"💭 Inner monologue  [dim]turn {turn.turn_number}[/dim]",
                        border_style="dim magenta",
                        padding=(0, 1),
                    ))
                    shown_any = True
                if inline and inline not in (native, tool_inner):
                    console.print()
                    console.print(Panel(
                        inline,
                        title=f"📋 Response draft  [dim]turn {turn.turn_number}[/dim]",
                        border_style="dim cyan",
                        padding=(0, 1),
                    ))
                    shown_any = True
                if not shown_any and turn.ai_thinking:
                    thinking_display = turn.ai_thinking
                    try:
                        parsed = json.loads(turn.ai_thinking)
                        if parsed.get("reasoning"):
                            thinking_display = f"🧠 {parsed['reasoning']}"
                        elif parsed.get("thoughts"):
                            # Backward compat with older conversations
                            thinking_display = f"💭 {parsed['thoughts']}"
                    except (json.JSONDecodeError, AttributeError):
                        pass
                    console.print()
                    console.print(Panel(
                        thinking_display,
                        title=f"🧠 AI thinking  [dim]turn {turn.turn_number}[/dim]",
                        border_style="dim yellow",
                        padding=(0, 1),
                    ))
                console.print(Panel(
                    turn.visible_text,
                    title=f"🤖 AI  [dim]turn {turn.turn_number}[/dim]",
                    border_style="green",
                    padding=(0, 1),
                ))

    def _generate_conversation_plan(self) -> str:
        """Generate the human's conversation plan before starting the dialogue."""
        console.print("  [dim]Generating conversation plan...[/dim]")
        plan_prompt = CONVERSATION_PLAN_PROMPT.format(**self.human_profile)
        if self.language != "english":
            plan_prompt += f"\n\nIMPORTANT: Write the entire plan in {self.language.upper()}."
        plan_messages = [
            {"role": "system", "content": "You are a creative writer preparing for a roleplay exercise."},
            {"role": "user", "content": plan_prompt},
        ]
        response = self.client.chat(
            model=self.human_model,
            messages=plan_messages,
            max_tokens=1500,
            temperature=0.95,
            provider=self.human_provider,
        )
        plan = response.content or ""
        console.print(f"  [dim]Plan generated ({_estimate_tokens(plan)} tokens)[/dim]")
        if self.verbose and plan:
            console.print()
            console.print(
                Panel(
                    plan,
                    title="📋 Conversation plan",
                    border_style="cyan",
                    padding=(0, 1),
                )
            )
        return plan

    def _init_human_context(self) -> None:
        """Initialize the human model's context after plan generation."""
        self._human_system_prompt = build_human_system_prompt(
            self.human_profile, self._conversation_plan, self.language
        )
        self._human_messages = [
            {"role": "system", "content": self._human_system_prompt},
            {
                "role": "user",
                "content": (
                    "[You just opened a chat with a new AI companion. "
                    "Send your first message — keep it casual and short, "
                    "like you'd text a new friend. Just say hi.]"
                ),
            },
        ]

    def _run_loop(self, start_turn: int, start_tokens: int) -> ConversationRecord:
        """Core conversation loop used by both generate() and resume().

        Alternates AI/human turns starting from the given turn number,
        accumulating tokens from start_tokens.
        """
        turn_number = start_turn
        accumulated_tokens = start_tokens
        consecutive_empty = 0
        max_consecutive_empty = 5

        # Determine whose turn it is: if last turn was human, AI goes next; vice versa.
        last_speaker = self._record.turns[-1].speaker if self._record.turns else None
        need_human_first = last_speaker == "ai" or last_speaker is None

        if need_human_first and last_speaker == "ai":
            # Human's turn — AI already spoke last
            turn_number += 1
            cost_before = self.client.total_cost
            human_text, human_reasoning, human_reasoning_details, human_usage = self._get_human_response()

            if not human_text or not human_text.strip():
                _got_response = False
                for _retry in range(1, max_consecutive_empty):
                    wait = min(2 ** _retry, 16)
                    console.print(f"[yellow]Human produced empty response ({_retry}/{max_consecutive_empty}), retrying in {wait}s with nudge...[/yellow]")
                    time.sleep(wait)
                    self._human_messages.append({
                        "role": "user",
                        "content": "(The AI just said something. Please respond naturally.)",
                    })
                    human_text, human_reasoning, human_reasoning_details, human_usage = self._get_human_response()
                    self._human_messages.pop(-1)
                    if human_text and human_text.strip():
                        _got_response = True
                        break
                if not _got_response:
                    console.print("[bold yellow]Human still empty after max retries — ending conversation.[/bold yellow]")
                    turn_number = self.max_turns  # skip the main while loop
            human_cost = self.client.total_cost - cost_before

            if turn_number < self.max_turns:
                consecutive_empty = 0
                self._add_human_turn_to_contexts(human_text, reasoning=human_reasoning, reasoning_details=human_reasoning_details)

                human_tokens = _estimate_tokens(human_text)
                accumulated_tokens += human_tokens

                human_ctx = (
                    (human_usage.prompt_tokens + human_usage.completion_tokens)
                    if human_usage and human_usage.prompt_tokens
                    else _estimate_context_tokens(self._human_messages)
                )
                human_turn = ConversationTurn(
                    turn_number=turn_number,
                    speaker="human",
                    visible_text=human_text,
                    human_reasoning=human_reasoning,
                    human_reasoning_details=human_reasoning_details,
                    token_estimate=human_tokens,
                    cost_usd=human_cost,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    ai_context_tokens=_estimate_context_tokens(self._ai_messages),
                    human_context_tokens=human_ctx,
                )
                self._record.turns.append(human_turn)
                self._log_turn(human_turn)

        while turn_number < self.max_turns:
            # AI turn
            turn_number += 1
            cost_before = self.client.total_cost
            ai_response = self._get_ai_response()

            if not ai_response.visible_text:
                consecutive_empty += 1
                wait = min(2 ** consecutive_empty, 32)
                reason = ai_response.rejection_reason or "empty response"
                console.print(f"[yellow]AI produced incorrect response ({reason}) — attempt {consecutive_empty}/{max_consecutive_empty}, retrying in {wait}s...[/yellow]")
                turn_number -= 1
                if consecutive_empty >= max_consecutive_empty:
                    console.print("[bold yellow]Too many consecutive empty AI responses — ending conversation.[/bold yellow]")
                    break
                time.sleep(wait)
                continue
            consecutive_empty = 0

            ai_cost = self.client.total_cost - cost_before
            tc_used = self._add_ai_turn_to_contexts(ai_response)

            ai_tokens = (
                _estimate_tokens(ai_response.visible_text)
                + _estimate_tokens(ai_response.assistant_content or "")
                + _estimate_tokens(ai_response.assistant_reasoning or "")
            )
            accumulated_tokens += ai_tokens

            # Use actual prompt+completion tokens from the API response for accurate context
            # tracking; fall back to character-based estimate when unavailable.
            ai_ctx = (
                (ai_response.usage.prompt_tokens + ai_response.usage.completion_tokens)
                if ai_response.usage and ai_response.usage.prompt_tokens
                else _estimate_context_tokens(self._ai_messages)
            )

            ai_turn = ConversationTurn(
                turn_number=turn_number,
                speaker="ai",
                visible_text=ai_response.visible_text,
                ai_thinking=ai_response.display_thinking,
                ai_content=ai_response.assistant_content,
                ai_reasoning=ai_response.assistant_reasoning,
                ai_tool_calls=copy.deepcopy(ai_response.tool_calls),
                token_estimate=ai_tokens,
                cost_usd=ai_cost,
                timestamp=datetime.now(timezone.utc).isoformat(),
                ai_context_tokens=ai_ctx,
                human_context_tokens=_estimate_context_tokens(self._human_messages),
            )
            self._record.turns.append(ai_turn)
            self._log_turn(ai_turn)

            # B3: Human emulator refresh — force a life event / topic change
            # (checked after AI turn where turn_number is even, so %80 works)
            if turn_number % 80 == 0 and turn_number > 0:
                injected, suppression_reason = self._queue_human_nudge(
                    turn_number=turn_number,
                    source="b3_refresh",
                    content=_B3_REFRESH_NOTE,
                )
                refresh_status = "nudge injected" if injected else f"nudge suppressed ({suppression_reason})"
                console.print(f"  [dim]Human refresh (turn {turn_number}): {refresh_status}[/dim]")

            # Topic judge: check for topic staleness
            # (checked after AI turn where turn_number is even, so %INTERVAL works)
            if turn_number % TOPIC_CHECK_INTERVAL == 0 and turn_number > 0:
                self._check_topic_staleness(turn_number)

            context_tokens = ai_ctx

            show_progress = (
                self.verbose
                or turn_number % 10 == 0
            )
            if show_progress:
                console.print(
                    f"  [dim]— progress: turn {turn_number} | {context_tokens:,}/{self.target_tokens:,} tok | ${self.client.total_cost:.4f}[/dim]"
                )

            if context_tokens >= self.target_tokens:
                console.print(
                    f"\n[bold green]✓ Reached target token count ({context_tokens:,} >= {self.target_tokens:,})[/bold green]"
                )
                break

            # Human turn
            turn_number += 1
            cost_before = self.client.total_cost
            human_text, human_reasoning, human_reasoning_details, human_usage = self._get_human_response()
            human_cost = self.client.total_cost - cost_before

            if not human_text or not human_text.strip():
                _got_response = False
                for _retry in range(1, max_consecutive_empty):
                    wait = min(2 ** _retry, 16)
                    console.print(f"[yellow]Human produced empty response ({_retry}/{max_consecutive_empty}), retrying in {wait}s with nudge...[/yellow]")
                    time.sleep(wait)
                    self._human_messages.append({
                        "role": "user",
                        "content": "(The AI just said something. Please respond naturally as yourself — share your thoughts, tell a story, bring up a new topic, or react to what they said.)",
                    })
                    human_text, human_reasoning, human_reasoning_details, human_usage = self._get_human_response()
                    self._human_messages.pop(-1)
                    if human_text and human_text.strip():
                        _got_response = True
                        break
                if not _got_response:
                    console.print("[bold yellow]Human still empty after max retries — ending conversation.[/bold yellow]")
                    turn_number -= 1
                    break
            consecutive_empty = 0

            self._add_human_turn_to_contexts(human_text, reasoning=human_reasoning, reasoning_details=human_reasoning_details)

            human_tokens = _estimate_tokens(human_text)
            accumulated_tokens += human_tokens

            human_ctx = (
                (human_usage.prompt_tokens + human_usage.completion_tokens)
                if human_usage and human_usage.prompt_tokens
                else _estimate_context_tokens(self._human_messages)
            )
            human_turn = ConversationTurn(
                turn_number=turn_number,
                speaker="human",
                visible_text=human_text,
                human_reasoning=human_reasoning,
                human_reasoning_details=human_reasoning_details,
                token_estimate=human_tokens,
                cost_usd=human_cost,
                timestamp=datetime.now(timezone.utc).isoformat(),
                ai_context_tokens=_estimate_context_tokens(self._ai_messages),
                human_context_tokens=human_ctx,
            )
            self._record.turns.append(human_turn)
            self._log_turn(human_turn)

        # Finalize
        self._record.total_tokens_estimate = accumulated_tokens
        self._record.total_cost_usd = self.client.total_cost
        self._record.finished_at = datetime.now(timezone.utc).isoformat()
        self._record.ai_messages_raw = self._ai_messages

        console.print(f"\n[bold]Conversation complete![/bold]")
        console.print(f"  Turns: {turn_number}")
        console.print(f"  Estimated tokens: {accumulated_tokens:,}")
        console.print(f"  Total cost: ${self.client.total_cost:.4f}")

        return self._record

    def generate(self) -> ConversationRecord:
        """Run the full conversation generation loop."""
        reset_tool_call_counter()

        console.print(f"\n[bold]Starting conversation with {self.human_profile['name']}[/bold]")
        console.print(f"  AI model: {self.ai_model}")
        console.print(f"  Human model: {self.human_model}")
        if self.ai_provider:
            console.print(f"  AI provider: {self.ai_provider}")
        if self.ai_reasoning:
            console.print(f"  AI reasoning: {self.ai_reasoning}")
        if self.human_provider:
            console.print(f"  Human provider: {self.human_provider}")
        if self.human_reasoning:
            console.print(f"  Human reasoning: {self.human_reasoning}")
        console.print(f"  Target: ~{self.target_tokens:,} tokens")
        console.print(f"  Seed: {', '.join(self._seed_words)}")

        self._conversation_plan = self._generate_conversation_plan()
        self._record.conversation_plan = self._conversation_plan
        self._init_human_context()
        console.print()

        # --- Bootstrap: AI sends its real first greeting ---
        console.print("  [dim]Getting AI initial greeting...[/dim]")
        ai_greeting: ParsedAIResponse | None = None
        for _attempt in range(5):
            resp = self._get_ai_response()
            if resp.visible_text and resp.tool_call_id:
                ai_greeting = resp
                break
            wait = min(2 ** (_attempt + 1), 16)
            reason = resp.rejection_reason or "empty response"
            console.print(
                f"  [yellow]AI greeting attempt {_attempt + 1}/5 failed ({reason}), "
                f"retrying in {wait}s...[/yellow]"
            )
            time.sleep(wait)

        if ai_greeting is not None:
            # Add the real greeting to AI context only — the human simulator
            # responds to its own trigger prompt, so the greeting must NOT be
            # injected into the human context.
            ai_msg = _build_ai_tool_message(
                ai_greeting.visible_text or "",
                ai_greeting.tool_call_id,
                thinking=ai_greeting.display_thinking,
                assistant_content=ai_greeting.assistant_content,
                assistant_reasoning=ai_greeting.assistant_reasoning,
                tool_calls=ai_greeting.tool_calls,
                use_reasoning_field=_uses_native_reasoning_field(self.ai_reasoning),
            )
            self._ai_messages.append(ai_msg)
            self._last_tool_call_id = ai_greeting.tool_call_id
        else:
            console.print("  [yellow]AI greeting failed after 5 attempts — using fallback greeting[/yellow]")
            greeting_msg, tc_id = make_ai_greeting_turn()
            self._ai_messages.append(greeting_msg)
            self._last_tool_call_id = tc_id

        # --- Turn 1: Human opens the conversation ---
        turn_number = 1
        cost_before = self.client.total_cost
        human_text, human_reasoning, human_reasoning_details, human_usage = self._get_human_response()

        if not human_text or not human_text.strip():
            _got_response = False
            for _retry in range(1, 5):
                wait = min(2 ** _retry, 16)
                console.print(f"[yellow]Human produced empty first response ({_retry}/5), retrying in {wait}s...[/yellow]")
                time.sleep(wait)
                human_text, human_reasoning, human_reasoning_details, human_usage = self._get_human_response()
                if human_text and human_text.strip():
                    _got_response = True
                    break
            if not _got_response:
                console.print("[yellow]Human produced empty first response after max retries, using fallback[/yellow]")
                human_text = f"Hey there! I'm {self.human_profile['name']}. Just wanted to say hi and see how you're doing. I'm really curious to get to know you."
                human_reasoning = None
                human_reasoning_details = None
        human_cost = self.client.total_cost - cost_before

        self._add_human_turn_to_contexts(human_text, is_first=True, reasoning=human_reasoning, reasoning_details=human_reasoning_details)

        human_tokens = _estimate_tokens(human_text)

        human_ctx = (
            (human_usage.prompt_tokens + human_usage.completion_tokens)
            if human_usage and human_usage.prompt_tokens
            else _estimate_context_tokens(self._human_messages)
        )
        human_turn = ConversationTurn(
            turn_number=1,
            speaker="human",
            visible_text=human_text,
            human_reasoning=human_reasoning,
            human_reasoning_details=human_reasoning_details,
            token_estimate=human_tokens,
            cost_usd=human_cost,
            timestamp=datetime.now(timezone.utc).isoformat(),
            ai_context_tokens=_estimate_context_tokens(self._ai_messages),
            human_context_tokens=human_ctx,
        )
        self._record.turns.append(human_turn)
        self._log_turn(human_turn)

        return self._run_loop(start_turn=1, start_tokens=human_tokens)

    @classmethod
    def resume(
        cls,
        client: OpenRouterClient,
        jsonl_path: str | Path,
        *,
        target_tokens: int = TARGET_TOKENS,
        verbose: bool = False,
        language_override: str | None = None,
        ai_model_override: str | None = None,
        human_model_override: str | None = None,
        ai_provider_override: object = _UNSET,
        human_provider_override: object = _UNSET,
        ai_temperature_override: float | None = None,
        human_temperature_override: float | None = None,
        ai_max_tokens_override: int | None = None,
        human_max_tokens_override: int | None = None,
    ) -> ConversationRecord:
        """Resume a conversation from a saved JSONL file."""
        jsonl_path = Path(jsonl_path)
        base = jsonl_path.stem  # e.g. conv_20260326_221953_michael
        raw_json_path = jsonl_path.parent / f"{base}_raw_ai_context.json"

        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
        if not raw_json_path.exists():
            raise FileNotFoundError(f"Raw AI context not found: {raw_json_path}")

        # Load metadata and turns from JSONL
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
        # Restore saved provider; fall back to current AI_PROVIDER for old files
        # that pre-date this field. Override wins if explicitly passed.
        saved_ai_provider = metadata.get("ai_provider", AI_PROVIDER)
        ai_provider = ai_provider_override if ai_provider_override is not _UNSET else saved_ai_provider
        saved_human_provider = metadata.get("human_provider", HUMAN_PROVIDER)
        human_provider = human_provider_override if human_provider_override is not _UNSET else saved_human_provider
        # Restore inference params; fall back to current config values for old files.
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

        # Load raw AI context
        with open(raw_json_path, "r", encoding="utf-8") as f:
            ai_messages = json.load(f)

        if not _is_restorable_ai_context(ai_messages):
            ai_messages = _rebuild_ai_context_from_turns(
                build_ai_system_prompt(
                    seed_words,
                    companion_mode=companion_mode,
                ),
                turns,
                use_reasoning_field=_uses_native_reasoning_field(ai_reasoning),
            )
            console.print(
                "  [dim yellow]Raw AI context missing or incomplete; rebuilt from saved turns.[/dim yellow]"
            )

        migrated_reasoning = _migrate_assistant_reasoning_fields(
            ai_messages,
            use_reasoning_field=_uses_native_reasoning_field(ai_reasoning),
        )
        if migrated_reasoning > 0:
            console.print(
                f"  [dim]Migrated {migrated_reasoning} assistant messages from content to reasoning.[/dim]"
            )

        # Clean up any Unicode-escaped tool arguments from older runs
        chars_saved = _normalize_tool_arguments(ai_messages)
        if chars_saved > 0:
            console.print(f"  [dim]Normalized Unicode escapes in context (saved ~{chars_saved:,} chars / ~{chars_saved // 4:,} tokens)[/dim]")

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
            console.print(f"  Human provider: {human_provider}" + (" [yellow](overridden)[/yellow]" if overridden_hp else ""))
        if human_reasoning:
            console.print(f"  Human reasoning: {human_reasoning}")
        temp_line = f"  AI temperature: {ai_temperature}" + (" [yellow](overridden)[/yellow]" if ai_temperature_override is not None else "")
        temp_line += f" / Human temperature: {human_temperature}" + (" [yellow](overridden)[/yellow]" if human_temperature_override is not None else "")
        console.print(temp_line)
        tokens_line = f"  AI max tokens: {ai_max_tokens}" + (" [yellow](overridden)[/yellow]" if ai_max_tokens_override is not None else "")
        tokens_line += f" / Human max tokens: {human_max_tokens}" + (" [yellow](overridden)[/yellow]" if human_max_tokens_override is not None else "")
        console.print(tokens_line)
        console.print(f"  Previous cost: ${previous_cost:.4f}")
        console.print(f"  New target: ~{target_tokens:,} tokens")
        console.print(f"  Seed: {', '.join(seed_words)}")

        # Find the highest tool call counter from the AI messages
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

        console.print(f"  Language: {language}")

        # Create generator instance
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

        # Restore cost from previous runs
        client.total_cost = previous_cost

        # Restore internal state
        gen._seed_words = seed_words
        gen._conversation_plan = conversation_plan
        gen._ai_messages = ai_messages
        gen._last_tool_call_id = last_tc_id
        topic_events = [e for e in events if e.get("event_type") == "topic_judge"]
        if topic_events:
            gen._current_topic = topic_events[-1].get("current_topic")
        nudge_events = [
            e for e in events
            if e.get("event_type") == "human_nudge" and e.get("nudge_injected")
        ]
        if nudge_events:
            gen._last_human_nudge_turn = max(e.get("turn_number", 0) for e in nudge_events)

        # Rebuild human context from turns
        gen._init_human_context()
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

        # Rebuild record
        gen._record.id = metadata["conversation_id"]
        gen._record.seed_words = seed_words
        gen._record.conversation_plan = conversation_plan
        gen._record.language = language
        gen._record.companion_mode = companion_mode
        gen._record.started_at = metadata["started_at"]
        for event_data in events:
            gen._record.events.append(ConversationEvent(
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
            ))
        for turn_data in turns:
            gen._record.turns.append(ConversationTurn(
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
            ))

        last_turn = turns[-1]["turn_number"] if turns else 0
        # Prefer the actual token count stored on the last AI turn (accurate since the
        # fix to use API-reported prompt_tokens).  Fall back to a character-based estimate
        # for older conversations that only have estimated values or no turns at all.
        last_ai_turn = next(
            (t for t in reversed(turns) if t.get("speaker") == "ai"),
            None,
        )
        stored_ctx = last_ai_turn.get("ai_context_tokens", 0) if last_ai_turn else 0
        existing_tokens = stored_ctx or _estimate_context_tokens(ai_messages)

        console.print(f"  Existing tokens: ~{existing_tokens:,}")
        console.print()

        return gen._run_loop(start_turn=last_turn, start_tokens=existing_tokens)


def load_conversation_record(jsonl_path: Path) -> ConversationRecord:
    """Load a ConversationRecord from a saved JSONL file."""
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
                turns.append(ConversationTurn(
                    turn_number=obj["turn_number"],
                    speaker=obj["speaker"],
                    visible_text=obj["visible_text"],
                    ai_thinking=obj.get("ai_thinking"),
                    ai_content=obj.get("ai_content"),
                    ai_reasoning=obj.get("ai_reasoning"),
                    ai_tool_calls=obj.get("ai_tool_calls"),
                    human_reasoning=obj.get("human_reasoning"),
                    human_reasoning_details=obj.get("human_reasoning_details"),
                    token_estimate=obj.get("token_estimate", 0),
                    cost_usd=obj.get("cost_usd", 0.0),
                    timestamp=obj.get("timestamp", ""),
                    ai_context_tokens=obj.get("ai_context_tokens", 0),
                    human_context_tokens=obj.get("human_context_tokens", 0),
                ))
            elif t == "event":
                events.append(ConversationEvent(
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
                ))

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


def save_conversation(record: ConversationRecord, output_dir: Path) -> Path:
    """Save a conversation record to JSONL and a readable markdown file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_name = record.human_profile["name"].lower()
    base = f"conv_{record.id}_{profile_name}"

    # Save JSONL (machine-readable, full data including AI context)
    jsonl_path = output_dir / f"{base}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        # First line: metadata
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
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        # Each turn as a line
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

    # Save readable markdown
    md_path = output_dir / f"{base}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        _write_conversation_markdown(f, record)

    # Save raw AI messages (for compression testing)
    raw_messages = record.ai_messages_raw
    if not _is_restorable_ai_context(raw_messages) and record.turns:
        raw_messages = _rebuild_ai_context_from_turns(
            build_ai_system_prompt(
                record.seed_words,
                companion_mode=record.companion_mode,
            ),
            _turns_to_context_rows(record.turns),
            use_reasoning_field=_uses_native_reasoning_field(AI_REASONING),
        )
        record.ai_messages_raw = raw_messages
    if record.turns and record.total_tokens_estimate <= 0:
        record.total_tokens_estimate = _estimate_context_tokens(raw_messages)

    raw_path = output_dir / f"{base}_raw_ai_context.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_messages, f, ensure_ascii=False, indent=2)

    console.print(f"\n[bold]Files saved:[/bold]")
    console.print(f"  📄 {jsonl_path}")
    console.print(f"  📖 {md_path}")
    console.print(f"  🔧 {raw_path}")

    return jsonl_path
