"""Context sanitization and validation helpers for conversation histories."""

from __future__ import annotations

import json
import re
from typing import Any

_HUMAN_HIDDEN_TAG_BLOCK_RE = re.compile(
    r"<\s*(?P<tag>thoughts?|think(?:ing)?)\b[^>]*>.*?</\s*(?P=tag)\s*>",
    re.IGNORECASE | re.DOTALL,
)


def sanitize_human_visible_text(text: str | None) -> str:
    """Strip hidden thinking blocks from human-visible text before AI sees it."""
    if not text:
        return ""

    cleaned = text
    while True:
        updated = _HUMAN_HIDDEN_TAG_BLOCK_RE.sub(" ", cleaned)
        if updated == cleaned:
            break
        cleaned = updated

    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"[ \t]*\n[ \t]*", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def sanitize_human_tool_messages(ai_messages: list[dict[str, Any]]) -> int:
    """Sanitize persisted human tool messages in AI context. Returns number changed."""
    changed = 0
    for msg in ai_messages:
        if msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        cleaned = sanitize_human_visible_text(content)
        if cleaned != content:
            msg["content"] = cleaned
            changed += 1
    return changed


def _is_restorable_ai_context(ai_messages: Any) -> bool:
    """Return True if saved ai_messages_raw looks complete enough to resume from."""
    if not isinstance(ai_messages, list) or len(ai_messages) < 3:
        return False
    if ai_messages[0].get("role") != "system":
        return False
    return any(msg.get("tool_calls") for msg in ai_messages)


def _looks_like_json_object(text: str | None) -> bool:
    """Fast check for strings that appear to be a single JSON object."""
    if not text:
        return False
    stripped = text.strip()
    return stripped.startswith("{") and stripped.endswith("}")


def response_is_missing_mandatory_reasoning(tool_calls: list[dict[str, Any]] | None) -> bool:
    """Return True when write_message_to_human omits the required reasoning argument."""
    if not tool_calls:
        return False
    for tc in tool_calls:
        func = tc.get("function", {})
        if func.get("name") != "write_message_to_human":
            continue
        args_str = func.get("arguments", "")
        try:
            args = json.loads(args_str)
        except (json.JSONDecodeError, TypeError):
            continue
        reasoning = args.get("reasoning")
        if not isinstance(reasoning, str) or not reasoning.strip():
            return True
    return False
