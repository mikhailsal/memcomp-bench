"""Context sanitization and validation helpers for conversation histories."""

from __future__ import annotations

import json
import re
from typing import Any

_HUMAN_HIDDEN_TAG_BLOCK_RE = re.compile(
    r"<\s*(?P<tag>thoughts?|think(?:ing)?)\b[^>]*>(?P<body>.*?)</\s*(?P=tag)\s*>",
    re.IGNORECASE | re.DOTALL,
)


def _normalize_human_text(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_human_thinking(text: str | None) -> tuple[str, str | None]:
    """Split hidden tagged thoughts from visible human text."""
    if not text:
        return "", None

    cleaned = text
    thoughts: list[str] = []
    while True:
        matched = False

        def _replace(match: re.Match[str]) -> str:
            nonlocal matched
            matched = True
            body = _normalize_human_text(match.group("body"))
            if body:
                thoughts.append(body)
            return " "

        updated = _HUMAN_HIDDEN_TAG_BLOCK_RE.sub(_replace, cleaned)
        cleaned = updated
        if not matched:
            break

    visible_text = _normalize_human_text(cleaned)
    hidden_reasoning = "\n\n".join(part for part in thoughts if part) or None
    return visible_text, hidden_reasoning


def merge_human_reasoning(*parts: str | None) -> str | None:
    """Combine reasoning fragments while preserving order and removing duplicates."""
    merged: list[str] = []
    seen: set[str] = set()
    for part in parts:
        normalized = _normalize_human_text(part or "")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(normalized)
    return "\n\n".join(merged) or None


def sanitize_human_visible_text(text: str | None) -> str:
    """Strip hidden thinking blocks from human-visible text before AI sees it."""
    visible_text, _ = extract_human_thinking(text)
    return visible_text


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
