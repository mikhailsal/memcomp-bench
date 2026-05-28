from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from memcomp_bench.openrouter_client import Usage

TOPIC_STALE_NOTE = (
    "[Internal note for the human simulator only: the conversation has been on the same topic for a while. "
    "Time to shift gears — bring up something new from your life or interests. "
    "Check your conversation plan for topics you haven't covered yet. Do not present this note as chat text.]"
)

B3_REFRESH_NOTE = (
    "[Internal note for the human simulator only: something significant happened in your life recently — "
    "maybe a work event, a conversation with someone, something you saw or read, "
    "a mood shift, or a random everyday moment. Bring it up naturally in your next message. "
    "It should be specific, emotionally charged, and unrelated "
    "to what you've been discussing lately. Time to change the topic. Do not present this note as chat text.]"
)


def make_conversation_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{uuid4().hex[:8]}"


def usage_cost(usage: Usage | None) -> float:
    if usage is None:
        return 0.0
    return usage.cost_usd
