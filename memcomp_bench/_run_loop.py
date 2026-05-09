"""Core conversation loop and generate() implementation — extracted from generator.py."""

from __future__ import annotations

import copy
import time
from datetime import datetime, timezone
from typing import Any

from rich.console import Console

from memcomp_bench.generator_helpers import (
    ConversationRecord,
    ConversationTurn,
    _build_ai_tool_message,
    _estimate_context_tokens,
    _estimate_tokens,
    _uses_native_reasoning_field,
)
from memcomp_bench.prompts import make_ai_greeting_turn, reset_tool_call_counter

console = Console()

_B3_REFRESH_NOTE = (
    "[System note: Something significant happened in your life recently — "
    "maybe a work event, a conversation with someone, something you saw or read, "
    "a mood shift, or a random everyday moment. Bring it up naturally in your "
    "next message. It should be specific, emotionally charged, and unrelated "
    "to what you've been discussing lately. Time to change the topic.]"
)


def do_generate(gen: Any) -> ConversationRecord:
    """Run the full conversation generation loop."""

    reset_tool_call_counter()
    console.print(f"\n[bold]Starting conversation with {gen.human_profile['name']}[/bold]")
    console.print(f"  AI model: {gen.ai_model}")
    console.print(f"  Human model: {gen.human_model}")
    if gen.ai_provider:
        console.print(f"  AI provider: {gen.ai_provider}")
    if gen.ai_reasoning:
        console.print(f"  AI reasoning: {gen.ai_reasoning}")
    if gen.human_provider:
        console.print(f"  Human provider: {gen.human_provider}")
    if gen.human_reasoning:
        console.print(f"  Human reasoning: {gen.human_reasoning}")
    console.print(f"  Target: ~{gen.target_tokens:,} tokens")
    console.print(f"  Seed: {', '.join(gen._seed_words)}")

    gen._conversation_plan = gen._generate_conversation_plan()
    gen._record.conversation_plan = gen._conversation_plan
    gen._init_human_context()
    console.print()

    _bootstrap_ai_greeting(gen)

    turn_number = 1
    cost_before = gen.client.total_cost
    human_text, human_reasoning, human_reasoning_details, human_usage = gen._get_human_response()

    if not human_text or not human_text.strip():
        human_text, human_reasoning, human_reasoning_details = _retry_first_human(gen)

    human_cost = gen.client.total_cost - cost_before
    gen._add_human_turn_to_contexts(
        human_text, is_first=True, reasoning=human_reasoning, reasoning_details=human_reasoning_details
    )
    human_tokens = _estimate_tokens(human_text)

    human_ctx = (
        (human_usage.prompt_tokens + human_usage.completion_tokens)
        if human_usage and human_usage.prompt_tokens
        else _estimate_context_tokens(gen._human_messages)
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
        ai_context_tokens=_estimate_context_tokens(gen._ai_messages),
        human_context_tokens=human_ctx,
    )
    gen._record.turns.append(human_turn)
    gen._log_turn(human_turn)

    return run_loop(gen, start_turn=1, start_tokens=human_tokens)


def _bootstrap_ai_greeting(gen: Any) -> None:
    """Get the AI's initial greeting via tool call."""
    console.print("  [dim]Getting AI initial greeting...[/dim]")
    ai_greeting = None
    for _attempt in range(5):
        resp = gen._get_ai_response()
        if resp.visible_text and resp.tool_call_id:
            ai_greeting = resp
            break
        wait = min(2 ** (_attempt + 1), 16)
        reason = resp.rejection_reason or "empty response"
        console.print(
            f"  [yellow]AI greeting attempt {_attempt + 1}/5 failed ({reason}), retrying in {wait}s...[/yellow]"
        )
        time.sleep(wait)

    if ai_greeting is not None:
        ai_msg = _build_ai_tool_message(
            ai_greeting.visible_text or "",
            ai_greeting.tool_call_id,
            thinking=ai_greeting.display_thinking,
            assistant_content=ai_greeting.assistant_content,
            assistant_reasoning=ai_greeting.assistant_reasoning,
            tool_calls=ai_greeting.tool_calls,
            reasoning_details=ai_greeting.reasoning_details,
            use_reasoning_field=_uses_native_reasoning_field(gen.ai_reasoning),
        )
        gen._ai_messages.append(ai_msg)
        gen._last_tool_call_id = ai_greeting.tool_call_id
    else:
        console.print("  [yellow]AI greeting failed after 5 attempts \u2014 using fallback greeting[/yellow]")
        greeting_msg, tc_id = make_ai_greeting_turn()
        gen._ai_messages.append(greeting_msg)
        gen._last_tool_call_id = tc_id


def _retry_first_human(gen: Any) -> tuple[str, str | None, list | None]:
    """Retry when the first human message is empty."""
    for _retry in range(1, 5):
        wait = min(2**_retry, 16)
        console.print(f"[yellow]Human produced empty first response ({_retry}/5), retrying in {wait}s...[/yellow]")
        time.sleep(wait)
        human_text, human_reasoning, human_reasoning_details, _ = gen._get_human_response()
        if human_text and human_text.strip():
            return human_text, human_reasoning, human_reasoning_details
    console.print("[yellow]Human produced empty first response after max retries, using fallback[/yellow]")
    fallback = f"Hey there! I'm {gen.human_profile['name']}. Just wanted to say hi and see how you're doing. I'm really curious to get to know you."
    return fallback, None, None


def _record_ai_turn(gen: Any, ai_response: Any, turn_number: int, ai_cost: float) -> tuple[int, int]:
    """Record an AI turn and return (ai_tokens, context_tokens)."""
    gen._add_ai_turn_to_contexts(ai_response)

    ai_tokens = (
        _estimate_tokens(ai_response.visible_text)
        + _estimate_tokens(ai_response.assistant_content or "")
        + _estimate_tokens(ai_response.assistant_reasoning or "")
    )
    ai_ctx = (
        (ai_response.usage.prompt_tokens + ai_response.usage.completion_tokens)
        if ai_response.usage and ai_response.usage.prompt_tokens
        else _estimate_context_tokens(gen._ai_messages)
    )

    ai_turn = ConversationTurn(
        turn_number=turn_number,
        speaker="ai",
        visible_text=ai_response.visible_text,
        ai_thinking=ai_response.display_thinking,
        ai_content=ai_response.assistant_content,
        ai_reasoning=ai_response.assistant_reasoning,
        ai_tool_calls=copy.deepcopy(ai_response.tool_calls),
        ai_reasoning_details=copy.deepcopy(ai_response.reasoning_details),
        token_estimate=ai_tokens,
        cost_usd=ai_cost,
        timestamp=datetime.now(timezone.utc).isoformat(),
        ai_context_tokens=ai_ctx,
        human_context_tokens=_estimate_context_tokens(gen._human_messages),
    )
    gen._record.turns.append(ai_turn)
    gen._log_turn(ai_turn)
    return ai_tokens, ai_ctx


def _check_periodic_events(gen: Any, turn_number: int) -> None:
    """Fire periodic nudges and topic checks."""
    from memcomp_bench.config import TOPIC_CHECK_INTERVAL

    if turn_number % 80 == 0 and turn_number > 0:
        injected, suppression_reason = gen._queue_human_nudge(
            turn_number=turn_number,
            source="b3_refresh",
            content=_B3_REFRESH_NOTE,
        )
        refresh_status = "nudge injected" if injected else f"nudge suppressed ({suppression_reason})"
        console.print(f"  [dim]Human refresh (turn {turn_number}): {refresh_status}[/dim]")

    if turn_number % TOPIC_CHECK_INTERVAL == 0 and turn_number > 0:
        gen._check_topic_staleness(turn_number)


def run_loop(gen: Any, start_turn: int, start_tokens: int) -> ConversationRecord:
    """Core conversation loop \u2014 alternates AI/human turns."""
    turn_number = start_turn
    accumulated_tokens = start_tokens
    consecutive_empty = 0
    max_consecutive_empty = 5

    last_speaker = gen._record.turns[-1].speaker if gen._record.turns else None
    if last_speaker == "ai":
        turn_number, accumulated_tokens = _handle_resume_human_turn(
            gen, turn_number, accumulated_tokens, max_consecutive_empty
        )

    while turn_number < gen.max_turns:
        turn_number += 1
        cost_before = gen.client.total_cost
        ai_response = gen._get_ai_response()

        if not ai_response.visible_text:
            consecutive_empty += 1
            wait = min(2**consecutive_empty, 32)
            reason = ai_response.rejection_reason or "empty response"
            console.print(
                f"[yellow]AI produced incorrect response ({reason}) \u2014 attempt {consecutive_empty}/{max_consecutive_empty}, retrying in {wait}s...[/yellow]"
            )
            turn_number -= 1
            if consecutive_empty >= max_consecutive_empty:
                console.print(
                    "[bold yellow]Too many consecutive empty AI responses \u2014 ending conversation.[/bold yellow]"
                )
                break
            time.sleep(wait)
            continue
        consecutive_empty = 0

        ai_tokens, context_tokens = _record_ai_turn(gen, ai_response, turn_number, gen.client.total_cost - cost_before)
        accumulated_tokens += ai_tokens

        _check_periodic_events(gen, turn_number)

        if gen.verbose or turn_number % 10 == 0:
            console.print(
                f"  [dim]\u2014 progress: turn {turn_number} | {context_tokens:,}/{gen.target_tokens:,} tok | ${gen.client.total_cost:.4f}[/dim]"
            )

        if context_tokens >= gen.target_tokens:
            console.print(
                f"\n[bold green]\u2713 Reached target token count ({context_tokens:,} >= {gen.target_tokens:,})[/bold green]"
            )
            break

        turn_number, accumulated_tokens, should_break = _do_human_turn(
            gen, turn_number, accumulated_tokens, max_consecutive_empty
        )
        if should_break:
            break

    return _finalize_record(gen, turn_number, accumulated_tokens)


def _finalize_record(gen: Any, turn_number: int, accumulated_tokens: int) -> ConversationRecord:
    """Seal the record and print the completion summary."""
    gen._record.total_tokens_estimate = accumulated_tokens
    gen._record.total_cost_usd = gen.client.total_cost
    gen._record.finished_at = datetime.now(timezone.utc).isoformat()
    gen._record.ai_messages_raw = gen._ai_messages

    console.print("\n[bold]Conversation complete![/bold]")
    console.print(f"  Turns: {turn_number}")
    console.print(f"  Estimated tokens: {accumulated_tokens:,}")
    console.print(f"  Total cost: ${gen.client.total_cost:.4f}")
    return gen._record


def _do_human_turn(
    gen: Any, turn_number: int, accumulated_tokens: int, max_consecutive_empty: int
) -> tuple[int, int, bool]:
    """Execute one human turn. Returns (turn_number, accumulated_tokens, should_break)."""
    turn_number += 1
    cost_before = gen.client.total_cost
    human_text, human_reasoning, human_reasoning_details, human_usage = gen._get_human_response()
    human_cost = gen.client.total_cost - cost_before

    if not human_text or not human_text.strip():
        human_text, human_reasoning, human_reasoning_details = _retry_empty_human(gen, max_consecutive_empty)
        if not human_text:
            return turn_number - 1, accumulated_tokens, True

    gen._add_human_turn_to_contexts(human_text, reasoning=human_reasoning, reasoning_details=human_reasoning_details)
    human_tokens = _estimate_tokens(human_text)
    accumulated_tokens += human_tokens

    human_ctx = (
        (human_usage.prompt_tokens + human_usage.completion_tokens)
        if human_usage and human_usage.prompt_tokens
        else _estimate_context_tokens(gen._human_messages)
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
        ai_context_tokens=_estimate_context_tokens(gen._ai_messages),
        human_context_tokens=human_ctx,
    )
    gen._record.turns.append(human_turn)
    gen._log_turn(human_turn)
    return turn_number, accumulated_tokens, False


def _retry_empty_human(gen: Any, max_retries: int) -> tuple[str, str | None, list | None]:
    """Retry the human model when it produces an empty response."""
    for _retry in range(1, max_retries):
        wait = min(2**_retry, 16)
        console.print(
            f"[yellow]Human produced empty response ({_retry}/{max_retries}), retrying in {wait}s with nudge...[/yellow]"
        )
        time.sleep(wait)
        gen._human_messages.append(
            {
                "role": "user",
                "content": "(The AI just said something. Please respond naturally as yourself \u2014 share your thoughts, tell a story, bring up a new topic, or react to what they said.)",
            }
        )
        human_text, human_reasoning, human_reasoning_details, _ = gen._get_human_response()
        gen._human_messages.pop(-1)
        if human_text and human_text.strip():
            return human_text, human_reasoning, human_reasoning_details
    console.print("[bold yellow]Human still empty after max retries \u2014 ending conversation.[/bold yellow]")
    return "", None, None


def _handle_resume_human_turn(
    gen: Any, turn_number: int, accumulated_tokens: int, max_consecutive_empty: int
) -> tuple[int, int]:
    """Handle the first human turn when resuming after an AI turn."""
    turn_number += 1
    cost_before = gen.client.total_cost
    human_text, human_reasoning, human_reasoning_details, human_usage = gen._get_human_response()

    if not human_text or not human_text.strip():
        human_text, human_reasoning, human_reasoning_details = _retry_empty_human(gen, max_consecutive_empty)
        if not human_text:
            return gen.max_turns, accumulated_tokens

    human_cost = gen.client.total_cost - cost_before
    gen._add_human_turn_to_contexts(human_text, reasoning=human_reasoning, reasoning_details=human_reasoning_details)
    human_tokens = _estimate_tokens(human_text)
    accumulated_tokens += human_tokens

    human_ctx = (
        (human_usage.prompt_tokens + human_usage.completion_tokens)
        if human_usage and human_usage.prompt_tokens
        else _estimate_context_tokens(gen._human_messages)
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
        ai_context_tokens=_estimate_context_tokens(gen._ai_messages),
        human_context_tokens=human_ctx,
    )
    gen._record.turns.append(human_turn)
    gen._log_turn(human_turn)
    return turn_number, accumulated_tokens
