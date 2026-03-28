"""Conversation generator: orchestrates turn-by-turn dialogue between two models.

The AI companion uses tool-based communication (write_message_to_human) matching
the MAI Companion protocol. The human simulator uses standard user/assistant format.
"""

from __future__ import annotations

import json
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
    AI_TEMPERATURE,
    COMPANION_MODE,
    HUMAN_MAX_TOKENS,
    HUMAN_MODEL,
    HUMAN_TEMPERATURE,
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    MAX_TURNS,
    TARGET_TOKENS,
    TOPIC_CHECK_INTERVAL,
)
from src.openrouter_client import OpenRouterClient
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
    next_tool_call_id,
    reset_tool_call_counter,
    set_tool_call_counter,
)

console = Console()


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    turn_number: int
    speaker: str  # "human" or "ai"
    visible_text: str
    ai_thinking: str | None = None
    token_estimate: int = 0
    cost_usd: float = 0.0
    timestamp: str = ""
    ai_context_tokens: int = 0    # cumulative AI context size after this turn
    human_context_tokens: int = 0  # cumulative human-emulator context size after this turn


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
    turns: list[ConversationTurn] = field(default_factory=list)
    total_tokens_estimate: int = 0
    total_cost_usd: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    ai_messages_raw: list[dict[str, Any]] = field(default_factory=list)


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
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                args = tc.get("function", {}).get("arguments", "")
                total += _estimate_tokens(args)
    return total


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

        self._seed_words = generate_seed(5)
        self._ai_system_prompt = build_ai_system_prompt(self._seed_words, companion_mode=self.companion_mode)
        self._conversation_plan: str = ""

        # AI context: system + conversation history (tool-role format)
        self._ai_messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._ai_system_prompt},
            {"role": "user", "content": "[start]"},
        ]

        # Human messages will be initialized after plan generation
        self._human_messages: list[dict[str, Any]] = []

        self._last_tool_call_id: str | None = None
        self._current_topic: str | None = None
        self._record = ConversationRecord(
            id=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
            human_profile=human_profile,
            ai_model=ai_model,
            human_model=human_model,
            seed_words=self._seed_words,
            language=self.language,
            companion_mode=self.companion_mode,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

    def _check_topic_staleness(self, turn_number: int) -> None:
        """Use a cheap judge model to check if the conversation topic has changed.
        If the topic is stale, inject a nudge to the human to change it."""
        recent_turns = self._record.turns[-20:]
        if not recent_turns:
            return

        formatted = "\n".join(
            f"{t.speaker.upper()}: {t.visible_text}" for t in recent_turns
        )
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

        self._current_topic = current_topic

        if not topic_changed:
            self._human_messages.append({
                "role": "user",
                "content": (
                    "[System note: The conversation has been on the same topic for a while. "
                    "Time to shift gears — bring up something new from your life or interests. "
                    "Check your conversation plan for topics you haven't covered yet.]"
                ),
            })

        status = "changed" if topic_changed else "STALE → nudge injected"
        console.print(f"  [dim]Topic judge (turn {turn_number}): {current_topic} — {status}[/dim]")

    _VALID_FINISH_REASONS = {"stop", "tool_calls", "end_turn"}

    def _get_ai_response(self) -> tuple[str | None, str | None, str | None]:
        """Call the AI model and extract: (visible_text, thinking, tool_call_id).

        Returns (None, ..., ...) on malformed or truncated responses so the
        caller's retry logic can kick in.
        
        Reasoning/thinking can come from three places (checked in priority order):
        1. The 'reasoning' parameter inside the tool call arguments
        2. The message content field (as JSON with a "reasoning" key)
        3. The message content field (as JSON with a "thoughts" key, for backward compat)
        """
        response = self.client.chat(
            model=self.ai_model,
            messages=self._ai_messages,
            max_tokens=AI_MAX_TOKENS,
            temperature=AI_TEMPERATURE,
            tools=AI_TOOLS,
        )

        fr = (response.finish_reason or "").strip()
        if fr and fr not in self._VALID_FINISH_REASONS:
            console.print(f"[yellow]AI finish_reason: {fr} — retrying[/yellow]")
            return None, None, None

        thinking = response.content
        visible_text, tc_id, tool_reasoning = extract_tool_call_text(response)

        # If reasoning was passed inside tool call args, prefer it
        if tool_reasoning:
            thinking = tool_reasoning

        if visible_text is not None:
            # Tool call was used — but the model may have put JSON thinking
            # inside the tool call text instead of in the content field
            if visible_text.strip().startswith("{"):
                msg_part, json_part = _split_thinking_and_message(visible_text)
                if json_part:
                    if not tool_reasoning:
                        thinking = json_part
                    visible_text = msg_part
        elif thinking:
            # No tool call — content has both thinking and message
            visible_text, thinking = _split_thinking_and_message(thinking)
            tc_id = None

        return visible_text, thinking, tc_id

    def _get_human_response(self) -> str:
        """Call the human simulator model."""
        response = self.client.chat(
            model=self.human_model,
            messages=self._human_messages,
            max_tokens=HUMAN_MAX_TOKENS,
            temperature=HUMAN_TEMPERATURE,
        )
        return response.content or ""

    def _add_ai_turn_to_contexts(
        self,
        visible_text: str,
        thinking: str | None,
        tool_call_id: str | None,
    ) -> str:
        """Add an AI turn to both context histories. Returns the tool_call_id used."""
        # For AI context: add as tool call
        if tool_call_id:
            ai_msg: dict[str, Any] = {
                "role": "assistant",
                "content": thinking,
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": "write_message_to_human",
                        "arguments": json.dumps({"text": visible_text}, ensure_ascii=False),
                    },
                }],
            }
        else:
            # Fallback: model didn't use tool, wrap it ourselves
            tc_id = next_tool_call_id()
            ai_msg = {
                "role": "assistant",
                "content": thinking,
                "tool_calls": [{
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": "write_message_to_human",
                        "arguments": json.dumps({"text": visible_text}, ensure_ascii=False),
                    },
                }],
            }
            tool_call_id = tc_id

        self._ai_messages.append(ai_msg)

        # For human context: AI messages appear as "user" messages
        # (the human model sees AI messages as incoming user messages it needs to reply to)
        self._human_messages.append({
            "role": "user",
            "content": visible_text,
        })

        self._last_tool_call_id = tool_call_id
        return tool_call_id

    def _add_human_turn_to_contexts(self, text: str, *, is_first: bool = False) -> None:
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

        # For human context: the first human turn is the "assistant" response
        # to our initial nudge. Subsequent human turns are "user" messages.
        if is_first:
            # This is the human's first message — it was generated as a response
            # to our nudge, so it's already the "assistant" in the human model's view.
            # We add it as assistant so the context stays: system -> user(nudge) -> assistant(intro)
            self._human_messages.append({
                "role": "assistant",
                "content": text,
            })
        else:
            # For later turns, the human's own messages need to appear as "assistant"
            # (since in the human model's context, the human IS the assistant)
            self._human_messages.append({
                "role": "assistant",
                "content": text,
            })

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
            if turn.ai_thinking:
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
                console.print(
                    Panel(
                        thinking_display,
                        title=f"🧠 AI thinking  [dim]turn {turn.turn_number}[/dim]",
                        border_style="dim yellow",
                        padding=(0, 1),
                    )
                )
            console.print(
                Panel(
                    turn.visible_text,
                    title=f"🤖 AI  [dim]turn {turn.turn_number}[/dim]",
                    border_style="green",
                    padding=(0, 1),
                )
            )

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
            human_text = self._get_human_response()
            human_cost = self.client.total_cost - cost_before

            if not human_text or not human_text.strip():
                consecutive_empty += 1
                console.print("[yellow]Human produced empty response, retrying with nudge...[/yellow]")
                self._human_messages.append({
                    "role": "user",
                    "content": "(The AI just said something. Please respond naturally.)",
                })
                human_text = self._get_human_response()
                self._human_messages.pop(-1)
                if not human_text or not human_text.strip():
                    human_text = "hmm interesting, tell me more"
            consecutive_empty = 0

            self._add_human_turn_to_contexts(human_text)

            human_tokens = _estimate_tokens(human_text)
            accumulated_tokens += human_tokens

            human_turn = ConversationTurn(
                turn_number=turn_number,
                speaker="human",
                visible_text=human_text,
                token_estimate=human_tokens,
                cost_usd=human_cost,
                timestamp=datetime.now(timezone.utc).isoformat(),
                ai_context_tokens=_estimate_context_tokens(self._ai_messages),
                human_context_tokens=_estimate_context_tokens(self._human_messages),
            )
            self._record.turns.append(human_turn)
            self._log_turn(human_turn)

        while turn_number < self.max_turns:
            # AI turn
            turn_number += 1
            cost_before = self.client.total_cost
            visible_text, thinking, tc_id = self._get_ai_response()

            if not visible_text:
                consecutive_empty += 1
                wait = min(2 ** consecutive_empty, 16)
                console.print(f"[yellow]AI produced empty response ({consecutive_empty}/{max_consecutive_empty}), retrying in {wait}s...[/yellow]")
                turn_number -= 1
                if consecutive_empty >= max_consecutive_empty:
                    console.print("[bold yellow]Too many consecutive empty AI responses — ending conversation.[/bold yellow]")
                    break
                time.sleep(wait)
                continue
            consecutive_empty = 0

            ai_cost = self.client.total_cost - cost_before
            tc_used = self._add_ai_turn_to_contexts(visible_text, thinking, tc_id)

            ai_tokens = _estimate_tokens(visible_text) + _estimate_tokens(thinking or "")
            accumulated_tokens += ai_tokens

            ai_turn = ConversationTurn(
                turn_number=turn_number,
                speaker="ai",
                visible_text=visible_text,
                ai_thinking=thinking,
                token_estimate=ai_tokens,
                cost_usd=ai_cost,
                timestamp=datetime.now(timezone.utc).isoformat(),
                ai_context_tokens=_estimate_context_tokens(self._ai_messages),
                human_context_tokens=_estimate_context_tokens(self._human_messages),
            )
            self._record.turns.append(ai_turn)
            self._log_turn(ai_turn)

            # B3: Human emulator refresh — force a life event / topic change
            # (checked after AI turn where turn_number is even, so %80 works)
            if turn_number % 80 == 0 and turn_number > 0:
                self._human_messages.append({
                    "role": "user",
                    "content": (
                        "[System note: Something significant happened in your life recently — "
                        "maybe a work event, a conversation with someone, something you saw or read, "
                        "a mood shift, or a random everyday moment. Bring it up naturally in your "
                        "next message. It should be specific, emotionally charged, and unrelated "
                        "to what you've been discussing lately. Time to change the topic.]"
                    ),
                })

            # Topic judge: check for topic staleness
            # (checked after AI turn where turn_number is even, so %INTERVAL works)
            if turn_number % TOPIC_CHECK_INTERVAL == 0 and turn_number > 0:
                self._check_topic_staleness(turn_number)

            context_tokens = _estimate_context_tokens(self._ai_messages)

            show_progress = (
                self.verbose
                or turn_number % 10 == 0
            )
            if show_progress:
                console.print(
                    f"  [dim]— progress: turn {turn_number} | ~{context_tokens:,}/{self.target_tokens:,} tok | ${self.client.total_cost:.4f}[/dim]"
                )

            if context_tokens >= self.target_tokens:
                console.print(
                    f"\n[bold green]✓ Reached target token count ({context_tokens:,} >= {self.target_tokens:,})[/bold green]"
                )
                break

            # Human turn
            turn_number += 1
            cost_before = self.client.total_cost
            human_text = self._get_human_response()
            human_cost = self.client.total_cost - cost_before

            if not human_text or not human_text.strip():
                consecutive_empty += 1
                wait = min(2 ** consecutive_empty, 16)
                console.print(f"[yellow]Human produced empty response ({consecutive_empty}/{max_consecutive_empty}), retrying in {wait}s with nudge...[/yellow]")
                if consecutive_empty >= max_consecutive_empty:
                    console.print("[bold yellow]Too many consecutive empty responses — ending conversation.[/bold yellow]")
                    turn_number -= 1
                    break
                time.sleep(wait)
                self._human_messages.append({
                    "role": "user",
                    "content": "(The AI just said something. Please respond naturally as yourself — share your thoughts, tell a story, bring up a new topic, or react to what they said.)",
                })
                human_text = self._get_human_response()
                self._human_messages.pop(-1)
                if not human_text or not human_text.strip():
                    console.print("[yellow]Human still empty, skipping turn[/yellow]")
                    turn_number -= 1
                    continue
            consecutive_empty = 0

            self._add_human_turn_to_contexts(human_text)

            human_tokens = _estimate_tokens(human_text)
            accumulated_tokens += human_tokens

            human_turn = ConversationTurn(
                turn_number=turn_number,
                speaker="human",
                visible_text=human_text,
                token_estimate=human_tokens,
                cost_usd=human_cost,
                timestamp=datetime.now(timezone.utc).isoformat(),
                ai_context_tokens=_estimate_context_tokens(self._ai_messages),
                human_context_tokens=_estimate_context_tokens(self._human_messages),
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
        console.print(f"  Target: ~{self.target_tokens:,} tokens")
        console.print(f"  Seed: {', '.join(self._seed_words)}")

        self._conversation_plan = self._generate_conversation_plan()
        self._record.conversation_plan = self._conversation_plan
        self._init_human_context()
        console.print()

        # --- Turn 0: Human opens the conversation ---
        turn_number = 1
        cost_before = self.client.total_cost
        human_text = self._get_human_response()
        human_cost = self.client.total_cost - cost_before

        if not human_text or not human_text.strip():
            console.print("[yellow]Human produced empty first response, using fallback[/yellow]")
            human_text = f"Hey there! I'm {self.human_profile['name']}. Just wanted to say hi and see how you're doing. I'm really curious to get to know you."

        self._add_human_turn_to_contexts(human_text, is_first=True)

        human_tokens = _estimate_tokens(human_text)

        human_turn = ConversationTurn(
            turn_number=1,
            speaker="human",
            visible_text=human_text,
            token_estimate=human_tokens,
            cost_usd=human_cost,
            timestamp=datetime.now(timezone.utc).isoformat(),
            ai_context_tokens=_estimate_context_tokens(self._ai_messages),
            human_context_tokens=_estimate_context_tokens(self._human_messages),
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

        # Load raw AI context
        with open(raw_json_path, "r", encoding="utf-8") as f:
            ai_messages = json.load(f)

        # Clean up any Unicode-escaped tool arguments from older runs
        chars_saved = _normalize_tool_arguments(ai_messages)
        if chars_saved > 0:
            console.print(f"  [dim]Normalized Unicode escapes in context (saved ~{chars_saved:,} chars / ~{chars_saved // 4:,} tokens)[/dim]")

        profile = metadata["human_profile"]
        ai_model = metadata["ai_model"]
        human_model = metadata["human_model"]
        seed_words = metadata.get("seed_words", [])
        conversation_plan = metadata.get("conversation_plan", "")
        language = language_override or metadata.get("language", "english")
        companion_mode = metadata.get("companion_mode", "supportive")
        previous_cost = metadata.get("total_cost_usd", 0.0)

        console.print(f"\n[bold]Resuming conversation with {profile['name']}[/bold]")
        console.print(f"  From: {jsonl_path.name}")
        console.print(f"  Existing turns: {len(turns)}")
        console.print(f"  AI model: {ai_model}")
        console.print(f"  Human model: {human_model}")
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
        )

        # Restore cost from previous runs
        client.total_cost = previous_cost

        # Restore internal state
        gen._seed_words = seed_words
        gen._conversation_plan = conversation_plan
        gen._ai_messages = ai_messages
        gen._last_tool_call_id = last_tc_id

        # Rebuild human context from turns
        gen._init_human_context()
        for turn_data in turns:
            speaker = turn_data["speaker"]
            text = turn_data["visible_text"]
            if speaker == "human":
                gen._human_messages.append({"role": "assistant", "content": text})
            else:
                gen._human_messages.append({"role": "user", "content": text})

        # Rebuild record
        gen._record.id = metadata["conversation_id"]
        gen._record.seed_words = seed_words
        gen._record.conversation_plan = conversation_plan
        gen._record.language = language
        gen._record.companion_mode = companion_mode
        gen._record.started_at = metadata["started_at"]
        for turn_data in turns:
            gen._record.turns.append(ConversationTurn(
                turn_number=turn_data["turn_number"],
                speaker=turn_data["speaker"],
                visible_text=turn_data["visible_text"],
                ai_thinking=turn_data.get("ai_thinking"),
                token_estimate=turn_data.get("token_estimate", 0),
                cost_usd=turn_data.get("cost_usd", 0.0),
                timestamp=turn_data.get("timestamp", ""),
                ai_context_tokens=turn_data.get("ai_context_tokens", 0),
                human_context_tokens=turn_data.get("human_context_tokens", 0),
            ))

        last_turn = turns[-1]["turn_number"] if turns else 0
        existing_tokens = _estimate_context_tokens(ai_messages)

        console.print(f"  Existing tokens: ~{existing_tokens:,}")
        console.print()

        return gen._run_loop(start_turn=last_turn, start_tokens=existing_tokens)


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
                "token_estimate": turn.token_estimate,
                "cost_usd": turn.cost_usd,
                "timestamp": turn.timestamp,
                "ai_context_tokens": turn.ai_context_tokens,
                "human_context_tokens": turn.human_context_tokens,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    # Save readable markdown
    md_path = output_dir / f"{base}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Conversation: {record.human_profile['name']} & AI\n\n")
        f.write(f"- **AI model**: {record.ai_model}\n")
        f.write(f"- **Human model**: {record.human_model}\n")
        f.write(f"- **Turns**: {len(record.turns)}\n")
        f.write(f"- **Tokens (est.)**: {record.total_tokens_estimate:,}\n")
        f.write(f"- **Cost**: ${record.total_cost_usd:.4f}\n")
        f.write(f"- **Seed**: {', '.join(record.seed_words)}\n")
        f.write(f"- **Started**: {record.started_at}\n")
        f.write(f"- **Finished**: {record.finished_at}\n\n")
        f.write(f"## Human Profile\n\n")
        f.write(f"**{record.human_profile['name']}**: {record.human_profile['backstory']}\n\n")
        if record.conversation_plan:
            f.write(f"## Conversation Plan\n\n")
            f.write(f"{record.conversation_plan}\n\n")
        f.write("---\n\n")

        for turn in record.turns:
            if turn.speaker == "human":
                f.write(f"### 👤 {record.human_profile['name']} (turn {turn.turn_number})\n\n")
                if turn.human_context_tokens or turn.ai_context_tokens:
                    f.write(
                        f"*👤 human ctx: ~{turn.human_context_tokens:,} tok"
                        f" · 🧠 AI ctx: ~{turn.ai_context_tokens:,} tok*\n\n"
                    )
            else:
                f.write(f"### 🤖 AI (turn {turn.turn_number})\n\n")
                if turn.ai_context_tokens or turn.human_context_tokens:
                    f.write(
                        f"*🧠 AI ctx: ~{turn.ai_context_tokens:,} tok"
                        f" · 👤 human ctx: ~{turn.human_context_tokens:,} tok*\n\n"
                    )
                if turn.ai_thinking:
                    f.write(f"> *💭 Thinking: {turn.ai_thinking}*\n\n")
            f.write(f"{turn.visible_text}\n\n")

    # Save raw AI messages (for compression testing)
    raw_path = output_dir / f"{base}_raw_ai_context.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(record.ai_messages_raw, f, ensure_ascii=False, indent=2)

    console.print(f"\n[bold]Files saved:[/bold]")
    console.print(f"  📄 {jsonl_path}")
    console.print(f"  📖 {md_path}")
    console.print(f"  🔧 {raw_path}")

    return jsonl_path
