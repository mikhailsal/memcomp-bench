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
from rich.panel import Panel
from rich.text import Text

from src.config import (
    AI_MAX_TOKENS,
    AI_MODEL,
    AI_TEMPERATURE,
    HUMAN_MAX_TOKENS,
    HUMAN_MODEL,
    HUMAN_TEMPERATURE,
    MAX_TURNS,
    TARGET_TOKENS,
)
from src.openrouter_client import OpenRouterClient
from src.prompts import (
    AI_TOOLS,
    CONVERSATION_PLAN_PROMPT,
    build_ai_system_prompt,
    build_human_system_prompt,
    extract_tool_call_text,
    generate_seed,
    make_ai_greeting_turn,
    make_ai_tool_call,
    make_human_tool_result,
    next_tool_call_id,
    reset_tool_call_counter,
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


@dataclass
class ConversationRecord:
    """Full record of a generated conversation."""
    id: str
    human_profile: dict[str, str]
    ai_model: str
    human_model: str
    seed_words: list[str] = field(default_factory=list)
    conversation_plan: str = ""
    turns: list[ConversationTurn] = field(default_factory=list)
    total_tokens_estimate: int = 0
    total_cost_usd: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    ai_messages_raw: list[dict[str, Any]] = field(default_factory=list)


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
    ) -> None:
        self.client = client
        self.human_profile = human_profile
        self.ai_model = ai_model
        self.human_model = human_model
        self.target_tokens = target_tokens
        self.max_turns = max_turns
        self.language = language.lower()

        self._seed_words = generate_seed(5)
        self._ai_system_prompt = build_ai_system_prompt(self._seed_words)
        self._conversation_plan: str = ""

        # AI context: system + conversation history (tool-role format)
        self._ai_messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._ai_system_prompt},
            {"role": "user", "content": "[start]"},
        ]

        # Human messages will be initialized after plan generation
        self._human_messages: list[dict[str, Any]] = []

        self._last_tool_call_id: str | None = None
        self._record = ConversationRecord(
            id=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
            human_profile=human_profile,
            ai_model=ai_model,
            human_model=human_model,
            seed_words=self._seed_words,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

    def _get_ai_response(self) -> tuple[str | None, str | None, str | None]:
        """Call the AI model and extract: (visible_text, thinking, tool_call_id)."""
        response = self.client.chat(
            model=self.ai_model,
            messages=self._ai_messages,
            max_tokens=AI_MAX_TOKENS,
            temperature=AI_TEMPERATURE,
            tools=AI_TOOLS,
        )

        thinking = response.content
        visible_text, tc_id = extract_tool_call_text(response)

        if visible_text is not None:
            # Tool call was used — but the model may have put JSON thinking
            # inside the tool call text instead of in the content field
            if visible_text.strip().startswith("{"):
                msg_part, json_part = _split_thinking_and_message(visible_text)
                if json_part:
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
                        "arguments": json.dumps({"text": visible_text}),
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
                        "arguments": json.dumps({"text": visible_text}),
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
        """Display a compact one-line summary per turn."""
        if turn.speaker == "human":
            label = f"[bold blue]👤 {self.human_profile['name']}[/bold blue]"
        else:
            label = "[bold green]🤖 AI[/bold green]"

        preview = turn.visible_text[:80].replace("\n", " ")
        if len(turn.visible_text) > 80:
            preview += "…"
        console.print(f"  {label} t{turn.turn_number}: {preview}")

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

        turn_number = 0
        accumulated_tokens = 0

        # --- Turn 0: Human opens the conversation ---
        turn_number += 1
        cost_before = self.client.total_cost
        human_text = self._get_human_response()
        human_cost = self.client.total_cost - cost_before

        if not human_text or not human_text.strip():
            console.print("[yellow]Human produced empty first response, using fallback[/yellow]")
            human_text = f"Hey there! I'm {self.human_profile['name']}. Just wanted to say hi and see how you're doing. I'm really curious to get to know you."

        self._add_human_turn_to_contexts(human_text, is_first=True)

        human_tokens = _estimate_tokens(human_text)
        accumulated_tokens += human_tokens

        human_turn = ConversationTurn(
            turn_number=turn_number,
            speaker="human",
            visible_text=human_text,
            token_estimate=human_tokens,
            cost_usd=human_cost,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._record.turns.append(human_turn)
        self._log_turn(human_turn)

        # --- Main loop: AI responds, then human responds ---
        consecutive_empty = 0
        max_consecutive_empty = 5

        while turn_number < self.max_turns:
            # AI turn
            turn_number += 1
            cost_before = self.client.total_cost
            visible_text, thinking, tc_id = self._get_ai_response()

            if not visible_text:
                consecutive_empty += 1
                console.print(f"[yellow]AI produced empty response ({consecutive_empty}/{max_consecutive_empty}), retrying...[/yellow]")
                turn_number -= 1
                if consecutive_empty >= max_consecutive_empty:
                    console.print("[bold yellow]Too many consecutive empty AI responses — ending conversation.[/bold yellow]")
                    break
                time.sleep(2)
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
            )
            self._record.turns.append(ai_turn)
            self._log_turn(ai_turn)

            context_tokens = _estimate_context_tokens(self._ai_messages)

            if turn_number % 10 == 0:
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
                console.print(f"[yellow]Human produced empty response ({consecutive_empty}/{max_consecutive_empty}), retrying with nudge...[/yellow]")
                if consecutive_empty >= max_consecutive_empty:
                    console.print("[bold yellow]Too many consecutive empty responses — ending conversation.[/bold yellow]")
                    turn_number -= 1
                    break
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
            else:
                f.write(f"### 🤖 AI (turn {turn.turn_number})\n\n")
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
