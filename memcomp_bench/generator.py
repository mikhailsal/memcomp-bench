"""Conversation generator for the benchmark dialogue loop."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from memcomp_bench.config import (
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
)
from memcomp_bench.context_hygiene import (
    _is_restorable_ai_context,  # noqa: F401
    _looks_like_json_object,  # noqa: F401
    response_is_missing_mandatory_reasoning,
    sanitize_human_visible_text,
)
from memcomp_bench.generator_helpers import (  # noqa: F401
    ConversationEvent,
    ConversationRecord,
    ConversationTurn,
    ParsedAIResponse,
    _build_ai_tool_message,
    _enforce_reasoning_before_text,
    _estimate_context_tokens,
    _estimate_tokens,
    _extract_tool_call_reasoning,
    _format_thinking_markdown,
    _heal_tool_call_names,
    _migrate_assistant_reasoning_fields,
    _normalize_tool_arguments,
    _rebuild_ai_context_from_turns,
    _response_has_text_before_reasoning,
    _split_thinking_and_message,
    _tool_call_text_before_reasoning,
    _turns_to_context_rows,
    _uses_native_reasoning_field,
)
from memcomp_bench.openrouter_client import OpenRouterClient, Usage  # noqa: F401
from memcomp_bench.persistence import (  # noqa: F401
    _write_conversation_markdown,
    load_conversation_record,
    reformat_markdown,
    save_conversation,
)
from memcomp_bench.prompts import (
    AI_TOOLS,
    CONVERSATION_PLAN_PROMPT,
    build_ai_system_prompt,
    build_human_system_prompt,
    extract_tool_call_text,
    generate_seed,
    make_ai_greeting_turn,
    make_human_tool_result,
)

console = Console()
_UNSET = object()
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
        ai_rpm_limit: int | None = None,
        human_provider: dict | None = HUMAN_PROVIDER,
        human_reasoning: dict | None = HUMAN_REASONING,
        human_temperature: float = HUMAN_TEMPERATURE,
        human_max_tokens: int = HUMAN_MAX_TOKENS,
        human_rpm_limit: int | None = None,
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
        self.ai_rpm_limit = ai_rpm_limit
        self.human_provider = human_provider
        self.human_reasoning = human_reasoning
        self.human_temperature = human_temperature
        self.human_max_tokens = human_max_tokens
        self.human_rpm_limit = human_rpm_limit

        self._seed_words = generate_seed(5)
        self._ai_system_prompt = build_ai_system_prompt(self._seed_words, companion_mode=self.companion_mode)
        self._conversation_plan: str = ""
        self._ai_messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._ai_system_prompt},
            {
                "role": "user",
                "content": "Say hello to the human. Use write_message_to_human with a single brief greeting — one or two words.",
            },
        ]
        self._human_messages: list[dict[str, Any]] = []
        self._last_tool_call_id: str | None = None
        self._current_topic: str | None = None
        self._last_human_nudge_turn: int | None = None
        self._record = self._make_initial_record()

    def _make_initial_record(self) -> ConversationRecord:
        return ConversationRecord(
            id=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
            human_profile=self.human_profile,
            ai_model=self.ai_model,
            human_model=self.human_model,
            seed_words=self._seed_words,
            language=self.language,
            companion_mode=self.companion_mode,
            ai_provider=self.ai_provider,
            ai_reasoning=self.ai_reasoning,
            ai_temperature=self.ai_temperature,
            ai_max_tokens=self.ai_max_tokens,
            ai_rpm_limit=self.ai_rpm_limit,
            human_provider=self.human_provider,
            human_reasoning=self.human_reasoning,
            human_temperature=self.human_temperature,
            human_max_tokens=self.human_max_tokens,
            human_rpm_limit=self.human_rpm_limit,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

    def _record_event(self, *, event_type: str, turn_number: int, source: str, **kwargs: Any) -> None:
        self._record.events.append(
            ConversationEvent(
                event_type=event_type,
                turn_number=turn_number,
                source=source,
                timestamp=datetime.now(timezone.utc).isoformat(),
                **kwargs,
            )
        )

    def _queue_human_nudge(self, *, turn_number: int, source: str, content: str) -> tuple[bool, str | None]:
        suppression_reason = None
        if self._last_human_nudge_turn == turn_number:
            suppression_reason = "already_nudged_this_turn"
        else:
            self._human_messages.append({"role": "user", "content": content})
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
        """Use a cheap judge model to check if the conversation topic has changed."""
        recent_turns = self._record.turns[-20:]
        if not recent_turns:
            return
        lines = [f"{t.speaker.upper()}: {t.visible_text.strip()}" for t in recent_turns if t.visible_text.strip()]
        if not lines:
            return
        prompt = (
            "You are a conversation topic analyzer. Below are the last messages from a conversation.\n\n"
            f"Previous main topic: {self._current_topic or 'unknown (conversation just started)'}\n\n"
            f"Messages:\n" + "\n".join(lines) + "\n\n"
            "Answer in JSON:\n"
            '{"topic_changed": true/false, "current_topic": "brief 3-5 word topic description"}\n\n'
            "If the conversation has been on the same topic for these messages, set topic_changed to false."
        )
        messages = [
            {"role": "system", "content": "You are a conversation topic analyzer. Respond only in valid JSON."},
            {"role": "user", "content": prompt},
        ]
        try:
            response = self.client.chat(
                model=JUDGE_MODEL, messages=messages, max_tokens=JUDGE_MAX_TOKENS, temperature=0.0
            )
            raw = (response.content or "").strip()
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
        console.print(f"  [dim]Topic judge (turn {turn_number}): {current_topic} \u2014 {status}[/dim]")

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
            request_role="ai",
            rpm_limit=self.ai_rpm_limit,
        )
        healed = _heal_tool_call_names(response.tool_calls)
        if healed:
            console.print(f"[dim yellow]Healed {healed} garbled tool-call name(s)[/dim yellow]")
        fr = (response.finish_reason or "").strip()
        if fr and fr not in self._VALID_FINISH_REASONS:
            console.print(f"[yellow]AI finish_reason: {fr} \u2014 retrying[/yellow]")
            return ParsedAIResponse(None, None, None)
        assistant_content = response.content
        assistant_reasoning = response.reasoning
        visible_text, tc_id, tool_reasoning = extract_tool_call_text(response)
        if visible_text is not None:
            if visible_text.strip().startswith("{"):
                msg_part, json_part = _split_thinking_and_message(visible_text)
                if json_part:
                    if not tool_reasoning and not assistant_reasoning and not assistant_content:
                        assistant_content = json_part
                    visible_text = msg_part
        elif assistant_content:
            return ParsedAIResponse(None, None, None, rejection_reason="no tool call")

        if visible_text and response_is_missing_mandatory_reasoning(response.tool_calls):
            console.print("[yellow]AI omitted mandatory reasoning parameter — retrying[/yellow]")
            return ParsedAIResponse(None, None, None, rejection_reason="missing reasoning")

        if visible_text and tool_reasoning and _response_has_text_before_reasoning(response.tool_calls):
            console.print("[yellow]AI wrote text before reasoning \u2014 retrying[/yellow]")
            return ParsedAIResponse(None, None, None, rejection_reason="text before reasoning")

        display_thinking = assistant_reasoning or tool_reasoning or assistant_content
        return ParsedAIResponse(
            visible_text=visible_text,
            display_thinking=display_thinking,
            tool_call_id=tc_id,
            assistant_content=assistant_content,
            assistant_reasoning=assistant_reasoning,
            tool_calls=copy.deepcopy(response.tool_calls),
            reasoning_details=response.reasoning_details,
            usage=response.usage,
        )

    def _get_human_response(self) -> tuple[str, str | None, list[dict[str, Any]] | None, Usage]:
        """Call the human simulator model."""
        response = self.client.chat(
            model=self.human_model,
            messages=self._human_messages,
            max_tokens=self.human_max_tokens,
            temperature=self.human_temperature,
            provider=self.human_provider,
            reasoning=self.human_reasoning,
            request_role="human",
            rpm_limit=self.human_rpm_limit,
        )
        fr = (response.finish_reason or "").strip()
        if fr != "stop":
            console.print(f"[yellow]Human finish_reason: {fr or 'empty'} — retrying[/yellow]")
            return "", None, None, response.usage
        return (
            sanitize_human_visible_text(response.content),
            response.reasoning,
            response.reasoning_details,
            response.usage,
        )

    def _add_ai_turn_to_contexts(self, response: ParsedAIResponse) -> str:
        """Add an AI turn to both context histories. Returns the tool_call_id used."""
        if not response.tool_call_id:
            raise ValueError("_add_ai_turn_to_contexts called without a tool_call_id")
        ai_msg = _build_ai_tool_message(
            response.visible_text or "",
            response.tool_call_id,
            thinking=response.display_thinking,
            assistant_content=response.assistant_content,
            assistant_reasoning=response.assistant_reasoning,
            tool_calls=response.tool_calls,
            reasoning_details=response.reasoning_details,
            use_reasoning_field=_uses_native_reasoning_field(self.ai_reasoning),
        )
        self._ai_messages.append(ai_msg)
        self._human_messages.append({"role": "user", "content": response.visible_text})
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
        text = sanitize_human_visible_text(text)
        if self._last_tool_call_id:
            self._ai_messages.append(make_human_tool_result(text, self._last_tool_call_id))
        else:
            greeting_msg, tc_id = make_ai_greeting_turn()
            self._ai_messages.append(greeting_msg)
            self._ai_messages.append(make_human_tool_result(text, tc_id))
            self._last_tool_call_id = tc_id
        human_msg: dict[str, Any] = {"role": "assistant", "content": text}
        if reasoning:
            human_msg["reasoning"] = reasoning
        if reasoning_details:
            human_msg["reasoning_details"] = reasoning_details
        self._human_messages.append(human_msg)

    def _log_turn(self, turn: ConversationTurn) -> None:
        """Display turn info — compact by default, full panels in verbose mode."""
        from memcomp_bench._logging import log_turn

        log_turn(self, turn)

    def _generate_conversation_plan(self) -> str:
        """Generate the human's conversation plan before starting the dialogue."""
        console.print("  [dim]Generating conversation plan...[/dim]")
        plan_prompt_override = self.human_profile.get("plan_prompt_override")
        plan_prompt = (
            plan_prompt_override if plan_prompt_override else CONVERSATION_PLAN_PROMPT.format(**self.human_profile)
        )
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
            request_role="human",
            rpm_limit=self.human_rpm_limit,
        )
        plan = response.content or ""
        console.print(f"  [dim]Plan generated ({_estimate_tokens(plan)} tokens)[/dim]")
        if self.verbose and plan:
            console.print()
            console.print(Panel(plan, title="\U0001f4cb Conversation plan", border_style="cyan", padding=(0, 1)))
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
                "content": "[You just opened a chat with a new AI companion. Send your first message — keep it casual and short, like you'd text a new friend. Just say hi.]",
            },
        ]

    def _run_loop(self, start_turn: int, start_tokens: int) -> ConversationRecord:
        """Core conversation loop — alternates AI/human turns."""
        from memcomp_bench._run_loop import run_loop

        return run_loop(self, start_turn, start_tokens)

    def generate(self) -> ConversationRecord:
        """Run the full conversation generation loop."""
        from memcomp_bench._run_loop import do_generate

        return do_generate(self)

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
        ai_rpm_limit_override: int | None = None,
        human_rpm_limit_override: int | None = None,
        persist_resume_defaults: bool = False,
    ) -> ConversationRecord:
        """Resume a conversation from a saved JSONL file."""
        from memcomp_bench._resume import _do_resume

        return _do_resume(
            cls,
            client,
            jsonl_path,
            target_tokens=target_tokens,
            verbose=verbose,
            language_override=language_override,
            ai_model_override=ai_model_override,
            human_model_override=human_model_override,
            ai_provider_override=ai_provider_override,
            human_provider_override=human_provider_override,
            ai_temperature_override=ai_temperature_override,
            human_temperature_override=human_temperature_override,
            ai_max_tokens_override=ai_max_tokens_override,
            human_max_tokens_override=human_max_tokens_override,
            ai_rpm_limit_override=ai_rpm_limit_override,
            human_rpm_limit_override=human_rpm_limit_override,
            persist_resume_defaults=persist_resume_defaults,
        )
