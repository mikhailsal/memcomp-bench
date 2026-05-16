"""Tests for private helper functions in memcomp_bench.generator."""

from __future__ import annotations

import json

from memcomp_bench.generator import (
    ConversationTurn,
    _build_ai_tool_message,
    _estimate_context_tokens,
    _estimate_tokens,
    _extract_tool_call_reasoning,
    _format_thinking_markdown,
    _heal_tool_call_names,
    _is_restorable_ai_context,
    _looks_like_json_object,
    _migrate_assistant_reasoning_fields,
    _normalize_tool_arguments,
    _rebuild_ai_context_from_turns,
    _split_thinking_and_message,
    _tool_call_text_before_reasoning,
    _turns_to_context_rows,
    _uses_native_reasoning_field,
    sanitize_human_visible_text,
)


class TestHealToolCallNames:
    def _tc(self, name: str) -> list[dict]:
        return [{"function": {"name": name}}]

    def test_valid_name_unchanged(self):
        tcs = self._tc("write_message_to_human")
        assert _heal_tool_call_names(tcs) == 0
        assert tcs[0]["function"]["name"] == "write_message_to_human"

    def test_strategy1_strip_slashes(self):
        tcs = self._tc("write_//message_to_human")
        assert _heal_tool_call_names(tcs) == 1
        assert tcs[0]["function"]["name"] == "write_message_to_human"

    def test_strategy2_substring_original(self):
        tcs = self._tc("prefix_write_message_to_human_suffix")
        assert _heal_tool_call_names(tcs) == 1
        assert tcs[0]["function"]["name"] == "write_message_to_human"

    def test_strategy3_substring_cleaned(self):
        tcs = self._tc("write_//message_to_human_extra123")
        assert _heal_tool_call_names(tcs) == 1
        assert tcs[0]["function"]["name"] == "write_message_to_human"

    def test_strategy4_last_resort(self):
        tcs = self._tc("totally_garbled_xyz")
        assert _heal_tool_call_names(tcs) == 1
        assert tcs[0]["function"]["name"] == "write_message_to_human"

    def test_none_input(self):
        assert _heal_tool_call_names(None) == 0

    def test_empty_list(self):
        assert _heal_tool_call_names([]) == 0

    def test_missing_function_key(self):
        assert _heal_tool_call_names([{}]) == 0


class TestNormalizeToolArguments:
    def test_unicode_escape_shrinks(self):
        cyrillic = json.dumps({"text": "Привет"}, ensure_ascii=True)
        msgs = [{"tool_calls": [{"function": {"arguments": cyrillic}}]}]
        saved = _normalize_tool_arguments(msgs)
        assert saved > 0
        new_args = msgs[0]["tool_calls"][0]["function"]["arguments"]
        assert "Привет" in new_args

    def test_already_clean_noop(self):
        clean = json.dumps({"text": "hello"}, ensure_ascii=False)
        msgs = [{"tool_calls": [{"function": {"arguments": clean}}]}]
        assert _normalize_tool_arguments(msgs) == 0

    def test_no_tool_calls_skipped(self):
        assert _normalize_tool_arguments([{"role": "user", "content": "hi"}]) == 0

    def test_empty_args_skipped(self):
        msgs = [{"tool_calls": [{"function": {"arguments": ""}}]}]
        assert _normalize_tool_arguments(msgs) == 0

    def test_malformed_json_skipped(self):
        msgs = [{"tool_calls": [{"function": {"arguments": "not json{{"}}]}]
        assert _normalize_tool_arguments(msgs) == 0


class TestEstimateTokens:
    def test_basic_estimate(self):
        assert _estimate_tokens("abcd") == 1
        assert _estimate_tokens("abcdefgh") == 2

    def test_empty_returns_zero(self):
        assert _estimate_tokens("") == 0
        assert _estimate_tokens(None) == 0

    def test_context_tokens_content(self):
        msgs = [{"content": "a" * 100}]
        assert _estimate_context_tokens(msgs) == 25

    def test_context_tokens_reasoning_and_tool_calls(self):
        msgs = [
            {"content": "a" * 40, "reasoning": "b" * 40},
            {"tool_calls": [{"function": {"arguments": "c" * 40}}]},
        ]
        assert _estimate_context_tokens(msgs) == 30  # 10 + 10 + 10


class TestSmallHelpers:
    def test_uses_reasoning_true(self):
        assert _uses_native_reasoning_field({"effort": "minimal"}) is True

    def test_uses_reasoning_false(self):
        assert _uses_native_reasoning_field(None) is False
        assert _uses_native_reasoning_field({}) is False

    def test_looks_like_json_object(self):
        assert _looks_like_json_object('{"key": 1}') is True
        assert _looks_like_json_object('  {"key": 1}') is True
        assert _looks_like_json_object("hello") is False
        assert _looks_like_json_object(None) is False
        assert _looks_like_json_object("") is False


class TestBuildAIToolMessage:
    def test_basic_tool_message(self):
        msg = _build_ai_tool_message("Hello", "tc_01")
        assert msg["role"] == "assistant"
        assert msg["content"] is None
        assert len(msg["tool_calls"]) == 1
        args = json.loads(msg["tool_calls"][0]["function"]["arguments"])
        assert args["text"] == "Hello"

    def test_with_thinking_as_content(self):
        msg = _build_ai_tool_message("Hello", "tc_01", thinking="pondering")
        assert msg["content"] == "pondering"

    def test_with_reasoning_field(self):
        msg = _build_ai_tool_message(
            "Hello",
            "tc_01",
            thinking="pondering",
            use_reasoning_field=True,
        )
        assert msg["reasoning"] == "pondering"
        assert msg["content"] is None

    def test_json_thinking_stays_in_content(self):
        msg = _build_ai_tool_message(
            "Hello",
            "tc_01",
            thinking='{"thoughts": "hmm"}',
            use_reasoning_field=True,
        )
        assert msg["content"] == '{"thoughts": "hmm"}'
        assert "reasoning" not in msg or msg.get("reasoning") is None

    def test_explicit_tool_calls_preserved(self):
        custom_tc = [
            {
                "id": "custom_01",
                "type": "function",
                "function": {"name": "write_message_to_human", "arguments": "{}"},
            }
        ]
        msg = _build_ai_tool_message("Hi", "tc_01", tool_calls=custom_tc)
        assert msg["tool_calls"][0]["id"] == "custom_01"

    def test_assistant_reasoning_takes_priority(self):
        msg = _build_ai_tool_message(
            "Hi",
            "tc_01",
            thinking="from thinking",
            assistant_reasoning="from reasoning field",
        )
        assert msg["reasoning"] == "from reasoning field"

    @staticmethod
    def _real_tc(tc_id: str, args: dict) -> list[dict]:
        return [
            {
                "id": tc_id,
                "type": "function",
                "function": {
                    "name": "write_message_to_human",
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
        ]

    def test_tool_call_reasoning_not_duplicated_to_message_fields(self):
        """Tool-call reasoning must NOT leak into message-level fields."""
        tc = self._real_tc("tc_001", {"reasoning": "inner thought", "text": "Hello!"})
        msg = _build_ai_tool_message(
            "Hello!",
            "tc_001",
            thinking="inner thought",
            tool_calls=tc,
            use_reasoning_field=True,
        )
        assert msg["content"] is None, "tool-call reasoning must not leak into content"
        assert msg.get("reasoning") is None, "tool-call reasoning must not leak into reasoning"

    def test_tool_calls_with_native_reasoning_preserved(self):
        """Native reasoning from the model should be preserved on the message."""
        tc = self._real_tc("tc_002", {"text": "Hi"})
        msg = _build_ai_tool_message(
            "Hi",
            "tc_002",
            thinking="display thinking",
            assistant_reasoning="model native reasoning",
            tool_calls=tc,
            use_reasoning_field=True,
        )
        assert msg["reasoning"] == "model native reasoning"

    def test_tool_calls_with_native_content_preserved(self):
        """Native content from the model should be preserved on the message."""
        tc = self._real_tc("tc_003", {"text": "Hi"})
        msg = _build_ai_tool_message(
            "Hi",
            "tc_003",
            thinking="display thinking",
            assistant_content="model draft content",
            tool_calls=tc,
            use_reasoning_field=True,
        )
        assert msg["content"] == "model draft content"
        assert msg.get("reasoning") is None

    def test_reasoning_details_preserved_on_message(self):
        """Encrypted reasoning_details from the model must be placed on the message."""
        tc = self._real_tc("tc_004", {"reasoning": "inner", "text": "Hi"})
        enc = [{"type": "reasoning.encrypted", "data": "abc123==", "format": "google-gemini-v1"}]
        msg = _build_ai_tool_message(
            "Hi",
            "tc_004",
            tool_calls=tc,
            reasoning_details=enc,
        )
        assert msg["reasoning_details"] == enc
        assert msg.get("reasoning") is None


class TestMigrateReasoningFields:
    def test_migrates_content_to_reasoning(self):
        msgs = [
            {
                "role": "assistant",
                "content": "inner thoughts",
                "tool_calls": [{"id": "tc"}],
            }
        ]
        count = _migrate_assistant_reasoning_fields(msgs, use_reasoning_field=True)
        assert count == 1
        assert msgs[0]["reasoning"] == "inner thoughts"
        assert msgs[0]["content"] is None

    def test_skips_when_disabled(self):
        msgs = [
            {
                "role": "assistant",
                "content": "inner thoughts",
                "tool_calls": [{"id": "tc"}],
            }
        ]
        assert _migrate_assistant_reasoning_fields(msgs, use_reasoning_field=False) == 0

    def test_skips_json_content(self):
        msgs = [
            {
                "role": "assistant",
                "content": '{"reasoning": "json blob"}',
                "tool_calls": [{"id": "tc"}],
            }
        ]
        assert _migrate_assistant_reasoning_fields(msgs, use_reasoning_field=True) == 0

    def test_skips_already_has_reasoning(self):
        msgs = [
            {
                "role": "assistant",
                "content": "text",
                "reasoning": "already set",
                "tool_calls": [{"id": "tc"}],
            }
        ]
        assert _migrate_assistant_reasoning_fields(msgs, use_reasoning_field=True) == 0


class TestIsRestorableAIContext:
    def test_valid_context(self):
        ctx = [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "start"},
            {"role": "assistant", "tool_calls": [{"id": "tc1"}]},
        ]
        assert _is_restorable_ai_context(ctx) is True

    def test_too_short(self):
        assert _is_restorable_ai_context([{"role": "system"}]) is False

    def test_no_system_role(self):
        ctx = [
            {"role": "user"},
            {"role": "user"},
            {"role": "assistant", "tool_calls": [{}]},
        ]
        assert _is_restorable_ai_context(ctx) is False

    def test_no_tool_calls(self):
        ctx = [
            {"role": "system", "content": "p"},
            {"role": "user", "content": "s"},
            {"role": "assistant", "content": "text"},
        ]
        assert _is_restorable_ai_context(ctx) is False

    def test_not_a_list(self):
        assert _is_restorable_ai_context("string") is False
        assert _is_restorable_ai_context(None) is False


class TestRebuildAIContext:
    def test_basic_rebuild(self):
        turns = [
            {"speaker": "human", "visible_text": "Hey"},
            {
                "speaker": "ai",
                "visible_text": "Hello!",
                "ai_thinking": None,
                "ai_content": None,
                "ai_reasoning": None,
                "ai_tool_calls": None,
            },
            {"speaker": "human", "visible_text": "How are you?"},
        ]
        ctx = _rebuild_ai_context_from_turns("System prompt", turns)
        assert ctx[0]["role"] == "system"
        assert ctx[1]["role"] == "user"
        roles = [m["role"] for m in ctx[2:]]
        assert "assistant" in roles
        assert "tool" in roles

    def test_rebuild_sanitizes_human_thinking_tags(self):
        turns = [
            {"speaker": "human", "visible_text": "<thought>private</thought>Visible reply"},
            {
                "speaker": "ai",
                "visible_text": "Hello!",
                "ai_thinking": None,
                "ai_content": None,
                "ai_reasoning": None,
                "ai_tool_calls": None,
            },
        ]

        ctx = _rebuild_ai_context_from_turns("System prompt", turns)

        tool_msgs = [msg for msg in ctx if msg.get("role") == "tool"]
        assert tool_msgs[0]["content"] == "Visible reply"


class TestSanitizeHumanVisibleText:
    def test_strips_hidden_thinking_blocks(self):
        text = "<thinking>internal</thinking>Hi there"
        assert sanitize_human_visible_text(text) == "Hi there"

    def test_preserves_visible_text_around_hidden_block(self):
        text = "Hello <thought>secret</thought> there"
        assert sanitize_human_visible_text(text) == "Hello there"


class TestSplitThinkingAndMessage:
    def test_plain_text(self):
        vis, think = _split_thinking_and_message("Hello world")
        assert vis == "Hello world"
        assert think is None

    def test_json_then_text(self):
        vis, think = _split_thinking_and_message('{"thoughts": "hmm"} Hello!')
        assert vis == "Hello!"
        assert think == '{"thoughts": "hmm"}'

    def test_json_only(self):
        vis, think = _split_thinking_and_message('{"thoughts": "hmm"}')
        assert vis is None
        assert think == '{"thoughts": "hmm"}'

    def test_truncated_json(self):
        vis, think = _split_thinking_and_message('{"thoughts": "hmm')
        assert vis is None
        assert think is not None


class TestFormatThinkingMarkdown:
    def test_default_label(self):
        result = _format_thinking_markdown("line1\nline2")
        assert result.startswith("> 💭 Thinking:")
        assert "> line1" in result
        assert "> line2" in result

    def test_custom_label(self):
        result = _format_thinking_markdown("text", label="Custom:")
        assert "> Custom:" in result


class TestToolCallReasoningExtraction:
    def _turn(self, args_dict: dict, tool_name="write_message_to_human") -> ConversationTurn:
        return ConversationTurn(
            turn_number=1,
            speaker="ai",
            visible_text="hi",
            ai_tool_calls=[
                {
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(args_dict),
                    },
                }
            ],
        )

    def test_reasoning_extracted(self):
        turn = self._turn({"text": "hello", "reasoning": "deep thought"})
        assert _extract_tool_call_reasoning(turn) == "deep thought"

    def test_no_reasoning(self):
        turn = self._turn({"text": "hello"})
        assert _extract_tool_call_reasoning(turn) is None

    def test_no_tool_calls(self):
        turn = ConversationTurn(turn_number=1, speaker="ai", visible_text="hi")
        assert _extract_tool_call_reasoning(turn) is None

    def test_text_before_reasoning(self):
        turn = self._turn({"text": "hi", "reasoning": "hmm"})
        assert _tool_call_text_before_reasoning(turn) is True

    def test_reasoning_before_text(self):
        args = '{"reasoning": "hmm", "text": "hi"}'
        turn = ConversationTurn(
            turn_number=1,
            speaker="ai",
            visible_text="hi",
            ai_tool_calls=[
                {
                    "function": {
                        "name": "write_message_to_human",
                        "arguments": args,
                    },
                }
            ],
        )
        assert _tool_call_text_before_reasoning(turn) is False

    def test_text_before_reasoning_no_tool_calls(self):
        turn = ConversationTurn(turn_number=1, speaker="ai", visible_text="hi")
        assert _tool_call_text_before_reasoning(turn) is False


class TestTurnsToContextRows:
    def test_converts_dataclass_to_dicts(self):
        turns = [
            ConversationTurn(
                turn_number=1,
                speaker="human",
                visible_text="Hey",
                ai_thinking="t",
                ai_content="c",
                ai_reasoning="r",
                ai_tool_calls=[{"id": "tc1"}],
            ),
        ]
        rows = _turns_to_context_rows(turns)
        assert len(rows) == 1
        assert rows[0]["speaker"] == "human"
        assert rows[0]["visible_text"] == "Hey"
        assert rows[0]["ai_tool_calls"] is not turns[0].ai_tool_calls  # deep copy
