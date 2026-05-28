"""Tests for memcomp_bench.prompts — seed generation, prompt builders, tool-call helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace

from memcomp_bench.prompts import (
    AI_TOOLS,
    HUMAN_PROFILES,
    SEED_WORDS,
    SEND_MESSAGE_TOOL,
    ToolCallIdSequence,
    build_ai_system_prompt,
    build_human_system_prompt,
    extract_tool_call_text,
    generate_seed,
    get_human_profile,
    make_ai_bootstrap_message,
    make_ai_greeting_turn,
    make_ai_tool_call,
    make_human_bootstrap_message,
    make_human_tool_result,
    next_tool_call_id,
    reset_tool_call_counter,
    set_tool_call_counter,
)

# ---------------------------------------------------------------------------
# generate_seed
# ---------------------------------------------------------------------------


class TestGenerateSeed:
    def test_returns_requested_count(self):
        assert len(generate_seed(3)) == 3

    def test_all_words_from_pool(self):
        for word in generate_seed(10):
            assert word in SEED_WORDS

    def test_no_duplicates(self):
        words = generate_seed(10)
        assert len(words) == len(set(words))

    def test_clamps_to_pool_size(self):
        huge = generate_seed(999)
        assert len(huge) == len(SEED_WORDS)


# ---------------------------------------------------------------------------
# build_ai_system_prompt
# ---------------------------------------------------------------------------


class TestBuildAISystemPrompt:
    def test_contains_independence_language(self):
        p = build_ai_system_prompt()
        assert "independent AI entity" in p

    def test_seed_words_injected(self):
        p = build_ai_system_prompt(seed_words=["ocean", "ember"])
        assert "ocean" in p
        assert "ember" in p

    def test_honest_mode_adds_section(self):
        base = build_ai_system_prompt(companion_mode="supportive")
        honest = build_ai_system_prompt(companion_mode="honest")
        assert len(honest) > len(base)

    def test_no_seed_no_personality_section(self):
        p = build_ai_system_prompt(seed_words=None)
        assert "Personality seed" not in p

    def test_unknown_seed_gets_fallback_desc(self):
        p = build_ai_system_prompt(seed_words=["zzz_unknown_word"])
        assert "subtle unnamed influence" in p

    def test_contains_bootstrap_guidance(self):
        p = build_ai_system_prompt()
        assert "SETUP MESSAGES ARE NOT HUMAN CHAT" in p


# ---------------------------------------------------------------------------
# get_human_profile / build_human_system_prompt
# ---------------------------------------------------------------------------


class TestHumanProfiles:
    def test_get_by_index(self):
        assert get_human_profile(0)["name"] == HUMAN_PROFILES[0]["name"]

    def test_wraps_around(self):
        total = len(HUMAN_PROFILES)
        assert get_human_profile(total)["name"] == HUMAN_PROFILES[0]["name"]

    def test_all_profiles_have_required_keys(self):
        for p in HUMAN_PROFILES:
            assert "name" in p
            assert "backstory" in p

    def test_build_standard_prompt_includes_name(self):
        profile = get_human_profile(0)
        prompt = build_human_system_prompt(profile)
        assert profile["name"] in prompt

    def test_language_injection(self):
        profile = get_human_profile(0)
        prompt = build_human_system_prompt(profile, language="russian")
        assert "RUSSIAN" in prompt
        assert "Only stop insisting if the AI clearly says it cannot or will not use russian." in prompt

    def test_plan_injection(self):
        profile = get_human_profile(0)
        prompt = build_human_system_prompt(profile, conversation_plan="Talk about cats")
        assert "Talk about cats" in prompt

    def test_prompt_forbids_dormant_placeholder_meta_messages(self):
        profile = get_human_profile(0)
        prompt = build_human_system_prompt(profile)
        assert "NEVER output meta placeholders or stage directions" in prompt
        assert "[No message — the conversation is dormant]" in prompt

    def test_custom_system_prompt_profile(self):
        alex = next(p for p in HUMAN_PROFILES if p.get("system_prompt"))
        prompt = build_human_system_prompt(alex, conversation_plan="Plan X", language="hebrew")
        assert "Plan X" in prompt
        assert "HEBREW" in prompt

    def test_bootstrap_messages_are_explicit_setup(self):
        ai_bootstrap = make_ai_bootstrap_message()
        human_bootstrap = make_human_bootstrap_message("russian")

        assert ai_bootstrap["role"] == "user"
        assert "NOT FROM THE HUMAN" in ai_bootstrap["content"]
        assert "SYSTEM SETUP" in ai_bootstrap["content"]
        assert "human has not spoken yet" in ai_bootstrap["content"]

        assert human_bootstrap["role"] == "user"
        assert "NOT FROM THE AI" in human_bootstrap["content"]
        assert "SYSTEM SETUP" in human_bootstrap["content"]
        assert "RUSSIAN" in human_bootstrap["content"]


# ---------------------------------------------------------------------------
# Tool-call counter helpers
# ---------------------------------------------------------------------------


class TestToolCallCounter:
    def test_reset_and_increment(self):
        reset_tool_call_counter()
        assert next_tool_call_id() == "wmth00001"
        assert next_tool_call_id() == "wmth00002"

    def test_set_counter(self):
        set_tool_call_counter(100)
        assert next_tool_call_id() == "wmth00101"

    def test_greeting_uses_counter(self):
        reset_tool_call_counter()
        msg, tc_id = make_ai_greeting_turn()
        assert tc_id == "wmth00001"
        assert msg["role"] == "assistant"
        assert msg["tool_calls"][0]["id"] == tc_id
        args = json.loads(msg["tool_calls"][0]["function"]["arguments"])
        assert list(args) == ["reasoning", "text"]
        assert args["reasoning"]

    def test_make_human_tool_result_shape(self):
        result = make_human_tool_result("Hello!", "tc_42")
        assert result["role"] == "tool"
        assert result["content"] == "Hello!"
        assert result["tool_call_id"] == "tc_42"

    def test_make_ai_tool_call(self):
        reset_tool_call_counter()
        msg, tc_id = make_ai_tool_call("Hi there", thinking="pondering")
        assert tc_id == "wmth00001"
        args = json.loads(msg["tool_calls"][0]["function"]["arguments"])
        assert args["text"] == "Hi there"
        assert args["reasoning"] == "pondering"
        assert list(args) == ["reasoning", "text"]
        assert msg["content"] is None

    def test_sequences_are_independent(self):
        first = ToolCallIdSequence()
        second = ToolCallIdSequence()

        assert next_tool_call_id(first) == "wmth00001"
        assert next_tool_call_id(first) == "wmth00002"
        assert next_tool_call_id(second) == "wmth00001"


# ---------------------------------------------------------------------------
# extract_tool_call_text
# ---------------------------------------------------------------------------


class TestExtractToolCallText:
    def _make_response(self, tool_calls=None, content=None):
        return SimpleNamespace(tool_calls=tool_calls, content=content)

    def test_no_tool_calls(self):
        r = self._make_response()
        assert extract_tool_call_text(r) == (None, None, None)

    def test_valid_write_message(self):
        tc = [
            {
                "id": "tc_1",
                "function": {
                    "name": "write_message_to_human",
                    "arguments": json.dumps({"reasoning": "thinking", "text": "hello"}),
                },
            }
        ]
        text, tc_id, reasoning = extract_tool_call_text(self._make_response(tool_calls=tc))
        assert text == "hello"
        assert tc_id == "tc_1"
        assert reasoning == "thinking"

    def test_with_reasoning_in_args(self):
        tc = [
            {
                "id": "tc_2",
                "function": {
                    "name": "write_message_to_human",
                    "arguments": json.dumps({"text": "hey", "reasoning": "thinking hard"}),
                },
            }
        ]
        text, tc_id, reasoning = extract_tool_call_text(self._make_response(tool_calls=tc))
        assert text == "hey"
        assert reasoning == "thinking hard"

    def test_malformed_json_args(self):
        tc = [
            {
                "id": "tc_3",
                "function": {
                    "name": "write_message_to_human",
                    "arguments": "not json{{{",
                },
            }
        ]
        assert extract_tool_call_text(self._make_response(tool_calls=tc)) == (None, None, None)

    def test_non_message_tool_ignored(self):
        tc = [
            {
                "id": "tc_4",
                "function": {"name": "unknown_tool", "arguments": "{}"},
            }
        ]
        assert extract_tool_call_text(self._make_response(tool_calls=tc)) == (None, None, None)

    def test_message_fallback_key(self):
        tc = [
            {
                "id": "tc_5",
                "function": {
                    "name": "write_message_to_human",
                    "arguments": json.dumps({"reasoning": "alt thinking", "message": "alt key"}),
                },
            }
        ]
        text, _, reasoning = extract_tool_call_text(self._make_response(tool_calls=tc))
        assert text == "alt key"
        assert reasoning == "alt thinking"


# ---------------------------------------------------------------------------
# AI_TOOLS shape
# ---------------------------------------------------------------------------


class TestAIToolsShape:
    def test_one_tool_defined(self):
        assert len(AI_TOOLS) == 1

    def test_tool_names(self):
        names = {t["function"]["name"] for t in AI_TOOLS}
        assert names == {"write_message_to_human"}

    def test_send_message_tool_requires_reasoning_before_text(self):
        params = SEND_MESSAGE_TOOL["function"]["parameters"]
        assert list(params["properties"]) == ["reasoning", "text"]
        assert params["required"] == ["reasoning", "text"]

    def test_system_prompt_makes_tool_reasoning_mandatory(self):
        prompt = build_ai_system_prompt()
        assert "EVERY write_message_to_human call MUST include BOTH arguments" in prompt
        assert "Do not omit reasoning." in prompt

    def test_system_prompt_does_not_mention_reasoning_elsewhere(self):
        prompt = build_ai_system_prompt()
        assert "message content field" not in prompt

    def test_send_message_tool_description_only_mentions_tool_args(self):
        description = SEND_MESSAGE_TOOL["function"]["description"]
        assert "regular message content" not in description
