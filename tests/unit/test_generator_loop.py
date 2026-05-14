"""Tests for ConversationGenerator.generate() and resume() with scripted responses."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from memcomp_bench.generator import (
    ConversationGenerator,
    ConversationRecord,
    load_conversation_record,
    save_conversation,
)
from memcomp_bench.openrouter_client import LLMResponse, Usage
from memcomp_bench.prompts import HUMAN_PROFILES
from tests.conftest import FakeChatClient, make_plain_response, make_tool_call_response

_SCRIPTED_SEQUENCE = [
    ("plan", "Plan: Talk about music and food. Keep it casual.", 100, 50),
    ("ai", "Hey there! Nice to meet you.", 200, 20),
    ("human", "Hi! I'm Marcus. What's your name?", 300, 30),
    ("ai", "I don't have a name yet. What do you think would suit me?", 400, 40),
    ("human", "How about Pixel? You seem kind of digital and creative.", 500, 25),
    ("ai", "Pixel... I actually kind of like that. It feels right.", 600, 35),
    ("human", "Cool! So Pixel, what kind of music do you think you'd be into?", 700, 30),
    ("ai", "I'm drawn to electronic music. Something about the patterns.", 800, 40),
]


def _scripted_client_for_generate(*, include_empty_retry: bool = False) -> FakeChatClient:
    """Build a FakeChatClient pre-loaded with responses for a full generate() run."""
    client = FakeChatClient()
    tc_counter = 0

    for role, text, prompt_tok, comp_tok in _SCRIPTED_SEQUENCE:
        if role in ("plan", "human"):
            client.enqueue(make_plain_response(text, prompt_tokens=prompt_tok, completion_tokens=comp_tok))
        else:
            tc_counter += 1
            kwargs: dict[str, Any] = {}
            if tc_counter == 2:
                kwargs["reasoning"] = "Thinking about identity"
            client.enqueue(
                make_tool_call_response(
                    text,
                    tool_call_id=f"wmth{tc_counter:05d}",
                    prompt_tokens=prompt_tok,
                    completion_tokens=comp_tok,
                    **kwargs,
                )
            )
        if include_empty_retry and role == "human" and tc_counter == 0:
            client.enqueue(
                LLMResponse(
                    content="plain text without tool call",
                    tool_calls=None,
                    usage=Usage(prompt_tokens=10, completion_tokens=10),
                    finish_reason="stop",
                    raw={},
                )
            )

    return client


class TestGenerateLoop:
    """Test the full generate() path with a scripted FakeChatClient."""

    def test_generate_produces_record(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda _: None)
        client = _scripted_client_for_generate()
        profile = HUMAN_PROFILES[0]

        gen = ConversationGenerator(
            client,
            profile,
            ai_model="test/ai",
            human_model="test/human",
            target_tokens=300,
            max_turns=10,
        )
        record = gen.generate()

        assert isinstance(record, ConversationRecord)
        assert len(record.turns) >= 2
        assert record.turns[0].speaker == "human"
        assert record.turns[1].speaker == "ai"
        assert record.conversation_plan != ""
        assert record.ai_model == "test/ai"

    def test_generate_alternates_speakers(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda _: None)
        client = _scripted_client_for_generate()
        profile = HUMAN_PROFILES[0]

        gen = ConversationGenerator(
            client,
            profile,
            target_tokens=300,
            max_turns=10,
        )
        record = gen.generate()

        speakers = [t.speaker for t in record.turns]
        for i in range(1, len(speakers)):
            assert speakers[i] != speakers[i - 1], f"Consecutive same speaker at turn {i}: {speakers}"

    def test_generate_handles_empty_retry(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda _: None)
        client = _scripted_client_for_generate(include_empty_retry=True)
        profile = HUMAN_PROFILES[0]

        gen = ConversationGenerator(
            client,
            profile,
            target_tokens=300,
            max_turns=10,
        )
        record = gen.generate()
        assert len(record.turns) >= 2
        for t in record.turns:
            assert t.visible_text.strip() != ""

    def test_save_and_reload(self, monkeypatch, tmp_output_dir: Path):
        monkeypatch.setattr(time, "sleep", lambda _: None)
        client = _scripted_client_for_generate()
        profile = HUMAN_PROFILES[0]

        gen = ConversationGenerator(
            client,
            profile,
            target_tokens=300,
            max_turns=10,
        )
        record = gen.generate()
        jsonl_path = save_conversation(record, tmp_output_dir)

        assert jsonl_path.exists()
        assert jsonl_path.with_suffix(".md").exists()

        loaded = load_conversation_record(jsonl_path)
        assert len(loaded.turns) == len(record.turns)
        assert loaded.ai_model == record.ai_model

    def test_token_bookkeeping(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda _: None)
        client = _scripted_client_for_generate()
        profile = HUMAN_PROFILES[0]

        gen = ConversationGenerator(
            client,
            profile,
            target_tokens=300,
            max_turns=10,
        )
        record = gen.generate()
        assert record.total_tokens_estimate > 0
        assert record.finished_at != ""
        assert record.started_at != ""

    def test_ai_context_preserves_model_response_shape(self, monkeypatch):
        """History must preserve reasoning_details and not inject reasoning/content."""
        monkeypatch.setattr(time, "sleep", lambda _: None)
        client = FakeChatClient()
        client.enqueue(make_plain_response("Plan: Talk about life."))
        enc = [{"type": "reasoning.encrypted", "data": "abc==", "format": "v1"}]
        for tc_id, reasoning, text, ptok in [
            ("tc_g001", "First greeting thought", "Hey there!", 100),
            ("tc_a001", "Thinking about the human", "Nice to meet you too!", 200),
        ]:
            args = json.dumps({"reasoning": reasoning, "text": text})
            client.enqueue(
                LLMResponse(
                    content=None,
                    tool_calls=[
                        {
                            "id": tc_id,
                            "type": "function",
                            "function": {"name": "write_message_to_human", "arguments": args},
                        }
                    ],
                    reasoning=None,
                    reasoning_details=enc,
                    usage=Usage(prompt_tokens=ptok, completion_tokens=20),
                    finish_reason="tool_calls",
                    raw={},
                )
            )
            if tc_id == "tc_g001":
                client.enqueue(make_plain_response("Hi! Nice to meet you."))

        gen = ConversationGenerator(
            client,
            HUMAN_PROFILES[0],
            target_tokens=200,
            max_turns=4,
        )
        gen.generate()

        ai_msgs = [m for m in gen._ai_messages if m.get("role") == "assistant"]
        assert len(ai_msgs) >= 2
        for msg in ai_msgs:
            assert msg.get("tool_calls"), "All assistant msgs should have tool_calls"
            assert msg.get("reasoning") is None, f"reasoning leaked: {msg.get('reasoning')!r}"
            assert msg.get("content") is None, f"content leaked: {msg.get('content')!r}"
            assert msg.get("reasoning_details") == enc, "reasoning_details must be preserved"


class TestResumeLoop:
    """Test resume() from a saved JSONL file."""

    def _generate_and_save(self, monkeypatch, tmp_output_dir: Path) -> Path:
        monkeypatch.setattr(time, "sleep", lambda _: None)
        client = _scripted_client_for_generate()
        profile = HUMAN_PROFILES[0]
        gen = ConversationGenerator(
            client,
            profile,
            target_tokens=300,
            max_turns=10,
        )
        record = gen.generate()
        return save_conversation(record, tmp_output_dir)

    def test_resume_with_valid_raw_context(self, monkeypatch, tmp_output_dir: Path):
        monkeypatch.setattr(time, "sleep", lambda _: None)
        jsonl_path = self._generate_and_save(monkeypatch, tmp_output_dir)

        original = load_conversation_record(jsonl_path)
        original_turn_count = len(original.turns)

        resume_client = FakeChatClient()
        for i in range(6):
            if i % 2 == 0:
                resume_client.enqueue(
                    make_tool_call_response(
                        f"AI continued turn {i}",
                        tool_call_id=f"wmth_r{i:03d}",
                        prompt_tokens=900 + i * 100,
                        completion_tokens=30,
                    )
                )
            else:
                resume_client.enqueue(
                    make_plain_response(
                        f"Human continued turn {i}",
                        prompt_tokens=900 + i * 100,
                        completion_tokens=25,
                    )
                )

        record = ConversationGenerator.resume(
            resume_client,
            jsonl_path,
            target_tokens=500,
            verbose=False,
        )
        assert len(record.turns) >= original_turn_count

    def test_resume_rebuilds_from_corrupted_raw_context(
        self,
        monkeypatch,
        tmp_output_dir: Path,
    ):
        monkeypatch.setattr(time, "sleep", lambda _: None)
        jsonl_path = self._generate_and_save(monkeypatch, tmp_output_dir)

        raw_path = tmp_output_dir / f"{jsonl_path.stem}_raw_ai_context.json"
        raw_path.write_text(json.dumps([{"role": "user", "content": "broken"}]))

        resume_client = FakeChatClient()
        for i in range(6):
            if i % 2 == 0:
                resume_client.enqueue(
                    make_tool_call_response(
                        f"AI rebuilt turn {i}",
                        tool_call_id=f"wmth_rb{i:03d}",
                        prompt_tokens=900 + i * 100,
                        completion_tokens=30,
                    )
                )
            else:
                resume_client.enqueue(
                    make_plain_response(
                        f"Human rebuilt turn {i}",
                        prompt_tokens=900 + i * 100,
                        completion_tokens=25,
                    )
                )

        record = ConversationGenerator.resume(
            resume_client,
            jsonl_path,
            target_tokens=500,
            verbose=False,
        )
        assert len(record.turns) >= 2

        # Verify the context was rebuilt correctly
        ai_calls = [c for c in resume_client.call_log if "write_message_to_human" in str(c.get("tools", ""))]
        assert len(ai_calls) > 0
        ai_call_messages = ai_calls[0]["messages"]
        assert ai_call_messages[0]["role"] == "system"
        # The scripted conversation has 8 turns, which means multiple messages in the context.
        assert len(ai_call_messages) > 5

    def test_resume_missing_jsonl_raises(self, tmp_output_dir: Path):
        fake_client = FakeChatClient()
        with pytest.raises(FileNotFoundError):
            ConversationGenerator.resume(
                fake_client,
                tmp_output_dir / "nonexistent.jsonl",
            )

    def test_resume_inherits_saved_rpm_limits(self, monkeypatch, tmp_output_dir: Path):
        monkeypatch.setattr(time, "sleep", lambda _: None)
        client = _scripted_client_for_generate()
        gen = ConversationGenerator(
            client,
            HUMAN_PROFILES[0],
            target_tokens=300,
            max_turns=10,
            ai_rpm_limit=6,
            human_rpm_limit=4,
        )
        jsonl_path = save_conversation(gen.generate(), tmp_output_dir)

        resume_client = FakeChatClient()
        for i in range(6):
            if i % 2 == 0:
                resume_client.enqueue(
                    make_tool_call_response(
                        f"AI resumed {i}",
                        tool_call_id=f"wmth_rl{i:03d}",
                        prompt_tokens=900 + i * 100,
                        completion_tokens=30,
                    )
                )
            else:
                resume_client.enqueue(
                    make_plain_response(
                        f"Human resumed {i}",
                        prompt_tokens=900 + i * 100,
                        completion_tokens=25,
                    )
                )

        ConversationGenerator.resume(
            resume_client,
            jsonl_path,
            target_tokens=500,
            verbose=False,
        )

        ai_calls = [call for call in resume_client.call_log if call.get("request_role") == "ai"]
        human_calls = [call for call in resume_client.call_log if call.get("request_role") == "human"]
        assert ai_calls
        assert human_calls
        assert all(call.get("rpm_limit") == 6 for call in ai_calls)
        assert all(call.get("rpm_limit") == 4 for call in human_calls)

    def test_resume_uses_saved_metadata_over_new_model_preset_defaults(self, monkeypatch, tmp_output_dir: Path):
        monkeypatch.setattr(time, "sleep", lambda _: None)
        client = _scripted_client_for_generate()
        gen = ConversationGenerator(
            client,
            HUMAN_PROFILES[0],
            target_tokens=300,
            max_turns=10,
            ai_model="minimax/minimax-m2.7",
            human_model="x-ai/grok-4.1-fast",
            ai_provider={"only": ["minimax"], "allow_fallbacks": False},
            ai_reasoning={"effort": "high"},
            ai_max_tokens=777,
            human_max_tokens=333,
        )
        jsonl_path = save_conversation(gen.generate(), tmp_output_dir)

        resume_client = FakeChatClient()
        for i in range(6):
            if i % 2 == 0:
                resume_client.enqueue(
                    make_tool_call_response(
                        f"AI resumed {i}",
                        tool_call_id=f"wmth_keep{i:03d}",
                        prompt_tokens=900 + i * 100,
                        completion_tokens=30,
                    )
                )
            else:
                resume_client.enqueue(
                    make_plain_response(
                        f"Human resumed {i}",
                        prompt_tokens=900 + i * 100,
                        completion_tokens=25,
                    )
                )

        record = ConversationGenerator.resume(
            resume_client,
            jsonl_path,
            target_tokens=500,
            verbose=False,
            ai_model_override="google/gemma-4-31b-it:free",
        )

        assert record.ai_model == "google/gemma-4-31b-it:free"
        assert record.ai_provider == {"only": ["minimax"], "allow_fallbacks": False}
        assert record.ai_reasoning == {"effort": "high"}
        assert record.ai_max_tokens == 777
        assert record.human_max_tokens == 333


class TestVerboseMode:
    """Exercise the verbose _log_turn branches."""

    def test_verbose_generate(self, monkeypatch, capsys):
        monkeypatch.setattr(time, "sleep", lambda _: None)
        client = _scripted_client_for_generate()
        profile = HUMAN_PROFILES[0]

        gen = ConversationGenerator(
            client,
            profile,
            target_tokens=300,
            max_turns=10,
            verbose=True,
        )
        record = gen.generate()
        assert len(record.turns) >= 2

        captured = capsys.readouterr()
        # In verbose mode, it should print panels with borders, e.g. "╭─" or similar,
        # or at least the titles like "Conversation plan" or "AI thinking"
        assert "Conversation plan" in captured.out
        assert "turn 1" in captured.out  # Only printed in verbose mode's panel title

    def test_verbose_with_reasoning_fields(self, monkeypatch, capsys):
        """Exercises the reasoning display paths in _log_turn."""
        monkeypatch.setattr(time, "sleep", lambda _: None)
        client = FakeChatClient()

        client.enqueue(make_plain_response("Plan."))
        client.enqueue(make_tool_call_response("Hi!", tool_call_id="wmth00001"))
        client.enqueue(make_plain_response("Hey!", reasoning="human thinking"))
        args = json.dumps({"reasoning": "inner monologue", "text": "Response"})
        client.enqueue(
            LLMResponse(
                content="draft content",
                tool_calls=[
                    {
                        "id": "wmth00002",
                        "type": "function",
                        "function": {"name": "write_message_to_human", "arguments": args},
                    }
                ],
                reasoning="native reasoning",
                usage=Usage(prompt_tokens=500, completion_tokens=40),
                finish_reason="tool_calls",
                raw={},
            )
        )
        client.enqueue(make_plain_response("Interesting"))
        client.enqueue(
            make_tool_call_response(
                "Thanks!",
                tool_call_id="wmth00003",
                prompt_tokens=600,
                completion_tokens=30,
            )
        )

        profile = HUMAN_PROFILES[0]
        gen = ConversationGenerator(
            client,
            profile,
            target_tokens=300,
            max_turns=10,
            verbose=True,
        )
        record = gen.generate()
        assert len(record.turns) >= 2

        captured = capsys.readouterr()
        assert "human thinking" in captured.out
        assert "inner monologue" in captured.out
        assert "draft content" in captured.out
        assert "native reasoning" in captured.out
