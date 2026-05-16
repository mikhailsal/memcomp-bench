"""Tests for disabled-model validation during resume."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from memcomp_bench.generator import ConversationGenerator, save_conversation
from memcomp_bench.prompts import HUMAN_PROFILES
from tests.conftest import FakeChatClient, make_plain_response, make_tool_call_response
from tests.unit.test_generator_loop import _scripted_client_for_generate


def test_resume_rejects_disabled_saved_model(monkeypatch, tmp_output_dir: Path):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    client = _scripted_client_for_generate()
    gen = ConversationGenerator(
        client,
        HUMAN_PROFILES[0],
        human_model="x-ai/grok-4.1-fast",
        target_tokens=300,
        max_turns=10,
    )
    jsonl_path = save_conversation(gen.generate(), tmp_output_dir)

    with pytest.raises(ValueError, match="Saved human model 'x-ai/grok-4.1-fast' is disabled"):
        ConversationGenerator.resume(
            FakeChatClient(),
            jsonl_path,
            target_tokens=500,
            verbose=False,
        )


def test_resume_allows_replacing_disabled_saved_model(monkeypatch, tmp_output_dir: Path):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    client = _scripted_client_for_generate()
    gen = ConversationGenerator(
        client,
        HUMAN_PROFILES[0],
        human_model="x-ai/grok-4.1-fast",
        target_tokens=300,
        max_turns=10,
    )
    jsonl_path = save_conversation(gen.generate(), tmp_output_dir)

    resume_client = FakeChatClient()
    for i in range(6):
        if i % 2 == 0:
            resume_client.enqueue(
                make_tool_call_response(
                    f"AI resumed {i}",
                    tool_call_id=f"wmth_ro{i:03d}",
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
        human_model_override="google/gemma-4-26b-a4b-it:free",
    )

    assert len(record.turns) >= 2
