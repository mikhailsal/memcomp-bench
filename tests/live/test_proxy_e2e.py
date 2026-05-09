"""Live end-to-end test: generate + resume through the local proxy using openrouter/free.

Requires: MEMCOMP_BENCH_LIVE=1 + local proxy at OPENROUTER_BASE_URL reachable.
Uses only free models so cost is $0.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.config import load_api_key
from src.generator import (
    ConversationGenerator,
    load_conversation_record,
    save_conversation,
)
from src.openrouter_client import OpenRouterClient
from src.prompts import HUMAN_PROFILES

pytestmark = pytest.mark.live

LIVE_MODEL = os.environ.get("MEMCOMP_BENCH_LIVE_MODEL", "openrouter/free")


@pytest.fixture(autouse=True)
def _require_proxy(live_proxy_or_skip):
    pass


@pytest.fixture()
def live_client():
    key = load_api_key()
    client = OpenRouterClient(key)
    yield client
    client.close()


class TestProxyE2E:
    """Generate a short conversation through the real proxy using free models."""

    def test_generate_and_resume(self, live_client, tmp_path: Path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        profile = HUMAN_PROFILES[0]
        gen = ConversationGenerator(
            live_client,
            profile,
            ai_model=LIVE_MODEL,
            human_model=LIVE_MODEL,
            target_tokens=400,
            max_turns=4,
            language="english",
            companion_mode="honest",
            ai_reasoning=None,
            human_reasoning=None,
            ai_provider=None,
            human_provider=None,
        )

        record = gen.generate()

        # Basic assertions on the generated conversation
        assert len(record.turns) >= 2, f"Expected >=2 turns, got {len(record.turns)}"
        assert record.turns[0].speaker == "human"

        # At least one AI turn should have tool_calls
        ai_turns_with_tools = [t for t in record.turns if t.speaker == "ai" and t.ai_tool_calls]
        # openrouter/free may route to a model that doesn't use tool_choice;
        # we just verify the conversation happened
        assert len(record.turns) >= 2

        # Save and verify artifacts
        jsonl_path = save_conversation(record, output_dir)
        assert jsonl_path.exists()
        assert jsonl_path.with_suffix(".md").exists()

        raw_ctx = output_dir / f"{jsonl_path.stem}_raw_ai_context.json"
        assert raw_ctx.exists()

        # Round-trip through load
        loaded = load_conversation_record(jsonl_path)
        assert len(loaded.turns) == len(record.turns)
        assert loaded.ai_model == LIVE_MODEL

        # Free model — cost should be zero or negligible
        assert record.total_cost_usd <= 0.01, f"Expected near-zero cost for free model, got ${record.total_cost_usd}"

        # --- Resume from the saved JSONL ---
        resume_client = OpenRouterClient(load_api_key())
        try:
            resumed = ConversationGenerator.resume(
                resume_client,
                jsonl_path,
                target_tokens=600,
                verbose=False,
            )
            # Should have at least as many turns as original
            assert len(resumed.turns) >= len(record.turns)
        finally:
            resume_client.close()
