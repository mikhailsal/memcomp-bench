from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from memcomp_bench.generator import ConversationGenerator, load_conversation_record, save_conversation
from memcomp_bench.interactive import scan_saved_conversations
from memcomp_bench.persistence_runtime import is_run_published
from memcomp_bench.prompts import HUMAN_PROFILES
from tests.conftest import FakeChatClient, make_plain_response, make_tool_call_response


def _short_generation_client(label: str) -> FakeChatClient:
    return FakeChatClient(
        [
            make_plain_response(f"Plan {label}"),
            make_tool_call_response(f"Hello {label}", tool_call_id="wmth00001"),
            make_plain_response(f"Human {label}"),
        ]
    )


def test_simultaneous_generations_publish_complete_runs(tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    fixed_now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    class FrozenDateTime:
        @staticmethod
        def now(tz=None):
            return fixed_now

    monkeypatch.setattr("memcomp_bench.generator.datetime", FrozenDateTime)

    barrier = threading.Barrier(2)

    def worker(label: str) -> Path:
        barrier.wait()
        generator = ConversationGenerator(
            _short_generation_client(label),
            HUMAN_PROFILES[0],
            target_tokens=10,
            max_turns=1,
        )
        record = generator.generate()
        return save_conversation(record, output_dir)

    with ThreadPoolExecutor(max_workers=2) as executor:
        paths = list(executor.map(worker, ["A", "B"]))

    assert len({path.name for path in paths}) == 2
    assert len(scan_saved_conversations(output_dir)) == 2

    for path in paths:
        assert is_run_published(path) is True
        assert path.with_suffix(".md").exists()
        assert (output_dir / f"{path.stem}_raw_ai_context.json").exists()
        assert load_conversation_record(path).turns
