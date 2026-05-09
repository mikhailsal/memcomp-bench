"""Tests for src.free_models with fake fetchers (no network)."""

from __future__ import annotations

from src.free_models import (
    ModelInfo,
    list_free_models,
    pick_best_free_model,
)

# ---------------------------------------------------------------------------
# Fake model data
# ---------------------------------------------------------------------------


def _fake_models_response() -> dict:
    return {
        "data": [
            {
                "id": "google/gemma-4-31b-it:free",
                "name": "Gemma 4 31B",
                "context_length": 262144,
                "pricing": {"prompt": "0", "completion": "0"},
                "supported_parameters": ["tools", "tool_choice", "max_tokens", "temperature"],
            },
            {
                "id": "meta/llama-small:free",
                "name": "Llama Small",
                "context_length": 8192,
                "pricing": {"prompt": "0", "completion": "0"},
                "supported_parameters": ["max_tokens", "temperature"],
            },
            {
                "id": "nvidia/nemotron-3-super:free",
                "name": "Nemotron 3 Super",
                "context_length": 262144,
                "pricing": {"prompt": "0", "completion": "0"},
                "supported_parameters": ["tools", "tool_choice", "max_tokens"],
            },
            {
                "id": "openai/gpt-5:paid",
                "name": "GPT-5",
                "context_length": 128000,
                "pricing": {"prompt": "0.01", "completion": "0.03"},
                "supported_parameters": ["tools", "tool_choice"],
            },
            {
                "id": "qwen/qwen3-next:free",
                "name": "Qwen3 Next",
                "context_length": 262144,
                "pricing": {"prompt": "0", "completion": "0"},
                "supported_parameters": ["tools", "tool_choice", "max_tokens"],
            },
        ]
    }


# ---------------------------------------------------------------------------
# list_free_models
# ---------------------------------------------------------------------------


class TestListFreeModels:
    def test_filters_free_only(self):
        models = list_free_models(require_tools=False, fetcher=_fake_models_response)
        ids = [m.id for m in models]
        assert "openai/gpt-5:paid" not in ids
        assert "meta/llama-small:free" in ids

    def test_filters_tools_required(self):
        models = list_free_models(require_tools=True, fetcher=_fake_models_response)
        ids = [m.id for m in models]
        assert "meta/llama-small:free" not in ids
        assert len(ids) == 3

    def test_sorted_by_context_length_desc(self):
        models = list_free_models(require_tools=True, fetcher=_fake_models_response)
        lengths = [m.context_length for m in models]
        assert lengths == sorted(lengths, reverse=True)

    def test_preferred_families_get_priority(self):
        models = list_free_models(require_tools=True, fetcher=_fake_models_response)
        top_ids = [m.id for m in models[:3]]
        # qwen3 and nemotron have preferred-family bonus
        has_preferred = any("qwen3" in mid or "nemotron" in mid or "gemma-4" in mid for mid in top_ids)
        assert has_preferred

    def test_model_info_fields(self):
        models = list_free_models(require_tools=True, fetcher=_fake_models_response)
        m = models[0]
        assert isinstance(m, ModelInfo)
        assert m.supports_tools is True
        assert m.context_length > 0

    def test_empty_data(self):
        models = list_free_models(fetcher=lambda: {"data": []})
        assert models == []

    def test_filters_out_free_prompt_paid_completion(self):
        """Model with prompt=0 but completion!=0 should be excluded."""
        data = {
            "data": [
                {
                    "id": "half/free:model",
                    "name": "Half Free",
                    "context_length": 100000,
                    "pricing": {"prompt": "0", "completion": "0.01"},
                    "supported_parameters": ["tools"],
                }
            ]
        }
        models = list_free_models(require_tools=False, fetcher=lambda: data)
        assert len(models) == 0


# ---------------------------------------------------------------------------
# pick_best_free_model
# ---------------------------------------------------------------------------


class TestPickBestFreeModel:
    def test_returns_top_model(self):
        result = pick_best_free_model(fetcher=_fake_models_response)
        assert ":free" in result

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("MEMCOMP_BENCH_LIVE_MODEL", "custom/my-model:free")
        result = pick_best_free_model(fetcher=_fake_models_response)
        assert result == "custom/my-model:free"

    def test_empty_env_override_ignored(self, monkeypatch):
        monkeypatch.setenv("MEMCOMP_BENCH_LIVE_MODEL", "")
        result = pick_best_free_model(fetcher=_fake_models_response)
        assert result != ""
        assert ":free" in result

    def test_network_failure_fallback(self):
        def failing_fetcher():
            raise ConnectionError("no internet")

        result = pick_best_free_model(fetcher=failing_fetcher)
        assert result == "openrouter/free"

    def test_empty_result_fallback(self):
        result = pick_best_free_model(
            require_tools=True,
            fetcher=lambda: {"data": []},
        )
        assert result == "openrouter/free"
