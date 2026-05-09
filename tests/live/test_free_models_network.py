"""Live test: fetch free models from the public OpenRouter /models endpoint.

Requires: MEMCOMP_BENCH_NETWORK=1  (no API key needed)
"""

from __future__ import annotations

import pytest

from memcomp_bench.free_models import list_free_models

pytestmark = pytest.mark.network


@pytest.fixture(autouse=True)
def _require_network(network_or_skip):
    pass


class TestFreeModelsNetwork:
    def test_at_least_one_free_model_with_tools(self):
        models = list_free_models(require_tools=True)
        assert len(models) >= 1, "Expected at least 1 free model with tool support"

    def test_returns_multiple_free_models(self):
        models = list_free_models(require_tools=False)
        assert len(models) >= 3, f"Expected >=3 free models, got {len(models)}"

    def test_models_have_context_length(self):
        models = list_free_models(require_tools=True)
        for m in models:
            assert m.context_length > 0, f"{m.id} has zero context_length"

    def test_tool_choice_support_present(self):
        models = list_free_models(require_tools=True)
        with_tc = [m for m in models if m.supports_tool_choice]
        assert len(with_tc) >= 1, "Expected at least 1 free model with tool_choice"
