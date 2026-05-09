"""Discover free models on OpenRouter without API keys.

Fetches the public /models endpoint directly (bypassing any local proxy)
and filters for free models that support tool calling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import httpx

OPENROUTER_PUBLIC_MODELS_URL = "https://openrouter.ai/api/v1/models"

PREFERRED_FAMILIES = ("qwen3", "nemotron", "gemma-4", "ring")


@dataclass
class ModelInfo:
    id: str
    name: str
    context_length: int
    supports_tools: bool
    supports_tool_choice: bool


def _default_fetcher(timeout: int = 10) -> dict[str, Any]:
    with httpx.Client(trust_env=False, timeout=timeout) as client:
        resp = client.get(OPENROUTER_PUBLIC_MODELS_URL)
        resp.raise_for_status()
        return resp.json()


def list_free_models(
    *,
    require_tools: bool = True,
    fetcher: Callable[[], dict[str, Any]] | None = None,
    timeout: int = 10,
) -> list[ModelInfo]:
    """Return free OpenRouter models sorted by capability (context_length desc)."""
    raw = fetcher() if fetcher else _default_fetcher(timeout)
    models = raw.get("data", [])

    results: list[ModelInfo] = []
    for m in models:
        pricing = m.get("pricing", {})
        if str(pricing.get("prompt", "1")) != "0":
            continue
        if str(pricing.get("completion", "1")) != "0":
            continue

        params = m.get("supported_parameters") or []
        has_tools = "tools" in params
        has_tool_choice = "tool_choice" in params

        if require_tools and not has_tools:
            continue

        results.append(
            ModelInfo(
                id=m["id"],
                name=m.get("name", m["id"]),
                context_length=m.get("context_length", 0),
                supports_tools=has_tools,
                supports_tool_choice=has_tool_choice,
            )
        )

    def _sort_key(info: ModelInfo) -> tuple[int, int, str]:
        family_bonus = 1 if any(f in info.id.lower() for f in PREFERRED_FAMILIES) else 0
        tc_bonus = 1 if info.supports_tool_choice else 0
        return (-info.context_length, -(family_bonus + tc_bonus), info.id)

    results.sort(key=_sort_key)
    return results


def pick_best_free_model(
    *,
    require_tools: bool = True,
    env_override: str = "MEMCOMP_BENCH_LIVE_MODEL",
    fetcher: Callable[[], dict[str, Any]] | None = None,
) -> str:
    """Return the best free model ID, or ``openrouter/free`` as fallback."""
    override = os.environ.get(env_override, "").strip()
    if override:
        return override

    try:
        models = list_free_models(require_tools=require_tools, fetcher=fetcher)
        if models:
            return models[0].id
    except Exception:
        pass

    return "openrouter/free"
