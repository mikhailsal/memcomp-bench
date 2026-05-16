"""Model preset loading for generate/resume defaults."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 test environment fallback
    import tomli as tomllib  # type: ignore[import-not-found]

from memcomp_bench.config import PROJECT_ROOT

MODELS_TOML_PATH = PROJECT_ROOT / "models.toml"
MISSING = object()


@dataclass(frozen=True)
class ModelPreset:
    disabled: bool = False
    provider: dict[str, Any] | None | object = MISSING
    reasoning: dict[str, Any] | None | object = MISSING
    tool_choice: bool | object = MISSING
    temperature: float | object = MISSING
    max_tokens: int | object = MISSING
    rpm_limit: int | None | object = MISSING


@dataclass(frozen=True)
class ModelCatalog:
    default_ai_model: str | None
    default_human_model: str | None
    models: dict[str, dict[str, Any]]


class DisabledModelError(ValueError):
    """Raised when a configured model is marked disabled in models.toml."""


def _is_disabled_model(model_data: dict[str, Any] | None) -> bool:
    return bool(model_data and model_data.get("disabled") is True)


def _normalize_provider(value: Any) -> dict[str, Any] | None | object:
    if value is MISSING:
        return MISSING
    if value is None:
        return None
    if isinstance(value, str):
        provider = value.strip()
        if not provider or provider.lower() == "auto":
            return None
        return {"only": [provider], "allow_fallbacks": False}
    if isinstance(value, dict):
        if not value or value.get("mode") == "auto":
            return None
        return dict(value)
    raise TypeError(f"Unsupported provider config: {value!r}")


def _normalize_reasoning(value: Any) -> dict[str, Any] | None | object:
    if value is MISSING:
        return MISSING
    if value is None or value is False:
        return None
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(f"Unsupported reasoning config: {value!r}")


def _normalize_tool_choice(value: Any) -> bool | object:
    if value is MISSING:
        return MISSING
    if isinstance(value, bool):
        return value
    raise TypeError(f"Unsupported tool_choice config: {value!r}")


@lru_cache(maxsize=1)
def load_model_catalog(path: Path = MODELS_TOML_PATH) -> ModelCatalog:
    if not path.exists():
        return ModelCatalog(default_ai_model=None, default_human_model=None, models={})

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    defaults = raw.get("defaults", {})
    models = raw.get("models", {})
    return ModelCatalog(
        default_ai_model=defaults.get("ai_model"),
        default_human_model=defaults.get("human_model"),
        models=models,
    )


def resolve_model_preset(model: str, role: str) -> ModelPreset:
    catalog = load_model_catalog()
    model_data = catalog.models.get(model)
    if not model_data:
        return ModelPreset()

    merged: dict[str, Any] = {
        key: value
        for key, value in model_data.items()
        if key in {"provider", "reasoning", "tool_choice", "temperature", "max_tokens", "rpm_limit"}
    }
    role_data = model_data.get("roles", {}).get(role, {})
    for key, value in role_data.items():
        if key in {"provider", "reasoning", "tool_choice", "temperature", "max_tokens", "rpm_limit"}:
            merged[key] = value

    return ModelPreset(
        disabled=_is_disabled_model(model_data),
        provider=_normalize_provider(merged.get("provider", MISSING)),
        reasoning=_normalize_reasoning(merged.get("reasoning", MISSING)),
        tool_choice=_normalize_tool_choice(merged.get("tool_choice", MISSING)),
        temperature=merged.get("temperature", MISSING),
        max_tokens=merged.get("max_tokens", MISSING),
        rpm_limit=merged.get("rpm_limit", MISSING),
    )


def model_uses_tool_choice(model: str, role: str) -> bool:
    tool_choice = resolve_model_preset(model, role).tool_choice
    return True if tool_choice is MISSING else bool(tool_choice)


def validate_model_enabled(model: str, role: str, *, usage: str, source: str) -> None:
    """Reject configured disabled models with a role- and usage-aware message."""
    model_data = load_model_catalog().models.get(model)
    if not _is_disabled_model(model_data):
        return

    role_label = "AI" if role == "ai" else "Human"
    if usage == "generate":
        raise DisabledModelError(
            f"{role_label} model '{model}' is disabled in models.toml and cannot be used for new generations."
        )

    if usage == "resume":
        if source == "saved":
            raise DisabledModelError(
                f"Saved {role} model '{model}' is disabled in models.toml and cannot be used to continue this run. "
                f"Choose a replacement with '--{role}-model'."
            )
        raise DisabledModelError(
            f"{role_label} model '{model}' is disabled in models.toml and cannot be used to continue this run."
        )

    raise ValueError(f"Unknown model validation usage: {usage}")


def default_model_for(role: str) -> str | None:
    catalog = load_model_catalog()
    if role == "ai":
        return catalog.default_ai_model
    if role == "human":
        return catalog.default_human_model
    raise ValueError(f"Unknown role: {role}")
