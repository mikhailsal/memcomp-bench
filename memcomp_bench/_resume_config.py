"""Config resolution helpers for conversation resume."""

from __future__ import annotations

from typing import Any

from memcomp_bench.config import (
    AI_MAX_TOKENS,
    AI_PROVIDER,
    AI_REASONING,
    AI_TEMPERATURE,
    HUMAN_MAX_TOKENS,
    HUMAN_PROVIDER,
    HUMAN_REASONING,
    HUMAN_TEMPERATURE,
)
from memcomp_bench.model_registry import MISSING, resolve_model_preset
from memcomp_bench.persistence import get_saved_resume_defaults


def _extract_resume_config(
    metadata: dict,
    *,
    ai_model_override: str | None,
    human_model_override: str | None,
    ai_provider_override: object,
    human_provider_override: object,
    ai_temperature_override: float | None,
    human_temperature_override: float | None,
    ai_max_tokens_override: int | None,
    human_max_tokens_override: int | None,
    ai_rpm_limit_override: int | None,
    human_rpm_limit_override: int | None,
    _UNSET: object,
    language_override: str | None,
) -> dict:
    """Extract and merge saved metadata with CLI overrides into a flat config dict."""
    saved_defaults = get_saved_resume_defaults(metadata)
    ai_model = ai_model_override or saved_defaults["ai_model"]
    human_model = human_model_override or saved_defaults["human_model"]
    ai_preset = resolve_model_preset(ai_model, "ai")
    human_preset = resolve_model_preset(human_model, "human")

    cfg = {
        "profile": metadata["human_profile"],
        "ai_model": ai_model,
        "human_model": human_model,
        "seed_words": metadata.get("seed_words", []),
        "conversation_plan": metadata.get("conversation_plan", ""),
        "language": language_override or saved_defaults.get("language", metadata.get("language", "english")),
        "companion_mode": metadata.get("companion_mode", "supportive"),
        "previous_cost": metadata.get("total_cost_usd", 0.0),
        "saved_resume_defaults": saved_defaults,
    }
    cfg.update(
        _resolve_resume_role_settings(
            saved_defaults,
            ai_preset=ai_preset,
            human_preset=human_preset,
            ai_provider_override=ai_provider_override,
            human_provider_override=human_provider_override,
            ai_temperature_override=ai_temperature_override,
            human_temperature_override=human_temperature_override,
            ai_max_tokens_override=ai_max_tokens_override,
            human_max_tokens_override=human_max_tokens_override,
            ai_rpm_limit_override=ai_rpm_limit_override,
            human_rpm_limit_override=human_rpm_limit_override,
            unset=_UNSET,
        )
    )
    cfg.update(
        _resume_override_metadata(
            ai_model_override=ai_model_override,
            human_model_override=human_model_override,
            ai_provider_override=ai_provider_override,
            human_provider_override=human_provider_override,
            ai_temperature_override=ai_temperature_override,
            human_temperature_override=human_temperature_override,
            ai_max_tokens_override=ai_max_tokens_override,
            human_max_tokens_override=human_max_tokens_override,
            ai_rpm_limit_override=ai_rpm_limit_override,
            human_rpm_limit_override=human_rpm_limit_override,
        )
    )
    return cfg


def _preset_default(preset: Any, field: str, fallback: Any) -> Any:
    value = getattr(preset, field)
    return fallback if value is MISSING else value


def _resolve_resume_value(metadata: dict, key: str, default: Any, override: Any, *, unset: object | None = None) -> Any:
    if unset is not None and override is unset:
        override = None
    if override is not None:
        return override
    if key in metadata:
        return metadata[key]
    return default


def _resolve_resume_setting(
    metadata: dict,
    key: str,
    preset: Any,
    field: str,
    fallback: Any,
    override: Any,
    *,
    unset: object | None = None,
) -> Any:
    return _resolve_resume_value(metadata, key, _preset_default(preset, field, fallback), override, unset=unset)


def _resolve_resume_role_settings(
    saved_defaults: dict,
    *,
    ai_preset: Any,
    human_preset: Any,
    ai_provider_override: object,
    human_provider_override: object,
    ai_temperature_override: float | None,
    human_temperature_override: float | None,
    ai_max_tokens_override: int | None,
    human_max_tokens_override: int | None,
    ai_rpm_limit_override: int | None,
    human_rpm_limit_override: int | None,
    unset: object,
) -> dict[str, Any]:
    specs = (
        ("ai_provider", ai_preset, "provider", AI_PROVIDER, ai_provider_override, unset),
        ("human_provider", human_preset, "provider", HUMAN_PROVIDER, human_provider_override, unset),
        ("ai_reasoning", ai_preset, "reasoning", AI_REASONING, None, None),
        ("human_reasoning", human_preset, "reasoning", HUMAN_REASONING, None, None),
        ("ai_temperature", ai_preset, "temperature", AI_TEMPERATURE, ai_temperature_override, None),
        ("human_temperature", human_preset, "temperature", HUMAN_TEMPERATURE, human_temperature_override, None),
        ("ai_max_tokens", ai_preset, "max_tokens", AI_MAX_TOKENS, ai_max_tokens_override, None),
        ("human_max_tokens", human_preset, "max_tokens", HUMAN_MAX_TOKENS, human_max_tokens_override, None),
        ("ai_rpm_limit", ai_preset, "rpm_limit", None, ai_rpm_limit_override, None),
        ("human_rpm_limit", human_preset, "rpm_limit", None, human_rpm_limit_override, None),
    )

    return {
        key: _resolve_resume_setting(saved_defaults, key, preset, field, fallback, override, unset=unset_value)
        for key, preset, field, fallback, override, unset_value in specs
    }


def _resume_override_metadata(**overrides: Any) -> dict[str, Any]:
    return overrides
