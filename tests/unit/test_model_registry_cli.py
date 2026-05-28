"""Tests for model registry defaults and CLI preset resolution."""

from __future__ import annotations

from types import SimpleNamespace

from tests.unit.test_config_and_cli import _make_generate_args, _make_generate_client


class TestModelRegistry:
    def test_defaults_and_role_overrides_load(self):
        from memcomp_bench.model_registry import default_model_for, resolve_model_preset

        assert default_model_for("ai") == "deepseek/deepseek-v4-flash"
        assert default_model_for("human") == "google/gemini-3.1-flash-lite"

        ai_preset = resolve_model_preset("deepseek/deepseek-v4-flash", "ai")
        assert ai_preset.provider == {"only": ["deepseek"], "allow_fallbacks": False}
        assert ai_preset.reasoning == {"effort": "minimal", "exclude": False, "enable": True}
        assert ai_preset.tool_choice is not False

        human_preset = resolve_model_preset("google/gemini-3.1-flash-lite", "human")
        assert human_preset.disabled is False
        assert human_preset.provider == {"only": ["google-direct"], "allow_fallbacks": False}
        assert human_preset.temperature == 0.9
        assert human_preset.max_tokens == 180

        hy3_preset = resolve_model_preset("tencent/hy3-preview", "ai")
        assert hy3_preset.tool_choice is False

    def test_validate_model_enabled_allows_unknown_models(self):
        from memcomp_bench.model_registry import validate_model_enabled

        validate_model_enabled("unknown/provider-model", "human", usage="generate", source="override")

    def test_validate_model_enabled_rejects_disabled_model(self):
        import pytest

        from memcomp_bench.model_registry import DisabledModelError, validate_model_enabled

        with pytest.raises(DisabledModelError, match="cannot be used for new generations"):
            validate_model_enabled("x-ai/grok-4.1-fast", "human", usage="generate", source="default")


class TestCmdGenerateModelRegistryIntegration:
    def test_cmd_generate_blocks_disabled_default_model(self, monkeypatch, tmp_path, capsys):
        called = False

        monkeypatch.setattr("memcomp_bench.config.ENV_PATH", tmp_path / ".env")
        monkeypatch.setenv("OPENROUTER_KEY", "test-key-for-generate")

        def fake_init(self, client, human_profile, **kwargs):
            nonlocal called
            called = True

        monkeypatch.setattr(
            "memcomp_bench.cli.default_model_for", lambda role: "x-ai/grok-4.1-fast" if role == "human" else None
        )
        monkeypatch.setattr("memcomp_bench.cli.OpenRouterClient", lambda key: _make_generate_client())
        monkeypatch.setattr("memcomp_bench.cli.ConversationGenerator.__init__", fake_init)

        from memcomp_bench.cli import cmd_generate

        cmd_generate(_make_generate_args(ai_model=None, human_model=None))

        assert called is False
        assert "Human model 'x-ai/grok-4.1-fast' is disabled in models.toml" in capsys.readouterr().out

    def test_cmd_generate_uses_models_toml_defaults_for_enabled_models(self, monkeypatch, tmp_path):
        seen: dict[str, object] = {}

        monkeypatch.setattr("memcomp_bench.config.ENV_PATH", tmp_path / ".env")
        monkeypatch.setenv("OPENROUTER_KEY", "test-key-for-generate")

        def fake_init(self, client, human_profile, **kwargs):
            seen.update(kwargs)
            self.client = client
            self.human_profile = human_profile
            self._record = SimpleNamespace(turns=[])

        def fake_generate(self):
            from memcomp_bench.generator import ConversationRecord

            return ConversationRecord(
                id="20260101_000000",
                human_profile=self.human_profile,
                ai_model=seen["ai_model"],
                human_model=seen["human_model"],
            )

        monkeypatch.setattr("memcomp_bench.cli.OpenRouterClient", lambda key: _make_generate_client())
        monkeypatch.setattr("memcomp_bench.cli.ConversationGenerator.__init__", fake_init)
        monkeypatch.setattr("memcomp_bench.cli.ConversationGenerator.generate", fake_generate)
        monkeypatch.setattr("memcomp_bench.cli.save_conversation", lambda record, output_dir: output_dir / "fake.jsonl")

        from memcomp_bench.cli import cmd_generate

        cmd_generate(_make_generate_args(ai_model=None, human_model="google/gemma-4-26b-a4b-it:free"))

        assert seen["ai_model"] == "deepseek/deepseek-v4-flash"
        assert seen["human_model"] == "google/gemma-4-26b-a4b-it:free"
        assert seen["ai_max_tokens"] == 2048
        assert seen["human_max_tokens"] == 800
        assert seen["ai_reasoning"] == {"effort": "minimal", "exclude": False, "enable": True}
        assert seen["human_reasoning"] == {"effort": "minimal", "exclude": False, "enable": True}

    def test_cmd_generate_cli_overrides_models_toml_defaults(self, monkeypatch, tmp_path):
        seen: dict[str, object] = {}

        monkeypatch.setattr("memcomp_bench.config.ENV_PATH", tmp_path / ".env")
        monkeypatch.setenv("OPENROUTER_KEY", "test-key-for-generate")

        def fake_init(self, client, human_profile, **kwargs):
            seen.update(kwargs)
            self.client = client
            self.human_profile = human_profile
            self._record = SimpleNamespace(turns=[])

        def fake_generate(self):
            from memcomp_bench.generator import ConversationRecord

            return ConversationRecord(
                id="20260101_000000",
                human_profile=self.human_profile,
                ai_model=seen["ai_model"],
                human_model=seen["human_model"],
            )

        monkeypatch.setattr("memcomp_bench.cli.OpenRouterClient", lambda key: _make_generate_client())
        monkeypatch.setattr("memcomp_bench.cli.ConversationGenerator.__init__", fake_init)
        monkeypatch.setattr("memcomp_bench.cli.ConversationGenerator.generate", fake_generate)
        monkeypatch.setattr("memcomp_bench.cli.save_conversation", lambda record, output_dir: output_dir / "fake.jsonl")

        from memcomp_bench.cli import cmd_generate

        cmd_generate(
            _make_generate_args(
                ai_model="google/gemma-4-31b-it:free",
                ai_provider="ai-studio",
                ai_max_tokens=999,
                human_model="google/gemma-4-26b-a4b-it:free",
                human_max_tokens=555,
            )
        )

        assert seen["ai_provider"] == {"only": ["ai-studio"], "allow_fallbacks": False}
        assert seen["ai_max_tokens"] == 999
        assert seen["human_max_tokens"] == 555

    def test_cmd_generate_blocks_explicit_disabled_model(self, monkeypatch, tmp_path, capsys):
        called = False

        monkeypatch.setattr("memcomp_bench.config.ENV_PATH", tmp_path / ".env")
        monkeypatch.setenv("OPENROUTER_KEY", "test-key-for-generate")

        def fake_init(self, client, human_profile, **kwargs):
            nonlocal called
            called = True

        monkeypatch.setattr("memcomp_bench.cli.OpenRouterClient", lambda key: _make_generate_client())
        monkeypatch.setattr("memcomp_bench.cli.ConversationGenerator.__init__", fake_init)

        from memcomp_bench.cli import cmd_generate

        cmd_generate(_make_generate_args(human_model="x-ai/grok-4.1-fast"))

        assert called is False
        out = capsys.readouterr().out
        assert "disabled in models.toml" in out
        assert "new generations" in out
