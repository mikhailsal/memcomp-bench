"""Tests for model registry defaults and CLI preset resolution."""

from __future__ import annotations

from types import SimpleNamespace

from tests.unit.test_config_and_cli import _make_generate_args, _make_generate_client


class TestModelRegistry:
    def test_defaults_and_role_overrides_load(self):
        from memcomp_bench.model_registry import default_model_for, resolve_model_preset

        assert default_model_for("ai") == "minimax/minimax-m2.7"
        assert default_model_for("human") == "x-ai/grok-4.1-fast"

        ai_preset = resolve_model_preset("minimax/minimax-m2.7", "ai")
        assert ai_preset.temperature == 1.1
        assert ai_preset.max_tokens == 2048
        assert ai_preset.provider is None

        human_preset = resolve_model_preset("x-ai/grok-4.1-fast", "human")
        assert human_preset.temperature == 0.9
        assert human_preset.max_tokens == 180


class TestCmdGenerateModelRegistryIntegration:
    def test_cmd_generate_uses_models_toml_defaults(self, monkeypatch, tmp_path):
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

        cmd_generate(_make_generate_args(ai_model=None, human_model=None))

        assert seen["ai_model"] == "minimax/minimax-m2.7"
        assert seen["human_model"] == "x-ai/grok-4.1-fast"
        assert seen["ai_max_tokens"] == 2048
        assert seen["human_max_tokens"] == 180
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
