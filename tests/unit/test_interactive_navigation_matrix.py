"""Navigation matrix coverage for interactive Enter/Esc behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from memcomp_bench import _interactive_prompts as prompt_module
from memcomp_bench import interactive as interactive_module
from memcomp_bench._interactive_display import format_run_line
from memcomp_bench._interactive_prompts import CANCEL, SORT_ACTION, PromptBack, TerminalMenuPrompter
from memcomp_bench.generator import save_conversation
from memcomp_bench.interactive import MODE_NEW, MODE_RESUME, MODE_RESUME_LAST, MODE_VIEW, scan_saved_conversations
from tests.unit.test_interactive import _console, _make_record

ESCAPE = object()
DEFAULT = object()

GENERATE_STAGE_ORDER = [
    "profile",
    "target_tokens",
    "language",
    "companion_mode",
    "verbose",
    "ai_model",
    "human_model",
    "ai_provider",
    "human_provider",
    "ai_temperature",
    "human_temperature",
    "ai_max_tokens",
    "human_max_tokens",
    "ai_rpm_limit",
    "human_rpm_limit",
]

RESUME_OVERRIDE_STAGE_ORDER = [
    "language",
    "ai_model",
    "human_model",
    "ai_provider",
    "human_provider",
    "ai_temperature",
    "human_temperature",
    "ai_max_tokens",
    "human_max_tokens",
    "ai_rpm_limit",
    "human_rpm_limit",
]

RESUME_OVERRIDE_STAGE_ATTRS = {
    "language": "language",
    "ai_model": "ai_model",
    "human_model": "human_model",
    "ai_provider": "ai_provider",
    "human_provider": "human_provider",
    "ai_temperature": "ai_temperature",
    "human_temperature": "human_temperature",
    "ai_max_tokens": "ai_max_tokens",
    "human_max_tokens": "human_max_tokens",
    "ai_rpm_limit": "ai_rpm_limit",
    "human_rpm_limit": "human_rpm_limit",
}

EXPLICIT_GENERATE_VALUES = {
    "profile": "2  James",
    "target_tokens": "900",
    "language": "spanish",
    "companion_mode": "honest",
    "verbose": True,
    "ai_model": "custom/ai",
    "human_model": "custom/human",
    "ai_provider": "openai",
    "human_provider": "anthropic",
    "ai_temperature": "0.8",
    "human_temperature": "1.0",
    "ai_max_tokens": "1024",
    "human_max_tokens": "512",
    "ai_rpm_limit": "20",
    "human_rpm_limit": "10",
}

EXPLICIT_RESUME_OVERRIDE_VALUES = {
    "language": "german",
    "ai_model": "override/ai",
    "human_model": "override/human",
    "ai_provider": "openai",
    "human_provider": "anthropic",
    "ai_temperature": "0.8",
    "human_temperature": "1.0",
    "ai_max_tokens": "1024",
    "human_max_tokens": "512",
    "ai_rpm_limit": "20",
    "human_rpm_limit": "10",
}


class ScriptedTerminalPrompter(TerminalMenuPrompter):
    def __init__(self, steps: list[tuple[str, object]]) -> None:
        self.steps = list(steps)
        self.select_prompts: list[str] = []
        self.menu_prompts: list[str] = []

    def _pop(self, expected_kind: str, prompt: str) -> object:
        if not self.steps:
            raise AssertionError(f"Unexpected {expected_kind}: {prompt}")
        kind, value = self.steps.pop(0)
        if kind != expected_kind:
            raise AssertionError(f"Expected {kind} but got {expected_kind} for prompt: {prompt}")
        return value

    def ask(self, prompt: str, *, default: str | None = None) -> str:
        value = self._pop("ask", prompt)
        if value is ESCAPE:
            raise PromptBack
        if value is DEFAULT:
            return "" if default is None else default
        answer = str(value)
        if answer == "" and default is not None:
            return default
        return answer

    def confirm(self, prompt: str, *, default: bool = False) -> bool:
        value = self._pop("confirm", prompt)
        if value is ESCAPE:
            raise PromptBack
        if value is DEFAULT:
            return default
        return bool(value)

    def select(self, prompt: str, choices: list[str], *, default: str | None = None) -> str:
        self.select_prompts.append(prompt)
        value = self._pop("select", prompt)
        if value == CANCEL:
            return CANCEL
        if value is DEFAULT:
            if default in choices:
                return default
            return choices[0]
        answer = str(value)
        if answer == "":
            if default in choices:
                return default
            return choices[0]
        if answer not in choices:
            raise AssertionError(f"Invalid select answer {answer!r} for prompt {prompt!r}")
        return answer

    def menu(
        self,
        prompt: str,
        choices: list[str],
        *,
        default: str | None = None,
        extra_accept_keys: tuple[str, ...] = (),
        status: str | None = None,
        skip_indices: list[int] | None = None,
    ) -> tuple[str, str | None]:
        del extra_accept_keys, status, skip_indices
        self.menu_prompts.append(prompt)
        value = self._pop("menu", prompt)
        if value == CANCEL:
            return CANCEL, None
        if value == SORT_ACTION:
            default_value = default if default in choices else choices[0]
            return default_value, "s"
        if value is DEFAULT:
            default_value = default if default in choices else choices[0]
            return default_value, "enter"
        answer = str(value)
        if answer == "":
            default_value = default if default in choices else choices[0]
            return default_value, "enter"
        if answer not in choices:
            raise AssertionError(f"Invalid menu answer {answer!r} for prompt {prompt!r}")
        return answer, "enter"


def _patch_terminal_width(monkeypatch) -> None:
    monkeypatch.setattr("memcomp_bench.interactive.terminal_width", lambda: 120)


def _main_prompt_count(prompter: ScriptedTerminalPrompter) -> int:
    return sum("What would you like to do?" in prompt for prompt in prompter.select_prompts)


def _run_picker_prompt_count(prompter: ScriptedTerminalPrompter) -> int:
    return sum(
        (" runs | " in prompt and " sorted: " in prompt) or (" run | " in prompt and " sorted: " in prompt)
        for prompt in prompter.menu_prompts
    )


def _make_saved_run(output_dir: Path) -> str:
    save_conversation(_make_record(), output_dir)
    summaries = scan_saved_conversations(output_dir)
    return format_run_line(1, 1, summaries[0], width=120)


def _run_generate_flow(prompter: ScriptedTerminalPrompter, output_dir: Path):
    called = {}
    console, _ = _console()
    interactive_module.run_interactive(
        lambda args: called.setdefault("generate", args),
        lambda args: called.setdefault("resume", args),
        output_dir=output_dir,
        console=console,
        prompter=prompter,
    )
    return called


def _run_resume_flow(prompter: ScriptedTerminalPrompter, output_dir: Path):
    called = {}
    console, _ = _console()
    interactive_module.run_interactive(
        lambda args: called.setdefault("generate", args),
        lambda args: called.setdefault("resume", args),
        output_dir=output_dir,
        console=console,
        prompter=prompter,
    )
    return called


def _generate_steps_for_default_stage(stage: str) -> list[tuple[str, object]]:
    steps: list[tuple[str, object]] = [("select", MODE_NEW)]
    for stage_name in GENERATE_STAGE_ORDER:
        value: object
        if stage_name == stage:
            value = DEFAULT
        else:
            value = EXPLICIT_GENERATE_VALUES[stage_name]
        kind = (
            "confirm" if stage_name == "verbose" else "select" if stage_name in {"profile", "companion_mode"} else "ask"
        )
        steps.append((kind, value))
    return steps


def _generate_steps_for_escape_stage(stage: str) -> list[tuple[str, object]]:
    steps: list[tuple[str, object]] = [("select", MODE_NEW)]
    for stage_name in GENERATE_STAGE_ORDER:
        if stage_name == stage:
            value: object = CANCEL if stage_name in {"profile", "companion_mode"} else ESCAPE
        else:
            value = EXPLICIT_GENERATE_VALUES[stage_name]
        kind = (
            "confirm" if stage_name == "verbose" else "select" if stage_name in {"profile", "companion_mode"} else "ask"
        )
        steps.append((kind, value))
        if stage_name == stage:
            break
    steps.append(("select", CANCEL))
    return steps


def _resume_steps_for_escape_stage(run_line: str, stage: str, *, use_last: bool) -> list[tuple[str, object]]:
    steps: list[tuple[str, object]] = [("select", MODE_RESUME_LAST if use_last else MODE_RESUME)]
    if not use_last:
        if stage == "run_picker":
            steps.extend([("menu", CANCEL), ("select", CANCEL)])
            return steps
        steps.append(("menu", run_line))

    if stage == "detail":
        steps.append(("menu", CANCEL))
        if use_last:
            steps.append(("select", CANCEL))
        else:
            steps.extend([("menu", CANCEL), ("select", CANCEL)])
        return steps

    steps.append(("menu", DEFAULT))

    if stage == "resume_mode":
        steps.append(("select", CANCEL))
        if use_last:
            steps.append(("select", CANCEL))
        else:
            steps.extend([("menu", CANCEL), ("select", CANCEL)])
        return steps

    edited_flow = stage in set(RESUME_OVERRIDE_STAGE_ORDER) | {"persist_defaults", "target_tokens_edited"}
    steps.append(("select", "Edit defaults before continuing" if edited_flow else "Continue with saved defaults"))

    if stage == "target_tokens_saved":
        steps.append(("ask", ESCAPE))
        if use_last:
            steps.append(("select", CANCEL))
        else:
            steps.extend([("menu", CANCEL), ("select", CANCEL)])
        return steps

    if edited_flow:
        for stage_name in RESUME_OVERRIDE_STAGE_ORDER:
            if stage_name == stage:
                steps.append(("ask", ESCAPE))
                break
            steps.append(("ask", EXPLICIT_RESUME_OVERRIDE_VALUES[stage_name]))
        else:
            if stage == "persist_defaults":
                steps.append(("confirm", ESCAPE))
            elif stage == "target_tokens_edited":
                steps.append(("confirm", False))
                steps.append(("ask", ESCAPE))
        if use_last:
            steps.append(("select", CANCEL))
        else:
            steps.extend([("menu", CANCEL), ("select", CANCEL)])
        return steps

    raise AssertionError(f"Unhandled resume escape stage: {stage}")


def _resume_steps_for_default_override(stage: str) -> list[tuple[str, object]]:
    steps: list[tuple[str, object]] = [
        ("select", MODE_RESUME_LAST),
        ("menu", DEFAULT),
        ("select", "Edit defaults before continuing"),
    ]
    for stage_name in RESUME_OVERRIDE_STAGE_ORDER:
        steps.append(("ask", DEFAULT if stage_name == stage else EXPLICIT_RESUME_OVERRIDE_VALUES[stage_name]))
    steps.extend(
        [
            ("confirm", False),
            ("ask", "700"),
        ]
    )
    return steps


@pytest.mark.parametrize("stage", GENERATE_STAGE_ORDER, ids=GENERATE_STAGE_ORDER)
def test_generate_enter_accepts_default_at_each_stage(stage: str):
    defaults = prompt_module._default_generate_values()
    prompter = ScriptedTerminalPrompter(_generate_steps_for_default_stage(stage))

    called = _run_generate_flow(prompter, Path("/tmp/nonexistent-interactive-output"))

    args = called["generate"]
    expected = {
        "profile": "0",
        "target_tokens": defaults["target_tokens"],
        "language": defaults["language"],
        "companion_mode": defaults["companion_mode"],
        "verbose": False,
        "ai_model": defaults["ai_model"],
        "human_model": defaults["human_model"],
        "ai_provider": defaults["ai_provider"],
        "human_provider": defaults["human_provider"],
        "ai_temperature": defaults["ai_temperature"],
        "human_temperature": defaults["human_temperature"],
        "ai_max_tokens": defaults["ai_max_tokens"],
        "human_max_tokens": defaults["human_max_tokens"],
        "ai_rpm_limit": defaults["ai_rpm_limit"],
        "human_rpm_limit": defaults["human_rpm_limit"],
    }[stage]
    attr = "profile" if stage == "profile" else stage
    assert getattr(args, attr) == expected


@pytest.mark.parametrize("stage", GENERATE_STAGE_ORDER, ids=GENERATE_STAGE_ORDER)
def test_generate_esc_returns_to_main_menu_from_each_stage(stage: str):
    prompter = ScriptedTerminalPrompter(_generate_steps_for_escape_stage(stage))

    called = _run_generate_flow(prompter, Path("/tmp/nonexistent-interactive-output"))

    assert called == {}
    assert _main_prompt_count(prompter) == 2


@pytest.mark.parametrize(
    "stage",
    [
        "run_picker",
        "detail",
        "resume_mode",
        *RESUME_OVERRIDE_STAGE_ORDER,
        "persist_defaults",
        "target_tokens_saved",
        "target_tokens_edited",
    ],
    ids=[
        "run_picker",
        "detail",
        "resume_mode",
        *RESUME_OVERRIDE_STAGE_ORDER,
        "persist_defaults",
        "target_tokens_saved",
        "target_tokens_edited",
    ],
)
def test_resume_esc_returns_to_parent_from_each_stage(stage: str, tmp_path: Path, monkeypatch):
    _patch_terminal_width(monkeypatch)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    run_line = _make_saved_run(output_dir)
    prompter = ScriptedTerminalPrompter(_resume_steps_for_escape_stage(run_line, stage, use_last=False))

    called = _run_resume_flow(prompter, output_dir)

    assert called == {}
    if stage == "run_picker":
        assert _main_prompt_count(prompter) == 2
    else:
        assert _run_picker_prompt_count(prompter) == 2


@pytest.mark.parametrize(
    "stage",
    [
        "detail",
        "resume_mode",
        *RESUME_OVERRIDE_STAGE_ORDER,
        "persist_defaults",
        "target_tokens_saved",
        "target_tokens_edited",
    ],
    ids=[
        "detail",
        "resume_mode",
        *RESUME_OVERRIDE_STAGE_ORDER,
        "persist_defaults",
        "target_tokens_saved",
        "target_tokens_edited",
    ],
)
def test_resume_last_esc_returns_to_main_menu_from_each_stage(stage: str, tmp_path: Path, monkeypatch):
    _patch_terminal_width(monkeypatch)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    run_line = _make_saved_run(output_dir)
    prompter = ScriptedTerminalPrompter(_resume_steps_for_escape_stage(run_line, stage, use_last=True))

    called = _run_resume_flow(prompter, output_dir)

    assert called == {}
    assert _main_prompt_count(prompter) == 2


@pytest.mark.parametrize("stage", RESUME_OVERRIDE_STAGE_ORDER, ids=RESUME_OVERRIDE_STAGE_ORDER)
def test_resume_last_enter_keeps_saved_value_at_each_override_stage(stage: str, tmp_path: Path, monkeypatch):
    _patch_terminal_width(monkeypatch)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _make_saved_run(output_dir)
    prompter = ScriptedTerminalPrompter(_resume_steps_for_default_override(stage))

    called = _run_resume_flow(prompter, output_dir)

    args = called["resume"]
    assert getattr(args, RESUME_OVERRIDE_STAGE_ATTRS[stage]) is None
    assert args.persist_resume_defaults is False
    assert args.target_tokens == 700


def test_view_esc_returns_to_parent_at_each_menu_stage(tmp_path: Path, monkeypatch):
    _patch_terminal_width(monkeypatch)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    run_line = _make_saved_run(output_dir)

    stage_scripts = {
        "run_picker": [("select", MODE_VIEW), ("menu", CANCEL), ("select", CANCEL)],
        "sort_order": [
            ("select", MODE_VIEW),
            ("menu", SORT_ACTION),
            ("select", CANCEL),
            ("menu", CANCEL),
            ("select", CANCEL),
        ],
        "detail": [
            ("select", MODE_VIEW),
            ("menu", run_line),
            ("menu", CANCEL),
            ("menu", CANCEL),
            ("select", CANCEL),
        ],
    }

    for stage, steps in stage_scripts.items():
        prompter = ScriptedTerminalPrompter(steps)
        called = _run_resume_flow(prompter, output_dir)
        assert called == {}, stage
        if stage == "run_picker":
            assert _main_prompt_count(prompter) == 2, stage
        else:
            assert _run_picker_prompt_count(prompter) == 2, stage
