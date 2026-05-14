"""CLI for the Memory Compression Benchmark conversation generator."""

from __future__ import annotations

import argparse
import sys
from typing import cast

from rich.console import Console

from memcomp_bench._interactive_prompts import Prompter
from memcomp_bench.config import (
    AI_MAX_TOKENS,
    AI_MODEL,
    AI_PROVIDER,
    AI_REASONING,
    AI_TEMPERATURE,
    HUMAN_MAX_TOKENS,
    HUMAN_MODEL,
    HUMAN_PROVIDER,
    HUMAN_REASONING,
    HUMAN_TEMPERATURE,
    OUTPUT_DIR,
    TARGET_TOKENS,
    ensure_dirs,
    load_api_key,
)
from memcomp_bench.generator import _UNSET, ConversationGenerator, reformat_markdown, save_conversation
from memcomp_bench.model_registry import MISSING, default_model_for, resolve_model_preset
from memcomp_bench.openrouter_client import OpenRouterClient
from memcomp_bench.prompts import HUMAN_PROFILES, get_human_profile

console = Console()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _resolve_profile(value: str) -> dict[str, str]:
    """Resolve a profile by name (case-insensitive) or numeric index."""
    # Try as number first
    try:
        idx = int(value)
        return get_human_profile(idx)
    except ValueError:
        pass
    # Try by name
    lower = value.lower()
    for p in HUMAN_PROFILES:
        if p["name"].lower() == lower:
            return p
    names = ", ".join(p["name"] for p in HUMAN_PROFILES)
    print(f"ERROR: Unknown profile '{value}'. Available: {names}", file=sys.stderr)
    sys.exit(1)


def _resolve_optional_setting(cli_value: object, preset_value: object, fallback: object) -> object:
    """Return CLI override first, then model preset, then hardcoded fallback."""
    if cli_value is not None:
        return cli_value
    if preset_value is not MISSING:
        return preset_value
    return fallback


def _resolve_provider_setting(provider_arg: str | None, preset_provider: object, fallback: object) -> object:
    """Resolve provider config with CLI taking precedence over preset/default."""
    provider = fallback if preset_provider is MISSING else preset_provider
    if provider_arg is None:
        return provider
    return {"only": [provider_arg], "allow_fallbacks": False} if provider_arg else None


def _build_generate_kwargs(args: argparse.Namespace) -> dict[str, object]:
    """Build ConversationGenerator kwargs for the generate command."""
    ai_model = args.ai_model or default_model_for("ai") or AI_MODEL
    human_model = args.human_model or default_model_for("human") or HUMAN_MODEL
    ai_preset = resolve_model_preset(ai_model, "ai")
    human_preset = resolve_model_preset(human_model, "human")

    return {
        "ai_model": ai_model,
        "human_model": human_model,
        "target_tokens": args.target_tokens or TARGET_TOKENS,
        "language": args.language,
        "companion_mode": args.companion_mode,
        "verbose": args.verbose,
        "ai_provider": _resolve_provider_setting(args.ai_provider, ai_preset.provider, AI_PROVIDER),
        "human_provider": _resolve_provider_setting(args.human_provider, human_preset.provider, HUMAN_PROVIDER),
    }


def _finalize_generate_kwargs(args: argparse.Namespace, kwargs: dict[str, object]) -> dict[str, object]:
    """Fill the remaining per-role generate settings from CLI, presets, and config defaults."""
    ai_model = cast(str, kwargs["ai_model"])
    human_model = cast(str, kwargs["human_model"])
    ai_preset = resolve_model_preset(ai_model, "ai")
    human_preset = resolve_model_preset(human_model, "human")

    kwargs.update(
        {
            "ai_reasoning": _resolve_optional_setting(None, ai_preset.reasoning, AI_REASONING),
            "human_reasoning": _resolve_optional_setting(None, human_preset.reasoning, HUMAN_REASONING),
            "ai_temperature": _resolve_optional_setting(args.ai_temperature, ai_preset.temperature, AI_TEMPERATURE),
            "human_temperature": _resolve_optional_setting(
                args.human_temperature,
                human_preset.temperature,
                HUMAN_TEMPERATURE,
            ),
            "ai_max_tokens": _resolve_optional_setting(args.ai_max_tokens, ai_preset.max_tokens, AI_MAX_TOKENS),
            "human_max_tokens": _resolve_optional_setting(
                args.human_max_tokens,
                human_preset.max_tokens,
                HUMAN_MAX_TOKENS,
            ),
            "ai_rpm_limit": _resolve_optional_setting(args.ai_rpm_limit, ai_preset.rpm_limit, None),
            "human_rpm_limit": _resolve_optional_setting(args.human_rpm_limit, human_preset.rpm_limit, None),
        }
    )
    return kwargs


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate a conversation."""
    ensure_dirs()
    api_key = load_api_key()
    client = OpenRouterClient(api_key)

    profile = _resolve_profile(args.profile)
    console.print(f"[bold]Using human profile: {profile['name']}[/bold]")

    generate_kwargs = _finalize_generate_kwargs(args, _build_generate_kwargs(args))
    generator = ConversationGenerator(
        client,
        profile,
        ai_model=cast(str, generate_kwargs["ai_model"]),
        human_model=cast(str, generate_kwargs["human_model"]),
        target_tokens=cast(int, generate_kwargs["target_tokens"]),
        language=cast(str, generate_kwargs["language"]),
        companion_mode=cast(str, generate_kwargs["companion_mode"]),
        verbose=cast(bool, generate_kwargs["verbose"]),
        ai_provider=cast(dict | None, generate_kwargs["ai_provider"]),
        ai_reasoning=cast(dict | None, generate_kwargs["ai_reasoning"]),
        ai_temperature=cast(float, generate_kwargs["ai_temperature"]),
        ai_max_tokens=cast(int, generate_kwargs["ai_max_tokens"]),
        ai_rpm_limit=cast(int | None, generate_kwargs["ai_rpm_limit"]),
        human_provider=cast(dict | None, generate_kwargs["human_provider"]),
        human_reasoning=cast(dict | None, generate_kwargs["human_reasoning"]),
        human_temperature=cast(float, generate_kwargs["human_temperature"]),
        human_max_tokens=cast(int, generate_kwargs["human_max_tokens"]),
        human_rpm_limit=cast(int | None, generate_kwargs["human_rpm_limit"]),
    )

    try:
        record = generator.generate()
        save_conversation(record, OUTPUT_DIR)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        # Save what we have
        record = generator._record
        if record.turns:
            console.print("[yellow]Saving partial conversation...[/yellow]")
            record.finished_at = "interrupted"
            record.total_cost_usd = client.total_cost
            save_conversation(record, OUTPUT_DIR)
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        raise
    finally:
        client.close()


def cmd_resume(args: argparse.Namespace) -> None:
    """Resume/extend an existing conversation."""
    ensure_dirs()
    api_key = load_api_key()
    client = OpenRouterClient(api_key)

    jsonl_path = args.file
    target = args.target_tokens or TARGET_TOKENS

    try:
        record = ConversationGenerator.resume(
            client,
            jsonl_path,
            target_tokens=target,
            verbose=args.verbose,
            language_override=args.language,
            ai_model_override=args.ai_model or None,
            human_model_override=args.human_model or None,
            # None = not specified (use saved); "" = clear; slug = lock to that provider
            ai_provider_override=(
                _UNSET
                if args.ai_provider is None
                else ({"only": [args.ai_provider], "allow_fallbacks": False} if args.ai_provider else None)
            ),
            human_provider_override=(
                _UNSET
                if args.human_provider is None
                else ({"only": [args.human_provider], "allow_fallbacks": False} if args.human_provider else None)
            ),
            ai_temperature_override=args.ai_temperature,
            human_temperature_override=args.human_temperature,
            ai_max_tokens_override=args.ai_max_tokens,
            human_max_tokens_override=args.human_max_tokens,
            ai_rpm_limit_override=args.ai_rpm_limit,
            human_rpm_limit_override=args.human_rpm_limit,
            persist_resume_defaults=args.persist_resume_defaults,
        )
        save_conversation(record, OUTPUT_DIR)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        raise
    finally:
        client.close()


def cmd_reformat(args: argparse.Namespace) -> None:
    """Reformat the markdown file for an existing conversation."""
    from pathlib import Path

    md_path = reformat_markdown(Path(args.file))
    console.print(f"[bold]Reformatted:[/bold] {md_path}")


def cmd_list_profiles(args: argparse.Namespace) -> None:
    """List available human profiles."""
    for i, profile in enumerate(HUMAN_PROFILES):
        console.print(f"  [bold]{i}[/bold]: {profile['name']}")
        console.print(f"     {profile['backstory'][:120]}...")
        console.print()


def cmd_interactive(
    args: argparse.Namespace,
    *,
    prompter: Prompter | None = None,
    console_override: Console | None = None,
) -> None:
    """Run the prompt-driven interactive benchmark interface."""
    from memcomp_bench.interactive import run_interactive

    run_interactive(
        cmd_generate,
        cmd_resume,
        output_dir=OUTPUT_DIR,
        console=console_override,
        prompter=prompter,
    )


def _add_common_model_args(parser: argparse.ArgumentParser) -> None:
    """Add model/provider/temperature arguments shared by generate and resume."""
    parser.add_argument("--ai-model", type=str, default=None, help="Override AI model")
    parser.add_argument("--human-model", type=str, default=None, help="Override human simulator model")
    parser.add_argument(
        "--ai-provider", dest="ai_provider", type=str, default=None, help="Force AI provider slug (e.g. 'minimax')"
    )
    parser.add_argument("--provider", dest="ai_provider", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--human-provider", type=str, default=None, help="Force human simulator provider slug")
    parser.add_argument("--ai-temperature", type=float, default=None, help="Override AI model temperature")
    parser.add_argument("--human-temperature", type=float, default=None, help="Override human simulator temperature")
    parser.add_argument("--ai-max-tokens", type=int, default=None, help="Override AI max tokens per response")
    parser.add_argument("--human-max-tokens", type=int, default=None, help="Override human max tokens per response")
    parser.add_argument("--ai-rpm-limit", type=_positive_int, default=None, help="Limit AI requests per minute")
    parser.add_argument(
        "--human-rpm-limit", type=_positive_int, default=None, help="Limit human simulator requests per minute"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show full messages and AI thinking")


def _build_generate_parser(sub: argparse._SubParsersAction) -> None:
    """Configure the 'generate' subcommand."""
    gen = sub.add_parser("generate", help="Generate a conversation")
    gen.add_argument(
        "--profile",
        type=str,
        default="0",
        help="Human profile name or index (see 'profiles' command). E.g. --profile vitaly",
    )
    gen.add_argument("--target-tokens", type=int, help=f"Target token count (default: {TARGET_TOKENS:,})")
    gen.add_argument(
        "--language",
        type=str,
        default="english",
        help="Language for conversation (default: english). E.g. 'russian', 'hebrew'",
    )
    gen.add_argument(
        "--companion-mode",
        type=str,
        default="honest",
        choices=["honest"],
        help="Companion mode (default: honest — values honesty over comfort)",
    )
    _add_common_model_args(gen)
    gen.set_defaults(func=cmd_generate)


def _build_resume_parser(sub: argparse._SubParsersAction) -> None:
    """Configure the 'resume' subcommand."""
    res = sub.add_parser("resume", help="Resume/extend an existing conversation")
    res.add_argument("file", type=str, help="Path to the JSONL file of the conversation to resume")
    res.add_argument("--target-tokens", type=int, help=f"New target token count (default: {TARGET_TOKENS:,})")
    res.add_argument(
        "--language", type=str, default=None, help="Override language (normally loaded from saved conversation)"
    )
    res.add_argument(
        "--persist-resume-defaults",
        action="store_true",
        help="Update the saved JSONL resume defaults with any overrides used for this continuation",
    )
    _add_common_model_args(res)
    res.set_defaults(func=cmd_resume)


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory Compression Benchmark — Conversation Generator")
    sub = parser.add_subparsers(dest="command")

    _build_generate_parser(sub)
    _build_resume_parser(sub)

    fmt = sub.add_parser("reformat", help="Reformat the markdown file for an existing conversation")
    fmt.add_argument("file", type=str, help="Path to the .jsonl conversation file")
    fmt.set_defaults(func=cmd_reformat)

    profiles = sub.add_parser("profiles", help="List human profiles")
    profiles.set_defaults(func=cmd_list_profiles)

    interactive = sub.add_parser("interactive", help="Browse saved runs and launch generate/resume flows")
    interactive.set_defaults(func=cmd_interactive)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
