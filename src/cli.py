"""CLI for the Memory Compression Benchmark conversation generator."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console

from src.config import (
    AI_MAX_TOKENS,
    AI_MODEL,
    AI_PROVIDER,
    AI_REASONING,
    AI_TEMPERATURE,
    HUMAN_MAX_TOKENS,
    HUMAN_MODEL,
    HUMAN_PROVIDER,
    HUMAN_TEMPERATURE,
    OUTPUT_DIR,
    TARGET_TOKENS,
    ensure_dirs,
    load_api_key,
)
from src.generator import _UNSET, ConversationGenerator, reformat_markdown, save_conversation
from src.openrouter_client import OpenRouterClient
from src.prompts import HUMAN_PROFILES, get_human_profile

console = Console()


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


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate a conversation."""
    ensure_dirs()
    api_key = load_api_key()
    client = OpenRouterClient(api_key)

    profile = _resolve_profile(args.profile)
    console.print(f"[bold]Using human profile: {profile['name']}[/bold]")

    ai_model = args.ai_model or AI_MODEL
    human_model = args.human_model or HUMAN_MODEL
    target = args.target_tokens or TARGET_TOKENS

    # Provider override: None = use config default; "" = clear; slug = lock to that provider
    ai_provider = AI_PROVIDER
    if args.provider is not None:
        ai_provider = {"only": [args.provider], "allow_fallbacks": False} if args.provider else None

    # Human provider override
    human_provider = HUMAN_PROVIDER
    if args.human_provider is not None:
        human_provider = {"only": [args.human_provider], "allow_fallbacks": False} if args.human_provider else None

    generator = ConversationGenerator(
        client,
        profile,
        ai_model=ai_model,
        human_model=human_model,
        target_tokens=target,
        language=args.language,
        companion_mode=args.companion_mode,
        verbose=args.verbose,
        ai_provider=ai_provider,
        ai_reasoning=AI_REASONING,
        ai_temperature=args.ai_temperature if args.ai_temperature is not None else AI_TEMPERATURE,
        ai_max_tokens=args.ai_max_tokens if args.ai_max_tokens is not None else AI_MAX_TOKENS,
        human_provider=human_provider,
        human_temperature=args.human_temperature if args.human_temperature is not None else HUMAN_TEMPERATURE,
        human_max_tokens=args.human_max_tokens if args.human_max_tokens is not None else HUMAN_MAX_TOKENS,
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
                if args.provider is None
                else ({"only": [args.provider], "allow_fallbacks": False} if args.provider else None)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory Compression Benchmark — Conversation Generator")
    sub = parser.add_subparsers(dest="command")

    # generate
    gen = sub.add_parser("generate", help="Generate a conversation")
    gen.add_argument(
        "--profile",
        type=str,
        default="0",
        help="Human profile name or index (see 'profiles' command). E.g. --profile vitaly",
    )
    gen.add_argument("--ai-model", type=str, help=f"AI model (default: {AI_MODEL})")
    gen.add_argument("--human-model", type=str, help=f"Human model (default: {HUMAN_MODEL})")
    gen.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Force a specific OpenRouter provider slug for the AI model (e.g. 'minimax')",
    )
    gen.add_argument(
        "--human-provider",
        type=str,
        default=None,
        help="Force a specific OpenRouter provider slug for the human simulator model",
    )
    gen.add_argument(
        "--ai-temperature",
        type=float,
        default=None,
        help=f"Override AI model temperature (default: {AI_TEMPERATURE})",
    )
    gen.add_argument(
        "--human-temperature",
        type=float,
        default=None,
        help=f"Override human simulator temperature (default: {HUMAN_TEMPERATURE})",
    )
    gen.add_argument(
        "--target-tokens",
        type=int,
        help=f"Target token count (default: {TARGET_TOKENS:,})",
    )
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
    gen.add_argument(
        "--ai-max-tokens",
        type=int,
        default=None,
        help=f"Override AI model max tokens per response (default: {AI_MAX_TOKENS})",
    )
    gen.add_argument(
        "--human-max-tokens",
        type=int,
        default=None,
        help=f"Override human simulator max tokens per response (default: {HUMAN_MAX_TOKENS})",
    )
    gen.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show full messages and AI thinking in real time",
    )
    gen.set_defaults(func=cmd_generate)

    # resume
    res = sub.add_parser("resume", help="Resume/extend an existing conversation")
    res.add_argument(
        "file",
        type=str,
        help="Path to the JSONL file of the conversation to resume",
    )
    res.add_argument(
        "--target-tokens",
        type=int,
        help=f"New target token count (default: {TARGET_TOKENS:,})",
    )
    res.add_argument(
        "--language",
        type=str,
        default=None,
        help="Override language (normally loaded from saved conversation)",
    )
    res.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Override AI provider slug (e.g. 'minimax'). Normally loaded from saved conversation.",
    )
    res.add_argument(
        "--human-provider",
        type=str,
        default=None,
        help="Override human simulator provider slug. Normally loaded from saved conversation.",
    )
    res.add_argument(
        "--ai-temperature",
        type=float,
        default=None,
        help="Override AI model temperature. Normally loaded from saved conversation.",
    )
    res.add_argument(
        "--human-temperature",
        type=float,
        default=None,
        help="Override human simulator temperature. Normally loaded from saved conversation.",
    )
    res.add_argument(
        "--ai-model",
        type=str,
        default=None,
        help="Override AI model (default: use model from saved conversation)",
    )
    res.add_argument(
        "--human-model",
        type=str,
        default=None,
        help="Override human simulator model (default: use model from saved conversation)",
    )
    res.add_argument(
        "--ai-max-tokens",
        type=int,
        default=None,
        help="Override AI model max tokens per response. Normally loaded from saved conversation.",
    )
    res.add_argument(
        "--human-max-tokens",
        type=int,
        default=None,
        help="Override human simulator max tokens per response. Normally loaded from saved conversation.",
    )
    res.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show full messages and AI thinking in real time",
    )
    res.set_defaults(func=cmd_resume)

    # reformat
    fmt = sub.add_parser("reformat", help="Reformat the markdown file for an existing conversation")
    fmt.add_argument("file", type=str, help="Path to the .jsonl conversation file")
    fmt.set_defaults(func=cmd_reformat)

    # profiles
    profiles = sub.add_parser("profiles", help="List human profiles")
    profiles.set_defaults(func=cmd_list_profiles)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
