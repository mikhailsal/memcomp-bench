"""CLI for the Memory Compression Benchmark conversation generator."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console

from src.config import (
    AI_MODEL,
    HUMAN_MODEL,
    OUTPUT_DIR,
    TARGET_TOKENS,
    ensure_dirs,
    load_api_key,
)
from src.generator import ConversationGenerator, save_conversation, _estimate_context_tokens
from src.openrouter_client import OpenRouterClient
from src.prompts import HUMAN_PROFILES, get_human_profile

console = Console()


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate a conversation."""
    ensure_dirs()
    api_key = load_api_key()
    client = OpenRouterClient(api_key)

    profile = get_human_profile(args.profile)
    console.print(f"[bold]Using human profile: {profile['name']}[/bold]")

    ai_model = args.ai_model or AI_MODEL
    human_model = args.human_model or HUMAN_MODEL
    target = args.target_tokens or TARGET_TOKENS

    generator = ConversationGenerator(
        client,
        profile,
        ai_model=ai_model,
        human_model=human_model,
        target_tokens=target,
        language=args.language,
        verbose=args.verbose,
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
        )
        save_conversation(record, OUTPUT_DIR)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        raise
    finally:
        client.close()


def cmd_list_profiles(args: argparse.Namespace) -> None:
    """List available human profiles."""
    for i, profile in enumerate(HUMAN_PROFILES):
        console.print(f"  [bold]{i}[/bold]: {profile['name']}")
        console.print(f"     {profile['backstory'][:120]}...")
        console.print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Memory Compression Benchmark — Conversation Generator"
    )
    sub = parser.add_subparsers(dest="command")

    # generate
    gen = sub.add_parser("generate", help="Generate a conversation")
    gen.add_argument(
        "--profile", type=int, default=0,
        help="Human profile index (see 'profiles' command)",
    )
    gen.add_argument("--ai-model", type=str, help=f"AI model (default: {AI_MODEL})")
    gen.add_argument("--human-model", type=str, help=f"Human model (default: {HUMAN_MODEL})")
    gen.add_argument(
        "--target-tokens", type=int,
        help=f"Target token count (default: {TARGET_TOKENS:,})",
    )
    gen.add_argument(
        "--language", type=str, default="english",
        help="Language for conversation (default: english). E.g. 'russian', 'hebrew'",
    )
    gen.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show full messages and AI thinking in real time",
    )
    gen.set_defaults(func=cmd_generate)

    # resume
    res = sub.add_parser("resume", help="Resume/extend an existing conversation")
    res.add_argument(
        "file", type=str,
        help="Path to the JSONL file of the conversation to resume",
    )
    res.add_argument(
        "--target-tokens", type=int,
        help=f"New target token count (default: {TARGET_TOKENS:,})",
    )
    res.add_argument(
        "--language", type=str, default=None,
        help="Override language (normally loaded from saved conversation)",
    )
    res.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show full messages and AI thinking in real time",
    )
    res.set_defaults(func=cmd_resume)

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
