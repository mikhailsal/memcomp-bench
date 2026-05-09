"""Configuration for the Memory Compression Benchmark conversation generator."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
ENV_PATH = PROJECT_ROOT / ".env"
CONFIGS_PATH = PROJECT_ROOT / "configs"

# Load .env early so env vars are available for config constants below
load_dotenv(ENV_PATH)

OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
API_CALL_TIMEOUT = 120

# AI companion model — high independence score, good value
AI_MODEL = "minimax/minimax-m2.7"
AI_TEMPERATURE = 1.1
AI_MAX_TOKENS = 2048

# Provider routing — force official MiniMax provider via OpenRouter
# Set to None to let OpenRouter auto-route, or a dict like {"only": ["minimax"]}
AI_PROVIDER: dict | None = None  # {"only": ["minimax"], "allow_fallbacks": False}

# Reasoning control — minimize native reasoning but keep it visible
# (MiniMax M2.7 requires reasoning enabled; cannot be disabled)
# Set to None to use model defaults, or a dict like {"effort": "minimal"}
AI_REASONING: dict | None = {"effort": "minimal", "exclude": False, "enable": True}

# Human simulator model — fast, cheap, excellent instruction following
HUMAN_MODEL = "x-ai/grok-4.1-fast"
HUMAN_TEMPERATURE = 0.9
HUMAN_MAX_TOKENS = 800

# Provider routing for human simulator — set to None for auto-routing
HUMAN_PROVIDER: dict | None = None

# Reasoning control for human simulator — set to None to use model defaults
HUMAN_REASONING: dict | None = {"effort": "minimal", "exclude": False, "enable": True}

# Companion mode: "honest" (default) — values honesty over comfort
COMPANION_MODE = "honest"

# Topic judge model
JUDGE_MODEL = "google/gemini-3.1-flash-lite-preview"
JUDGE_MAX_TOKENS = 200
TOPIC_CHECK_INTERVAL = 40  # Check topic every N turns

# Conversation parameters
TARGET_TOKENS = 70_000
MAX_TURNS = 10_000
TURN_WARN_THRESHOLD = 250


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_api_key() -> str:
    load_dotenv(ENV_PATH)
    key = os.environ.get("OPENROUTER_KEY", "").strip()
    if not key or key == "your-key-here":
        print(
            "ERROR: OPENROUTER_KEY is not set.\n"
            f"  Create a .env file at {ENV_PATH} with:\n"
            "  OPENROUTER_KEY=sk-or-...\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return key
