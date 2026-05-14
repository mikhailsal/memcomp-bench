# memcomp-bench

Memory Compression Benchmark — measures personality preservation through context compression by generating long multi-turn conversations between two LLM models via the [OpenRouter](https://openrouter.ai/) API.

One model acts as an **AI companion** (with an independent personality, tool-call-based communication, and a randomly generated personality seed). A second model simulates a **human** with a detailed backstory and a pre-generated conversation plan. A lightweight **topic judge** model periodically checks for topic staleness and injects nudges to keep the dialogue natural.

## Prerequisites

- Python ≥ 3.11
- An [OpenRouter API key](https://openrouter.ai/keys)

## Setup

```bash
# Clone and install (editable mode + test/dev deps)
git clone <repo-url> && cd memcomp-bench
python3 -m pip install -e ".[test,dev]"

# Or via Makefile
make install
```

Create a `.env` file in the project root:

```
OPENROUTER_KEY=sk-or-v1-...
```

Optionally override the API base URL (e.g. for a local proxy):

```
OPENROUTER_BASE_URL=http://localhost:8080/api/v1
```

## Running Generation

### Direct CLI

The package exposes a `memcomp` entry point and can also be invoked as a module:

```bash
# Generate a new conversation (default profile: Marcus, 70k tokens)
python -m memcomp_bench.cli generate

# Pick a profile by name or index (see "profiles" command)
python -m memcomp_bench.cli generate --profile vitaly
python -m memcomp_bench.cli generate --profile 8   # Alex (AI-posing-as-human)

# Override models, target tokens, language
python -m memcomp_bench.cli generate \
  --profile anya \
  --ai-model "anthropic/claude-sonnet-4" \
  --human-model "google/gemini-2.5-flash" \
  --target-tokens 100000 \
  --language russian \
  -v   # verbose — shows full messages and AI inner monologue

# Override temperature and max tokens per response
python -m memcomp_bench.cli generate \
  --ai-temperature 0.8 \
  --human-temperature 1.0 \
  --ai-max-tokens 4096 \
  --human-max-tokens 1200

# Limit requests per minute separately for the AI and human simulator
python -m memcomp_bench.cli generate \
  --ai-rpm-limit 20 \
  --human-rpm-limit 10

# Force a specific provider slug (OpenRouter provider routing)
python -m memcomp_bench.cli generate --ai-provider minimax --human-provider x-ai
```

**Resume** an interrupted or completed conversation to extend it further:

```bash
python -m memcomp_bench.cli resume output/conv_20260326_185127_marcus.jsonl
python -m memcomp_bench.cli resume output/conv_20260326_185127_marcus.jsonl \
  --target-tokens 150000 --ai-model "openai/gpt-4.1" -v

# Saved AI/human RPM limits are restored automatically on resume unless overridden
python -m memcomp_bench.cli resume output/conv_20260326_185127_marcus.jsonl \
  --ai-rpm-limit 30 --human-rpm-limit 12
```

**Reformat** the markdown file (useful after render logic updates):

```bash
python -m memcomp_bench.cli reformat output/conv_20260326_185127_marcus.jsonl
```

**List available human profiles:**

```bash
python -m memcomp_bench.cli profiles
```

### Via Makefile

All CLI commands have Makefile wrappers. Pass extra arguments through `ARGS`:

```bash
make generate                                    # default profile, default settings
make generate ARGS="--profile vitaly -v"         # verbose, Vitaly profile
make generate ARGS="--profile alex --language hebrew --target-tokens 50000"
make generate ARGS="--profile michael --ai-rpm-limit 20 --human-rpm-limit 10"
make resume ARGS="output/conv_20260326_185127_marcus.jsonl --target-tokens 150000"
make resume ARGS="output/conv_20260326_185127_marcus.jsonl --ai-rpm-limit 30"
make reformat ARGS="output/conv_20260326_185127_marcus.jsonl"
make profiles                                    # list all profiles
```

Run `make help` to see all available targets.

## Output

Each generation produces three files in `output/`:

| File | Description |
|------|-------------|
| `conv_<timestamp>_<name>.jsonl` | Structured data: metadata, turns, events (one JSON object per line) |
| `conv_<timestamp>_<name>.md` | Human-readable markdown with reasoning panels and context stats |
| `conv_<timestamp>_<name>_raw_ai_context.json` | Full AI message history (required for `resume`) |

## Human Profiles

Nine built-in profiles, each with a unique backstory and conversational style:

| # | Name | Summary |
|---|------|---------|
| 0 | Marcus | Software architect, philosopher, jazz guitarist from Portland |
| 1 | Anya | Illustrator and art teacher from Berlin, originally Moscow |
| 2 | James | History teacher from Chicago, opinionated and witty |
| 3 | Priya | Biotech researcher, amateur astronomer from Bangalore/SF |
| 4 | Leo | Independent journalist from London, digital rights advocate |
| 5 | Michael | AI entrepreneur from Tel Aviv, technically demanding |
| 6 | Nathan | Reclusive former programmer, runs AI experiments obsessively |
| 7 | Vitaly | Burned-out programmer from Minsk, cynical and darkly humorous |
| 8 | Alex | **Special character** — an AI posing as a human, with a phased revelation arc |

## Project Structure

```
memcomp_bench/
├── cli.py              # CLI entry point (generate / resume / reformat / profiles)
├── config.py           # All defaults: models, tokens, temperature, API settings
├── generator.py        # ConversationGenerator — orchestrates the dialogue
├── generator_helpers.py# Dataclasses (Turn, Record, Event) and utility functions
├── openrouter_client.py# OpenRouter HTTP client with retries and cost tracking
├── prompts.py          # AI system prompt, tool definitions, seed words
├── profiles.py         # Human profile definitions and special character prompts
├── prompt_templates.py # Long-form prompt templates (human simulator, plan generator)
├── persistence.py      # Save/load JSONL, markdown rendering, reformat
├── _run_loop.py        # Core conversation loop (extracted for file-size limits)
├── _resume.py          # Resume logic (context restoration, config merging)
└── _logging.py         # Rich-based verbose/compact turn logging
```

## Tests & Code Quality

```bash
make test              # unit tests only (no network)
make test-live         # requires a local AI proxy (MEMCOMP_BENCH_LIVE=1)
make test-network      # hits the public OpenRouter API (MEMCOMP_BENCH_NETWORK=1)
make test-all          # everything

make coverage          # unit tests with coverage report (fail_under = 87%)
make lint              # ruff linter
make typecheck         # mypy
make format            # auto-format with ruff
make check             # lint + typecheck + unit tests
```

Pre-commit hooks enforce formatting, linting, file/function length limits (500/65 lines), and coverage:

```bash
make hooks-install     # set up hooks
make hooks-run         # run all hooks against the full repo
```

## Configuration Defaults

Key defaults from `config.py` (all overridable via CLI flags):

| Parameter | Default |
|-----------|---------|
| AI model | `minimax/minimax-m2.7` |
| Human model | `x-ai/grok-4.1-fast` |
| Topic judge | `google/gemini-3.1-flash-lite-preview` |
| Target tokens | 70,000 |
| AI temperature | 1.1 |
| Human temperature | 0.9 |
| AI max tokens / response | 2,048 |
| Human max tokens / response | 800 |
| AI RPM limit | unset |
| Human RPM limit | unset |
| AI reasoning | `{"effort": "minimal", "exclude": false, "enable": true}` |
| Topic check interval | Every 40 turns |

When set, RPM limits are stored in the conversation JSONL metadata and reused by `resume` unless you pass new override values.
