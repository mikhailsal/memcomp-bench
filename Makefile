.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

PYTHON   ?= python3
PYTEST    = $(PYTHON) -m pytest
ARGS     ?=

SRC_DIRS  = src/ tests/
COV_OPTS  = --cov=src --cov-report=term-missing --cov-report=html

# ---------------------------------------------------------------------------
# Help  (default target — prints all documented targets)
# ---------------------------------------------------------------------------

.PHONY: help
help: ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\n\033[1mUsage:\033[0m  make \033[36m<target>\033[0m [ARGS=\"...\"]\n\n\033[1mTargets:\033[0m\n"} \
		/^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

.PHONY: install
install: ## Install package in editable mode with test + dev dependencies
	$(PYTHON) -m pip install -e ".[test,dev]"

# ---------------------------------------------------------------------------
# Benchmark CLI  (pass extra arguments via ARGS="...")
# ---------------------------------------------------------------------------

.PHONY: generate resume reformat profiles

generate: ## Generate a conversation          (ARGS="--profile vitaly -v")
	$(PYTHON) -m src.cli generate $(ARGS)

resume: ## Resume an existing conversation   (ARGS="output/conv_xxx.jsonl")
	$(PYTHON) -m src.cli resume $(ARGS)

reformat: ## Reformat markdown for a conversation (ARGS="output/conv_xxx.jsonl")
	$(PYTHON) -m src.cli reformat $(ARGS)

profiles: ## List available human profiles
	$(PYTHON) -m src.cli profiles

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

.PHONY: test test-unit test-live test-network test-all

test: test-unit ## Run unit tests (safe default, no network needed)

test-unit: ## Run unit tests only
	$(PYTEST) tests/unit/ $(ARGS)

test-live: ## Run live proxy tests   (requires local AI proxy)
	MEMCOMP_BENCH_LIVE=1 $(PYTEST) -m live tests/live/ $(ARGS)

test-network: ## Run network tests     (hits public OpenRouter API)
	MEMCOMP_BENCH_NETWORK=1 $(PYTEST) -m network tests/live/ $(ARGS)

test-all: ## Run every test (unit + live + network)
	MEMCOMP_BENCH_LIVE=1 MEMCOMP_BENCH_NETWORK=1 $(PYTEST) $(ARGS)

# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

.PHONY: coverage coverage-open

coverage: ## Run unit tests with coverage report
	$(PYTEST) tests/unit/ $(COV_OPTS) $(ARGS)

coverage-open: coverage ## Open HTML coverage report in browser
	$(PYTHON) -m webbrowser htmlcov/index.html

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------

.PHONY: lint typecheck format check

lint: ## Run ruff linter on src and tests
	$(PYTHON) -m ruff check $(SRC_DIRS)

typecheck: ## Run mypy type checker on src
	$(PYTHON) -m mypy src/

format: ## Auto-format code with ruff
	$(PYTHON) -m ruff format $(SRC_DIRS)
	$(PYTHON) -m ruff check --fix $(SRC_DIRS)

check: lint typecheck test ## Run lint + typecheck + unit tests

# ---------------------------------------------------------------------------
# Pre-commit hooks
# ---------------------------------------------------------------------------

.PHONY: hooks-install hooks-uninstall hooks-run hooks-update

hooks-install: ## Install pre-commit hooks into .git/hooks
	pre-commit install

hooks-uninstall: ## Remove pre-commit hooks from .git/hooks
	pre-commit uninstall

hooks-run: ## Run all pre-commit hooks against all files
	pre-commit run --all-files $(ARGS)

hooks-update: ## Update pre-commit hook revisions to latest tags
	pre-commit autoupdate

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

.PHONY: clean

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/ *.egg-info dist/ build/
