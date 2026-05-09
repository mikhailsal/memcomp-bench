#!/usr/bin/env python3
"""Check that no function/method exceeds the maximum allowed line count.

Usage: python scripts/check_function_length.py [--max N] [file ...]

When invoked without file arguments, checks all tracked *.py files via git.
Exit code 0 = pass, 1 = at least one function exceeds the limit.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys

DEFAULT_MAX = 65


def _get_tracked_py_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "*.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [f for f in result.stdout.strip().splitlines() if f]


def check_file(path: str, max_lines: int) -> list[str]:
    """Return a list of violation messages for *path*."""
    with open(path, encoding="utf-8") as f:
        source = f.read()
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError:
        return []
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            length = node.end_lineno - node.lineno + 1
            if length > max_lines:
                violations.append(f"FAIL  {path}:{node.lineno}  {node.name}() is {length} lines (max {max_lines})")
    return violations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max", type=int, default=DEFAULT_MAX, help=f"Max lines per function (default: {DEFAULT_MAX})")
    parser.add_argument("files", nargs="*", help="Files to check (default: all tracked *.py)")
    args = parser.parse_args()

    files = args.files or _get_tracked_py_files()
    all_violations: list[str] = []
    for path in files:
        all_violations.extend(check_file(path, args.max))

    if all_violations:
        for v in all_violations:
            print(v)
        sys.exit(1)


if __name__ == "__main__":
    main()
