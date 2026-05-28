from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class ConcurrentRunOperationError(RuntimeError):
    """Raised when another process is already operating on the same saved run."""


class StaleRunRevisionError(RuntimeError):
    """Raised when a resumed run is being saved over a newer published revision."""


class RunOperationLock:
    def __init__(self, jsonl_path: Path) -> None:
        self.jsonl_path = jsonl_path
        self.lock_path = jsonl_path.with_suffix(".lock")
        self._fd: int | None = None

    def acquire(self) -> None:
        try:
            self._fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise ConcurrentRunOperationError(f"Saved run is busy: {self.jsonl_path}") from exc
        os.write(self._fd, str(os.getpid()).encode("ascii"))

    def release(self) -> None:
        if self._fd is None:
            return
        os.close(self._fd)
        self._fd = None
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass

    def matches(self, jsonl_path: Path) -> bool:
        return self.jsonl_path == jsonl_path

    def __enter__(self) -> RunOperationLock:
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()


@contextmanager
def lock_saved_run(jsonl_path: Path, run_lock: RunOperationLock | None = None) -> Iterator[RunOperationLock]:
    if run_lock is not None:
        if not run_lock.matches(jsonl_path):
            raise ValueError(f"Lock target mismatch for {jsonl_path}")
        yield run_lock
        return
    with RunOperationLock(jsonl_path) as acquired_lock:
        yield acquired_lock


def base_name_for_record(record_id: str, profile_name: str) -> str:
    return f"conv_{record_id}_{profile_name.lower()}"


def jsonl_path_for_record(record_id: str, profile_name: str, output_dir: Path) -> Path:
    return output_dir / f"{base_name_for_record(record_id, profile_name)}.jsonl"


def raw_context_path(jsonl_path: Path) -> Path:
    return jsonl_path.parent / f"{jsonl_path.stem}_raw_ai_context.json"


def ready_marker_path(jsonl_path: Path) -> Path:
    return jsonl_path.with_suffix(".ready")


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def read_run_revision(jsonl_path: Path, *, run_lock: RunOperationLock | None = None) -> str | None:
    if run_lock is not None and not run_lock.matches(jsonl_path):
        raise ValueError(f"Lock target mismatch for {jsonl_path}")
    ready_path = ready_marker_path(jsonl_path)
    if not ready_path.exists():
        return None
    return ready_path.read_text(encoding="utf-8").strip() or None


def is_run_published(jsonl_path: Path) -> bool:
    ready_path = ready_marker_path(jsonl_path)
    if ready_path.exists():
        return True
    return raw_context_path(jsonl_path).exists() and not jsonl_path.with_suffix(".lock").exists()
