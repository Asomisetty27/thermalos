"""
Atomic file I/O helpers — used by baseline.py, calibrate.py, and config writes.

Audit finding P0: agent state files (baselines.json, calibration.json) were
written with `Path.write_text()`, which can leave a torn file on power loss
or process kill. Next startup reads the half-written file and either crashes
or silently discards all per-GPU history.

This module provides `atomic_write_text()` which uses the standard
temp-file-plus-rename pattern: write to a sibling tempfile, fsync it, rename
over the target. On POSIX, the rename is atomic — either the old or new
file is visible to readers, never a partial one. On crash recovery, leftover
.tmp files in the directory are pruned by `cleanup_stale_tmpfiles()`.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write content to path atomically. Parent dir is created if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fp:
            fp.write(content)
            fp.flush()
            try:
                os.fsync(fp.fileno())
            except OSError:
                pass  # fsync unsupported on some FS — best-effort
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def cleanup_stale_tmpfiles(directory: Path, *, older_than_sec: float = 3600.0) -> int:
    """Remove orphaned .tmp files in `directory` older than the threshold.

    Returns count of files removed. Safe to call on startup — won't touch
    fresh tmpfiles that another write might be in the middle of creating.
    """
    if not directory.exists():
        return 0
    now = time.time()
    removed = 0
    for entry in directory.iterdir():
        if not entry.name.endswith(".tmp"):
            continue
        try:
            age = now - entry.stat().st_mtime
        except OSError:
            continue
        if age > older_than_sec:
            try:
                entry.unlink()
                removed += 1
            except OSError:
                pass
    return removed
