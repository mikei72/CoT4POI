from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


def run_command(command: list[str], cwd: Path | None = None, dry_run: bool = False) -> int:
    rendered = " ".join(shlex.quote(part) for part in command)
    print(f"[run] {rendered}")
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=str(cwd) if cwd else None, check=False)
    return completed.returncode
