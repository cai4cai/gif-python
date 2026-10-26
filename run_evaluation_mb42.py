"""Moved to scripts/mindboggle42/evaluate_mb42.py (same arguments)."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "scripts" / "mindboggle42" / "evaluate_mb42.py"),
        run_name="__main__",
    )
