"""Moved to scripts/mindboggle42/run_mb42_with_mb59_atlases.py (same arguments)."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "scripts" / "mindboggle42" / "run_mb42_with_mb59_atlases.py"),
        run_name="__main__",
    )
