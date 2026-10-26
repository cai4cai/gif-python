"""Running external command line tools (NiftyReg, NiftySeg)."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Sequence, Union

logger = logging.getLogger(__name__)

#: number of trailing output lines included in an error message
_TAIL_LINES = 20


class ExternalToolError(RuntimeError):
    """An external tool exited with a non-zero status."""


def run_tool(args: Sequence[Union[str, Path]]) -> subprocess.CompletedProcess:
    """Run ``args`` (no shell) and raise :class:`ExternalToolError` if it fails.

    The output of the tool is captured and logged at DEBUG level; use ``--verbose`` of the command
    line interface, or ``logging.basicConfig(level=logging.DEBUG)``, to see it. Messages such as
    NiftyReg's ``[NiftyReg WARNING] The folding correction scheme failed`` are informational and do
    not make the tool fail.
    """
    cmd = [str(a) for a in args]
    logger.debug("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError as exc:
        raise ExternalToolError(f"Cannot execute {cmd[0]!r}: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        tail = "\n".join((exc.stdout or "").strip().splitlines()[-_TAIL_LINES:])
        raise ExternalToolError(
            f"{Path(cmd[0]).name} failed with exit status {exc.returncode}.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Last output:\n{tail}"
        ) from exc
    tool = Path(cmd[0]).name
    if result.stdout and result.stdout.strip():
        logger.debug("%s output:\n%s", tool, result.stdout.rstrip())
        n_warnings = sum(1 for line in result.stdout.splitlines() if "WARNING" in line)
        if n_warnings:
            logger.info(
                "%s finished successfully and printed %d warning line(s) (for example 'The folding correction "
                "scheme failed'); these are informational, see docs/tuning-registration.md. "
                "Use -vv (or DEBUG logging) to see them.",
                tool,
                n_warnings,
            )
    return result
