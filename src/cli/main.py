"""Entry point of the ``gif-python`` command."""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from typing import List, Optional, Sequence

#: sub-command name -> module that provides ``HELP``, ``add_arguments(parser)`` and ``run(args)``
COMMANDS = {
    "segment": "src.cli.segment",
}


def _common_options() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="more output (-v: progress details; -vv: also the output of the NiftyReg/NiftySeg tools)",
    )
    return common


def build_parser() -> argparse.ArgumentParser:
    from src import __version__

    parser = argparse.ArgumentParser(
        prog="gif-python",
        description="Multi-atlas segmentation with the GIF algorithm (NiftyReg/NiftySeg based).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True
    for name, module_name in COMMANDS.items():
        module = importlib.import_module(module_name)
        sub = subparsers.add_parser(
            name,
            help=module.HELP,
            description=module.HELP,
            parents=[_common_options()],
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        module.add_arguments(sub)
        sub.set_defaults(handler=module.run)
    return parser


def configure_logging(verbosity: int) -> None:
    level = logging.INFO if verbosity == 0 else logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(message)s" if verbosity else "%(message)s")
    if verbosity < 2:
        # the output of the external tools is logged at DEBUG level by src.utils.process
        logging.getLogger("src.utils.process").setLevel(logging.INFO)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command line interface; returns the process exit status."""
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    try:
        return int(args.handler(args) or 0)
    except KeyboardInterrupt:
        return 130
    except (FileNotFoundError, ValueError, RuntimeError, NotImplementedError) as exc:
        if args.verbose:
            raise
        print(f"gif-python {args.command}: error: {exc}", file=sys.stderr)
        return 1


def entry_point() -> None:  # pragma: no cover - thin wrapper for the console script
    sys.exit(main())


__all__: List[str] = ["main", "build_parser"]
