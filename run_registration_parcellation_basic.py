"""Deprecated entry point, kept so that old commands keep working. Use ``gif-python segment``."""

import sys

from src.cli.main import main

if __name__ == "__main__":
    sys.exit(main(["segment", *sys.argv[1:]]))
