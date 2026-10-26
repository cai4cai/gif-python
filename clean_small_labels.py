"""Deprecated entry point, kept so that old commands keep working. Use ``gif-python clean-labels``.

The input files are now arguments (the script used to contain a hardcoded folder)::

    python clean_small_labels.py "data/atlases/Mindboggle101/*/labels.DKT31.manual+aseg.nii.gz"
"""

import sys

from src.cli.main import main

if __name__ == "__main__":
    sys.exit(main(["clean-labels", *sys.argv[1:]]))
