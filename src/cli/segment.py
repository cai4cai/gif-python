"""``gif-python segment``: multi-atlas segmentation of one image."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from src.utils.config import GifConfig, default_config, field_names

logger = logging.getLogger(__name__)

HELP = "Segment an image with a set of atlases (registration, label fusion, tissue EM)."

#: file names of the Mindboggle-101 atlases as laid out by ``gif-python fetch-mindboggle``
DEFAULT_ATLAS_IMG = "t1weighted_brain.nii.gz"
DEFAULT_ATLAS_SEG = "labels.DKT31.manual+aseg_cleaned.nii.gz"

EPILOG = """\
Atlas folder layout: ATLAS_DIR/<atlas name>/<atlas image> and <atlas segmentation>, plus
structures_info.csv and tissues_info.csv in ATLAS_DIR (or given with --structure-info/--tissue-info).
The defaults of --atlas-img/--atlas-seg are those of the Mindboggle-101 layout.

Parameters of the pipeline can be changed with --config NAME=VALUE (repeatable); --print-config
shows the names and the values in effect. The NiftyReg/NiftySeg executables are looked up in
--niftyreg-path/--niftyseg-path, the environment variables NIFTYREG_PATH/NIFTYSEG_PATH, and PATH.

Example:
  gif-python segment --img subject/t1weighted_brain.nii.gz --atlas-dir data/atlases/Mindboggle101 \\
      --exclude-atlas subject --out outputs/subject --config reg_f3d_lncc=7
"""


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.epilog = EPILOG
    io = parser.add_argument_group("input and output")
    io.add_argument("--img", type=Path, help="image to segment")
    io.add_argument("--mask", type=Path, default=None, help="mask of the image to segment (optional)")
    io.add_argument("--atlas-dir", type=Path, help="folder with one sub-folder per atlas")
    io.add_argument("--atlas-img", default=DEFAULT_ATLAS_IMG, help="atlas image file name (default: %(default)s)")
    io.add_argument("--atlas-seg", default=DEFAULT_ATLAS_SEG, help="atlas segmentation file name (default: %(default)s)")
    io.add_argument("--atlas-mask", default=None, help="atlas mask file name (optional)")
    io.add_argument(
        "--exclude-atlas", action="append", default=[], metavar="NAME", help="leave this atlas out (repeatable)"
    )
    io.add_argument("--structure-info", type=Path, default=None, help="default: ATLAS_DIR/structures_info.csv")
    io.add_argument("--tissue-info", type=Path, default=None, help="default: ATLAS_DIR/tissues_info.csv")
    io.add_argument("--out", type=Path, help="output folder")
    io.add_argument("--skip-existing", action="store_true", help="do nothing if OUT/final_parcellation.nii.gz exists")
    io.add_argument("--view", action="store_true", help="open the result in ITK-SNAP when done (if installed)")

    cfg = parser.add_argument_group("pipeline parameters")
    cfg.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="set a parameter of the pipeline, e.g. reg_f3d_lncc=7 (repeatable; see --print-config)",
    )
    cfg.add_argument("--print-config", action="store_true", help="print the parameters in effect and exit")
    cfg.add_argument("--no-multiprocessing", action="store_true", help="process the atlases one after the other")
    cfg.add_argument("--pools", type=int, default=None, help="number of atlases processed in parallel")
    cfg.add_argument("--omp", type=int, default=None, help="OpenMP threads of each NiftyReg process")
    cfg.add_argument(
        "--no-reuse",
        action="store_true",
        help="recompute registrations and weights even if they are found in the output folder",
    )
    cfg.add_argument(
        "--keep-registration-files",
        action="store_true",
        help="keep affine, control point grid and warped atlas images in OUT/registered_atlases/<atlas>/",
    )
    cfg.add_argument("--niftyreg-path", default=None, help="folder with reg_aladin, reg_f3d and reg_resample")
    cfg.add_argument("--niftyseg-path", default=None, help="folder with seg_EM")


def build_config(args: argparse.Namespace) -> GifConfig:
    """Combine the default configuration with ``--config`` and the dedicated options."""
    overrides = {}
    for item in args.config:
        name, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"--config expects NAME=VALUE, got {item!r}. Valid names: {', '.join(field_names())}")
        overrides[name.strip()] = value
    config = default_config().with_overrides(overrides)
    changes = {}
    if args.no_multiprocessing:
        changes["multiprocessing"] = False
    if args.pools is not None:
        changes["num_pools"] = args.pools
    if args.omp is not None:
        changes["omp"] = args.omp
    if args.no_reuse:
        changes["use_old_results"] = False
    if args.keep_registration_files:
        changes["keep_registration_files"] = True
    if args.niftyreg_path:
        changes["niftyreg_path"] = args.niftyreg_path
    if args.niftyseg_path:
        changes["niftyseg_path"] = args.niftyseg_path
    return config.replace(**changes)


def _require(args: argparse.Namespace, *names: str) -> None:
    missing = [f"--{n.replace('_', '-')}" for n in names if getattr(args, n) is None]
    if missing:
        raise ValueError("missing required option(s): " + ", ".join(missing))


def run(args: argparse.Namespace) -> int:
    from src.multi_atlas.multi_atlas_segmentation import multi_atlas_segmentation
    from src.multi_atlas.utils import find_atlases

    config = build_config(args)
    if args.print_config:
        print(json.dumps(config.to_dict(), indent=2, default=str))
        print(f"# resolved NiftyReg directory: {config.niftyreg_dir()}")
        print(f"# resolved NiftySeg directory: {config.niftyseg_dir()}")
        return 0

    _require(args, "img", "atlas_dir", "out")
    final_parcellation = args.out / "final_parcellation.nii.gz"
    if args.skip_existing and final_parcellation.is_file():
        logger.info("%s exists - skipping", final_parcellation)
        return 0

    atlases = find_atlases(args.atlas_dir, args.atlas_img, args.atlas_seg, args.atlas_mask, args.exclude_atlas)
    structure_info = args.structure_info or args.atlas_dir / "structures_info.csv"
    tissue_info = args.tissue_info or args.atlas_dir / "tissues_info.csv"
    for path, option in ((structure_info, "--structure-info"), (tissue_info, "--tissue-info")):
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"{path} does not exist. Pass {option}, or put the file into the atlas folder "
                "(for Mindboggle-101, 'gif-python fetch-mindboggle' copies it there)."
            )

    logger.info("Segmenting %s with %d atlases", args.img, len(atlases))
    time_0 = time.time()
    multi_atlas_segmentation(
        img_path=args.img,
        mask_path=args.mask,
        atlas_paths_dicts_list=atlases,
        structure_info_csv_path=structure_info,
        tissue_info_csv_path=tissue_info,
        save_dir=args.out,
        config=config,
    )
    logger.info("Done in %.1f s. Final parcellation: %s", time.time() - time_0, final_parcellation)

    if args.view:
        from src.cli.viewer import open_in_itksnap

        open_in_itksnap(args.img, final_parcellation)
    return 0

