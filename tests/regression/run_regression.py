"""Fixed small multi-atlas run used to check that refactoring does not change the output.

The script runs ``multi_atlas_segmentation`` for one target and a few atlases and
writes a manifest of SHA-256 hashes of the final parcellation, the tissue
segmentation (and prior), the argmax parcellation, every warped atlas
segmentation and every fusion weight map. NIfTI files are hashed on their
*uncompressed* bytes so that gzip metadata cannot influence the result.

It works with the code before and after the configuration refactoring:

* old code (module-level ``src.utils.definitions``): the settings are patched into
  that module before the pipeline is imported;
* new code (``src.utils.config.GifConfig``): a config object is passed in.

Run (see README.md for the full recipe)::

    python run_regression.py run   --code-root PATH_WITH_src --out OUT [options]
    python run_regression.py check --manifest OUT/sha256.txt --out OTHER_OUT
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import os
import shutil
import sys
from pathlib import Path

MANIFEST_NAME = "sha256.txt"
FINAL_FILES = [
    "final_parcellation.nii.gz",
    "multi_atlas_tissue_seg.nii.gz",
    "multi_atlas_tissue_prior.nii.gz",
    "multi_atlas_proba_seg_argmax.nii.gz",
]


def hash_nifti(path: Path) -> str:
    """SHA-256 of the uncompressed bytes of a (possibly gzipped) file."""
    opener = gzip.open if path.suffix == ".gz" else open
    h = hashlib.sha256()
    with opener(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(out: Path) -> dict[str, str]:
    """Hash all result files below ``out`` that make up the regression record."""
    files = [out / n for n in FINAL_FILES]
    files += sorted((out / "registered_atlases").glob("*/warped_atlas_seg.nii.gz"))
    files += sorted((out / "captured_weights").glob("*/weights.nii.gz"))
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise SystemExit(f"missing result files: {missing}")
    return {str(f.relative_to(out)): hash_nifti(f) for f in files}


def write_manifest(out: Path) -> Path:
    hashes = collect(out)
    manifest = out / MANIFEST_NAME
    manifest.write_text("".join(f"{h}  {name}\n" for name, h in hashes.items()))
    return manifest


def read_manifest(path: Path) -> dict[str, str]:
    return dict(line.strip().split("  ", 1)[::-1] for line in path.read_text().splitlines() if line.strip())


def cmd_check(args: argparse.Namespace) -> int:
    expected = read_manifest(args.manifest)
    actual = collect(args.out)
    ok = True
    for name in sorted(set(expected) | set(actual)):
        e, a = expected.get(name), actual.get(name)
        status = "OK  " if e == a else "DIFF"
        ok &= e == a
        print(f"{status} {name}  expected={e}  actual={a}")
    print("IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


def capture_weights(out: Path) -> None:
    """Copy each ``weights.nii.gz`` before the pipeline deletes it (os.remove / Path.unlink)."""
    real_remove, real_unlink = os.remove, Path.unlink

    def keep(path) -> None:
        p = Path(path)
        if p.name == "weights.nii.gz" and p.is_file():
            dest = out / "captured_weights" / p.parent.name
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest / p.name)

    def remove(path, *a, **k):
        keep(path)
        return real_remove(path, *a, **k)

    def unlink(self, *a, **k):
        keep(self)
        return real_unlink(self, *a, **k)

    os.remove = remove
    Path.unlink = unlink


def cmd_run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.code_root.resolve()))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(out / ".numba_cache"))

    config_kwargs = dict(
        niftyreg_path=str(args.niftyreg_path),
        niftyseg_path=str(args.niftyseg_path),
        omp=args.omp,
        num_pools=args.pools,
        multiprocessing=args.pools > 1,
    )
    new_api = importlib.util.find_spec("src.utils.config") is not None
    run_kwargs = {}
    if new_api:
        from src.utils.config import GifConfig

        run_kwargs["config"] = GifConfig(**config_kwargs)
    else:
        import src.utils.definitions as d

        d.NIFTYREG_PATH, d.NIFTYSEG_PATH = config_kwargs["niftyreg_path"], config_kwargs["niftyseg_path"]
        d.OMP, d.NUM_POOLS, d.MULTIPROCESSING = args.omp, args.pools, args.pools > 1
    capture_weights(out)
    from src.multi_atlas.multi_atlas_segmentation import multi_atlas_segmentation

    atlases = [
        {
            "name": name,
            "img_path": str(args.atlas_img_dir / f"{name}.nii"),
            "seg_path": str(args.atlas_seg_dir / f"{name}.nii"),
        }
        for name in args.atlases
    ]
    multi_atlas_segmentation(
        img_path=str(args.img),
        mask_path=None,
        atlas_paths_dicts_list=atlases,
        structure_info_csv_path=str(args.structure_info),
        tissue_info_csv_path=str(args.tissue_info),
        save_dir=str(out),
        **run_kwargs,
    )
    manifest = write_manifest(out)
    print(f"API: {'GifConfig' if new_api else 'definitions globals'}")
    print(manifest.read_text())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the pipeline and write sha256.txt")
    r.add_argument("--code-root", type=Path, required=True, help="directory that contains the 'src' package")
    r.add_argument("--img", type=Path, required=True)
    r.add_argument("--atlas-img-dir", type=Path, required=True)
    r.add_argument("--atlas-seg-dir", type=Path, required=True)
    r.add_argument("--atlases", nargs="+", required=True, help="atlas names (file stems)")
    r.add_argument("--structure-info", type=Path, required=True)
    r.add_argument("--tissue-info", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--niftyreg-path", type=Path, required=True)
    r.add_argument("--niftyseg-path", type=Path, required=True)
    r.add_argument("--omp", type=int, default=1, help="threads per NiftyReg process")
    r.add_argument("--pools", type=int, default=3, help="atlases registered in parallel")
    r.set_defaults(func=cmd_run)
    c = sub.add_parser("check", help="compare a result directory with a manifest")
    c.add_argument("--manifest", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.set_defaults(func=cmd_check)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
