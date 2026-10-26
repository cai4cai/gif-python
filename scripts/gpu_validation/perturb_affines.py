"""Add eps * (+-1) to the 12 entries of every <name>/affine.txt of SRC and write the result to DST (noise floor experiments).

usage: python perturb_affines.py SRC DST --eps 1e-7 --seed 1
"""

import argparse
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--eps", type=float, default=1e-7)
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
for f in sorted(Path(a.src).glob("*/affine.txt")):
    m = np.loadtxt(f)
    rng = np.random.default_rng(a.seed + sum(map(ord, str(f.parent.name))))
    m[:3, :] += a.eps * rng.choice([-1.0, 1.0], size=(3, 4))
    out = Path(a.dst) / f.parent.name
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "affine.txt", m, fmt="%.9f")
