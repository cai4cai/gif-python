"""Run `gif-python segment` (CPU) with a tiny perturbation of the affine found by reg_aladin.

After every reg_aladin call the affine text file is rewritten with `eps * (1 + k)` added to its 12 entries
(different sign pattern per --seed), then reg_f3d starts from that perturbed affine.  This estimates how much the
CPU result changes under a ~1e-7 change of the starting point (the noise floor of the registration).

usage: python cpu_perturbed.py --eps 1e-7 --seed 1 -- segment <gif-python segment arguments>
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
import src.multi_atlas.atlas_propagation as ap
from src.cli.main import main

parser = argparse.ArgumentParser()
parser.add_argument("--eps", type=float, default=1e-7)
parser.add_argument("--seed", type=int, default=1)
opts, rest = parser.parse_known_args()
rest = [r for r in rest if r != "--"]

_orig = ap.run_tool


def run_tool(args):
    result = _orig(args)
    args = [str(a) for a in args]
    if Path(args[0]).name == "reg_aladin":
        aff = Path(args[args.index("-aff") + 1])
        m = np.loadtxt(aff)
        rng = np.random.default_rng(opts.seed + sum(map(ord, str(aff))))
        m[:3, :] += opts.eps * rng.choice([-1.0, 1.0], size=(3, 4))
        np.savetxt(aff, m, fmt="%.9f")
    return result


ap.run_tool = run_tool
sys.exit(main(rest))
