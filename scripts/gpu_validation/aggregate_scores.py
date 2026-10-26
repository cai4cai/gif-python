"""Summarise the CSV written by native/bench/reg_scores (rows labelled `cpu` and `gpu`): mean scores and paired differences.

usage: python aggregate_scores.py scores.csv   (several runs of reg_scores can be concatenated; repeated header lines are skipped)
"""

import sys

import pandas as pd

d = pd.read_csv(sys.argv[1])
d = d[d.label.isin(["cpu", "gpu"])].copy()
for c in ["template", "ncc", "lncc", "nmi", "jac_min", "jac_neg_frac"]:
    d[c] = pd.to_numeric(d[c])
print(d.groupby(["target", "label"])[["ncc", "lncc", "nmi", "jac_neg_frac"]].mean().round(4))
print(d.groupby("label")[["ncc", "lncc", "nmi", "jac_neg_frac"]].mean().round(4))
p = d.pivot_table(index=["target", "template"], columns="label", values=["ncc", "lncc", "nmi"])
for m in ["ncc", "lncc", "nmi"]:
    diff = p[(m, "gpu")] - p[(m, "cpu")]
    print(f"{m}: gpu - cpu mean {diff.mean():+.4f}  sd {diff.std():.4f}  median {diff.median():+.4f}  n={len(diff)}")
