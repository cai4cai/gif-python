"""Leave-one-out evaluation against expert labels, CPU vs GPU backend (paired).

For every held-out subject the final parcellations of the CPU and the GPU run (folders RUNS_CPU/<subject> and
RUNS_GPU/<subject>) are compared with the expert labels of the subject (``--gt-pattern``, ``{name}`` is replaced by the
subject).  Dice is computed per label (labels present in the expert labels, background excluded) and averaged per subject.
The paired CPU - GPU difference of the per-subject mean Dice is reported with a 95% confidence interval (t distribution),
and, as a check that does not rely on normality over subjects, a bootstrap over subjects.

usage: python loo_eval.py RUNS_CPU RUNS_GPU --subjects OASIS-TRT-20-1 ... --gt-pattern '/data/{name}/labels.nii.gz'
"""

import argparse
import json

import nibabel as nib
import numpy as np
from scipy import stats


def dice_labels(pred, gt):
    out = {}
    for lab in np.unique(gt):
        if lab == 0:
            continue
        a, b = pred == lab, gt == lab
        out[int(lab)] = 2.0 * np.logical_and(a, b).sum() / (a.sum() + b.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cpu_runs")
    ap.add_argument("gpu_runs")
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--gt-pattern", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = []
    for name in a.subjects:
        gt = np.rint(np.asarray(nib.load(a.gt_pattern.format(name=name)).dataobj)).astype(np.int32)
        d = {}
        for tag, root in (("cpu", a.cpu_runs), ("gpu", a.gpu_runs)):
            pred = np.asarray(nib.load(f"{root}/{name}/final_parcellation.nii.gz").dataobj).astype(np.int32)
            d[tag] = dice_labels(pred, gt)
        labs = sorted(set(d["cpu"]) & set(d["gpu"]))
        cpu = np.array([d["cpu"][label] for label in labs])
        gpu = np.array([d["gpu"][label] for label in labs])
        rows.append((name, cpu.mean(), gpu.mean(), cpu, gpu))
        print(
            f"{name}: mean Dice CPU {cpu.mean():.4f}  GPU {gpu.mean():.4f}  diff {gpu.mean() - cpu.mean():+.4f}  ({len(labs)} labels)"
        )
    c = np.array([r[1] for r in rows])
    g = np.array([r[2] for r in rows])
    diff = g - c
    n = len(diff)
    sem = diff.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    ci = (diff.mean() - tcrit * sem, diff.mean() + tcrit * sem)
    rng = np.random.default_rng(0)
    boots = [diff[rng.integers(0, n, n)].mean() for _ in range(20000)]
    bci = tuple(np.percentile(boots, [2.5, 97.5]))
    t, p = stats.ttest_rel(g, c)
    summary = {
        "n_subjects": n,
        "cpu_mean_dice": float(c.mean()),
        "gpu_mean_dice": float(g.mean()),
        "mean_diff_gpu_minus_cpu": float(diff.mean()),
        "ci95_t": [float(ci[0]), float(ci[1])],
        "ci95_bootstrap": [float(bci[0]), float(bci[1])],
        "paired_t_p": float(p),
        "per_subject": {r[0]: [float(r[1]), float(r[2])] for r in rows},
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "per_subject"}, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
