"""Compare two output folders of the multi-atlas segmentation (CPU vs GPU, or CPU vs CPU).

usage: python compare_runs.py REF_DIR TEST_DIR [--atlases]
Prints the fraction of differing voxels and the mean Dice of final_parcellation, agreement of the other outputs.
"""

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def load(path, dtype=None):
    img = nib.load(str(path))
    return np.asarray(img.dataobj) if dtype is None else np.asarray(img.dataobj).astype(dtype)


def dice_per_label(a, b, labels):
    out = {}
    for lab in labels:
        ma, mb = a == lab, b == lab
        s = ma.sum() + mb.sum()
        if s:
            out[int(lab)] = 2.0 * np.logical_and(ma, mb).sum() / s
    return out


def compare(ref_dir, test_dir, atlases=False):
    ref_dir, test_dir = Path(ref_dir), Path(test_dir)
    res = {}
    fa = load(ref_dir / "final_parcellation.nii.gz", np.int32)
    fb = load(test_dir / "final_parcellation.nii.gz", np.int32)
    labels = [lab for lab in np.union1d(np.unique(fa), np.unique(fb)) if lab != 0]
    d = dice_per_label(fa, fb, labels)
    brain = (fa > 0) | (fb > 0)
    res["final_differing_fraction_of_volume"] = float((fa != fb).mean())
    res["final_differing_fraction_of_labelled_voxels"] = float((fa != fb).sum() / max(brain.sum(), 1))
    res["final_mean_dice_labels"] = float(np.mean(list(d.values())))
    res["final_min_dice_label"] = float(np.min(list(d.values())))
    res["n_labels"] = len(d)
    aa = load(ref_dir / "multi_atlas_proba_seg_argmax.nii.gz", np.int32)
    ab = load(test_dir / "multi_atlas_proba_seg_argmax.nii.gz", np.int32)
    res["argmax_differing_fraction"] = float((aa != ab).mean())
    pa = load(ref_dir / "multi_atlas_tissue_prior.nii.gz", np.float32)
    pb = load(test_dir / "multi_atlas_tissue_prior.nii.gz", np.float32)
    res["prior_max_abs_diff"] = float(np.abs(pa - pb).max())
    res["prior_mean_abs_diff"] = float(np.abs(pa - pb).mean())
    res["prior_exact_equal_fraction"] = float((pa == pb).mean())
    ta = load(ref_dir / "multi_atlas_tissue_seg.nii.gz", np.float32)
    tb = load(test_dir / "multi_atlas_tissue_seg.nii.gz", np.float32)
    res["tissue_post_max_abs_diff"] = float(np.abs(ta - tb).max())
    res["tissue_post_mean_abs_diff"] = float(np.abs(ta - tb).mean())

    # tissue segmentation as used by the pipeline (argmax of the float16 posterior with the overrides)
    def tseg(post, prior):
        p = post.astype(np.float16).copy()
        p[prior[..., 0] == 1] = 1
        p[[0, -1], :, :] = 1
        p[:, [0, -1], :] = 1
        p[:, :, [0, -1]] = 1
        return np.argmax(p, axis=-1)

    sa, sb = tseg(ta, pa), tseg(tb, pb)
    res["tissue_seg_differing_fraction"] = float((sa != sb).mean())
    res["final_differing_fraction_where_tissue_seg_equal"] = float(((fa != fb) & (sa == sb)).mean())
    res["tissue_seg_differing_fraction_in_brain"] = float((sa != sb).sum() / max((sa > 0).sum(), 1))
    if atlases:
        diffs = []
        for sub in sorted((ref_dir / "registered_atlases").iterdir()):
            wa = load(sub / "warped_atlas_seg.nii.gz")
            wb = load(test_dir / "registered_atlases" / sub.name / "warped_atlas_seg.nii.gz")
            wa = np.nan_to_num(wa).astype(np.float16).astype(np.uint16)
            wb = np.nan_to_num(wb).astype(np.float16).astype(np.uint16)
            diffs.append(float((wa != wb).mean()))
        res["warped_seg_differing_fraction_mean"] = float(np.mean(diffs))
        res["warped_seg_differing_fraction_max"] = float(np.max(diffs))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--atlases", action="store_true")
    a = ap.parse_args()
    print(json.dumps(compare(a.ref, a.test, a.atlases), indent=1))
    sys.exit(0)
