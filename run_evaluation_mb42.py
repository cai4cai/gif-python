"""
Evaluate Mindboggle42 GIF predictions with Dice.

Example:
    python run_evaluation_mb42.py \
        --pred-dir outputs/mb59_to_mb42_mindboggle101_gif \
        --gt-dir /nfs/home/jwang/datasets/Mindboggle101/subjects
"""
from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from pathlib import Path

import nibabel as nib
import numpy as np


PREDICTION_NAME = "final_parcellation.nii.gz"
GT_NAME = "labels.DKT31.manual+aseg_cleaned.nii.gz"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate average Dice for MB42 GIF predictions."
    )
    parser.add_argument(
        "--pred-dir",
        required=True,
        type=Path,
        help="Directory containing per-subject prediction folders.",
    )
    parser.add_argument(
        "--gt-dir",
        required=True,
        type=Path,
        help="Directory containing per-subject ground-truth folders.",
    )
    parser.add_argument(
        "--prediction-name",
        default=PREDICTION_NAME,
        help=f"Prediction filename inside each subject folder. Default: {PREDICTION_NAME}",
    )
    parser.add_argument(
        "--gt-name",
        default=GT_NAME,
        help=f"Ground-truth filename inside each subject folder. Default: {GT_NAME}",
    )
    parser.add_argument(
        "--exclude-background",
        action="store_true",
        help="Exclude label 0 from the Dice calculation.",
    )
    return parser.parse_args(argv)


def load_label_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    return np.asarray(img.dataobj), img.affine


def labels_for_pair(pred: np.ndarray, gt: np.ndarray, exclude_background: bool) -> list[int]:
    labels = set(np.unique(pred).astype(int))
    labels.update(np.unique(gt).astype(int))
    if exclude_background:
        labels.discard(0)
    return sorted(labels)


def subject_dice(
    pred_path: Path,
    gt_path: Path,
    exclude_background: bool,
) -> list[tuple[int, float]]:
    pred, pred_affine = load_label_image(pred_path)
    gt, gt_affine = load_label_image(gt_path)

    if pred.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch for {pred_path.parent.name}: "
            f"prediction {pred.shape}, ground truth {gt.shape}"
        )
    if not np.allclose(pred_affine, gt_affine):
        raise ValueError(f"Affine mismatch for {pred_path.parent.name}")

    labels = labels_for_pair(pred, gt, exclude_background)
    max_label = max(labels)
    pred_flat = pred.astype(np.int64, copy=False).ravel()
    gt_flat = gt.astype(np.int64, copy=False).ravel()
    pred_counts = np.bincount(pred_flat, minlength=max_label + 1)
    gt_counts = np.bincount(gt_flat, minlength=max_label + 1)
    matching_counts = np.bincount(
        pred_flat[pred_flat == gt_flat],
        minlength=max_label + 1,
    )

    scores = []
    for label in labels:
        denom = pred_counts[label] + gt_counts[label]
        if denom == 0:
            scores.append((label, 1.0))
        else:
            scores.append((label, 2.0 * matching_counts[label] / denom))
    return scores


def collect_prediction_paths(pred_dir: Path, prediction_name: str) -> list[Path]:
    return sorted(path for path in pred_dir.glob(f"*/{prediction_name}") if path.is_file())


def write_score_csv(rows: list[tuple[str, int, float]], output_path: Path) -> None:
    with output_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["subject", "label", "dice"])
        writer.writerows(rows)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    pred_paths = collect_prediction_paths(args.pred_dir, args.prediction_name)
    if not pred_paths:
        raise FileNotFoundError(
            f"No prediction files named {args.prediction_name} found under {args.pred_dir}"
        )

    score_rows = []
    evaluated_subjects = 0
    for pred_path in pred_paths:
        subject = pred_path.parent.name
        gt_path = args.gt_dir / subject / args.gt_name
        if not gt_path.is_file():
            print(f"{subject}: missing ground truth - skipping", flush=True)
            continue

        subject_scores = subject_dice(pred_path, gt_path, args.exclude_background)
        scores = [score for _, score in subject_scores]
        score_rows.extend((subject, label, score) for label, score in subject_scores)
        evaluated_subjects += 1
        print(
            f"{subject}: mean Dice = {np.mean(scores):.4f} over {len(scores)} labels",
            flush=True,
        )

    if not score_rows:
        raise RuntimeError("No Dice scores were calculated.")

    output_dir = args.pred_dir
    score_csv_path = output_dir / "score.csv"
    write_score_csv(score_rows, score_csv_path)

    dice_scores = [score for _, _, score in score_rows]
    dice_array = np.asarray(dice_scores, dtype=np.float64)
    print(
        f"Average Dice over {evaluated_subjects} subjects and {len(dice_scores)} labels: "
        f"{dice_array.mean():.4f} +- {dice_array.std():.4f}",
        flush=True,
    )
    print(f"Saved class-wise Dice scores to {score_csv_path}", flush=True)


if __name__ == "__main__":
    main()
