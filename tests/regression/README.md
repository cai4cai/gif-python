# Regression check for refactoring

`run_regression.py` runs one small, fixed multi-atlas segmentation (1 target, 3 atlases) and
records SHA-256 hashes of

* the final parcellation, the tissue segmentation, the tissue prior and the argmax parcellation,
* every warped atlas segmentation,
* every fusion weight map (copied before the pipeline deletes it).

NIfTI files are hashed on their uncompressed bytes, so gzip metadata does not matter.
It needs the NiftyReg and NiftySeg binaries and real data, so it is not part of the pytest suite;
it is a manual check to run before and after a refactoring.

## Recipe

```bash
# 1. inputs: structures_info.csv / tissues_info.csv from a GIF labels.xml
python tests/regression/prepare_inputs.py --labels-xml DB/labels.xml --label-map label_map.tsv --out-dir inputs/

# 2. run (reference code and refactored code are separate checkouts; --code-root is the
#    directory that contains the `src` package)
python tests/regression/run_regression.py run --code-root /path/to/checkout \
    --img target.nii.gz --atlas-img-dir DB/T1s --atlas-seg-dir DB/labels \
    --atlases OASIS-TRT-20-1 OASIS-TRT-20-2 OASIS-TRT-20-3 \
    --structure-info inputs/structures_info.csv --tissue-info inputs/tissues_info.csv \
    --niftyreg-path /path/to/bin --niftyseg-path /path/to/bin \
    --omp 2 --pools 3 --out OUT_NEW        # writes OUT_NEW/sha256.txt

# 3. compare with the recorded reference
python tests/regression/run_regression.py check --manifest OUT_REF/sha256.txt --out OUT_NEW
```

The script works with both configuration styles (module globals in `src.utils.definitions`,
and `GifConfig`), so the same script produces the reference with the code before the refactoring
and the check with the code after it.

The multi-threaded NiftyReg registrations must be deterministic for this to be meaningful: run the
reference twice and check that the two manifests are identical before trusting it.

## Recorded result

Reference recorded before the refactoring and the check after it (2026-10-26, host `sie271-pc`, aarch64 Linux,
NiftyReg 1.5.77 built without SSE, NiftySeg `seg_EM`, Python 3.12, NumPy 2.2.6, nibabel 5.4.2; one IXI T1 target
`IXI002-Guys-0828-T1.nii.gz`, three atlases `OASIS-TRT-20-1..3` of the benchmark database with 112 labels, `--omp 2
--pools 3`):

* Reference code: upstream `main` plus the one-line NumPy 2 fix of `get_structure_seg_from_tissue_seg`
  (the original line raises an `OverflowError` with NumPy 2, so the unmodified code cannot run on this machine).
* The reference was run twice and both runs gave identical hashes.
* The refactored code (`GifConfig`, no shell, logging, pathlib; final commit of the branch) gives identical hashes
  for all ten files, both with the multiprocessing pools and with `--pools 1` (no multiprocessing).

| SHA-256 (uncompressed bytes) | file |
|---|---|
| `c0bb83729eb5db5b6eaab1e8009ec3c1cc8afef1198bcbb6719f975b72e9985d` | `final_parcellation.nii.gz` |
| `fd568fae9c71d4ec63ad3b2d4b572c74ccc93ea89e9f06a903b89bee9144ca02` | `multi_atlas_tissue_seg.nii.gz` |
| `db392f2842c3a62a7f1d8229f881f5a04ccf391f4671d358b23f8cec3f2c4e76` | `multi_atlas_tissue_prior.nii.gz` |
| `a423c29bcbd2c97606092d48cd434981431a036f21317ad991ea9f7180c44e75` | `multi_atlas_proba_seg_argmax.nii.gz` |
| `6498a08e1602e1556d844d7651e43f822aff4ade20b6c89ab0e8947155ed79f4` | `registered_atlases/OASIS-TRT-20-1/warped_atlas_seg.nii.gz` |
| `82985241068c518fc673df99efc5ee5466142054cd36d86d59d4af184d6d63f0` | `registered_atlases/OASIS-TRT-20-2/warped_atlas_seg.nii.gz` |
| `b9499a41cffbd7c8bd8b1a1bcdc6b52b1c1a00a5da18f5ea55727bf05afcef9d` | `registered_atlases/OASIS-TRT-20-3/warped_atlas_seg.nii.gz` |
| `5bb791559ffab34352ebdf917b09b5caaca26f6ad8b247deb9dfe8150c988fec` | `captured_weights/OASIS-TRT-20-1/weights.nii.gz` |
| `52ad9589c0d1249fbc95e59b76db74ec601cdbf9be3666f74bb54c72fd1d18e4` | `captured_weights/OASIS-TRT-20-2/weights.nii.gz` |
| `5e627d308774fa5686946e8aa9061e9189810df4b461ef05901167dca6eab29b` | `captured_weights/OASIS-TRT-20-3/weights.nii.gz` |

On that host the reference outputs, the manifest (`baseline/REFERENCE.sha256`) and a description are in
`~/fastgif/gif-regress/baseline/`. The hashes are valid for that NiftyReg build and architecture; another
machine (for example x86 with SSE) gives its own reference, because the registration is sensitive to
floating-point details. Record a new reference there with the code before the change, then check the code after it.
