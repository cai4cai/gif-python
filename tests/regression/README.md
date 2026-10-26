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

See the pull request description for the data set, the machine, the reference hashes and the
comparison.
