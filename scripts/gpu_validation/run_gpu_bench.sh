#!/bin/bash
# GPU timing and outputs for the IXI targets: best of N runs of the command line (end to end, Python included) and of the
# gif_gpu program itself (stage times in JSON).  Run on an idle GPU.
#   WORK=~/gifpy-work REPS=3 run_gpu_bench.sh [numerics]
set -u
W=${WORK:-$HOME/gifpy-work}
REPS=${REPS:-3}
NUM=${1:-0}
cd "$W/gif-python"
PY=${PY:-$HOME/.venvs/ml/bin/python}
ATL=$(cat "$W/data/atlas_args.txt")
for t in IXI002-Guys-0828-T1 IXI012-HH-1211-T1 IXI035-IOP-0873-T1; do
  img=$HOME/fastgif/data/raw/ixi/$t.nii.gz
  for k in $(seq 1 "$REPS"); do
    out=$W/runs/gpu_bench/num${NUM}_$t/cli_$k
    rm -rf "$out"; mkdir -p "$out"
    s=$(date +%s.%N)
    PYTHONPATH=. $PY -m src.cli segment --backend gpu --config gpu_numerics=$NUM --img "$img" --atlas-dir "$W/data/atlases" \
        --atlas-img t1.nii.gz --atlas-seg labels.nii.gz --out "$out" > /dev/null 2>&1
    e=$(date +%s.%N)
    echo "cli $t num$NUM rep$k $(echo "$e - $s" | bc)" | tee -a "$W/runs/gpu_bench/cli_times.txt"
  done
  for k in $(seq 1 "$REPS"); do
    out=$W/runs/gpu_bench/num${NUM}_$t/prog_$k
    rm -rf "$out"; mkdir -p "$out"
    (cd "$W" && ./gif-python/native/build/gif_gpu --img "$img" --out "$out" --plan "$W/data/plan.txt" $ATL --numerics "$NUM" \
        --timing-json "$out/timing.json" > /dev/null)
  done
done
