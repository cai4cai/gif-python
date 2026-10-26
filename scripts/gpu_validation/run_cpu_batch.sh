#!/bin/bash
# CPU baseline jobs of the validation of the GPU backend (sequential): the IXI targets, two runs of IXI002 with a 1e-7
# perturbation of the affine (noise floor), and the leave-one-out runs of 8 Mindboggle OASIS-TRT-20 subjects.
# Layout: $WORK/data/atlases/<name>/{t1.nii.gz,labels.nii.gz} (+ structures_info.csv, tissues_info.csv), $WORK/gif-python = checkout.
#   WORK=~/gifpy-work NIFTYREG_PATH=... NIFTYSEG_PATH=... run_cpu_batch.sh
set -u
W=${WORK:-$HOME/gifpy-work}
cd $W/gif-python
PY=${PY:-python}
COMMON="--atlas-dir $W/data/atlases --atlas-img t1.nii.gz --atlas-seg labels.nii.gz --pools 2 --omp 8 --keep-registration-files"
run() { # name, extra args...
  name=$1; shift
  /usr/bin/time -f "%e s wall" -o $W/runs/cpu_$name.time nice -n 10 env PYTHONPATH=. $PY "$@" > $W/runs/cpu_$name.log 2>&1
}
for t in IXI002-Guys-0828-T1 IXI012-HH-1211-T1 IXI035-IOP-0873-T1; do
  run $t -m src.cli segment -v --img ${IXI_DIR:?set IXI_DIR}/$t.nii.gz $COMMON --out $W/runs/cpu/$t
done
for s in 1 2; do
  run IXI002_pert$s $W/gif-python/scripts/gpu_validation/cpu_perturbed.py --eps 1e-7 --seed $s -- segment -v --img ${IXI_DIR:?set IXI_DIR}/IXI002-Guys-0828-T1.nii.gz $COMMON --out $W/runs/cpu/IXI002_pert$s
done
for i in 1 3 5 7 9 11 13 15; do
  n=OASIS-TRT-20-$i
  run loo_$n -m src.cli segment -v --img $W/data/atlases/$n/t1.nii.gz --exclude-atlas $n $COMMON --out $W/runs/cpu_loo/$n
done
echo done > $W/runs/cpu_batch.done
