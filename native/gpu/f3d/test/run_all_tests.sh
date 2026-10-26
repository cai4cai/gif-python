#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Runs the stage tests against NiftyReg (needs the NiftyReg static libraries, see Makefile) on one target / template pair.
# usage: test/run_all_tests.sh target.nii.gz template.nii affine.txt
cd "$(dirname "$0")/.."
[ $# -eq 3 ] || { echo "usage: $0 target.nii[.gz] template.nii affine.txt"; exit 1; }
T=$1; F=$2; A=$3
echo "### stage 1/2: pyramid, grid init/refine, B-spline deformation + resampling + image gradient"; ./build/test_stage1 $T $F $A
echo "### stage 3/4: LNCC value, voxel gradient, spline smoothing, node gradient"; ./build/test_stage3 $T $F $A
echo "### stage 3/4 with a reference mask"; TEST_MASK=1 ./build/test_stage3 $T $F $A | grep "LNCC value\|node gradient"
echo "### stage 5: BE, LE, Jacobian penalties and folding correction (amp = control point perturbation in mm)"
for amp in 3 8; do ./build/test_stage5 $T $F $A $amp; done
echo "### float-float warp vs double precision warp"; ./build/test_warpdf $T $F $A 20
