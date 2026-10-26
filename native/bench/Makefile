# SPDX-License-Identifier: BSD-3-Clause
# reg_scores: registration quality scores with NiftyReg's measure classes.   make NIFTYREG_PREFIX=/path/to/install
NR ?= $(NIFTYREG_PREFIX)
build/reg_scores: reg_scores.cpp
	@mkdir -p build
	g++ -O2 -std=c++17 -fopenmp -I$(NR)/include reg_scores.cpp -o $@ \
	  $(NR)/lib/lib_reg_f3d.a $(NR)/lib/lib_reg_localTrans.a $(NR)/lib/lib_reg_measure.a $(NR)/lib/lib_reg_resampling.a \
	  $(NR)/lib/lib_reg_globalTrans.a $(NR)/lib/lib_reg_ReadWriteImage.a $(NR)/lib/libreg_png.a $(NR)/lib/lib_reg_tools.a \
	  $(NR)/lib/lib_reg_maths.a $(NR)/lib/libreg_nifti.a $(NR)/lib/libpng.a -lz
clean:
	rm -rf build
.PHONY: clean
