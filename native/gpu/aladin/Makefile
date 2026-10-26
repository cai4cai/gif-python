# SPDX-License-Identifier: BSD-3-Clause
# GPU batched symmetric aladin.  Needs CUDA (nvcc) and, for the comparison tools, NiftyReg static libraries.
#   make                build the driver (build/aladin_cli)
#   make tools-cpu      build the NiftyReg comparison tools (maths_check, cpu_trace, stage_check)
# Variables: NVCC, ARCH, NIFTYREG_PREFIX (install prefix of a NiftyReg build with static libraries; tools-cpu only).
NVCC ?= /usr/local/cuda/bin/nvcc
ARCH ?= -gencode arch=compute_121,code=sm_121
FLAGS = -O3 -std=c++17 $(ARCH) -lineinfo
NRP ?= $(NIFTYREG_PREFIX)
NRINC = -I$(NRP)/include -I$(NRP)/include/nifti
NRLIBS = -L$(NRP)/lib -l_reg_maths -l_reg_tools -lreg_nifti -lz
NRCPULIBS = -L$(NRP)/lib -l_reg_aladin -l_reg_blockMatching -l_reg_globalTrans -l_reg_resampling -l_reg_tools -l_reg_maths -l_reg_ReadWriteImage -lreg_nifti -lreg_png -lz

all: build/aladin_cli
build/aladin_cli: test/aladin_cli.cu aladin.cu aladin.h mat4.h
	mkdir -p build && $(NVCC) $(FLAGS) -o $@ test/aladin_cli.cu aladin.cu
build/maths_check: test/maths_check.cpp mat4.h
	mkdir -p build && g++ -O2 -fopenmp -std=c++11 $(NRINC) -o $@ test/maths_check.cpp $(NRLIBS)
build/cpu_trace: test/cpu_trace.cpp
	mkdir -p build && g++ -O2 -fopenmp -std=c++11 $(NRINC) -o $@ $< $(NRCPULIBS)
build/stage_check: test/stage_check.cpp
	mkdir -p build && g++ -O2 -fopenmp -std=c++11 $(NRINC) -o $@ $< $(NRCPULIBS)
tools-cpu: build/maths_check build/cpu_trace build/stage_check
clean:
	rm -rf build
.PHONY: all tools-cpu clean
