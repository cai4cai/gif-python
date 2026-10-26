# SPDX-License-Identifier: BSD-3-Clause
# GPU NiftySeg EM + mask morphology.  Needs CUDA (nvcc); the tests need the NiftySeg static libraries.
#   make                 libsegem.a
#   make tests           test_morph, test_em
# Variables: NVCC, ARCH, NIFTYSEG_PREFIX (install prefix of a NiftySeg build with static libraries).
NVCC     ?= /usr/local/cuda/bin/nvcc
ARCH     ?= -gencode arch=compute_121,code=sm_121
NVFLAGS  ?= -O3 -std=c++17 $(ARCH) -Iinclude -lineinfo
PREFIX   ?= $(NIFTYSEG_PREFIX)
BUILD    ?= build

SRC := src/context.cu src/morph.cu src/em.cu
OBJ := $(patsubst src/%.cu,$(BUILD)/%.o,$(SRC))
SEGLIBS = -L$(PREFIX)/lib -l_seg_tools -l_seg_EM -l_seg_nifti -l_seg_FMM -l_seg_tools -l_seg_EM -l_seg_nifti -lz -lgomp

all: $(BUILD)/libsegem.a
tests: $(BUILD)/test_morph $(BUILD)/test_em

$(BUILD)/%.o: src/%.cu include/segem.h src/internal.cuh
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) -dc -c $< -o $@

$(BUILD)/libsegem.a: $(OBJ)
	$(NVCC) $(ARCH) -lib $(OBJ) -o $@

# random-mask equivalence test of the morphology against the NiftySeg library functions
$(BUILD)/test_morph: tests/test_morph.cu $(BUILD)/libsegem.a
	$(NVCC) $(NVFLAGS) -Xcompiler -fopenmp -I$(PREFIX)/include tests/test_morph.cu $(BUILD)/libsegem.a -o $@ $(SEGLIBS)

# EM equivalence test against seg_EM on synthetic volumes
$(BUILD)/test_em: tests/test_em.cu $(BUILD)/libsegem.a
	$(NVCC) $(NVFLAGS) -Xcompiler -fopenmp -I$(PREFIX)/include tests/test_em.cu $(BUILD)/libsegem.a -o $@ $(SEGLIBS)

clean:
	rm -rf $(BUILD)
.PHONY: all tests clean
