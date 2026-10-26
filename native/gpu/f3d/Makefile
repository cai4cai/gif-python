# SPDX-License-Identifier: BSD-3-Clause
# GPU F3D (NiftyReg F3D + LNCC port).  Build with:  make -j4 all
# Variables: NVCC, ARCH, NIFTYREG_PREFIX (install prefix of a NiftyReg build with static libraries; needed for tests and tools).
NVCC     ?= /usr/local/cuda/bin/nvcc
ARCH     ?= -gencode arch=compute_121,code=sm_121
NR       ?= $(NIFTYREG_PREFIX)
NVFLAGS  = $(NVEXTRA) -O3 -std=c++17 $(ARCH) -Iinclude -Xcompiler -ffp-contract=off,-fopenmp -lineinfo
CXXFLAGS = -O2 -std=c++17 -fopenmp -ffp-contract=off -Iinclude -Itest -I$(NR)/include -I/usr/local/cuda/include
NRLIBS   = $(NR)/lib/lib_reg_f3d.a $(NR)/lib/lib_reg_localTrans.a $(NR)/lib/lib_reg_measure.a $(NR)/lib/lib_reg_resampling.a \
           $(NR)/lib/lib_reg_globalTrans.a $(NR)/lib/lib_reg_ReadWriteImage.a $(NR)/lib/libreg_png.a $(NR)/lib/lib_reg_tools.a \
           $(NR)/lib/lib_reg_maths.a $(NR)/lib/libreg_nifti.a $(NR)/lib/libpng.a -lz
OBJ = build/f3d_conv.o build/f3d_transform.o build/f3d_lncc.o build/f3d_gradient.o build/f3d_regul.o build/f3d_lncc_stream.o build/f3d_lncc_fast.o build/f3d_optim.o build/f3d_batch.o
LIB = build/libf3d.a

lib: $(LIB)

all: $(LIB) build/f3d_run build/test_stage1 build/test_stage3 build/test_stage5 build/test_stage6 build/test_fma_le build/test_fma_be build/emul_be build/emul_be_value build/test_m3 build/emul_le build/test_batch build/bench_lncc build/test_warpdf build/eval_cpp build/ref_f3d

build/%.o: src/%.cu include/*.h
	@mkdir -p build
	$(NVCC) $(NVFLAGS) -c $< -o $@

$(LIB): $(OBJ)
	ar rcs $@ $^

build/%: test/%.cpp $(LIB) include/*.h test/test_common.h test/test_config.h
	@mkdir -p build
	g++ $(CXXFLAGS) $< -o $@ $(LIB) $(NRLIBS) -L/usr/local/cuda/lib64 -lcudart

# NiftyReg-only reference programs (no GPU code)
build/ref_f3d: test/ref_f3d.cpp test/test_config.h
	@mkdir -p build
	g++ -O2 -fopenmp -Itest -I$(NR)/include test/ref_f3d.cpp -o $@ $(NRLIBS)
build/ref_f3d_dbg: test/ref_f3d_dbg.cpp test/test_config.h
	@mkdir -p build
	g++ -O2 -fopenmp -Itest -I$(NR)/include test/ref_f3d_dbg.cpp -o $@ $(NRLIBS)

build/replica_be: test/replica_be.cpp $(LIB) include/*.h
	g++ -O3 -fopenmp -DNDEBUG -std=c++17 -Iinclude -Itest -I$(NR)/include -I/usr/local/cuda/include test/replica_be.cpp -o $@ $(LIB) $(NRLIBS) -L/usr/local/cuda/lib64 -lcudart

build/f3d_run: tools/f3d_run.cpp $(LIB) include/*.h
	@mkdir -p build
	g++ $(CXXFLAGS) tools/f3d_run.cpp -o $@ $(LIB) $(NRLIBS) -L/usr/local/cuda/lib64 -lcudart

clean:
	rm -rf build
.PHONY: all lib clean
