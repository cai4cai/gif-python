# SPDX-License-Identifier: BSD-3-Clause
# Common host utilities.   make test
CXX ?= g++
build/gzwrite.o: gzwrite.cpp gzwrite.h
	@mkdir -p build
	$(CXX) -O2 -std=c++17 -fopenmp -c $< -o $@
build/libgpucommon.a: build/gzwrite.o
	ar rcs $@ $^
build/test_gzwrite: test_gzwrite.cpp build/libgpucommon.a
	$(CXX) -O2 -std=c++17 -fopenmp test_gzwrite.cpp build/libgpucommon.a -lz -o $@
test: build/test_gzwrite
	./build/test_gzwrite
clean:
	rm -rf build
.PHONY: test clean
