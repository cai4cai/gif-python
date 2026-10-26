# SPDX-License-Identifier: BSD-3-Clause
# GPU backend of gif-python:  cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release && cmake --build native/build -j
# Produces native/build/gif_gpu.  Needs the CUDA toolkit (nvcc), OpenMP and zlib.  The comparison tests of the ported NiftyReg / NiftySeg
# parts (gpu/*/test, bench) need NiftyReg / NiftySeg static libraries and are built with the Makefiles of the sub-directories instead.
cmake_minimum_required(VERSION 3.18)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
  endif()
endif()
project(gif_gpu LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(ZLIB REQUIRED)

# the ported kernels were validated with these flags (no floating point contraction on the host, line info for profiling)
set(GIF_CUDA_FLAGS -O3 -lineinfo "-Xcompiler=-ffp-contract=off,-fopenmp")

add_library(gif_common STATIC gpu/common/gzwrite.cpp)
target_include_directories(gif_common PUBLIC gpu/common)
target_compile_options(gif_common PRIVATE -O2 -ffp-contract=off)
target_link_libraries(gif_common PUBLIC OpenMP::OpenMP_CXX ZLIB::ZLIB)

file(GLOB F3D_SOURCES CONFIGURE_DEPENDS gpu/f3d/src/*.cu)
add_library(gif_f3d STATIC ${F3D_SOURCES})
target_include_directories(gif_f3d PUBLIC gpu/f3d/include)
target_compile_options(gif_f3d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${GIF_CUDA_FLAGS}>)
target_link_libraries(gif_f3d PUBLIC CUDA::cudart OpenMP::OpenMP_CXX)

add_library(gif_aladin STATIC gpu/aladin/aladin.cu)
target_include_directories(gif_aladin PUBLIC gpu/aladin)
target_compile_options(gif_aladin PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)
target_link_libraries(gif_aladin PUBLIC CUDA::cudart)

add_library(gif_segem STATIC gpu/seg/src/context.cu gpu/seg/src/morph.cu gpu/seg/src/em.cu)
target_include_directories(gif_segem PUBLIC gpu/seg/include PRIVATE gpu/seg/src)
set_target_properties(gif_segem PROPERTIES CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)
target_compile_options(gif_segem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)
target_link_libraries(gif_segem PUBLIC CUDA::cudart)

add_executable(gif_gpu gif/gif_gpu.cpp gif/gif_io.cpp gif/gif_stages.cu)
target_include_directories(gif_gpu PRIVATE gif)
set_target_properties(gif_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(gif_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo> $<$<COMPILE_LANGUAGE:CXX>:-O2>)
target_link_libraries(gif_gpu PRIVATE gif_aladin gif_f3d gif_segem gif_common CUDA::cudart ZLIB::ZLIB OpenMP::OpenMP_CXX)
