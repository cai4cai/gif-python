// SPDX-License-Identifier: BSD-3-Clause
// gif_io.h - minimal NIfTI-1 reader / header builder for the GPU backend (written for gif-python; no NiftyReg code).
//
// Reader: .nii / .nii.gz, native byte order, data types uint8 int8 uint16 int16 uint32 int32 float32 float64, scl_slope / scl_inter
// applied (a zero or non-finite slope means "no scaling", as in nibabel).  The voxel data are returned as float, x fastest.
// The affine is the sform if sform_code > 0, else the qform (the rule of nibabel and NiftyReg).
#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace gifio {

struct Image {
    std::string path;
    int dim[8] = {0, 1, 1, 1, 1, 1, 1, 1};   // NIfTI dim[]: dim[0] = number of dimensions, dim[1..7] sizes
    float pixdim[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    int datatype = 16;                       // NIfTI data type code of the FILE
    float xyz[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};   // voxel -> world
    std::vector<float> data;
    size_t nvox3() const { return (size_t)dim[1] * dim[2] * dim[3]; }
    size_t nvox() const
    {
        size_t n = 1;
        for (int i = 1; i <= dim[0] && i <= 7; i++) n *= (size_t)(dim[i] > 0 ? dim[i] : 1);
        return n;
    }
    bool integer_type() const { return datatype != 16 && datatype != 64; }
};

enum { DT_U8 = 2, DT_I16 = 4, DT_I32 = 8, DT_F32 = 16, DT_F64 = 64, DT_I8 = 256, DT_U16 = 512, DT_U32 = 768 };

// Throws std::runtime_error with a readable message.
void read_image(const std::string &path, Image &im);

// 348 byte header + 4 byte extension flag (vox_offset = 352) for a file with dimensions dim[1..dim[0]].  The affine is written as sform
// (code 2) and qform (code 2), like nibabel does for Nifti1Image(data, affine).
std::vector<uint8_t> make_header(const int dim[8], const float pixdim[8], int datatype, const float xyz[4][4], int intent_code = 0,
                                 const char *intent_name = nullptr, float intent_p1 = 0.f);

int bytes_per_voxel(int datatype);

}  // namespace gifio
