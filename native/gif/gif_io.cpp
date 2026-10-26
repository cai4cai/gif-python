// SPDX-License-Identifier: BSD-3-Clause
// gif_io.cpp - minimal NIfTI-1 reader / header builder (see gif_io.h).  Field offsets follow the public NIfTI-1 header specification.
#include "gif_io.h"
#include <zlib.h>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace gifio {

namespace {

template <class T> T rd(const uint8_t *h, size_t off)
{
    T v;
    memcpy(&v, h + off, sizeof(T));
    return v;
}
template <class T> void wr(uint8_t *h, size_t off, T v) { memcpy(h + off, &v, sizeof(T)); }

// quaternion (b,c,d) -> rotation matrix scaled by the voxel size (NIfTI-1 specification, "method 2")
void quatern_to_xyz(float b, float c, float d, float qx, float qy, float qz, float dx, float dy, float dz, float qfac, float m[4][4])
{
    double a = 1.0 - (double)b * b - (double)c * c - (double)d * d;
    if (a < 1e-7) {
        double s = 1.0 / std::sqrt((double)b * b + (double)c * c + (double)d * d);
        b = (float)(b * s); c = (float)(c * s); d = (float)(d * s);
        a = 0.0;
    } else {
        a = std::sqrt(a);
    }
    if (dx <= 0.f) dx = 1.f;
    if (dy <= 0.f) dy = 1.f;
    if (dz <= 0.f) dz = 1.f;
    if (qfac < 0.f) dz = -dz;
    m[0][0] = (float)((a * a + (double)b * b - (double)c * c - (double)d * d) * dx);
    m[0][1] = (float)(2.0 * ((double)b * c - a * d) * dy);
    m[0][2] = (float)(2.0 * ((double)b * d + a * c) * dz);
    m[1][0] = (float)(2.0 * ((double)b * c + a * d) * dx);
    m[1][1] = (float)((a * a + (double)c * c - (double)b * b - (double)d * d) * dy);
    m[1][2] = (float)(2.0 * ((double)c * d - a * b) * dz);
    m[2][0] = (float)(2.0 * ((double)b * d - a * c) * dx);
    m[2][1] = (float)(2.0 * ((double)c * d + a * b) * dy);
    m[2][2] = (float)((a * a + (double)d * d - (double)c * c - (double)b * b) * dz);
    m[0][3] = qx; m[1][3] = qy; m[2][3] = qz;
    m[3][0] = m[3][1] = m[3][2] = 0.f; m[3][3] = 1.f;
}

// rotation/scaling matrix -> quaternion (NIfTI-1 specification)
void xyz_to_quatern(const float M[4][4], float &qb, float &qc, float &qd, float &qfac)
{
    double r[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) r[i][j] = M[i][j];
    for (int j = 0; j < 3; j++) {
        double n = std::sqrt(r[0][j] * r[0][j] + r[1][j] * r[1][j] + r[2][j] * r[2][j]);
        if (n > 0) for (int i = 0; i < 3; i++) r[i][j] /= n;
    }
    double det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0]) +
                 r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
    qfac = 1.f;
    if (det < 0) {
        for (int i = 0; i < 3; i++) r[i][2] = -r[i][2];
        qfac = -1.f;
    }
    double a = r[0][0] + r[1][1] + r[2][2] + 1.0, b, c, d;
    if (a > 0.5) {
        a = 0.5 * std::sqrt(a);
        b = 0.25 * (r[2][1] - r[1][2]) / a;
        c = 0.25 * (r[0][2] - r[2][0]) / a;
        d = 0.25 * (r[1][0] - r[0][1]) / a;
    } else {
        double xd = 1.0 + r[0][0] - (r[1][1] + r[2][2]);
        double yd = 1.0 + r[1][1] - (r[0][0] + r[2][2]);
        double zd = 1.0 + r[2][2] - (r[0][0] + r[1][1]);
        if (xd > 1.0) {
            b = 0.5 * std::sqrt(xd);
            a = 0.25 * (r[2][1] - r[1][2]) / b;
            c = 0.25 * (r[0][1] + r[1][0]) / b;
            d = 0.25 * (r[0][2] + r[2][0]) / b;
        } else if (yd > 1.0) {
            c = 0.5 * std::sqrt(yd);
            a = 0.25 * (r[0][2] - r[2][0]) / c;
            b = 0.25 * (r[0][1] + r[1][0]) / c;
            d = 0.25 * (r[1][2] + r[2][1]) / c;
        } else {
            d = 0.5 * std::sqrt(zd);
            a = 0.25 * (r[1][0] - r[0][1]) / d;
            b = 0.25 * (r[0][2] + r[2][0]) / d;
            c = 0.25 * (r[1][2] + r[2][1]) / d;
        }
        if (a < 0.0) { a = -a; b = -b; c = -c; d = -d; }
    }
    qb = (float)b; qc = (float)c; qd = (float)d;
}

}  // namespace

int bytes_per_voxel(int dt)
{
    switch (dt) {
        case DT_U8: case DT_I8: return 1;
        case DT_I16: case DT_U16: return 2;
        case DT_I32: case DT_U32: case DT_F32: return 4;
        case DT_F64: return 8;
        default: return 0;
    }
}

void read_image(const std::string &path, Image &im)
{
    im.path = path;
    gzFile f = gzopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    gzbuffer(f, 1 << 20);
    uint8_t h[348];
    if (gzread(f, h, 348) != 348) { gzclose(f); throw std::runtime_error("cannot read the NIfTI header of " + path); }
    if (rd<int32_t>(h, 0) != 348) {
        gzclose(f);
        throw std::runtime_error(path + ": not a native byte order NIfTI-1 file (header size != 348)");
    }
    for (int i = 0; i < 8; i++) {
        im.dim[i] = rd<int16_t>(h, 40 + 2 * i);
        im.pixdim[i] = rd<float>(h, 76 + 4 * i);
    }
    im.datatype = rd<int16_t>(h, 70);
    const int bpv = bytes_per_voxel(im.datatype);
    if (!bpv) { gzclose(f); throw std::runtime_error(path + ": unsupported NIfTI data type " + std::to_string(im.datatype)); }
    if (im.dim[0] < 3) { gzclose(f); throw std::runtime_error(path + ": not a 3D image"); }
    float slope = rd<float>(h, 112), inter = rd<float>(h, 116);
    if (!std::isfinite(slope) || slope == 0.f) { slope = 1.f; inter = 0.f; }
    if (!std::isfinite(inter)) inter = 0.f;
    const size_t voxoff = (size_t)rd<float>(h, 108);
    const int qcode = rd<int16_t>(h, 252), scode = rd<int16_t>(h, 254);
    if (scode > 0) {
        for (int c = 0; c < 4; c++) {
            im.xyz[0][c] = rd<float>(h, 280 + 4 * c);
            im.xyz[1][c] = rd<float>(h, 296 + 4 * c);
            im.xyz[2][c] = rd<float>(h, 312 + 4 * c);
        }
        im.xyz[3][0] = im.xyz[3][1] = im.xyz[3][2] = 0.f; im.xyz[3][3] = 1.f;
    } else if (qcode > 0) {
        quatern_to_xyz(rd<float>(h, 256), rd<float>(h, 260), rd<float>(h, 264), rd<float>(h, 268), rd<float>(h, 272), rd<float>(h, 276),
                       im.pixdim[1], im.pixdim[2], im.pixdim[3], im.pixdim[0] < 0.f ? -1.f : 1.f, im.xyz);
    } else {
        gzclose(f);
        throw std::runtime_error(path + ": the image has neither an sform nor a qform");
    }
    // skip the extension bytes up to the data
    {
        size_t pos = 348;
        std::vector<uint8_t> skip(1 << 16);
        while (pos < voxoff) {
            size_t n = std::min(skip.size(), voxoff - pos);
            if (gzread(f, skip.data(), (unsigned)n) != (int)n) { gzclose(f); throw std::runtime_error(path + ": truncated file"); }
            pos += n;
        }
    }
    const size_t nv = im.nvox();
    std::vector<uint8_t> raw(nv * bpv);
    size_t got = 0;
    while (got < raw.size()) {
        size_t n = std::min<size_t>(raw.size() - got, 1 << 28);
        int r = gzread(f, raw.data() + got, (unsigned)n);
        if (r <= 0) break;
        got += (size_t)r;
    }
    gzclose(f);
    if (got != raw.size()) throw std::runtime_error(path + ": truncated data");
    im.data.resize(nv);
    float *o = im.data.data();
    const bool scale = (slope != 1.f || inter != 0.f);
#define CONVERT(T)                                                                  \
    {                                                                               \
        const T *p = reinterpret_cast<const T *>(raw.data());                       \
        if (scale) for (size_t i = 0; i < nv; i++) o[i] = (float)((double)p[i] * slope + inter); \
        else for (size_t i = 0; i < nv; i++) o[i] = (float)p[i];                    \
    } break;
    switch (im.datatype) {
        case DT_U8: CONVERT(uint8_t)
        case DT_I8: CONVERT(int8_t)
        case DT_I16: CONVERT(int16_t)
        case DT_U16: CONVERT(uint16_t)
        case DT_I32: CONVERT(int32_t)
        case DT_U32: CONVERT(uint32_t)
        case DT_F32: CONVERT(float)
        case DT_F64: CONVERT(double)
    }
#undef CONVERT
}

std::vector<uint8_t> make_header(const int dim[8], const float pixdim[8], int datatype, const float xyz[4][4], int intent_code,
                                 const char *intent_name, float intent_p1)
{
    std::vector<uint8_t> hb(352, 0);
    uint8_t *h = hb.data();
    wr<int32_t>(h, 0, 348);
    wr<int32_t>(h, 32, 16384);   // extents
    wr<char>(h, 38, 'r');
    for (int i = 0; i < 8; i++) wr<int16_t>(h, 40 + 2 * i, (int16_t)dim[i]);
    wr<float>(h, 56, intent_p1);
    wr<int16_t>(h, 68, (int16_t)intent_code);
    wr<int16_t>(h, 70, (int16_t)datatype);
    wr<int16_t>(h, 72, (int16_t)(8 * bytes_per_voxel(datatype)));
    float qfac = 1.f, qb, qc, qd;
    xyz_to_quatern(xyz, qb, qc, qd, qfac);
    float pd[8];
    for (int i = 0; i < 8; i++) pd[i] = pixdim[i];
    pd[0] = qfac;
    for (int i = 0; i < 3; i++) {
        double n = std::sqrt((double)xyz[0][i] * xyz[0][i] + (double)xyz[1][i] * xyz[1][i] + (double)xyz[2][i] * xyz[2][i]);
        pd[1 + i] = (float)n;
    }
    for (int i = 0; i < 8; i++) wr<float>(h, 76 + 4 * i, pd[i]);
    wr<float>(h, 108, 352.f);
    wr<float>(h, 112, 1.f);   // scl_slope
    wr<char>(h, 123, 10);     // xyzt_units: mm, sec
    wr<int16_t>(h, 252, 2);   // qform_code (aligned)
    wr<int16_t>(h, 254, 2);   // sform_code (aligned)
    wr<float>(h, 256, qb); wr<float>(h, 260, qc); wr<float>(h, 264, qd);
    wr<float>(h, 268, xyz[0][3]); wr<float>(h, 272, xyz[1][3]); wr<float>(h, 276, xyz[2][3]);
    for (int c = 0; c < 4; c++) {
        wr<float>(h, 280 + 4 * c, xyz[0][c]);
        wr<float>(h, 296 + 4 * c, xyz[1][c]);
        wr<float>(h, 312 + 4 * c, xyz[2][c]);
    }
    if (intent_name) memcpy(h + 328, intent_name, std::min<size_t>(strlen(intent_name), 15));
    memcpy(h + 344, "n+1", 4);
    return hb;
}

}  // namespace gifio
