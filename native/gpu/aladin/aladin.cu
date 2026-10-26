// SPDX-License-Identifier: BSD-3-Clause
// GPU batched symmetric aladin (block matching + LTS) -- see aladin.h and README.md.
#include <cub/cub.cuh>
#include <type_traits>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "aladin.h"

#define dfree_s_(p) dfree((p), s)
#define CK(x)                                                                              \
  do {                                                                                     \
    cudaError_t e_ = (x);                                                                  \
    if (e_ != cudaSuccess) {                                                               \
      fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      abort();                                                                             \
    }                                                                                      \
  } while (0)

namespace aladin {

// ======================================================================================
// Image pyramid: 5-tap gaussian (sigma 0.7355 voxel, radius 2, border-renormalised) followed by
// nearest-neighbour sub-sampling of every second voxel, exactly reg_downsampleImage(type=1).
// The three separable passes are only evaluated at the voxels that survive the sub-sampling.
// ======================================================================================
__global__ void conv_axis_kernel(const float* const* inPtrs, const float* inBase, size_t inStride, float* out,
                                 size_t outStride, int nxi, int nyi, int nzi, int nxo, int nyo, int nzo, int axis,
                                 int ds, int radius, const float* __restrict__ w) {
  size_t voxo = (size_t)nxo * nyo * nzo;
  size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (idx >= voxo) return;
  int t = blockIdx.y;
  const float* in = inPtrs ? inPtrs[t] : inBase + (size_t)t * inStride;
  int xo = (int)(idx % nxo), yo = (int)((idx / nxo) % nyo), zo = (int)(idx / ((size_t)nxo * nyo));
  int c[3] = {xo, yo, zo};
  if (ds) c[axis] *= 2;
  int dimAxis = axis == 0 ? nxi : (axis == 1 ? nyi : nzi);
  long stride = axis == 0 ? 1 : (axis == 1 ? nxi : (long)nxi * nyi);
  long base = ((long)c[2] * nyi + c[1]) * nxi + c[0];
  float acc = 0.f, norm = 0.f;
  for (int k = -radius; k <= radius; ++k) {
    int pos = c[axis] + k;
    if (pos < 0 || pos >= dimAxis) continue;
    float wk = w[k + radius];
    acc += wk * __ldg(in + base + k * stride);
    norm += wk;
  }
  out[(size_t)t * outStride + idx] = acc / norm;
}

// Bit-faithful variant of the pass above: reproduces the CPU arithmetic of reg_tools_kernelConvolution_core
// (float products accumulated in double, float storage between the passes, density image convolved alongside
// and the intensity/density division only after the third pass).  The density of an interior voxel is a constant;
// near the borders it depends on the number of valid taps, so it is recomputed from the source coordinates.
struct ConvGeom {
  int nx0, ny0, nz0;  // source (full) dimensions of this down-sampling step
  int ds[3];
};
__device__ inline float dens_pass(float din, const float* w, int radius, int c, int dimAxis) {
  double dsum = 0.0;
  for (int k = -radius; k <= radius; ++k) {
    int pos = c + k;
    if (pos < 0 || pos >= dimAxis) continue;
    dsum += (double)(w[k + radius] * din);
  }
  return (float)dsum;
}
__global__ void conv_axis_exact_kernel(const float* const* inPtrs, const float* inBase, size_t inStride, float* out,
                                       size_t outStride, int nxi, int nyi, int nzi, int nxo, int nyo, int nzo, int axis,
                                       ConvGeom g, int radius, const float* __restrict__ w) {
  size_t voxo = (size_t)nxo * nyo * nzo;
  size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (idx >= voxo) return;
  int t = blockIdx.y;
  const float* in = inPtrs ? inPtrs[t] : inBase + (size_t)t * inStride;
  int xo = (int)(idx % nxo), yo = (int)((idx / nxo) % nyo), zo = (int)(idx / ((size_t)nxo * nyo));
  int c[3] = {xo, yo, zo};
  if (g.ds[axis]) c[axis] *= 2;
  int dimAxis = axis == 0 ? nxi : (axis == 1 ? nyi : nzi);
  long stride = axis == 0 ? 1 : (axis == 1 ? nxi : (long)nxi * nyi);
  long base = ((long)c[2] * nyi + c[1]) * nxi + c[0];
  double acc = 0.0;
  for (int k = -radius; k <= radius; ++k) {
    int pos = c[axis] + k;
    if (pos < 0 || pos >= dimAxis) continue;
    acc += (double)(w[k + radius] * __ldg(in + base + k * stride));
  }
  float v = (float)acc;
  if (axis == 2) {
    int cx = g.ds[0] ? 2 * xo : xo, cy = g.ds[1] ? 2 * yo : yo, cz = g.ds[2] ? 2 * zo : zo;
    float d1 = dens_pass(1.0f, w, radius, cx, g.nx0);
    float d2 = dens_pass(d1, w, radius, cy, g.ny0);
    float d3 = dens_pass(d2, w, radius, cz, g.nz0);
    v = v / d3;
  }
  out[(size_t)t * outStride + idx] = v;
}

// Block statistics, one thread per 4x4x4 block (CPU _reg_set_active_blocks: sequential float sums)
__global__ void block_var_kernel(const float* data, size_t vox, int nx, int ny, int nz, int bnx, int bny, int bnz,
                                 float* var, int* unusable) {
  int total = bnx * bny * bnz;
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= total) return;
  int t = blockIdx.y;
  const float* img = data + (size_t)t * vox;
  int bi = b % bnx, bj = (b / bnx) % bny, bk = b / (bnx * bny);
  float v[64];
  float mean = 0.f, num = 0.f;
  int c = 0;
  for (int z = bk * 4; z < bk * 4 + 4; ++z)
    for (int y = bj * 4; y < bj * 4 + 4; ++y)
      for (int x = bi * 4; x < bi * 4 + 4; ++x, ++c) {
        float val = nanf("");
        if (x < nx && y < ny && z < nz) val = img[((size_t)z * ny + y) * nx + x];
        v[c] = val;
        if (val == val) {
          mean += val;
          num += 1.f;
        }
      }
  mean /= num;
  float variance = 0.f;
  for (int i = 0; i < 64; ++i)
    if (v[i] == v[i]) variance += (mean - v[i]) * (mean - v[i]);
  variance /= num;
  float outv;
  if (num > 32.f && variance > 0.f) {
    outv = variance;
  } else {
    outv = -1.f;
    atomicAdd(&unusable[t], 1);
  }
  var[(size_t)t * total + b] = outv;
}
__global__ void iota_kernel(int* idx, int total, int n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= (size_t)total * n) return;
  idx[i] = (int)(i % total);
}
__global__ void offsets_kernel(int* offs, int total, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= n) offs[i] = i * total;
}

// Stream-ordered pool allocation (the default memory pool keeps freed blocks, so repeated set-ups do not pay for cudaMalloc)
static void* dalloc(size_t bytes, cudaStream_t s) {
  static const bool pool_init = [] {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaMemPool_t pool;
    if (cudaDeviceGetDefaultMemPool(&pool, dev) == cudaSuccess) {
      uint64_t thr = UINT64_MAX;
      cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &thr);
    }
    return true;
  }();
  (void)pool_init;
  void* p = nullptr;
  CK(cudaMallocAsync(&p, bytes, s));
  return p;
}
static void dfree(void* p, cudaStream_t s) {
  if (p) cudaFreeAsync(p, s);
}

struct LevelData {
  int nx = 0, ny = 0, nz = 0;
  size_t vox = 0;
  float* data = nullptr;  // [n][vox]
  float* xyz = nullptr;   // [n][16]
  float* ijk = nullptr;   // [n][16]
  int bnx = 0, bny = 0, bnz = 0, total = 0;
  int* active = nullptr;  // [n][total] block indices ranked by decreasing variance (first K[t] valid)
  int* K = nullptr;       // [n] device
  int Kmax = 0;
  std::vector<int> hK;
  std::vector<Mat44> hxyz;
};

struct ImageSet {
  int n = 0;
  Params p;
  cudaStream_t s = 0;
  std::vector<LevelData> lv;
  float ms = 0;
  const char* tag = "set";
  int fnx = 0, fny = 0, fnz = 0;  // full-resolution geometry (for align-centre)
  std::vector<Mat44> fxyz;

  ~ImageSet() {
    for (auto& l : lv) {
      dfree(l.data, s);
      dfree(l.xyz, s);
      dfree(l.ijk, s);
      dfree(l.active, s);
      dfree(l.K, s);
    }
  }

  static Mat44 inverse(const Mat44& m) { return nifti_inverse(m); }

  void build(const Image* imgs, int n_, const Params& p_, cudaStream_t s_) {
    n = n_;
    p = p_;
    s = s_;
    const int L = p.levels_to_perform, NL = p.num_levels;
    lv.assign(L, LevelData());
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0, s));

    for (int i = 1; i < n; ++i)
      if (imgs[i].nx != imgs[0].nx || imgs[i].ny != imgs[0].ny || imgs[i].nz != imgs[0].nz) {
        fprintf(stderr, "aladin: all images of a set must have identical dimensions\n");
        abort();
      }
    // gaussian kernel (reg_tools_kernelConvolution_core: sigma<0 => voxels; radius=(int)(sigma*3))
    double sg = fabs((double)(-p.smoothing_sigma_voxel));
    int radius = (int)(sg * 3.0f);
    float hw[5] = {0, 0, 0, 0, 0};
    for (int i = -radius; i <= radius; ++i) hw[radius + i] = (float)(exp(-(double)(i * i) / (2.0 * sg * sg)) / (sg * 2.506628274631));
    float* dw;
    dw = (decltype(dw))dalloc(5 * sizeof(float), s);
    CK(cudaMemcpyAsync(dw, hw, sizeof(hw), cudaMemcpyHostToDevice, s));

    std::vector<const float*> hptr(n);
    for (int i = 0; i < n; ++i) hptr[i] = imgs[i].dev;
    const float** dptr;
    dptr = (decltype(dptr))dalloc(n * sizeof(float*), s);
    CK(cudaMemcpyAsync(dptr, hptr.data(), n * sizeof(float*), cudaMemcpyHostToDevice, s));
    CK(cudaStreamSynchronize(s));  // hptr lifetime

    std::vector<Mat44> xyz(n);
    for (int i = 0; i < n; ++i) xyz[i] = imgs[i].sform;
    fxyz = xyz;
    fnx = imgs[0].nx; fny = imgs[0].ny; fnz = imgs[0].nz;
    int cnx = imgs[0].nx, cny = imgs[0].ny, cnz = imgs[0].nz;
    float* curBuf = nullptr;  // current source buffer (null => input pointer table)
    bool curOwned = false;

    auto keep_level = [&](int li, float* buf, int nx, int ny, int nz) {
      LevelData& l = lv[li];
      l.nx = nx; l.ny = ny; l.nz = nz;
      l.vox = (size_t)nx * ny * nz;
      l.data = buf;
      l.hxyz = xyz;
    };

    if (NL == L) {
      float* buf;
      buf = (decltype(buf))dalloc((size_t)n * cnx * cny * cnz * sizeof(float), s);
      for (int i = 0; i < n; ++i)
        CK(cudaMemcpyAsync(buf + (size_t)i * cnx * cny * cnz, imgs[i].dev, (size_t)cnx * cny * cnz * sizeof(float),
                           cudaMemcpyDeviceToDevice, s));
      keep_level(L - 1, buf, cnx, cny, cnz);
    }
    for (int k = 1; k <= NL - 1; ++k) {
      bool dsa[3] = {true, true, true};
      if (cnx / 2 < 32) dsa[0] = false;
      if (cny / 2 < 32) dsa[1] = false;
      if (cnz / 2 < 32) dsa[2] = false;
      int nxo = dsa[0] ? (cnx + 1) / 2 : cnx, nyo = dsa[1] ? (cny + 1) / 2 : cny, nzo = dsa[2] ? (cnz + 1) / 2 : cnz;
      float *tA, *tB, *out;
      tA = (decltype(tA))dalloc((size_t)n * nxo * cny * cnz * sizeof(float), s);
      tB = (decltype(tB))dalloc((size_t)n * nxo * nyo * cnz * sizeof(float), s);
      out = (decltype(out))dalloc((size_t)n * nxo * nyo * nzo * sizeof(float), s);
      const float* const* ip = curBuf ? nullptr : dptr;
      size_t istr = (size_t)cnx * cny * cnz;
      ConvGeom geom;
      geom.nx0 = cnx; geom.ny0 = cny; geom.nz0 = cnz;
      for (int a = 0; a < 3; ++a) geom.ds[a] = dsa[a] ? 1 : 0;
      auto launch = [&](const float* const* ptrs, const float* base, size_t bstride, float* o, int nxi, int nyi, int nzi, int nxx,
                        int nyy, int nzz, int axis, bool ds) {
        size_t vo = (size_t)nxx * nyy * nzz;
        int dimAxis = axis == 0 ? nxi : (axis == 1 ? nyi : nzi);
        if (p.exact)
          conv_axis_exact_kernel<<<dim3((unsigned)((vo + 255) / 256), n), 256, 0, s>>>(ptrs, base, bstride, o, vo, nxi, nyi, nzi, nxx, nyy,
                                                                                         nzz, axis, geom, dimAxis > 1 ? radius : 0, dw);
        else
          conv_axis_kernel<<<dim3((unsigned)((vo + 255) / 256), n), 256, 0, s>>>(ptrs, base, bstride, o, vo, nxi, nyi, nzi, nxx, nyy,
                                                                                   nzz, axis, ds ? 1 : 0, dimAxis > 1 ? radius : 0, dw);
      };
      launch(ip, curBuf, istr, tA, cnx, cny, cnz, nxo, cny, cnz, 0, dsa[0]);
      launch(nullptr, tA, (size_t)nxo * cny * cnz, tB, nxo, cny, cnz, nxo, nyo, cnz, 1, dsa[1]);
      launch(nullptr, tB, (size_t)nxo * nyo * cnz, out, nxo, nyo, cnz, nxo, nyo, nzo, 2, dsa[2]);
      CK(cudaGetLastError());
      CK(cudaStreamSynchronize(s));
      dfree_s_(tA);
      dfree_s_(tB);
      if (curOwned) {
        bool kept = false;
        for (auto& l : lv) if (l.data == curBuf) kept = true;
        if (!kept) dfree_s_(curBuf);
      }
      // sform update (reg_downsampleImage: columns scaled by 2.f)
      for (int i = 0; i < n; ++i)
        for (int ax = 0; ax < 3; ++ax)
          if (dsa[ax])
            for (int r = 0; r < 3; ++r) xyz[i].m[r][ax] *= 2.f;
      cnx = nxo; cny = nyo; cnz = nzo;
      curBuf = out;
      curOwned = true;
      if (k >= NL - L) keep_level((L - 1) - (k - (NL - L)), out, cnx, cny, cnz);
    }
    if (curOwned) {
      bool kept = false;
      for (auto& l : lv) if (l.data == curBuf) kept = true;
      if (!kept) dfree_s_(curBuf);
    }
    dfree_s_(dw);
    dfree_s_(dptr);

    // per level: matrices, block variance and ranking
    for (int li = 0; li < L; ++li) {
      LevelData& l = lv[li];
      std::vector<float> hx(n * 16), hi(n * 16);
      for (int i = 0; i < n; ++i) {
        Mat44 inv = inverse(l.hxyz[i]);
        for (int a = 0; a < 4; ++a)
          for (int b = 0; b < 4; ++b) {
            hx[i * 16 + a * 4 + b] = l.hxyz[i].m[a][b];
            hi[i * 16 + a * 4 + b] = inv.m[a][b];
          }
      }
      l.xyz = (decltype(l.xyz))dalloc(n * 16 * sizeof(float), s);
      l.ijk = (decltype(l.ijk))dalloc(n * 16 * sizeof(float), s);
      CK(cudaMemcpyAsync(l.xyz, hx.data(), n * 16 * sizeof(float), cudaMemcpyHostToDevice, s));
      CK(cudaMemcpyAsync(l.ijk, hi.data(), n * 16 * sizeof(float), cudaMemcpyHostToDevice, s));
      l.bnx = (l.nx + 3) / 4; l.bny = (l.ny + 3) / 4; l.bnz = (l.nz + 3) / 4;
      l.total = l.bnx * l.bny * l.bnz;
      float *var, *var_sorted;
      int *idx, *unus, *offs;
      var = (decltype(var))dalloc((size_t)n * l.total * sizeof(float), s);
      var_sorted = (decltype(var_sorted))dalloc((size_t)n * l.total * sizeof(float), s);
      idx = (decltype(idx))dalloc((size_t)n * l.total * sizeof(int), s);
      l.active = (decltype(l.active))dalloc((size_t)n * l.total * sizeof(int), s);
      unus = (decltype(unus))dalloc(n * sizeof(int), s);
      offs = (decltype(offs))dalloc((n + 1) * sizeof(int), s);
      CK(cudaMemsetAsync(unus, 0, n * sizeof(int), s));
      block_var_kernel<<<dim3((l.total + 127) / 128, n), 128, 0, s>>>(l.data, l.vox, l.nx, l.ny, l.nz, l.bnx, l.bny, l.bnz, var, unus);
      iota_kernel<<<(unsigned)(((size_t)n * l.total + 255) / 256), 256, 0, s>>>(idx, l.total, n);
      offsets_kernel<<<(n + 1 + 127) / 128, 128, 0, s>>>(offs, l.total, n);
      size_t tb = 0;
      cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, tb, var, var_sorted, idx, l.active, n * l.total, n, offs, offs + 1, 0, 32, s);
      void* tmp;
      tmp = (decltype(tmp))dalloc(tb, s);
      cub::DeviceSegmentedRadixSort::SortPairsDescending(tmp, tb, var, var_sorted, idx, l.active, n * l.total, n, offs, offs + 1, 0, 32, s);
      std::vector<int> hu(n);
      CK(cudaMemcpyAsync(hu.data(), unus, n * sizeof(int), cudaMemcpyDeviceToHost, s));
      CK(cudaStreamSynchronize(s));
      int act0 = (int)((double)l.total * ((double)p.block_percentage / (double)100));
      l.hK.resize(n);
      l.Kmax = 0;
      for (int i = 0; i < n; ++i) {
        l.hK[i] = std::min(act0, l.total - hu[i]);
        l.Kmax = std::max(l.Kmax, l.hK[i]);
      }
      l.K = (decltype(l.K))dalloc(n * sizeof(int), s);
      CK(cudaMemcpyAsync(l.K, l.hK.data(), n * sizeof(int), cudaMemcpyHostToDevice, s));
      dfree_s_(tmp); dfree_s_(var); dfree_s_(var_sorted); dfree_s_(idx); dfree_s_(unus); dfree_s_(offs);
    }
    CK(cudaEventRecord(e1, s));
    CK(cudaEventSynchronize(e1));
    CK(cudaEventElapsedTime(&ms, e0, e1));
    if (const char* dd = getenv("ALADIN_DUMP")) {  // debug: dump image 0 of each level
      for (int li = 0; li < L; ++li) {
        LevelData& l = lv[li];
        std::vector<float> h(l.vox);
        CK(cudaMemcpy(h.data(), l.data, l.vox * 4, cudaMemcpyDeviceToHost));
        char fn[512];
        snprintf(fn, sizeof(fn), "%s/%s_L%d.f32", dd, tag, li);
        FILE* f = fopen(fn, "wb");
        fwrite(h.data(), 4, l.vox, f);
        fclose(f);
        snprintf(fn, sizeof(fn), "%s/%s_L%d.txt", dd, tag, li);
        f = fopen(fn, "w");
        fprintf(f, "%d %d %d K=%d total=%d\n", l.nx, l.ny, l.nz, l.hK[0], l.total);
        for (int a = 0; a < 3; ++a) fprintf(f, "%.9g %.9g %.9g %.9g\n", l.hxyz[0].m[a][0], l.hxyz[0].m[a][1], l.hxyz[0].m[a][2], l.hxyz[0].m[a][3]);
        fclose(f);
      }
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
  }
};

// ======================================================================================
// Per-iteration kernels
// ======================================================================================
// Folded voxel->voxel matrices: Mf[d][t] (3x4) = ijk_flo * M_d * xyz_ref   (double, stored as float)
struct FoldArgs {
  const float* refXyz[2];
  int refXyzStride[2];
  const float* floIjk[2];
  int floIjkStride[2];
  const float* Mcur[2];  // [n][16]
  float* Mf;             // [2][n][36]
  int n;
};
__device__ inline void fold_one(const FoldArgs& a, int d, int t) {
  const float* X = a.refXyz[d] + (size_t)t * a.refXyzStride[d];
  const float* I = a.floIjk[d] + (size_t)t * a.floIjkStride[d];
  const float* M = a.Mcur[d] + t * 16;
  double A[16], B[16];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      double s = 0;
      for (int k = 0; k < 4; ++k) s += (double)M[i * 4 + k] * (double)X[k * 4 + j];
      A[i * 4 + j] = s;
    }
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j) {
      double s = 0;
      for (int k = 0; k < 4; ++k) s += (double)I[i * 4 + k] * A[k * 4 + j];
      B[i * 4 + j] = s;
    }
  float* o = a.Mf + ((size_t)d * a.n + t) * 36;
  // [0..11]: float(T * xyz_ref) (reg_mat44_mul semantics), [12..23]: ijk_flo, [24..35]: both folded (fast path)
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j) {
      o[i * 4 + j] = (float)A[i * 4 + j];
      o[12 + i * 4 + j] = I[i * 4 + j];
      o[24 + i * 4 + j] = (float)B[i * 4 + j];
    }
}
__global__ void fold_kernel(FoldArgs a) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.n) return;
  fold_one(a, 0, t);
  fold_one(a, 1, t);
}

// Resampling (trilinear, NaN padding like reg_resampleImage with padding=NaN)
// Warped images are stored with a NaN border (3 voxels below, 6 above, x pitch rounded to 4 floats) so that the
// block-matching patch (10^3 voxels around a 4^3 block, also for blocks sticking out of the image) can be read without
// bounds checks and with aligned 16-byte loads.
struct PadGeom {
  int pnx, pny, pnz;
  size_t pvox;
};
static PadGeom pad_geom(int nx, int ny, int nz) {
  PadGeom g;
  g.pnx = ((nx + 9) + 3) / 4 * 4;
  g.pny = ny + 9;
  g.pnz = nz + 9;
  g.pvox = (size_t)g.pnx * g.pny * g.pnz + 16;
  return g;
}
struct WarpDir {
  const float* flo;  // [n or 1][floVox]
  size_t floStride;  // 0 for the target
  int fx, fy, fz;
  float* warped;     // [n][pvox] padded
  int rx, ry, rz;
  int pnx, pny;
  size_t pvox;
};
constexpr int WARP_TX = 256;
struct WarpArgs {
  WarpDir d[2];
  const float* Mf;
  int n;
  int maxnz;
  int ntz;
};
__global__ void __launch_bounds__(WARP_TX) warp_kernel(WarpArgs a) {
  // 3D tiles (8x8x4 threads): neighbouring threads touch neighbouring source voxels even when the two images
  // have permuted axes (images stored with different axis orders), which keeps the gathers in L1.
  // grid: x = tile columns, y = tile rows, z = tile slice + ntz * (template + n * direction)
  const int ntz = a.ntz;
  const int zz = blockIdx.z;
  const int d = zz / (ntz * a.n);
  const int rem = zz - d * ntz * a.n;
  const int t = rem / ntz;
  const int z = (rem - t * ntz) * 4 + (threadIdx.x >> 6);
  const int y = blockIdx.y * 8 + ((threadIdx.x >> 3) & 7);
  const int x = blockIdx.x * 8 + (threadIdx.x & 7);
  const WarpDir& P = a.d[d];
  if (x >= P.rx || y >= P.ry || z >= P.rz) return;
  const float* M = a.Mf + ((size_t)d * a.n + t) * 36 + 24;
  float fx = (float)x, fy = (float)y, fz = (float)z;
  float px = fmaf(M[0], fx, fmaf(M[1], fy, fmaf(M[2], fz, M[3])));
  float py = fmaf(M[4], fx, fmaf(M[5], fy, fmaf(M[6], fz, M[7])));
  float pz = fmaf(M[8], fx, fmaf(M[9], fy, fmaf(M[10], fz, M[11])));
  float flx = floorf(px), fly = floorf(py), flz = floorf(pz);
  int x0 = (int)flx, y0 = (int)fly, z0 = (int)flz;
  float val = nanf("");
  if (x0 >= 0 && x0 + 1 < P.fx && y0 >= 0 && y0 + 1 < P.fy && z0 >= 0 && z0 + 1 < P.fz) {
    float rx = px - flx, ry = py - fly, rz = pz - flz;
    const float* f = P.flo + (size_t)t * P.floStride + ((size_t)z0 * P.fy + y0) * P.fx + x0;
    size_t sy = P.fx, sz = (size_t)P.fx * P.fy;
    float c000 = __ldg(f), c100 = __ldg(f + 1), c010 = __ldg(f + sy), c110 = __ldg(f + sy + 1);
    float c001 = __ldg(f + sz), c101 = __ldg(f + sz + 1), c011 = __ldg(f + sz + sy), c111 = __ldg(f + sz + sy + 1);
    float a00 = c000 + rx * (c100 - c000), a10 = c010 + rx * (c110 - c010);
    float a01 = c001 + rx * (c101 - c001), a11 = c011 + rx * (c111 - c011);
    float b0 = a00 + ry * (a10 - a00), b1 = a01 + ry * (a11 - a01);
    val = b0 + rz * (b1 - b0);
  }
  P.warped[(size_t)t * P.pvox + ((size_t)(z + 3) * P.pny + (y + 3)) * P.pnx + (x + 3)] = val;
}

// Bit-faithful variant: replicates reg_affine_getDeformationField (double, stored as float) followed by
// reg_resampleImage (float world -> float voxel position, double linear interpolation).  Slow on this GPU
// (fp64 is 1/55 of fp32) but removes the pyramid/warp rounding noise that otherwise makes the discrete
// block-matching decisions diverge from the CPU.
__global__ void __launch_bounds__(WARP_TX) warp_exact_kernel(WarpArgs a) {
  const int ntz = a.ntz;
  const int zz = blockIdx.z;
  const int d = zz / (ntz * a.n);
  const int rem = zz - d * ntz * a.n;
  const int t = rem / ntz;
  const int z = (rem - t * ntz) * 4 + (threadIdx.x >> 6);
  const int y = blockIdx.y * 8 + ((threadIdx.x >> 3) & 7);
  const int x = blockIdx.x * 8 + (threadIdx.x & 7);
  const WarpDir& P = a.d[d];
  if (x >= P.rx || y >= P.ry || z >= P.rz) return;
  size_t idx = ((size_t)(z + 3) * P.pny + (y + 3)) * P.pnx + (x + 3);
  const float* TM = a.Mf + ((size_t)d * a.n + t) * 36;
  const float* IJ = TM + 12;
  double vx = x, vy = y, vz = z;
  float w0 = (float)((double)TM[0] * vx + (double)TM[1] * vy + (double)TM[2] * vz + (double)TM[3]);
  float w1 = (float)((double)TM[4] * vx + (double)TM[5] * vy + (double)TM[6] * vz + (double)TM[7]);
  float w2 = (float)((double)TM[8] * vx + (double)TM[9] * vy + (double)TM[10] * vz + (double)TM[11]);
  float px = (float)((double)IJ[0] * (double)w0 + (double)IJ[1] * (double)w1 + (double)IJ[2] * (double)w2 + (double)IJ[3]);
  float py = (float)((double)IJ[4] * (double)w0 + (double)IJ[5] * (double)w1 + (double)IJ[6] * (double)w2 + (double)IJ[7]);
  float pz = (float)((double)IJ[8] * (double)w0 + (double)IJ[9] * (double)w1 + (double)IJ[10] * (double)w2 + (double)IJ[11]);
  int x0 = (int)floorf(px), y0 = (int)floorf(py), z0 = (int)floorf(pz);
  float val = nanf("");
  if (x0 >= 0 && x0 + 1 < P.fx && y0 >= 0 && y0 + 1 < P.fy && z0 >= 0 && z0 + 1 < P.fz) {
    double rx = (double)px - (double)x0, ry = (double)py - (double)y0, rz = (double)pz - (double)z0;
    if (rx < 0.0) rx = 0.0;
    if (ry < 0.0) ry = 0.0;
    if (rz < 0.0) rz = 0.0;
    double xb0 = 1.0 - rx, xb1 = rx, yb0 = 1.0 - ry, yb1 = ry, zb0 = 1.0 - rz, zb1 = rz;
    const float* f = P.flo + (size_t)t * P.floStride + ((size_t)z0 * P.fy + y0) * P.fx + x0;
    size_t sy = P.fx, sz = (size_t)P.fx * P.fy;
    double inten = 0.0;
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      double yt = 0.0;
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        const float* q = f + c * sz + b * sy;
        double xt = 0.0;
        xt += (double)__ldg(q) * xb0;
        xt += (double)__ldg(q + 1) * xb1;
        yt += xt * (b ? yb1 : yb0);
      }
      inten += yt * (c ? zb1 : zb0);
    }
    val = (float)inten;
  }
  P.warped[(size_t)t * P.pvox + idx] = val;
}

// ---------------------------------------------------------------- block matching
struct BMDir {
  const float* ref;      // reference image(s)
  size_t refStride;      // 0 for the target
  const float* warped;   // [n][vox]
  const float* refXyz;   // sform of reference
  int refXyzStride;
  const int* active;
  size_t activeStride;
  const int* K;
  int Kstride;
  int nx, ny, nz, bnx, bny;
  int Kmax;
  int pnx, pny;
  size_t pvox;
  float* outRef;   // [n][Kmax*3]
  float* outWarp;  // [n][Kmax*3]
};
struct BMArgs {
  BMDir d[2];
  int step;
};

constexpr int BM_THREADS = 64;
#ifndef BM_MINBLOCKS
#define BM_MINBLOCKS 12
#endif

// ---- Block matching, per 4x4x4 reference block: NCC of the block against the 343 (7^3) shifted blocks of the warped
// image.  Two stages:
//  A) approximate NCC of all candidates from box sums (separable window sums of the shifted patch d = w - c, c = block
//     mean) and a sliding correlation with the reference block: ~3.5x fewer instructions than evaluating the CPU two-pass
//     formula for every candidate;
//  B) every candidate whose approximate NCC is within BM_MARGIN of the maximum is re-evaluated with the CPU's two-pass
//     float arithmetic (sequential sums in voxel order) and the CPU's "first maximum with 1e-7 hysteresis" rule is
//     applied to them.  The winner is therefore the one the CPU rule selects whenever the approximation error is below
//     BM_MARGIN (it is ~1e-6).
__device__ unsigned long long g_bm_stats[8];
#ifndef BM_MARGIN_OVERRIDE
#define BM_MARGIN_OVERRIDE 2e-5f
#endif
constexpr float BM_MARGIN = BM_MARGIN_OVERRIDE;

// General (rare) path: some voxels of the reference block or of the candidate window are undefined (NaN).
__device__ __noinline__ float cand_cc_general(const float* sP, const float* sR, int ox, int oy, int oz) {
  float rm = 0.f, wm = 0.f, vn = 0.f;
  for (int a = 0; a < 64; ++a) {
    int zz = a >> 4, yy = (a >> 2) & 3, xx = a & 3;
    float w = sP[((zz + oz) * 10 + (yy + oy)) * 12 + xx + ox];
    float r = sR[a];
    if (r == r && w == w) {
      rm += r;
      wm += w;
      vn += 1.f;
    }
  }
  if (!(vn > 32.f)) return -1.f;
  rm /= vn;
  wm /= vn;
  float rv = 0.f, wv = 0.f, cc = 0.f;
  for (int a = 0; a < 64; ++a) {
    int zz = a >> 4, yy = (a >> 2) & 3, xx = a & 3;
    float w = sP[((zz + oz) * 10 + (yy + oy)) * 12 + xx + ox];
    float r = sR[a];
    if (r == r && w == w) {
      float rt = r - rm, wt = w - wm;
      rv += rt * rt;
      wv += wt * wt;
      cc += rt * wt;
    }
  }
  float prod = rv * wv;
  return prod > 0.f ? fabsf(cc / sqrtf(prod)) : 0.f;
}

// CPU two-pass float arithmetic for one candidate (reference block fully defined).
__device__ inline float cand_cc_exact(const float* sP, const float* sR, const float4* sRc4, bool refAll, float refVar, int ox, int oy, int oz) {
  if (!refAll) return cand_cc_general(sP, sR, ox, oy, oz);
  float s = 0.f;
#pragma unroll 1
  for (int r = 0; r < 16; ++r) {
    const float* row = sP + (((r >> 2) + oz) * 10 + ((r & 3) + oy)) * 12 + ox;
    s += row[0];
    s += row[1];
    s += row[2];
    s += row[3];
  }
  if (s != s) return cand_cc_general(sP, sR, ox, oy, oz);
  const float wm = s / 64.f;
  float wv = 0.f, cc = 0.f;
#pragma unroll 1
  for (int r = 0; r < 16; ++r) {
    const float* row = sP + (((r >> 2) + oz) * 10 + ((r & 3) + oy)) * 12 + ox;
    float4 rt = sRc4[r];
    float t0 = row[0] - wm, t1 = row[1] - wm, t2 = row[2] - wm, t3 = row[3] - wm;
    wv = fmaf(t0, t0, wv); cc = fmaf(rt.x, t0, cc);
    wv = fmaf(t1, t1, wv); cc = fmaf(rt.y, t1, cc);
    wv = fmaf(t2, t2, wv); cc = fmaf(rt.z, t2, cc);
    wv = fmaf(t3, t3, wv); cc = fmaf(rt.w, t3, cc);
  }
  float prod = refVar * wv;
  return prod > 0.f ? fabsf(cc / sqrtf(prod)) : 0.f;
}

template <int STEP>
__global__ void __launch_bounds__(BM_THREADS, BM_MINBLOCKS) bm_kernel(BMArgs a) {
  const int d = blockIdx.z, t = blockIdx.y, rank = blockIdx.x;
  const BMDir& P = a.d[d];
  const int K = P.K[t * P.Kstride];
  if (rank >= K) return;
  __shared__ __align__(16) float sP[1200];  // 100 rows (z,y) x pitch 12
  __shared__ float sBd[490], sBq[490];  // 2-D (x,y) window sums of d and d^2 for each of the 10 slices: [pz][y'][x']
  __shared__ float sBc[490];            // number of defined voxels in each 2-D window
  __shared__ float sR[64];
  __shared__ __align__(16) float sRc[64];
  __shared__ float sCC[343], sLb[343];
  __shared__ int sList[343];
  __shared__ float sEx[343];
  __shared__ float sInfo[4];  // refVar, refAll, mean, sum(rt)
  __shared__ int sNc;
  const int tid = threadIdx.x;
  const int bidx = P.active[t * P.activeStride + rank];
  const int bi = bidx % P.bnx, bj = (bidx / P.bnx) % P.bny, bk = bidx / (P.bnx * P.bny);
  const int x0 = bi * 4, y0 = bj * 4, z0 = bk * 4;
  const float* ref = P.ref + (size_t)t * P.refStride;
  const float* wr = P.warped + (size_t)t * P.pvox;
  {
    int xx = tid & 3, yy = (tid >> 2) & 3, zz = tid >> 4;
    int x = x0 + xx, y = y0 + yy, z = z0 + zz;
    float v = nanf("");
    if (x < P.nx && y < P.ny && z < P.nz) v = ref[((size_t)z * P.ny + y) * P.nx + x];
    sR[tid] = v;
  }
  // patch of the warped image: 100 rows (z,y) of 12 floats from the NaN-padded image, 16-byte loads
  for (int i = tid; i < 300; i += BM_THREADS) {
    const int row = i / 3, q = i - row * 3;
    const int pz = row / 10, py = row - pz * 10;
    const float4* src = reinterpret_cast<const float4*>(wr + ((size_t)(z0 + pz) * P.pny + (y0 + py)) * P.pnx + x0) + q;
    reinterpret_cast<float4*>(sP)[row * 3 + q] = __ldg(src);
  }
  __syncthreads();
  if (tid == 0) {  // reference block statistics in the order of the CPU code
    bool all = true;
    for (int i = 0; i < 64; ++i) all = all && (sR[i] == sR[i]);
    float rv = 0.f, m = 0.f, srt = 0.f;
    if (all) {
      float s = 0.f;
      for (int i = 0; i < 64; ++i) s += sR[i];
      m = s / 64.f;
      for (int i = 0; i < 64; ++i) {
        float rt = sR[i] - m;
        sRc[i] = rt;
        rv = fmaf(rt, rt, rv);
        srt += rt;
      }
    }
    sInfo[0] = rv;
    sInfo[1] = all ? 1.f : 0.f;
    sInfo[2] = m;
    sInfo[3] = srt;
  }
  __syncthreads();
  const float refVar = sInfo[0];
  const bool refAll = sInfo[1] != 0.f;
  const float cshift = sInfo[2];
  const float srt = sInfo[3];
  constexpr int step = STEP;
  constexpr int cnt = 6 / STEP + 1;
  constexpr int ncand = cnt * cnt * cnt;
  const float4* sRc4 = reinterpret_cast<const float4*>(sRc);

  if (refAll) {
    // ---- 2-D window sums of d = w - c and d^2 (items: slice pz, row offset y')
    for (int item = tid; item < 70; item += BM_THREADS) {
      const int pz = item / 7, yp = item - pz * 7;
      float bd[7], bq[7], bc[7];
#pragma unroll
      for (int k = 0; k < 7; ++k) { bd[k] = 0.f; bq[k] = 0.f; bc[k] = 0.f; }
#pragma unroll
      for (int dy = 0; dy < 4; ++dy) {
        const float* row = sP + (pz * 10 + yp + dy) * 12;
        float dv[10], qv[10], mv[10];
#pragma unroll
        for (int k = 0; k < 10; ++k) {
          const float v = row[k];
          dv[k] = v - cshift;
          qv[k] = dv[k] * dv[k];
          mv[k] = (v == v) ? 1.f : 0.f;
        }
#pragma unroll
        for (int k = 0; k < 7; ++k) {
          bd[k] += (dv[k] + dv[k + 1]) + (dv[k + 2] + dv[k + 3]);
          bq[k] += (qv[k] + qv[k + 1]) + (qv[k + 2] + qv[k + 3]);
          bc[k] += (mv[k] + mv[k + 1]) + (mv[k + 2] + mv[k + 3]);
        }
      }
#pragma unroll
      for (int k = 0; k < 7; ++k) {
        sBd[item * 7 + k] = bd[k];
        sBq[item * 7 + k] = bq[k];
        sBc[item * 7 + k] = bc[k];
      }
    }
  }
  __syncthreads();

  // ---- stage A: thread (m,n) evaluates the CNT candidates along x (sliding over the 16 rows of the block)
  if (tid < cnt * cnt) {
    const int mi = tid % cnt, ni = tid / cnt;
    const int oy = mi * step, oz = ni * step;
    float cc[7], lbv[7];
    if (!refAll) {
#pragma unroll 1
      for (int l = 0; l < cnt; ++l) { cc[l] = cand_cc_general(sP, sR, l * step, oy, oz); lbv[l] = cc[l]; }
    } else {
      float cr[7];
#pragma unroll
      for (int l = 0; l < 7; ++l) cr[l] = 0.f;
#pragma unroll 1
      for (int r = 0; r < 16; ++r) {
        const float* row = sP + (((r >> 2) + oz) * 10 + ((r & 3) + oy)) * 12;
        float v[10];
#pragma unroll
        for (int k = 0; k < 10; ++k) v[k] = row[k];
        const float4 rt = sRc4[r];
#pragma unroll
        for (int l = 0; l < 7; ++l) {
          if (l < cnt) {  // l*step + 3 <= 9 always holds for step in {1,2,3}
            const int o = l * step;
            cr[l] = fmaf(rt.x, v[o], cr[l]);
            cr[l] = fmaf(rt.y, v[o + 1], cr[l]);
            cr[l] = fmaf(rt.z, v[o + 2], cr[l]);
            cr[l] = fmaf(rt.w, v[o + 3], cr[l]);
          }
        }
      }
#pragma unroll
      for (int l = 0; l < 7; ++l) {
        if (l < cnt) {
          const int o = l * step;
          float sd = 0.f, sq = 0.f, sc = 0.f;
#pragma unroll
          for (int dz = 0; dz < 4; ++dz) {
            const int idx = ((oz + dz) * 7 + oy) * 7 + o;
            sd += sBd[idx];
            sq += sBq[idx];
            sc += sBc[idx];
          }
          if (sc < 64.f) {
            // window contains undefined voxels: with at most 32 voxels in common with the (fully defined) reference
            // block the CPU code ignores the candidate, otherwise the overlap-masked formula is evaluated
            cc[l] = (sc > 32.f) ? cand_cc_general(sP, sR, o, oy, oz) : -1.f;
            lbv[l] = cc[l];
          } else {
            const float wm = cshift + sd * (1.f / 64.f);
            const float V = sq - sd * sd * (1.f / 64.f);
            const float cross = cr[l] - wm * srt;
            {  // rigorous-ish bounds: summation error of V below 8e-7*sq, of the cross term 3e-6 relative
              const float E = 8e-7f * sq;
              const float acr = fabsf(cross);
              const float Vh = V + E, Vl = V - E;
              lbv[l] = (Vh > 0.f && refVar > 0.f) ? fmaxf(acr * (1.f - 3e-6f) * rsqrtf(refVar * Vh), 0.f) : 0.f;
              cc[l] = (Vl > 1e-30f && refVar > 0.f) ? fminf(acr * (1.f + 3e-6f) * rsqrtf(refVar * Vl), 1.0001f) : 1.0001f;
              if (!(refVar > 0.f)) { lbv[l] = 0.f; cc[l] = 0.f; }
            }
          }
        }
      }
    }
#pragma unroll
    for (int l = 0; l < cnt; ++l) { sCC[tid * cnt + l] = cc[l]; sLb[tid * cnt + l] = lbv[l]; }
  }
  __syncthreads();

  // ---- stage B: contenders (approximate NCC within BM_MARGIN of the maximum) re-evaluated with the CPU arithmetic
  if (tid < 32) {
    const int per = (ncand + 31) / 32;
    const int lo = tid * per, hi = min(lo + per, ncand);
    float mx = -2.f;
    for (int c = lo; c < hi; ++c) mx = fmaxf(mx, sLb[c]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    const float thr = mx - BM_MARGIN;
    int myc = 0;
    for (int c = lo; c < hi; ++c) myc += (sCC[c] >= thr && mx > 0.f);
    int incl = myc;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      int nv = __shfl_up_sync(0xffffffffu, incl, o);
      if (tid >= o) incl += nv;
    }
    int pos = incl - myc;
    for (int c = lo; c < hi; ++c)
      if (sCC[c] >= thr && mx > 0.f) sList[pos++] = c;
    if (tid == 31) sNc = incl;
    __syncwarp();
    const int nc = __shfl_sync(0xffffffffu, incl, 31);
#ifdef ALADIN_STATS
    if (tid == 0) { atomicAdd(&g_bm_stats[0], 1ull); atomicAdd(&g_bm_stats[1], (unsigned long long)nc); atomicAdd(&g_bm_stats[2], (unsigned long long)(refAll ? 0 : 1)); atomicAdd(&g_bm_stats[3 + (nc == 0 ? 0 : nc == 1 ? 1 : nc <= 4 ? 2 : nc <= 32 ? 3 : 4)], 1ull); }
#endif
    // a single contender wins whatever its exact value is (every other candidate is below it by more than the
    // approximation error), provided its NCC is positive
    if (nc > 1) {
      for (int k = tid; k < nc; k += 32) {
        const int c = sList[k];
        const int il = c % cnt, im = (c / cnt) % cnt, in_ = c / (cnt * cnt);
        sEx[k] = cand_cc_exact(sP, sR, sRc4, refAll, refVar, il * step, im * step, in_ * step);
      }
    } else if (nc == 1 && tid == 0) {
      sEx[0] = 1.f;  // placeholder > 0; mx (largest lower bound) > 0 was required for it to be listed
    }
    __syncwarp();
    if (tid == 0) {
      float bestCC = 0.f;
      int best = -1;
      for (int k = 0; k < nc; ++k) {
        float v = sEx[k];
        if (v > bestCC) {
          bestCC = v + 1.0e-7f;
          best = sList[k];
        }
      }
      float rp[3] = {(float)x0, (float)y0, (float)z0};
      float bp[3];
      if (best >= 0) {
        int il = best % cnt, im = (best / cnt) % cnt, in_ = best / (cnt * cnt);
        bp[0] = (float)(-3 + step * il) + rp[0];
        bp[1] = (float)(-3 + step * im) + rp[1];
        bp[2] = (float)(-3 + step * in_) + rp[2];
      } else {
        bp[0] = nanf("");
        bp[1] = 0.f + rp[1];
        bp[2] = 0.f + rp[2];
      }
      const float* X = P.refXyz + (size_t)t * P.refXyzStride;
      float* orf = P.outRef + ((size_t)t * P.Kmax + rank) * 3;
      float* owp = P.outWarp + ((size_t)t * P.Kmax + rank) * 3;
      for (int r = 0; r < 3; ++r) {
        orf[r] = (float)((double)X[r * 4] * rp[0] + (double)X[r * 4 + 1] * rp[1] + (double)X[r * 4 + 2] * rp[2] + (double)X[r * 4 + 3]);
        owp[r] = (float)((double)X[r * 4] * bp[0] + (double)X[r * 4 + 1] * bp[1] + (double)X[r * 4 + 2] * bp[2] + (double)X[r * 4 + 3]);
      }
    }
  }
}

// ---------------------------------------------------------------- LTS (optimize_3D)
constexpr int LTS_THREADS = 256;
constexpr int LTS_NV = 22;
#ifndef LTS_DP_MAX_K_OVERRIDE
#define LTS_DP_MAX_K_OVERRIDE 2048
#endif
constexpr int LTS_DP_MAX_K = LTS_DP_MAX_K_OVERRIDE;  // levels with at most this many correspondences use the double precision solver

struct LtsArgs {
  const float* bmRef[2];
  const float* bmWarp[2];
  int Kmax[2];
  const int* K[2];
  int Kstride[2];
  const float* Mcur[2];  // [n][16]
  float* Mnew[2];        // [n][16]
  float* scratch;
  size_t scratchStride;  // floats per (d,t)
  int Kall;
  int n;
  int affine;
  int percent;
  int* status;
};

// 3x3 SVD by one-sided Jacobi (Hestenes): H = U diag(S) V^T, in precision T (double for the coarse level, where the
// CPU reference is followed closely; float for the large levels, fp64 is 1/64 rate on this GPU).
template <typename T>
__device__ void svd3t(const T* H, T* U, T* S, T* V) {
  const T eps = sizeof(T) == 8 ? (T)1e-15 : (T)1e-7;
  T A[9];
  for (int i = 0; i < 9; ++i) { A[i] = H[i]; V[i] = (i % 4 == 0) ? (T)1 : (T)0; }
  for (int sweep = 0; sweep < 40; ++sweep) {
    bool rotated = false;
    for (int p = 0; p < 2; ++p)
      for (int q = p + 1; q < 3; ++q) {
        T alpha = 0, beta = 0, gamma = 0;
        for (int i = 0; i < 3; ++i) {
          alpha += A[i * 3 + p] * A[i * 3 + p];
          beta += A[i * 3 + q] * A[i * 3 + q];
          gamma += A[i * 3 + p] * A[i * 3 + q];
        }
        if (gamma == (T)0 || fabs((double)gamma) <= (double)eps * sqrt((double)(alpha * beta))) continue;
        rotated = true;
        T zeta = (beta - alpha) / ((T)2 * gamma);
        T tt = (zeta >= 0 ? (T)1 : (T)-1) / (fabs(zeta) + sqrt((T)1 + zeta * zeta));
        T c = (T)1 / sqrt((T)1 + tt * tt), s = c * tt;
        for (int i = 0; i < 3; ++i) {
          T ap = A[i * 3 + p], aq = A[i * 3 + q];
          A[i * 3 + p] = c * ap - s * aq;
          A[i * 3 + q] = s * ap + c * aq;
          T vp = V[i * 3 + p], vq = V[i * 3 + q];
          V[i * 3 + p] = c * vp - s * vq;
          V[i * 3 + q] = s * vp + c * vq;
        }
      }
    if (!rotated) break;
  }
  for (int j = 0; j < 3; ++j) {
    T nrm = sqrt(A[j] * A[j] + A[3 + j] * A[3 + j] + A[6 + j] * A[6 + j]);
    S[j] = nrm;
    for (int i = 0; i < 3; ++i) U[i * 3 + j] = nrm > (T)1e-30 ? A[i * 3 + j] / nrm : (i == j ? (T)1 : (T)0);
  }
}

// Moments (accumulated on shifted coordinates p-c0p, q-c0q; see lts_kernel):
// tot: [0..5] = xx,xy,xz,yy,yz,zz of p'   [6..8] = sum p'   [9] = n   [10..18] = sum p'_i q'_j (i,j = 0..2)   [19..21] = sum q'
// c0: [0..2] shift of p, [3..5] shift of q
__device__ void fit_affine(const double* tot, const double* c0, float* out) {
  const double n = tot[9];
  double mp[3] = {tot[6] / n, tot[7] / n, tot[8] / n}, mq[3] = {tot[19] / n, tot[20] / n, tot[21] / n};
  double C[3][3] = {{tot[0] - n * mp[0] * mp[0], tot[1] - n * mp[0] * mp[1], tot[2] - n * mp[0] * mp[2]},
                    {0, tot[3] - n * mp[1] * mp[1], tot[4] - n * mp[1] * mp[2]},
                    {0, 0, tot[5] - n * mp[2] * mp[2]}};
  C[1][0] = C[0][1]; C[2][0] = C[0][2]; C[2][1] = C[1][2];
  double M[3][6];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      M[i][j] = C[i][j];
      M[i][3 + j] = tot[10 + i * 3 + j] - n * mp[i] * mq[j];
    }
  }
  for (int c = 0; c < 3; ++c) {
    int p = c;
    double best = fabs(M[c][c]);
    for (int r = c + 1; r < 3; ++r)
      if (fabs(M[r][c]) > best) { best = fabs(M[r][c]); p = r; }
    if (p != c)
      for (int j = 0; j < 6; ++j) { double tmp = M[c][j]; M[c][j] = M[p][j]; M[p][j] = tmp; }
    double ip = 1.0 / M[c][c];
    for (int j = c; j < 6; ++j) M[c][j] *= ip;
    for (int r = 0; r < 3; ++r) {
      if (r == c) continue;
      double f = M[r][c];
      if (f != 0.0)
        for (int j = c; j < 6; ++j) M[r][j] -= f * M[c][j];
    }
  }
  // X = M[:,3..5]: q_j - mq_j = sum_i X[i][j] (p_i - mp_i)  => A[j][i] = X[i][j]
  for (int j = 0; j < 3; ++j) {
    double t = c0[3 + j] + mq[j];
    for (int i = 0; i < 3; ++i) {
      out[j * 4 + i] = (float)M[i][3 + j];
      t -= M[i][3 + j] * (c0[i] + mp[i]);
    }
    out[j * 4 + 3] = (float)t;
  }
  out[12] = 0.f; out[13] = 0.f; out[14] = 0.f; out[15] = 1.f;
}
template <bool DP>
__device__ void fit_rigid(const double* tot, const double* c0, float* out) {
  typedef typename std::conditional<DP, double, float>::type T;
  const double n = tot[9];
  double mp[3] = {tot[6] / n, tot[7] / n, tot[8] / n}, mq[3] = {tot[19] / n, tot[20] / n, tot[21] / n};
  float c1[3], c2[3];
  for (int i = 0; i < 3; ++i) {
    c1[i] = (float)(c0[i] + mp[i]);
    c2[i] = (float)(c0[3 + i] + mq[i]);
  }
  // H = sum (p-c1)(q-c2)^T (the matrix `u` of the CPU code, stored as float there); float SVD input is normalised by n
  T H[9];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      double h = tot[10 + i * 3 + j] - n * mp[i] * mq[j];
      H[i * 3 + j] = DP ? (T)(double)(float)h : (T)(float)(h / n);
    }
  T U[9], S[3], V[9];
  svd3t<T>(H, U, S, V);
  int imin = 0;
  for (int j = 1; j < 3; ++j)
    if (S[j] < S[imin]) imin = j;
  if (DP) {  // the CPU code stores U and V as float
    for (int k = 0; k < 9; ++k) { U[k] = (T)(double)(float)U[k]; V[k] = (T)(double)(float)V[k]; }
  }
  T R[9];
  auto compose = [&]() {
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        T r = V[i * 3] * U[j * 3] + V[i * 3 + 1] * U[j * 3 + 1] + V[i * 3 + 2] * U[j * 3 + 2];
        R[i * 3 + j] = DP ? (T)(double)(float)r : r;
      }
  };
  compose();
  T det = R[0] * (R[4] * R[8] - R[5] * R[7]) - R[1] * (R[3] * R[8] - R[5] * R[6]) + R[2] * (R[3] * R[7] - R[4] * R[6]);
  if (det < (T)0) {
    for (int i = 0; i < 3; ++i) V[i * 3 + imin] = -V[i * 3 + imin];
    compose();
  }
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) out[i * 4 + j] = (float)R[i * 3 + j];
    out[i * 4 + 3] = (float)((double)c2[i] - ((double)out[i * 4] * (double)c1[0] + (double)out[i * 4 + 1] * (double)c1[1] + (double)out[i * 4 + 2] * (double)c1[2]));
  }
  out[12] = 0.f; out[13] = 0.f; out[14] = 0.f; out[15] = 1.f;
}

// DP = true: double precision accumulation, SVD and transforms (follows the CPU arithmetic closely; used for the coarse
// level where there are only ~1000 correspondences).  DP = false: single precision (large levels).
template <bool DP>
__global__ void __launch_bounds__(LTS_THREADS) lts_kernel(LtsArgs a) {
  typedef typename std::conditional<DP, double, float>::type AccT;
  const int t = blockIdx.x, d = blockIdx.y, tid = threadIdx.x;
  typedef cub::BlockScan<int, LTS_THREADS> BS;
  __shared__ typename BS::TempStorage ts;
  __shared__ double sred[(LTS_THREADS / 32) * LTS_NV];
  __shared__ double sTot[LTS_NV];
  __shared__ double sC0[6];
  __shared__ float sFinal[16], sLast[16];
  __shared__ unsigned hist[256];
  __shared__ unsigned sB, sRem, sEq;
  __shared__ int sBase, sStop;
  __shared__ double sLastDist;

  const int Kd = a.K[d][t * a.Kstride[d]];
  const int Kmax = a.Kmax[d];
  const float* bmRef = a.bmRef[d] + (size_t)t * Kmax * 3;
  const float* bmWarp = a.bmWarp[d] + (size_t)t * Kmax * 3;
  const float* Mc = a.Mcur[d] + t * 16;
  float* S = a.scratch + ((size_t)d * a.n + t) * a.scratchStride;
  float* P1 = S;
  float* P2 = S + 3 * (size_t)a.Kall;
  float* key = S + 6 * (size_t)a.Kall;
  unsigned char* sel = (unsigned char*)(S + 7 * (size_t)a.Kall);
  float* Mn = a.Mnew[d] + t * 16;

  // ---- compaction of defined correspondences + transformation of the warped positions by the current matrix
  if (tid == 0) sBase = 0;
  __syncthreads();
  for (int c0 = 0; c0 < Kd; c0 += LTS_THREADS) {
    int i = c0 + tid;
    bool v = false;
    float wx = 0, wy = 0, wz = 0;
    if (i < Kd) {
      wx = bmWarp[3 * i]; wy = bmWarp[3 * i + 1]; wz = bmWarp[3 * i + 2];
      v = (wx == wx);
    }
    int pos, tot;
    BS(ts).ExclusiveSum(v ? 1 : 0, pos, tot);
    if (v) {
      int o = sBase + pos;
      P1[3 * o] = bmRef[3 * i]; P1[3 * o + 1] = bmRef[3 * i + 1]; P1[3 * o + 2] = bmRef[3 * i + 2];
      // reg_mat44_mul(float): double accumulation, float result (only 1 + 9 DP ops per correspondence)
      P2[3 * o] = (float)((double)Mc[0] * wx + (double)Mc[1] * wy + (double)Mc[2] * wz + (double)Mc[3]);
      P2[3 * o + 1] = (float)((double)Mc[4] * wx + (double)Mc[5] * wy + (double)Mc[6] * wz + (double)Mc[7]);
      P2[3 * o + 2] = (float)((double)Mc[8] * wx + (double)Mc[9] * wy + (double)Mc[10] * wz + (double)Mc[11]);
    }
    __syncthreads();
    if (tid == 0) sBase += tot;
    __syncthreads();
  }
  const int n = sBase;
  const int need = a.affine ? 8 : 4;
  if (n < need) {
    if (tid < 16) Mn[tid] = Mc[tid];
    if (tid == 0) a.status[d * a.n + t] = 1;
    return;
  }
  const unsigned ntk = (unsigned)((float)n * ((float)a.percent / 100.0f));
  if (ntk < 1) {
    if (tid < 16) Mn[tid] = Mc[tid];
    if (tid == 0) a.status[d * a.n + t] = 2;
    return;
  }

  // ---- shift for the moment accumulation (centroid of all correspondences, once)
  {
    double v[6] = {0, 0, 0, 0, 0, 0};
    for (int i = tid; i < n; i += LTS_THREADS) {
      v[0] += P1[3 * i]; v[1] += P1[3 * i + 1]; v[2] += P1[3 * i + 2];
      v[3] += P2[3 * i]; v[4] += P2[3 * i + 1]; v[5] += P2[3 * i + 2];
    }
    for (int k = 0; k < 6; ++k) {
      double x = v[k];
      for (int o = 16; o > 0; o >>= 1) x += __shfl_down_sync(0xffffffffu, x, o);
      if ((tid & 31) == 0) sred[(tid >> 5) * 6 + k] = x;
    }
    __syncthreads();
    if (tid < 6) {
      double x = 0;
      for (int w = 0; w < LTS_THREADS / 32; ++w) x += sred[w * 6 + tid];
      sC0[tid] = (double)(float)(x / n);
    }
    __syncthreads();
  }
  const float c0p0 = (float)sC0[0], c0p1 = (float)sC0[1], c0p2 = (float)sC0[2];
  const float c0q0 = (float)sC0[3], c0q1 = (float)sC0[4], c0q2 = (float)sC0[5];

  auto accumulate = [&](bool useSel) {
    AccT v[LTS_NV];
#pragma unroll
    for (int k = 0; k < LTS_NV; ++k) v[k] = 0;
    for (int i = tid; i < n; i += LTS_THREADS) {
      if (useSel && !sel[i]) continue;
      AccT x = (AccT)P1[3 * i] - (AccT)c0p0, y = (AccT)P1[3 * i + 1] - (AccT)c0p1, z = (AccT)P1[3 * i + 2] - (AccT)c0p2;
      AccT q0 = (AccT)P2[3 * i] - (AccT)c0q0, q1 = (AccT)P2[3 * i + 1] - (AccT)c0q1, q2 = (AccT)P2[3 * i + 2] - (AccT)c0q2;
      v[0] += x * x; v[1] += x * y; v[2] += x * z;
      v[3] += y * y; v[4] += y * z; v[5] += z * z;
      v[6] += x; v[7] += y; v[8] += z; v[9] += 1;
      v[10] += x * q0; v[11] += x * q1; v[12] += x * q2;
      v[13] += y * q0; v[14] += y * q1; v[15] += y * q2;
      v[16] += z * q0; v[17] += z * q1; v[18] += z * q2;
      v[19] += q0; v[20] += q1; v[21] += q2;
    }
#pragma unroll
    for (int k = 0; k < LTS_NV; ++k) {
      AccT x = v[k];
      for (int o = 16; o > 0; o >>= 1) x += __shfl_down_sync(0xffffffffu, x, o);
      if ((tid & 31) == 0) sred[(tid >> 5) * LTS_NV + k] = (double)x;
    }
    __syncthreads();
    if (tid < LTS_NV) {
      double x = 0;
      for (int w = 0; w < LTS_THREADS / 32; ++w) x += sred[w * LTS_NV + tid];
      sTot[tid] = x;
    }
    __syncthreads();
  };
  auto fit = [&](float* dst) {
    if (tid == 0) {
      if (a.affine) fit_affine(sTot, sC0, dst); else fit_rigid<DP>(sTot, sC0, dst);
    }
    __syncthreads();
  };

  // ---- initial fit with all correspondences
  accumulate(false);
  fit(sFinal);
  if (tid < 16) sLast[tid] = 0.f;
  if (tid == 0) sLastDist = DBL_MAX;
  __syncthreads();

  int lts_iters_dbg = 0;
  for (int count = 0; count < 30; ++count) {
    ++lts_iters_dbg;
    // distances of the transformed reference points to the (transformed) warped positions (reg_mat44_mul result
    // rounded to float; evaluated here with float fma, i.e. within 1 ulp of the CPU value)
    {
      const float f0 = sFinal[0], f1 = sFinal[1], f2 = sFinal[2], f3 = sFinal[3], f4 = sFinal[4], f5 = sFinal[5], f6 = sFinal[6],
                  f7 = sFinal[7], f8 = sFinal[8], f9 = sFinal[9], f10 = sFinal[10], f11 = sFinal[11];
      for (int i = tid; i < n; i += LTS_THREADS) {
        float x = P1[3 * i], y = P1[3 * i + 1], z = P1[3 * i + 2];
        // single precision (within 1 ulp of the CPU's double-accumulated, float-rounded value; changing this to the CPU
        // evaluation order does not change the registration trajectory)
        float nx = fmaf(f0, x, fmaf(f1, y, fmaf(f2, z, f3)));
        float ny = fmaf(f4, x, fmaf(f5, y, fmaf(f6, z, f7)));
        float nz = fmaf(f8, x, fmaf(f9, y, fmaf(f10, z, f11)));
        float dx = nx - P2[3 * i], dy = ny - P2[3 * i + 1], dz = nz - P2[3 * i + 2];
        float ssq = dx * dx;
        ssq = ssq + dy * dy;
        ssq = ssq + dz * dz;
        key[i] = sqrtf(ssq);
      }
    }
    __syncthreads();
    // k-th smallest key by 4-pass radix select
    unsigned prefix = 0, remk = ntk;
    for (int pass = 0; pass < 4; ++pass) {
      const int shift = 24 - 8 * pass;
      hist[tid] = 0;  // LTS_THREADS == 256
      __syncthreads();
      for (int i = tid; i < n; i += LTS_THREADS) {
        unsigned kb = __float_as_uint(key[i]);
        if (pass == 0 || (kb >> (shift + 8)) == prefix) atomicAdd(&hist[(kb >> shift) & 255u], 1u);
      }
      __syncthreads();
      {
        int h = (int)hist[tid], incl, tot_;
        BS(ts).InclusiveSum(h, incl, tot_);
        int excl = incl - h;
        if (excl < (int)remk && incl >= (int)remk) {
          sB = tid; sRem = remk - excl; sEq = h;
        }
        __syncthreads();
      }
      prefix = (prefix << 8) | sB;
      remk = sRem;
      __syncthreads();
    }
    const unsigned kv = prefix;
    const unsigned eqTotal = sEq;
    // selection (ties resolved by index order, like the stable multimap of the CPU code)
    float dsum = 0.f, dcomp = 0.f;  // Neumaier-compensated per-thread sum
    auto addd = [&](float x) {
      float tt = dsum + x;
      dcomp += (fabsf(dsum) >= fabsf(x)) ? ((dsum - tt) + x) : ((x - tt) + dsum);
      dsum = tt;
    };
    if (remk == eqTotal) {
      for (int i = tid; i < n; i += LTS_THREADS) {
        unsigned kb = __float_as_uint(key[i]);
        bool s = kb <= kv;
        sel[i] = s;
        if (s) addd(key[i]);
      }
    } else {
      int running = 0;
      for (int c0 = 0; c0 < n; c0 += LTS_THREADS) {
        int i = c0 + tid;
        unsigned kb = i < n ? __float_as_uint(key[i]) : 0u;
        bool eq = i < n && kb == kv;
        int pos, tot;
        BS(ts).ExclusiveSum(eq ? 1 : 0, pos, tot);
        if (i < n) {
          bool s = kb < kv || (eq && (unsigned)(running + pos) < remk);
          sel[i] = s;
          if (s) addd(key[i]);
        }
        running += tot;
        __syncthreads();
      }
    }
    {
      double x = (double)dsum + (double)dcomp;
      for (int o = 16; o > 0; o >>= 1) x += __shfl_down_sync(0xffffffffu, x, o);
      if ((tid & 31) == 0) sred[tid >> 5] = x;
      __syncthreads();
      if (tid == 0) {
        double tt = 0;
        for (int w = 0; w < LTS_THREADS / 32; ++w) tt += sred[w];
        bool stop = (tt > sLastDist) || (sLastDist - tt) < 0.001;
        sStop = stop;
        if (stop) {
          for (int k = 0; k < 16; ++k) sFinal[k] = sLast[k];
        } else {
          sLastDist = tt;
          for (int k = 0; k < 16; ++k) sLast[k] = sFinal[k];
        }
      }
      __syncthreads();
    }
    if (sStop) break;
    accumulate(true);
    fit(sFinal);
  }
  if (tid == 0 && a.status) atomicAdd(&a.status[2 * a.n + (a.affine ? 1 : 0)], lts_iters_dbg);
  if (tid < 16) Mn[tid] = sFinal[tid];
}

// ---------------------------------------------------------------- symmetric update
__global__ void __launch_bounds__(128) sym_update_kernel(const float* Mnew0, const float* Mnew1, float* T, float* B, FoldArgs fa) {
  // One CTA per template; the four matrix logarithms run in four warps (fp64 issue is per warp), then two exponentials.
  __shared__ Mat44 sTn, sBn, sIn[4];  // sIn: arguments of the four logarithms
  __shared__ Mat44 sLog[4];
  __shared__ Mat44 sOut[2];
  const int t = blockIdx.x, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x == 0) {
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) {
        sTn.m[i][j] = Mnew0[t * 16 + i * 4 + j];
        sBn.m[i][j] = Mnew1[t * 16 + i * 4 + j];
      }
    sIn[0] = sTn;                  // forward: avg2(T', inv(B'))
    sIn[1] = nifti_inverse(sBn);
    sIn[2] = nifti_inverse(sTn);   // backward: avg2(inv(T'), B')
    sIn[3] = sBn;
  }
  __syncthreads();
  if (lane == 0) sLog[w] = reg_logm(sIn[w]);
  __syncthreads();
  if (lane == 0 && w < 2) {
    Mat44 la = sLog[2 * w], lb = sLog[2 * w + 1];
    for (int i = 0; i < 4; ++i) la.m[3][i] = lb.m[3][i] = 0.f;
    Mat44 sm;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) {
        float sum = (float)((double)la.m[i][j] + (double)lb.m[i][j]);
        sm.m[i][j] = (float)((double)sum * 0.5);
      }
    Mat44 r = reg_expm(sm);
    for (int i = 0; i < 3; ++i) r.m[3][i] = 0.f;
    r.m[3][3] = 1.f;
    sOut[w] = r;
  }
  __syncthreads();
  if (threadIdx.x < 16) {
    const int i = threadIdx.x >> 2, j = threadIdx.x & 3;
    T[t * 16 + i * 4 + j] = sOut[0].m[i][j];
    B[t * 16 + i * 4 + j] = sOut[1].m[i][j];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    fold_one(fa, 0, t);
    fold_one(fa, 1, t);
  }
}

// ======================================================================================
// Host side
// ======================================================================================
struct TemplateSet::Impl {
  ImageSet set;
  Impl() {}
};
TemplateSet::TemplateSet(const Image* templates, int n, const Params& p, cudaStream_t stream) : impl_(new Impl) {
  impl_->set.tag = "tmpl";
  impl_->set.build(templates, n, p, stream);
}
TemplateSet::~TemplateSet() {}
int TemplateSet::size() const { return impl_->set.n; }
float TemplateSet::setup_ms() const { return impl_->set.ms; }

struct AladinBatch::Impl {
  Image target;
  Params p;
  cudaStream_t s;
  ImageSet tgt;
  // work buffers (grown on demand)
  int cap_n = 0;
  float *dT = nullptr, *dB = nullptr, *dMnew[2] = {nullptr, nullptr}, *dMf = nullptr, *dScratch = nullptr;
  int* dStatus = nullptr;
  float *dWarp[2] = {nullptr, nullptr}, *dBmRef[2] = {nullptr, nullptr}, *dBmWarp[2] = {nullptr, nullptr};
  size_t capWarp[2] = {0, 0}, capBm[2] = {0, 0}, capScratch = 0;
  cudaEvent_t ev[16];
  int nev = 0;

  void release() {
    cudaFree(dT); cudaFree(dB); cudaFree(dMnew[0]); cudaFree(dMnew[1]); cudaFree(dMf); cudaFree(dScratch); cudaFree(dStatus);
    for (int d = 0; d < 2; ++d) { cudaFree(dWarp[d]); cudaFree(dBmRef[d]); cudaFree(dBmWarp[d]); }
    dT = dB = dMf = dScratch = nullptr; dStatus = nullptr; dMnew[0] = dMnew[1] = nullptr;
    for (int d = 0; d < 2; ++d) { dWarp[d] = dBmRef[d] = dBmWarp[d] = nullptr; capWarp[d] = capBm[d] = 0; }
    capScratch = 0; cap_n = 0;
  }
};

AladinBatch::AladinBatch(const Image& target, const Params& p, cudaStream_t stream) : impl_(new Impl) {
  impl_->target = target;
  impl_->p = p;
  impl_->s = stream;
  impl_->tgt.tag = "tgt";
  impl_->tgt.build(&target, 1, p, stream);
  tm_.target_setup = impl_->tgt.ms;
  for (int i = 0; i < 16; ++i) CK(cudaEventCreate(&impl_->ev[i]));
}
AladinBatch::~AladinBatch() {
  if (impl_) {
    impl_->release();
    for (int i = 0; i < 16; ++i) cudaEventDestroy(impl_->ev[i]);
  }
}


void AladinBatch::run(const Image* templates, int n, Mat44* out_host, const Mat44* init) {
  TemplateSet ts(templates, n, impl_->p, impl_->s);
  run(ts, out_host, init);
}

// float version of reg_mat44_mul(mat44, float*, float*)
static void mul_pt(const Mat44& m, const float* in, float* out) {
  for (int i = 0; i < 3; ++i)
    out[i] = (float)((double)m.m[i][0] * in[0] + (double)m.m[i][1] * in[1] + (double)m.m[i][2] * in[2] + (double)m.m[i][3]);
}

void AladinBatch::run(const TemplateSet& tset, Mat44* out_host, const Mat44* init, float* out_dev) {
  Impl& I = *impl_;
  const ImageSet& TS = tset.impl_->set;
  const int n = TS.n;
  const int L = I.p.levels_to_perform;
  cudaStream_t s = I.s;
  tm_.template_setup = TS.ms;
  tm_.levels = L;

  // ---- allocate work buffers (grown on demand, kept between calls)
  if (n > I.cap_n) {
    I.release();
    I.cap_n = n;
    CK(cudaMalloc(&I.dT, n * 16 * sizeof(float)));
    CK(cudaMalloc(&I.dB, n * 16 * sizeof(float)));
    CK(cudaMalloc(&I.dMnew[0], n * 16 * sizeof(float)));
    CK(cudaMalloc(&I.dMnew[1], n * 16 * sizeof(float)));
    CK(cudaMalloc(&I.dMf, 2 * n * 36 * sizeof(float)));
    CK(cudaMalloc(&I.dStatus, (2 * n + 4) * sizeof(int)));
  }
  size_t needWarp[2] = {0, 0}, needBm[2] = {0, 0}, needScr = 0;
  for (int l = 0; l < L; ++l) {
    needWarp[0] = std::max(needWarp[0], (size_t)n * pad_geom(I.tgt.lv[l].nx, I.tgt.lv[l].ny, I.tgt.lv[l].nz).pvox);
    needWarp[1] = std::max(needWarp[1], (size_t)n * pad_geom(TS.lv[l].nx, TS.lv[l].ny, TS.lv[l].nz).pvox);
    needBm[0] = std::max(needBm[0], (size_t)n * I.tgt.lv[l].Kmax * 3);
    needBm[1] = std::max(needBm[1], (size_t)n * TS.lv[l].Kmax * 3);
    int Kall = std::max(I.tgt.lv[l].Kmax, TS.lv[l].Kmax);
    needScr = std::max(needScr, (size_t)(2 * n) * (8 * (size_t)Kall + 4));
  }
  for (int d = 0; d < 2; ++d) {
    if (needWarp[d] > I.capWarp[d]) { cudaFree(I.dWarp[d]); CK(cudaMalloc(&I.dWarp[d], needWarp[d] * sizeof(float))); I.capWarp[d] = needWarp[d]; }
    if (needBm[d] > I.capBm[d]) {
      cudaFree(I.dBmRef[d]); cudaFree(I.dBmWarp[d]);
      CK(cudaMalloc(&I.dBmRef[d], needBm[d] * sizeof(float)));
      CK(cudaMalloc(&I.dBmWarp[d], needBm[d] * sizeof(float)));
      I.capBm[d] = needBm[d];
    }
  }
  if (needScr > I.capScratch) { cudaFree(I.dScratch); CK(cudaMalloc(&I.dScratch, needScr * sizeof(float))); I.capScratch = needScr; }
  CK(cudaMemsetAsync(I.dStatus, 0, (2 * n + 4) * sizeof(int), s));

  // ---- initial transformation (host, tiny): align the centres of the full-resolution images; backward = inverse
  const ImageSet& TG = I.tgt;
  std::vector<float> hT(n * 16), hB(n * 16);
  for (int t = 0; t < n; ++t) {
    Mat44 T;
    if (init) {
      T = init[t];
    } else {
      std::memset(&T, 0, sizeof(T));
      for (int i = 0; i < 4; ++i) T.m[i][i] = 1.f;
      if (I.p.align_centre) {
        float fc[3] = {(float)TS.fnx / 2.0f, (float)TS.fny / 2.0f, (float)TS.fnz / 2.0f};
        float rc[3] = {(float)TG.fnx / 2.0f, (float)TG.fny / 2.0f, (float)TG.fnz / 2.0f};
        float fr[3], rr[3];
        mul_pt(TS.fxyz[t], fc, fr);
        mul_pt(TG.fxyz[0], rc, rr);
        for (int i = 0; i < 3; ++i) T.m[i][3] = fr[i] - rr[i];
      }
    }
    Mat44 B = nifti_inverse(T);
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) {
        hT[t * 16 + i * 4 + j] = T.m[i][j];
        hB[t * 16 + i * 4 + j] = B.m[i][j];
      }
  }
  CK(cudaMemcpyAsync(I.dT, hT.data(), n * 16 * sizeof(float), cudaMemcpyHostToDevice, s));
  CK(cudaMemcpyAsync(I.dB, hB.data(), n * 16 * sizeof(float), cudaMemcpyHostToDevice, s));
  CK(cudaStreamSynchronize(s));  // host buffers go out of scope

  CK(cudaEventRecord(I.ev[0], s));
  for (int l = 0; l < L; ++l) {
    const LevelData& tl = TG.lv[l];
    const LevelData& ml = TS.lv[l];

    FoldArgs fa;
    fa.refXyz[0] = tl.xyz; fa.refXyzStride[0] = 0;
    fa.floIjk[0] = ml.ijk; fa.floIjkStride[0] = 16;
    fa.Mcur[0] = I.dT;
    fa.refXyz[1] = ml.xyz; fa.refXyzStride[1] = 16;
    fa.floIjk[1] = tl.ijk; fa.floIjkStride[1] = 0;
    fa.Mcur[1] = I.dB;
    fa.Mf = I.dMf;
    fa.n = n;

    const PadGeom pt = pad_geom(tl.nx, tl.ny, tl.nz), pm = pad_geom(ml.nx, ml.ny, ml.nz);
    CK(cudaMemsetAsync(I.dWarp[0], 0xFF, (size_t)n * pt.pvox * sizeof(float), s));  // NaN borders
    CK(cudaMemsetAsync(I.dWarp[1], 0xFF, (size_t)n * pm.pvox * sizeof(float), s));
    WarpArgs wa;
    wa.d[0] = WarpDir{ml.data, ml.vox, ml.nx, ml.ny, ml.nz, I.dWarp[0], tl.nx, tl.ny, tl.nz, pt.pnx, pt.pny, pt.pvox};
    wa.d[1] = WarpDir{tl.data, 0, tl.nx, tl.ny, tl.nz, I.dWarp[1], ml.nx, ml.ny, ml.nz, pm.pnx, pm.pny, pm.pvox};
    wa.Mf = I.dMf;
    wa.n = n;

    BMArgs ba;
    ba.step = I.p.block_step;
    ba.d[0] = BMDir{tl.data, 0, I.dWarp[0], tl.xyz, 0, tl.active, 0, tl.K, 0, tl.nx, tl.ny, tl.nz, tl.bnx, tl.bny, tl.Kmax,
                    pt.pnx, pt.pny, pt.pvox, I.dBmRef[0], I.dBmWarp[0]};
    ba.d[1] = BMDir{ml.data, ml.vox, I.dWarp[1], ml.xyz, 16, ml.active, (size_t)ml.total, ml.K, 1, ml.nx, ml.ny, ml.nz, ml.bnx, ml.bny,
                    ml.Kmax, pm.pnx, pm.pny, pm.pvox, I.dBmRef[1], I.dBmWarp[1]};

    LtsArgs la;
    la.Kall = std::max(tl.Kmax, ml.Kmax);
    for (int d = 0; d < 2; ++d) {
      la.bmRef[d] = I.dBmRef[d];
      la.bmWarp[d] = I.dBmWarp[d];
      la.Mnew[d] = I.dMnew[d];
    }
    la.Kmax[0] = tl.Kmax; la.Kmax[1] = ml.Kmax;
    la.K[0] = tl.K; la.K[1] = ml.K;
    la.Kstride[0] = 0; la.Kstride[1] = 1;
    la.Mcur[0] = I.dT; la.Mcur[1] = I.dB;
    la.scratch = I.dScratch;
    la.scratchStride = 8 * (size_t)la.Kall + 4;
    la.n = n;
    la.percent = I.p.inlier_lts;
    la.status = I.dStatus;

    const int maxnx = std::max(tl.nx, ml.nx), maxny = std::max(tl.ny, ml.ny), maxnz = std::max(tl.nz, ml.nz);
    const int ntz = (maxnz + 3) / 4;
    const dim3 gWarp((maxnx + 7) / 8, (maxny + 7) / 8, (unsigned)ntz * n * 2);
    wa.maxnz = maxnz;
    wa.ntz = ntz;
    const dim3 gBm(std::max(tl.Kmax, ml.Kmax), n, 2);
    const dim3 gLts(n, 2);

    // iteration plan of reg_aladin::Run
    const int maxIt = (l == 0) ? I.p.max_iter * 2 : I.p.max_iter;
    int rigidIters = 0, affineIters = 0;
    if ((I.p.rigid && !I.p.affine) || (I.p.affine && I.p.rigid && l == 0)) {
      int ratio = (I.p.affine && I.p.rigid && l == 0) ? 4 : 1;
      rigidIters = maxIt * ratio;
    }
    if (I.p.affine) affineIters = maxIt;

    fold_kernel<<<(n + 31) / 32, 32, 0, s>>>(fa);
    static int traceit = 0;
    static const int dumpIt = getenv("ALADIN_DUMP_IT") ? atoi(getenv("ALADIN_DUMP_IT")) : -1;
    static int globalIt = -1;
    static const int trace = getenv("ALADIN_TRACE") ? atoi(getenv("ALADIN_TRACE")) : -1;
    for (int it = 0; it < rigidIters + affineIters; ++it) {
      la.affine = (it >= rigidIters) ? 1 : 0;
      ++globalIt;
      if (I.p.exact) warp_exact_kernel<<<gWarp, WARP_TX, 0, s>>>(wa); else warp_kernel<<<gWarp, WARP_TX, 0, s>>>(wa);
      if (I.p.block_step == 1) bm_kernel<1><<<gBm, BM_THREADS, 0, s>>>(ba);
      else if (I.p.block_step == 2) bm_kernel<2><<<gBm, BM_THREADS, 0, s>>>(ba);
      else bm_kernel<3><<<gBm, BM_THREADS, 0, s>>>(ba);
      if (dumpIt >= 0 && globalIt == dumpIt) {  // debug: dump everything of template 0 at this iteration
        CK(cudaStreamSynchronize(s));
        const char* dd = getenv("ALADIN_DUMP");
        auto dump = [&](const char* nm, const void* dev, size_t bytes) {
          std::vector<char> h(bytes);
          CK(cudaMemcpy(h.data(), dev, bytes, cudaMemcpyDeviceToHost));
          char fn[512];
          snprintf(fn, sizeof(fn), "%s/it%d_%s.bin", dd, dumpIt, nm);
          FILE* f = fopen(fn, "wb");
          fwrite(h.data(), 1, bytes, f);
          fclose(f);
        };
        auto dump_host = [&](const char* nm, const void* h, size_t bytes) {
          char fn[512];
          snprintf(fn, sizeof(fn), "%s/it%d_%s.bin", dd, dumpIt, nm);
          FILE* f = fopen(fn, "wb");
          fwrite(h, 1, bytes, f);
          fclose(f);
        };
        dump("T", I.dT, 64); dump("B", I.dB, 64);
        {
          for (int dd2 = 0; dd2 < 2; ++dd2) {
            const LevelData& L_ = dd2 == 0 ? tl : ml;
            const PadGeom& G_ = dd2 == 0 ? pt : pm;
            std::vector<float> hp(G_.pvox), hu(L_.vox);
            CK(cudaMemcpy(hp.data(), I.dWarp[dd2], G_.pvox * 4, cudaMemcpyDeviceToHost));
            for (int z = 0; z < L_.nz; ++z)
              for (int y = 0; y < L_.ny; ++y)
                for (int x = 0; x < L_.nx; ++x) hu[((size_t)z * L_.ny + y) * L_.nx + x] = hp[((size_t)(z + 3) * G_.pny + (y + 3)) * G_.pnx + (x + 3)];
            dump_host(dd2 == 0 ? "warp0" : "warp1", hu.data(), L_.vox * 4);
          }
        }
        dump("bmref0", I.dBmRef[0], (size_t)tl.Kmax * 12); dump("bmwarp0", I.dBmWarp[0], (size_t)tl.Kmax * 12);
        dump("bmref1", I.dBmRef[1], (size_t)ml.Kmax * 12); dump("bmwarp1", I.dBmWarp[1], (size_t)ml.Kmax * 12);
      }
      if (!I.p.fast_solver && la.Kall <= LTS_DP_MAX_K) lts_kernel<true><<<gLts, LTS_THREADS, 0, s>>>(la); else lts_kernel<false><<<gLts, LTS_THREADS, 0, s>>>(la);
      if (dumpIt >= 0 && globalIt == dumpIt) {
        CK(cudaStreamSynchronize(s));
        const char* dd = getenv("ALADIN_DUMP");
        for (int dd_ = 0; dd_ < 2; ++dd_) {
          float h[16];
          CK(cudaMemcpy(h, I.dMnew[dd_], 64, cudaMemcpyDeviceToHost));
          char fn[512];
          snprintf(fn, sizeof(fn), "%s/it%d_Mnew%d.bin", dd, dumpIt, dd_);
          FILE* f = fopen(fn, "wb");
          fwrite(h, 4, 16, f);
          fclose(f);
        }
      }
      sym_update_kernel<<<n, 128, 0, s>>>(I.dMnew[0], I.dMnew[1], I.dT, I.dB, fa);
      if (trace >= 0 && trace < n) {  // (debug trace)
        CK(cudaStreamSynchronize(s));
        float a0[16], a1[16], a2[16];
        CK(cudaMemcpy(a0, I.dMnew[0] + trace * 16, 64, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(a1, I.dMnew[1] + trace * 16, 64, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(a2, I.dT + trace * 16, 64, cudaMemcpyDeviceToHost));
        printf("it %d L%d %s\n  fwd:", traceit, l, la.affine ? "aff" : "rig");
        for (int i = 0; i < 12; ++i) printf(" %.5f", a0[i]);
        printf("\n  bwd:");
        for (int i = 0; i < 12; ++i) printf(" %.5f", a1[i]);
        printf("\n  T  :");
        for (int i = 0; i < 12; ++i) printf(" %.5f", a2[i]);
        printf("\n");
        ++traceit;
      }
    }
    CK(cudaGetLastError());
    CK(cudaEventRecord(I.ev[l + 1], s));
  }
  CK(cudaEventSynchronize(I.ev[L]));
  tm_.registration = 0;
  for (int l = 0; l < L; ++l) {
    CK(cudaEventElapsedTime(&tm_.level[l], I.ev[l], I.ev[l + 1]));
    tm_.registration += tm_.level[l];
  }
  std::vector<float> res(n * 16);
  // stream-ordered copies (the legacy null stream would not wait for work queued in a non-blocking stream)
  CK(cudaMemcpyAsync(res.data(), I.dT, n * 16 * sizeof(float), cudaMemcpyDeviceToHost, s));
  if (out_dev) CK(cudaMemcpyAsync(out_dev, I.dT, n * 16 * sizeof(float), cudaMemcpyDeviceToDevice, s));
  for (int t = 0; t < n; ++t)
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) out_host[t].m[i][j] = res[t * 16 + i * 4 + j];
  status_.assign(n, 0);
  std::vector<int> st(2 * n + 4);
  CK(cudaMemcpyAsync(st.data(), I.dStatus, (2 * n + 4) * sizeof(int), cudaMemcpyDeviceToHost, s));
  CK(cudaStreamSynchronize(s));
#ifdef ALADIN_STATS
  if (getenv("ALADIN_LTSSTATS")) {
    unsigned long long h[8];
    CK(cudaMemcpyFromSymbol(h, g_bm_stats, sizeof(h)));
    printf("BM blocks %llu, contenders %llu (%.2f per block), blocks with partial ref %llu; nc histogram: 0:%llu 1:%llu 2-4:%llu 5-32:%llu >32:%llu\n", h[0], h[1], (double)h[1] / (double)h[0], h[2], h[3], h[4], h[5], h[6], h[7]);
  }
#endif
  if (getenv("ALADIN_LTSSTATS")) printf("LTS loop iterations total: rigid %d, affine %d (calls: rigid %d, affine %d per level sum)\n", st[2 * n], st[2 * n + 1], 40 * n * 2, 15 * n * 2);
  for (int t = 0; t < n; ++t) status_[t] = st[t] | st[n + t];
}

}  // namespace aladin
