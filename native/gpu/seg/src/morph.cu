// SPDX-License-Identifier: BSD-3-Clause
// morph.cu - mask morphology: Dillate, Close_Forground_ConnectComp, Erosion, GaussianFilter4D_cArray (NiftySeg _seg_tools.cpp)
//
// Dillate / Erosion: separable max / min over a window of +-r voxels, clamped at the volume border (no padding),
// applied along x, y, z in turn. For a binary mask this is an exact OR / AND (byte-vectorised version for nx % 8 == 0).
//
// Close_Forground_ConnectComp: fills the holes of the (dilated) mask. The CPU routine labels the *background* with a
// Gauss-Seidel min-label propagation that is only run on the strictly interior voxels (1..n-2) and then keeps every background
// component whose final label equals the label of some border voxel.  This is NOT "fill every component that does not touch
// the border": a border voxel only shares its label with its inward interior neighbour when that label is strictly the smallest
// one the neighbour sees in the first sweep (raster order).  The GPU version reproduces that rule exactly:
//   (1) the sweep-1 labels L in the three planes x==1, y==1, z==1 (three 2-D wavefronts),
//   (2) 6-connected union-find on the interior background voxels (4x4x4 tiles that are entirely background are pre-merged),
//   (3) "tie" flags (border voxel label strictly minimal for its inward neighbour) OR-ed onto the component roots,
//   (4) output: mask | (interior background voxel whose component carries no tie flag).
// It reproduces the CPU output exactly (tests/test_morph.cu: thousands of random masks, ~50% of which are filled differently
// from a textbook border flood fill).
#include "internal.cuh"
#include <algorithm>

namespace segem {

// ---------------------------------------------------------------- separable binary dilate / erode
// Scalar version (any nx).
template <bool DIL>
__global__ void k_pass(const uint8_t *in, uint8_t *out, int nx, int ny, int nz, int dir, int r) {
    size_t N = (size_t)nx * ny * nz;
    size_t v = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;
    int x = v % nx, y = (v / nx) % ny, z = (int)(v / ((size_t)nx * ny));
    int pos, dim; size_t stride;
    if (dir == 0) { pos = x; dim = nx; stride = 1; }
    else if (dir == 1) { pos = y; dim = ny; stride = nx; }
    else { pos = z; dim = nz; stride = (size_t)nx * ny; }
    int lo = pos < r ? -pos : -r;
    int hi = (pos >= dim - r) ? dim - pos - 1 : r;
    uint8_t res = DIL ? 0 : 1;
    for (int s = lo; s <= hi; s++) {
        uint8_t val = in[v + (ptrdiff_t)s * (ptrdiff_t)stride];
        if (DIL) { if (val) { res = 1; break; } }
        else { if (!val) { res = 0; break; } }
    }
    out[v] = res;
}

// Vectorised version for nx % 8 == 0: every thread handles 8 consecutive voxels (one uint64 of 0/1 bytes).
// y / z: OR (dilate) or AND (erode) of the clamped window of rows; x: byte shifts across the three neighbouring words.
template <bool DIL>
__global__ void k_pass_vec(const uint64_t *in, uint64_t *out, int nx8, int ny, int nz, int dir, int r) {
    size_t N8 = (size_t)nx8 * ny * nz;
    size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= N8) return;
    int x8 = w % nx8, y = (w / nx8) % ny, z = (int)(w / ((size_t)nx8 * ny));
    const uint64_t ONES = 0x0101010101010101ull;
    if (dir == 0) {
        uint64_t C = in[w];
        uint64_t Lw = x8 > 0 ? in[w - 1] : (DIL ? 0ull : ONES);
        uint64_t Rw = x8 < nx8 - 1 ? in[w + 1] : (DIL ? 0ull : ONES);
        uint64_t acc = C;
        for (int s = 1; s <= r; s++) {
            // byte j of the result takes byte j+s (right neighbours) and byte j-s (left neighbours)
            uint64_t right = (C >> (8 * s)) | (Rw << (64 - 8 * s));
            uint64_t left = (C << (8 * s)) | (Lw >> (64 - 8 * s));
            if (s >= 8) { right = Rw >> (8 * (s - 8)); left = Lw << (8 * (s - 8)); }
            if (DIL) acc |= right | left; else acc &= right & left;
        }
        out[w] = acc;
    } else {
        size_t stride = dir == 1 ? (size_t)nx8 : (size_t)nx8 * ny;
        int pos = dir == 1 ? y : z, dim = dir == 1 ? ny : nz;
        int lo = pos < r ? -pos : -r;
        int hi = (pos >= dim - r) ? dim - pos - 1 : r;
        uint64_t acc = DIL ? 0ull : ONES;
        for (int s = lo; s <= hi; s++) {
            uint64_t val = in[w + (ptrdiff_t)s * (ptrdiff_t)stride];
            if (DIL) acc |= val; else acc &= val;
        }
        out[w] = acc;
    }
}

// out = dilate/erode(in); tmp is scratch. in is preserved. (x: in->out, y: out->tmp, z: tmp->out)
void launch_dilate_erode_u8(const uint8_t *in, uint8_t *out, uint8_t *tmp, const Dims &d, int r, bool dil, cudaStream_t s) {
    if (r <= 0) { CUDA_CHECK(cudaMemcpyAsync(out, in, d.N, cudaMemcpyDeviceToDevice, s)); return; }
    if (d.nx % 8 == 0 && r <= 7) {
        int nx8 = d.nx / 8;
        size_t N8 = d.N / 8;
        int blocks = (int)((N8 + 255) / 256);
        const uint64_t *i0 = (const uint64_t *)in; uint64_t *o0 = (uint64_t *)out, *t0 = (uint64_t *)tmp;
        if (dil) {
            k_pass_vec<true><<<blocks, 256, 0, s>>>(i0, o0, nx8, d.ny, d.nz, 0, r);
            k_pass_vec<true><<<blocks, 256, 0, s>>>(o0, t0, nx8, d.ny, d.nz, 1, r);
            k_pass_vec<true><<<blocks, 256, 0, s>>>(t0, o0, nx8, d.ny, d.nz, 2, r);
        } else {
            k_pass_vec<false><<<blocks, 256, 0, s>>>(i0, o0, nx8, d.ny, d.nz, 0, r);
            k_pass_vec<false><<<blocks, 256, 0, s>>>(o0, t0, nx8, d.ny, d.nz, 1, r);
            k_pass_vec<false><<<blocks, 256, 0, s>>>(t0, o0, nx8, d.ny, d.nz, 2, r);
        }
        return;
    }
    int blocks = (int)((d.N + 255) / 256);
    if (dil) {
        k_pass<true><<<blocks, 256, 0, s>>>(in, out, d.nx, d.ny, d.nz, 0, r);
        k_pass<true><<<blocks, 256, 0, s>>>(out, tmp, d.nx, d.ny, d.nz, 1, r);
        k_pass<true><<<blocks, 256, 0, s>>>(tmp, out, d.nx, d.ny, d.nz, 2, r);
    } else {
        k_pass<false><<<blocks, 256, 0, s>>>(in, out, d.nx, d.ny, d.nz, 0, r);
        k_pass<false><<<blocks, 256, 0, s>>>(out, tmp, d.nx, d.ny, d.nz, 1, r);
        k_pass<false><<<blocks, 256, 0, s>>>(tmp, out, d.nx, d.ny, d.nz, 2, r);
    }
}

// ---------------------------------------------------------------- hole filling
__device__ __forceinline__ bool is_interior(int x, int y, int z, int nx, int ny, int nz) {
    return x >= 1 && x <= nx - 2 && y >= 1 && y <= ny - 2 && z >= 1 && z <= nz - 2;
}

// Sweep-1 label DP of the CPU algorithm. fg = background of the dilated mask (mask1 == 0).
//   L[v] = min(v+1, cand(x-1), cand(y-1), cand(z-1)),  cand(n) = L[n] if n is an interior background voxel,
//   n+1 if n is a border background voxel, none otherwise.
// The tie test only looks at the back neighbours of the interior voxels of the planes x==1, y==1 and z==1, and the monotone
// back paths of those voxels never leave their plane (a step to x-1 / y-1 / z-1 = 0 ends in a border voxel, which does not
// propagate), so L is only needed in these three planes: three independent 2-D wavefronts, one thread block each.
__global__ void __launch_bounds__(1024) k_dp_planes(const uint8_t *mask1, int *L, int nx, int ny, int nz) {
    const int plane = blockIdx.x;            // 0: x == 1 (axes y,z)   1: y == 1 (axes x,z)   2: z == 1 (axes x,y)
    const int fixedc = 1;
    int na, nb;                              // extents of the two in-plane axes (interior range 1..n-2)
    size_t sa, sb, base;                     // linear strides of the two axes and the plane origin
    if (plane == 0) { na = ny; nb = nz; sa = nx; sb = (size_t)nx * ny; base = fixedc; }
    else if (plane == 1) { na = nx; nb = nz; sa = 1; sb = (size_t)nx * ny; base = (size_t)nx * fixedc; }
    else { na = nx; nb = ny; sa = 1; sb = nx; base = (size_t)nx * ny * fixedc; }
    const int ia = na - 2, ib = nb - 2;      // number of interior positions per axis
    if (ia <= 0 || ib <= 0) return;
    extern __shared__ int prevd[];           // labels of the previous anti-diagonal, indexed by a
    int *curd = prevd + ia;
    // position (a,b), a in [1,na-2], b in [1,nb-2]; voxel = base + a*sa + b*sb ; the fixed coordinate is 1 (interior)
    for (int d = 2; d <= (na - 2) + (nb - 2); d++) {
        // a + b = d
        int alo = max(1, d - (nb - 2)), ahi = min(na - 2, d - 1);
        for (int a = alo + threadIdx.x; a <= ahi; a += blockDim.x) {
            int b = d - a;
            size_t v = base + (size_t)a * sa + (size_t)b * sb;
            int res = 0x7fffffff;
            if (!mask1[v]) {
                int best = (int)v + 1;
                // back neighbour along the fixed axis (coordinate 0): a border voxel
                size_t nf = plane == 0 ? v - 1 : plane == 1 ? v - nx : v - (size_t)nx * ny;
                if (!mask1[nf]) best = min(best, (int)nf + 1);
                // back neighbour along a: interior if a-1 >= 1 (previous diagonal), else a border voxel
                size_t nA = v - sa;
                if (!mask1[nA]) best = min(best, a - 1 >= 1 ? prevd[a - 1] : (int)nA + 1);
                size_t nB = v - sb;
                if (!mask1[nB]) best = min(best, b - 1 >= 1 ? prevd[a] : (int)nB + 1);
                res = best;
                L[v] = best;
            }
            curd[a] = res;
        }
        __syncthreads();
        // swap by copying (the diagonal is short)
        for (int a = alo + threadIdx.x; a <= ahi; a += blockDim.x) prevd[a] = curd[a];
        __syncthreads();
    }
}

// ---- tile-uniform union-find over the interior background voxels
constexpr int kTX = 4, kTY = 4, kTZ = 4;       // 64-voxel tiles
struct TileGeom { int ntx, nty, ntz; };
__device__ __forceinline__ int tile_id(int x, int y, int z, const TileGeom &t) { return x / kTX + t.ntx * (y / kTY + t.nty * (z / kTZ)); }

// tile is "uniform" if every voxel is inside the volume, interior, and background
__global__ void __launch_bounds__(kTX * kTY * kTZ) k_tile_class(const uint8_t *mask1, uint8_t *tu, int nx, int ny, int nz, TileGeom t) {
    int tid = blockIdx.x;
    int tx = tid % t.ntx, ty = (tid / t.ntx) % t.nty, tz = tid / (t.ntx * t.nty);
    int lx = threadIdx.x % kTX, ly = (threadIdx.x / kTX) % kTY, lz = threadIdx.x / (kTX * kTY);
    int x = tx * kTX + lx, y = ty * kTY + ly, z = tz * kTZ + lz;
    bool ok = x < nx && y < ny && z < nz && is_interior(x, y, z, nx, ny, nz) && !mask1[(size_t)x + (size_t)nx * (y + (size_t)ny * z)];
    int all = __syncthreads_and(ok);
    if (threadIdx.x == 0) tu[tid] = all ? 1 : 0;
}

__global__ void k_uf_init2(const uint8_t *tu, int *parent, int nx, int ny, int nz, TileGeom t) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y, z = blockIdx.z;
    if (x >= nx) return;
    size_t v = (size_t)x + (size_t)nx * (y + (size_t)ny * z);
    int tid = tile_id(x, y, z, t);
    if (tu[tid]) {
        int tx = tid % t.ntx, ty = (tid / t.ntx) % t.nty, tz = tid / (t.ntx * t.nty);
        parent[v] = (tx * kTX) + nx * (ty * kTY + (size_t)ny * (tz * kTZ));
    } else parent[v] = (int)v;
}

__device__ __forceinline__ int uf_find_v(volatile int *p, int a) {
    while (true) {
        int pa = p[a];
        if (pa == a) return a;
        int ppa = p[pa];
        p[a] = ppa;
        a = ppa;
    }
}
__device__ __forceinline__ void uf_union(int *p, int a, int b) {
    while (true) {
        a = uf_find_v(p, a);
        b = uf_find_v(p, b);
        if (a == b) return;
        if (a < b) { int t = a; a = b; b = t; }
        if (atomicCAS(&p[a], a, b) == a) return;
    }
}

__global__ void k_uf_hook2(const uint8_t *mask1, const uint8_t *tu, int *parent, int nx, int ny, int nz, TileGeom t) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y, z = blockIdx.z;
    if (x >= nx) return;
    size_t v = (size_t)x + (size_t)nx * (y + (size_t)ny * z);
    if (!is_interior(x, y, z, nx, ny, nz)) return;
    int tid = tile_id(x, y, z, t);
    bool uni = tu[tid];
    if (!uni && mask1[v]) return;
    // uniform tile interior (not on a low face): nothing to do
    bool lx0 = (x % kTX) == 0, ly0 = (y % kTY) == 0, lz0 = (z % kTZ) == 0;
    if (uni && !(lx0 || ly0 || lz0)) return;
    int cx[3] = {x - 1, x, x}, cy[3] = {y, y - 1, y}, cz[3] = {z, z, z - 1};
    size_t nb[3] = {v - 1, v - nx, v - (size_t)nx * ny};
#pragma unroll
    for (int k = 0; k < 3; k++) {
        if (!is_interior(cx[k], cy[k], cz[k], nx, ny, nz) || mask1[nb[k]]) continue;
        int nt = tile_id(cx[k], cy[k], cz[k], t);
        if (nt == tid) {
            if (uni) continue;                     // already merged
        } else if (uni && tu[nt]) {
            // two uniform tiles: one union per tile pair (at the first voxel of the shared face) suffices
            bool first = (k == 0 ? (ly0 && lz0) : k == 1 ? (lx0 && lz0) : (lx0 && ly0));
            // v is on the low face of its tile in direction k (the neighbour is in the other tile)
            bool onface = (k == 0 ? lx0 : k == 1 ? ly0 : lz0);
            if (!onface || !first) continue;
        }
        uf_union(parent, (int)v, (int)nb[k]);
    }
}

// Pass 1 compresses the trees (path halving; the stores are valid ancestors but may race with each other, so they do not
// necessarily leave a root behind), pass 2 reads the roots without writing anything but the final value.
__global__ void k_uf_flatten2(const uint8_t *mask1, int *parent, int nx, int ny, int nz, int pass) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y, z = blockIdx.z;
    if (x >= nx) return;
    size_t v = (size_t)x + (size_t)nx * (y + (size_t)ny * z);
    if (mask1[v] || !is_interior(x, y, z, nx, ny, nz)) return;
    if (pass == 0) { uf_find_v(parent, (int)v); return; }
    int r = (int)v;
    while (true) { int pr = __ldcg(&parent[r]); if (pr == r) break; r = pr; }
    parent[v] = r;
}

// tie flag (see header comment): interior background voxel u adjacent to a near-face (coordinate 0) border background voxel b
// whose label idx(b)+1 is strictly the smallest label u sees at its sweep-1 visit; flags the component root.
__global__ void k_tie_flag(const uint8_t *mask1, const int *L, const int *parent, uint8_t *rootflag, int nx, int ny, int nz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y, z = blockIdx.z;
    if (x >= nx) return;
    size_t v = (size_t)x + (size_t)nx * (y + (size_t)ny * z);
    if (mask1[v] || !is_interior(x, y, z, nx, ny, nz) || !(x == 1 || y == 1 || z == 1)) return;
    int best = (int)v + 1;
    size_t nb[3] = {v - 1, v - nx, v - (size_t)nx * ny};
    int cx[3] = {x - 1, x, x}, cy[3] = {y, y - 1, y}, cz[3] = {z, z, z - 1};
    int bl[3] = {-1, -1, -1};
#pragma unroll
    for (int k = 0; k < 3; k++) {
        if (mask1[nb[k]]) continue;
        bool in = is_interior(cx[k], cy[k], cz[k], nx, ny, nz);
        int cand = in ? L[nb[k]] : (int)nb[k] + 1;
        if (!in) bl[k] = cand;
        if (cand < best) best = cand;
    }
    bool t = false;
#pragma unroll
    for (int k = 0; k < 3; k++)
        if (bl[k] == best) t = true;
    if (t) rootflag[parent[v]] = 1;
}

__global__ void k_fill(const uint8_t *mask1, const int *parent, const uint8_t *rootflag, uint8_t *out, int nx, int ny, int nz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y, z = blockIdx.z;
    if (x >= nx) return;
    size_t v = (size_t)x + (size_t)nx * (y + (size_t)ny * z);
    uint8_t r = mask1[v];
    if (!r && is_interior(x, y, z, nx, ny, nz) && !rootflag[parent[v]]) r = 1;
    out[v] = r;
}


static void launch_fill(EmContext::Impl &I, const uint8_t *mask1, uint8_t *out) {
    const Dims &d = I.d;
    cudaStream_t s = I.stream;
    dim3 g3((d.nx + 127) / 128, d.ny, d.nz);
    int *L = I.wsI0, *parent = I.wsI1;
    uint8_t *rootflag = I.wsB1;
    if (d.nx < 3 || d.ny < 3 || d.nz < 3) { CUDA_CHECK(cudaMemcpyAsync(out, mask1, d.N, cudaMemcpyDeviceToDevice, s)); return; }
    TileGeom tg{(d.nx + kTX - 1) / kTX, (d.ny + kTY - 1) / kTY, (d.nz + kTZ - 1) / kTZ};
    int ntiles = tg.ntx * tg.nty * tg.ntz;
    uint8_t *tu = I.wsB0;     // ntiles bytes (<< N)
    // (1) sweep-1 label DP (three 2-D wavefronts in the planes x==1, y==1, z==1)
    {
        int maxa = std::max(std::max(d.nx, d.ny), d.nz);
        size_t smem = 2 * (size_t)maxa * sizeof(int);
        k_dp_planes<<<3, 1024, smem, s>>>(mask1, L, d.nx, d.ny, d.nz);
    }
    // (2) union-find over the interior background voxels (uniform tiles are pre-merged)
    k_tile_class<<<ntiles, kTX * kTY * kTZ, 0, s>>>(mask1, tu, d.nx, d.ny, d.nz, tg);
    k_uf_init2<<<g3, 128, 0, s>>>(tu, parent, d.nx, d.ny, d.nz, tg);
    k_uf_hook2<<<g3, 128, 0, s>>>(mask1, tu, parent, d.nx, d.ny, d.nz, tg);
    k_uf_flatten2<<<g3, 128, 0, s>>>(mask1, parent, d.nx, d.ny, d.nz, 0);
    k_uf_flatten2<<<g3, 128, 0, s>>>(mask1, parent, d.nx, d.ny, d.nz, 1);
    // (3) tie flags on the component roots, (4) output
    CUDA_CHECK(cudaMemsetAsync(rootflag, 0, d.N, s));
    k_tie_flag<<<g3, 128, 0, s>>>(mask1, L, parent, rootflag, d.nx, d.ny, d.nz);
    k_fill<<<g3, 128, 0, s>>>(mask1, parent, rootflag, out, d.nx, d.ny, d.nz);
    CUDA_CHECK(cudaGetLastError());
}

void launch_morphology(EmContext::Impl &I, const uint8_t *mask0, int dilate_radius, int erode_radius) {
    const Dims &d = I.d;
    if (mask0 != I.maskS[0]) CUDA_CHECK(cudaMemcpyAsync(I.maskS[0], mask0, d.N, cudaMemcpyDeviceToDevice, I.stream));
    launch_dilate_erode_u8(I.maskS[0], I.maskS[1], I.wsB0, d, dilate_radius, true, I.stream);
    launch_fill(I, I.maskS[1], I.maskS[2]);
    launch_dilate_erode_u8(I.maskS[2], I.maskS[3], I.wsB0, d, erode_radius, false, I.stream);
}

// ---------------------------------------------------------------- smooth mask for the bias corrected image (seg_EM::GetBiasCorrected)
// brainmask = mask (0 where the image is NaN); Dillate(7); Erosion(3); GaussianFilter4D_cArray(sigma=3 voxels)
__global__ void k_bm_init(const uint8_t *mask, const float *target, uint8_t *out, size_t N) {
    size_t v = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;
    out[v] = (mask[v] && target[v] == target[v]) ? 1 : 0;   // NaN voxels excluded (inside the mask T is NaN iff the input is)
}
__global__ void k_u8_to_f(const uint8_t *in, float *out, size_t N) {
    size_t v = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;
    out[v] = in[v] ? 1.f : 0.f;
}
struct GaussTaps { float k[64]; int shift; };

__global__ void k_gauss(const float *in, float *out, GaussTaps g, int nx, int ny, int nz, int dir) {
    size_t N = (size_t)nx * ny * nz;
    size_t v = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;
    int x = v % nx, y = (v / nx) % ny, z = (int)(v / ((size_t)nx * ny));
    int pos, dim; size_t stride;
    if (dir == 0) { pos = x; dim = nx; stride = 1; }
    else if (dir == 1) { pos = y; dim = ny; stride = nx; }
    else { pos = z; dim = nz; stride = (size_t)nx * ny; }
    int ks = g.shift;
    int lo = pos < ks ? -pos : -ks;
    int hi = (pos >= dim - ks) ? dim - pos - 1 : ks;
    float acc = 0.f, kd = 0.f;
    for (int s = lo; s <= hi; s++) {
        float k = g.k[s + ks];
        acc = __fadd_rn(acc, __fmul_rn(k, in[v + (ptrdiff_t)s * (ptrdiff_t)stride]));
        kd = __fadd_rn(kd, k);
    }
    out[v] = __fdiv_rn(acc, kd);
}

// Writes the smoothed mask into I.wsF0 (the final field of GetBiasCorrected).
void launch_smooth_mask(EmContext::Impl &I) {
    const Dims &d = I.d;
    cudaStream_t s = I.stream;
    int blocks = (int)((d.N + 255) / 256);
    k_bm_init<<<blocks, 256, 0, s>>>(I.emMask, I.targetPtr, I.wsB1, d.N);
    launch_dilate_erode_u8(I.wsB1, I.wsB2, I.wsB0, d, 7, true, s);
    launch_dilate_erode_u8(I.wsB2, I.wsB1, I.wsB0, d, 3, false, s);
    // gaussian taps as in the NiftySeg gaussian smoothing (sigma = 3 voxels)
    GaussTaps g;
    float gauss_std = 3.0f;
    int ksize = (int)(gauss_std * 6.0f) % 2 != 0 ? (int)(gauss_std * 6.0f) : (int)(gauss_std * 6.0f) + 1;
    int kshift = (int)(ksize / 2.0f);
    float kernelsum = 0;
    memset(g.k, 0, sizeof(g.k));
    for (int i = 0; i < ksize; i++) {
        g.k[i] = expf((float)(-0.5f * powf((float)((float)i - (float)kshift) / gauss_std, 2.0f))) / (sqrtf(2.0f * 3.14159265 * powf(gauss_std, 2)));
        kernelsum += g.k[i];
    }
    for (int i = 0; i < ksize; i++) g.k[i] /= kernelsum;
    g.shift = kshift;
    k_u8_to_f<<<blocks, 256, 0, s>>>(I.wsB1, I.wsF1, d.N);
    k_gauss<<<blocks, 256, 0, s>>>(I.wsF1, I.wsF0, g, d.nx, d.ny, d.nz, 0);
    k_gauss<<<blocks, 256, 0, s>>>(I.wsF0, I.wsF1, g, d.nx, d.ny, d.nz, 1);
    k_gauss<<<blocks, 256, 0, s>>>(I.wsF1, I.wsF0, g, d.nx, d.ny, d.nz, 2);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace segem
