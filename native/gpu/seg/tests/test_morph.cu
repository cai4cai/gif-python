// SPDX-License-Identifier: BSD-3-Clause
// test_morph.cu - exactness test of the GPU mask morphology against the NiftySeg library routines
// (Dillate, Close_Forground_ConnectComp, Erosion) on random masks with complex topology.
//   usage: test_morph [ncases=300] [seed=1]
#include "segem.h"
#include <dirent.h>
#include "nifti1_io.h"
#include "_seg_common.h"
#include "_seg_tools.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include <cstring>
using namespace segem;

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)

static std::vector<uint8_t> random_mask(int nx, int ny, int nz, std::mt19937 &rng, double q, int blur, double faceBias) {
    size_t N = (size_t)nx * ny * nz;
    std::uniform_real_distribution<float> U(0, 1);
    std::vector<float> a(N), b(N);
    for (auto &x : a) x = U(rng);
    for (int it = 0; it < blur; it++) {
        for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++) {
            float s = 0; int c = 0;
            for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
                int X = x + dx, Y = y + dy, Z = z + dz;
                if (X < 0 || Y < 0 || Z < 0 || X >= nx || Y >= ny || Z >= nz) continue;
                s += a[X + (size_t)nx * (Y + (size_t)ny * Z)]; c++;
            }
            b[x + (size_t)nx * (y + (size_t)ny * z)] = s / c;
        }
        a.swap(b);
    }
    std::vector<float> s(a);
    std::sort(s.begin(), s.end());
    float thr = s[(size_t)(q * (N - 1))];
    std::vector<uint8_t> m(N);
    for (size_t i = 0; i < N; i++) m[i] = a[i] > thr;
    // optionally force some face voxels
    if (faceBias > 0) {
        for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++) {
            bool face = x == 0 || y == 0 || z == 0 || x == nx - 1 || y == ny - 1 || z == nz - 1;
            if (face && U(rng) < faceBias) m[x + (size_t)nx * (y + (size_t)ny * z)] = U(rng) < 0.5;
        }
    }
    return m;
}

int main(int argc, char **argv) {
    int ncases = argc > 1 ? atoi(argv[1]) : 300;
    unsigned seed = argc > 2 ? atoi(argv[2]) : 1;
    std::mt19937 rng(seed);
    int bad = 0, total = 0, anyfilled = 0, quirkCases = 0; long long quirkVox = 0;
    long long diffvox = 0;
    for (int cs = 0; cs < ncases; cs++) {
        int nx = 5 + rng() % 28, ny = 5 + rng() % 28, nz = 5 + rng() % 24;
        if (cs % 25 == 24) { nx = 64; ny = 70; nz = 40; }
        double q = 0.25 + 0.5 * (rng() % 1000) / 1000.0;
        if (cs % 10 == 9) { nx = 96; ny = 72; nz = 48; q = 0.88 + 0.1 * (rng() % 1000) / 1000.0; }   // mostly background: exercises uniform tiles
        int blur = rng() % 4;
        if (cs % 10 == 9) blur = 2 + rng() % 3;
        double fb = (rng() % 3 == 0) ? 0.3 : 0.0;
        std::vector<uint8_t> m0 = random_mask(nx, ny, nz, rng, q, blur, fb);
        size_t N = m0.size();
        for (int mode = 0; mode < 2; mode++) {
            const int dil = mode == 0 ? 0 : 4, ero = mode == 0 ? 0 : 3;   // mode 0: hole filling alone, mode 1: dilate, fill, erode
            EmContext ctx(nx, ny, nz, 2);
            uint8_t *d; CK(cudaMalloc(&d, N)); CK(cudaMemcpy(d, m0.data(), N, cudaMemcpyHostToDevice));
            ctx.morphology(d, dil, ero);
            std::vector<uint8_t> g[4];
            for (int k = 0; k < 4; k++) { g[k].resize(N); CK(cudaMemcpy(g[k].data(), ctx.maskStage(k), N, cudaMemcpyDeviceToHost)); }
            cudaFree(d);
            // CPU
            std::vector<float> f(N), fill(N);
            for (size_t i = 0; i < N; i++) f[i] = m0[i];
            ImageSize cs_; memset(&cs_, 0, sizeof(cs_));
            cs_.numel = N; cs_.xsize = nx; cs_.ysize = ny; cs_.zsize = nz; cs_.usize = 1; cs_.tsize = 1;
            if (mode == 1) Dillate(f.data(), 4, &cs_);
            std::vector<float> after_dil(f);
            Close_Forground_ConnectComp<float, float>(f.data(), fill.data(), &cs_);
            std::vector<float> after_fill(fill);
            if (mode == 1) Erosion(fill.data(), 3, &cs_);
            long long d1 = 0, d2 = 0, d3 = 0;
            for (size_t i = 0; i < N; i++) {
                d1 += (after_dil[i] > 0) != (g[1][i] != 0) && mode == 1;
                // note: CPU Close_ modifies its input (inverts); the stage-1 mask for mode 0 is the input itself
                d2 += (after_fill[i] > 0) != (g[2][i] != 0);
                d3 += (fill[i] > 0) != (g[3][i] != 0);
                if (mode == 0 && after_fill[i] > 0 && !m0[i]) anyfilled++;
            }
            if (mode == 0) {   // informational: how often does the CPU routine differ from a standard 'fill holes not touching the border' ?
                std::vector<uint8_t> vis(N, 0), stdfill(m0);
                std::vector<size_t> q;
                auto push = [&](int x, int y, int z) { size_t i = x + (size_t)nx * (y + (size_t)ny * z); if (!m0[i] && !vis[i]) { vis[i] = 1; q.push_back(i); } };
                for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++)
                    if (x == 0 || y == 0 || z == 0 || x == nx - 1 || y == ny - 1 || z == nz - 1) push(x, y, z);
                while (!q.empty()) {
                    size_t i = q.back(); q.pop_back();
                    int x = i % nx, y = (i / nx) % ny, z = i / ((size_t)nx * ny);
                    if (x > 0) push(x - 1, y, z); if (x < nx - 1) push(x + 1, y, z);
                    if (y > 0) push(x, y - 1, z); if (y < ny - 1) push(x, y + 1, z);
                    if (z > 0) push(x, y, z - 1); if (z < nz - 1) push(x, y, z + 1);
                }
                long long dq = 0;
                for (size_t i = 0; i < N; i++) { uint8_t r = m0[i] || !vis[i]; dq += (r != (after_fill[i] > 0)); }
                if (dq) { quirkCases++; quirkVox += dq; }
            }
            total++;
            if (d1 || d2 || d3) {
                bad++; diffvox += d1 + d2 + d3;
                if (bad <= 10 && d2) {
                    int shown = 0;
                    for (size_t i = 0; i < N && shown < 4; i++)
                        if ((after_fill[i] > 0) != (g[2][i] != 0)) {
                            int x = i % nx, y = (i / nx) % ny, z = i / ((size_t)nx * ny);
                            printf("   voxel (%d,%d,%d): cpu=%d gpu=%d  input=%d; neighbours(-x,+x,-y,+y,-z,+z) input:", x, y, z, after_fill[i] > 0, g[2][i] != 0, m0[i]);
                            int dx[6] = {-1, 1, 0, 0, 0, 0}, dy[6] = {0, 0, -1, 1, 0, 0}, dz[6] = {0, 0, 0, 0, -1, 1};
                            for (int k = 0; k < 6; k++) { int X = x + dx[k], Y = y + dy[k], Z = z + dz[k]; printf(" %d", (X < 0 || Y < 0 || Z < 0 || X >= nx || Y >= ny || Z >= nz) ? -1 : m0[X + (size_t)nx * (Y + (size_t)ny * Z)]); }
                            printf("\n"); shown++;
                        }
                }
                if (bad <= 10) printf("MISMATCH case %d mode %d dims %dx%dx%d q=%.2f blur=%d fb=%.1f: dilate %lld fill %lld erode %lld voxels\n", cs, mode, nx, ny, nz, q, blur, fb, d1, d2, d3);
            }
        }
    }
    printf("(CPU routine differs from standard border-flood hole filling in %d of %d fill tests, %lld voxels)\n", quirkCases, total / 2, quirkVox);
    printf("morphology test: %d runs, %d mismatching, %lld mismatching voxels total; (%d filled voxels over all fill tests)\n", total, bad, diffvox, anyfilled);
    return bad ? 1 : 0;
}
