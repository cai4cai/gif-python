// SPDX-License-Identifier: BSD-3-Clause
// context.cu - EmContext: buffers, orchestration and timing
#include "internal.cuh"
#include <algorithm>
#include <vector>
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>

namespace segem {

void EmContext::Impl::allocate() {
    size_t N = d.N;
    expec = dalloc<float>((size_t)C * N);
    expec2 = dalloc<float>((size_t)C * N);
    T = dalloc<float>(N);
    bc = dalloc<float>(N);
    for (int i = 0; i < 4; i++) maskS[i] = dalloc<uint8_t>(N);
    wsI0 = dalloc<int32_t>(N);
    wsI1 = dalloc<int32_t>(N);
    wsB0 = dalloc<uint8_t>(N);
    wsB1 = dalloc<uint8_t>(N);
    wsB2 = dalloc<uint8_t>(N);
    wsF0 = dalloc<float>(N);
    wsF1 = dalloc<float>(N);
    voxList = dalloc<int>(N);
    pmin = dalloc<float>(kEmGrid);
    pmax = dalloc<float>(kEmGrid);
    numActive = dalloc<int>(1);
    CUDA_CHECK(cudaMemsetAsync(numActive, 0, sizeof(int), allocStream));
    st = dalloc<EMState>(1);
    red = dalloc<double>((size_t)kEmGrid * 3 * kMaxClasses);
    int nly = (d.ny + 1) / 2, nlz = (d.nz + 1) / 2;
    lineAcc = dalloc<double>((size_t)nly * nlz * 14);
    momAcc = dalloc<double>((size_t)((nly * nlz + kLinesPerBlock - 1) / kLinesPerBlock) * 256);
    // dormant planes of the posterior buffers are never written by the main kernels: keep them exactly zero
    CUDA_CHECK(cudaMemsetAsync(expec, 0, (size_t)C * N * sizeof(float), allocStream));
    CUDA_CHECK(cudaMemsetAsync(expec2, 0, (size_t)C * N * sizeof(float), allocStream));
    eFinal = expec;
    redS = dalloc<double>((size_t)kSpecGrid * 3 * kMaxClasses);
    redSL = dalloc<double>(kSpecGrid);
    pT = dalloc<double>((size_t)kEmGrid * 3);
    {
        size_t nb = N - (size_t)std::max(d.nx - 2, 0) * std::max(d.ny - 2, 0) * std::max(d.nz - 2, 0);
        specList = dalloc<int>(nb + 1);
        numSpec = dalloc<int>(1);
        dormFlag = dalloc<uint8_t>(N);
        CUDA_CHECK(cudaMemsetAsync(dormFlag, 0, N, allocStream));
        CUDA_CHECK(cudaMemsetAsync(numSpec, 0, sizeof(int), allocStream));
    }
    CUDA_CHECK(cudaMallocHost(&hostSt, sizeof(EMState)));
    for (int i = 0; i < 10; i++) CUDA_CHECK(cudaEventCreate(&ev[i]));
}

void EmContext::Impl::freeAll() {
    void *ptrs[] = {expec, expec2, T, bc, maskS[0], maskS[1], maskS[2], maskS[3], wsI0, wsI1, wsB0, wsB1, wsB2, wsF0, wsF1,
                    voxList, pmin, pmax, numActive, st, red, lineAcc, momAcc,
                    cubTmp, specList, numSpec, dormFlag, redS, redSL, pT, cubTmp2};
    for (void *p : ptrs) if (p) cudaFreeAsync(p, allocStream);
    if (hostSt) cudaFreeHost(hostSt);
    for (int i = 0; i < 10; i++) cudaEventDestroy(ev[i]);
}

EmContext::EmContext(int nx, int ny, int nz, int numClasses, const EmParams &p, unsigned liveMask, cudaStream_t allocStream) {
    p_ = new Impl();
    Impl &I = *p_;
    I.allocStream = allocStream;
    I.d.nx = nx; I.d.ny = ny; I.d.nz = nz; I.d.N = (size_t)nx * ny * nz;
    I.d.ntilesX = (nx + kTileX - 1) / kTileX;
    I.prm = p;
    if (I.prm.bias_order > kMaxOrder) I.prm.bias_order = kMaxOrder;
    I.C = numClasses;
    if (I.C < 2 || I.C > kMaxClasses) { fprintf(stderr, "segem: unsupported number of classes %d\n", I.C); abort(); }
    const unsigned all = (1u << I.C) - 1u;
    I.liveMask = liveMask ? ((liveMask | 1u) & all) : all;
    I.hasDormant = (I.liveMask != all) ? 1 : 0;
    I.allocate();
}

EmContext::~EmContext() { p_->freeAll(); delete p_; }

const float *EmContext::seg() const { return p_->eFinal; }
const float *EmContext::biasCorrected() const { return p_->bc; }
const uint8_t *EmContext::maskStage(int k) const { return p_->maskS[k]; }
const StageTimes &EmContext::times() const { return p_->tm; }
int EmContext::numClasses() const { return p_->C; }
size_t EmContext::numVoxels() const { return p_->d.N; }
size_t EmContext::deviceBytes() const { return p_->bytes; }

__global__ void k_live(const float *prior, int C, size_t N, unsigned *out) {
    size_t v = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; v < N; v += (size_t)gridDim.x * blockDim.x)
        for (int c = 0; c < C; c++) {
            float x = prior[(size_t)c * N + v];
            if (x > 0.f && x <= 1000.f) atomicOr(out, 1u << c);   // seg_EM ignores negative, NaN and > 1000 prior values
        }
}

unsigned EmContext::liveClasses(const float *prior, int C, size_t N, cudaStream_t s) {
    unsigned *d = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d, sizeof(unsigned), s));
    CUDA_CHECK(cudaMemsetAsync(d, 0, sizeof(unsigned), s));
    k_live<<<1024, 256, 0, s>>>(prior, C, N, d);
    unsigned h = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h, d, sizeof h, cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaFreeAsync(d, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    return h | 1u;
}

void EmContext::morphology(const uint8_t *mask0, int dilate_radius, int erode_radius, cudaStream_t s) {
    p_->stream = s;
    launch_morphology(*p_, mask0, dilate_radius, erode_radius);
}

void EmContext::run(const float *prior, const uint8_t *mask, const float *target, cudaStream_t s, bool biasCorrected) {
    Impl &I = *p_;
    I.stream = s;
    I.prior = prior;
    I.emMask = mask;
    I.targetPtr = target;
    cudaEventRecord(I.ev[0], s);
    launch_em(I, target);                 // records ev[3] (end of initialisation) itself
    cudaEventRecord(I.ev[4], s);
    if (biasCorrected) launch_bias_corrected(I);
    cudaEventRecord(I.ev[5], s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    float t3, t4, t5, tt;
    cudaEventElapsedTime(&t3, I.ev[0], I.ev[3]);
    cudaEventElapsedTime(&t4, I.ev[3], I.ev[4]);
    cudaEventElapsedTime(&t5, I.ev[4], I.ev[5]);
    cudaEventElapsedTime(&tt, I.ev[0], I.ev[5]);
    I.tm.em_init = t3; I.tm.em = t4; I.tm.bias_corrected = t5; I.tm.total = tt;
    I.tm.em_iterations = I.iters;
}

} // namespace segem
