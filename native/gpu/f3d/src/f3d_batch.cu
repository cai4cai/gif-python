// SPDX-License-Identifier: BSD-3-Clause
// F3DBatch: orchestration of the multi-level, multi-template registration.
#include "f3d.h"
#include "f3d_internal.h"
#include "f3d_state.h"
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <algorithm>

namespace f3d {

// -------------------------------------------------------------- mask pyramid helper
__global__ void mask_down_k(const int *in, int *out, int nx, int ny, int nz, int ox, int oy, int oz, int sx, int sy, int sz)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t No = (size_t)ox * oy * oz;
    if (i >= No) return;
    int x = i % ox, y = (i / ox) % oy, z = i / ((size_t)ox * oy);
    int X = x * sx, Y = y * sy, Z = z * sz;
    int v = -1;
    if (X < nx && Y < ny && Z < nz) v = in[((size_t)Z * ny + Y) * nx + X];
    out[i] = v;
}
__global__ void mask_bin_k(const unsigned char *in, int *out, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] ? 1 : -1;
}

__global__ void mask_any_neg_k(const int *m, size_t n, int *flag)
{
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        if (m[i] < 0) { *flag = 1; return; }
}

struct F3DBatch::Impl {
    F3DParams P;
    cudaStream_t st;
    ImgGeom tg;
    int maxB;
    std::vector<float *> refLv;      // rescaled reference pyramid
    std::vector<ImgGeom> refG;
    std::vector<int *> maskLv;       // nullable
    GridGeom gridL[8];
    double simWeight;
    std::vector<std::vector<std::vector<LogRec>>> traces;  // [level][tpl]
    std::vector<std::vector<std::vector<EvalRec>>> evtrace;
    std::vector<cudaStream_t> gstream;
    std::vector<cudaEvent_t> evGroup;
    cudaEvent_t evMain = nullptr;
    int *hflag = nullptr;
    void ensure_streams(int G)
    {
        if (!evMain) { CK(cudaEventCreateWithFlags(&evMain, cudaEventDisableTiming)); CK(cudaMallocHost((void **)&hflag, 64 * sizeof(int))); }
        while ((int)gstream.size() < G) {
            cudaStream_t s; cudaEvent_t e;
            CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
            gstream.push_back(s); evGroup.push_back(e);
        }
    }
};

static void level_weights(const F3DParams &P, int l, float &be, float &le)
{
    float b0 = P.beWeight, l0 = P.leWeight;
    for (int k = 0; k <= l; k++) {
        if (k == 0) {
            b0 = b0 / static_cast<float>(powf(16.0f, (float)(P.levelNumber - 1)));
            l0 = l0 / static_cast<float>(powf(3.0f, (float)(P.levelNumber - 1)));
        } else {
            b0 = b0 * static_cast<float>(16);
            l0 = l0 * static_cast<float>(3);
        }
    }
    be = b0; le = l0;
}

F3DBatch::F3DBatch(const float *d_target, const ImgGeom &tg, const unsigned char *d_mask, const F3DParams &p, int maxBatch,
                   cudaStream_t st)
{
    p_ = new Impl;
    p_->P = p; p_->st = st; p_->tg = tg; p_->maxB = maxBatch;
    int L = p.levelToPerform;
    build_pyramid(d_target, tg, p.levelNumber, L, p_->refLv, p_->refG, st);
    for (int l = 0; l < L; l++) rescale01(p_->refLv[l], p_->refG[l].nvox(), st);
    p_->maskLv.assign(L, nullptr);
    if (d_mask) {
        // binarise, downsample without smoothing
        size_t n = tg.nvox();
        int *cur = dalloc<int>(n);
        mask_bin_k<<<cdiv(n, 256), 256, 0, st>>>(d_mask, cur, n);
        ImgGeom cg = tg;
        for (int l = L; l < p.levelNumber; l++) {
            bool ax[3]; downsample_axes(cg, ax);
            ImgGeom ng = geom_downsample(cg, ax);
            int *nx = dalloc<int>(ng.nvox());
            mask_down_k<<<cdiv(ng.nvox(), 256), 256, 0, st>>>(cur, nx, cg.n[0], cg.n[1], cg.n[2], ng.n[0], ng.n[1], ng.n[2], ax[0] ? 2 : 1, ax[1] ? 2 : 1, ax[2] ? 2 : 1);
            pool_free(cur); cur = nx; cg = ng;
        }
        p_->maskLv[L - 1] = cur;
        for (int l = L - 2; l >= 0; l--) {
            bool ax[3]; downsample_axes(cg, ax);
            ImgGeom ng = geom_downsample(cg, ax);
            int *nx = dalloc<int>(ng.nvox());
            mask_down_k<<<cdiv(ng.nvox(), 256), 256, 0, st>>>(p_->maskLv[l + 1], nx, cg.n[0], cg.n[1], cg.n[2], ng.n[0], ng.n[1], ng.n[2], ax[0] ? 2 : 1, ax[1] ? 2 : 1, ax[2] ? 2 : 1);
            p_->maskLv[l] = nx; cg = ng;
        }
    }
    // a mask that keeps every voxel is the same as no mask (and enables the specialised kernels)
    for (int l = 0; l < L; l++) {
        if (!p_->maskLv[l]) continue;
        int *dflag = dalloc_zero<int>(1);
        mask_any_neg_k<<<64, 256, 0, st>>>(p_->maskLv[l], p_->refG[l].nvox(), dflag);
        int f = 0;
        CK(cudaMemcpyAsync(&f, dflag, sizeof(int), cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        pool_free(dflag);
        if (!f) { pool_free(p_->maskLv[l]); p_->maskLv[l] = nullptr; }
    }
    // grids
    float sp[3];
    for (int a = 0; a < 3; a++) {
        float s = p.spacingVox;
        if (s < 0) s *= -1.0f * tg.d[a];
        sp[a] = s * powf(2.0f, (float)(p.levelNumber - 1));
    }
    if (tg.n[2] <= 1) sp[2] = 1.0f;
    p_->gridL[0] = create_grid(p_->refG[0], sp);
    for (int l = 1; l < L; l++) p_->gridL[l] = refine_grid_geom(p_->gridL[l - 1], p_->refG[l]);
    float pen = p.beWeight + p.leWeight + p.jlWeight + 0.f;
    p_->simWeight = (pen >= 1.0) ? 0.0 : 1.0 - (double)pen;
    p_->traces.assign(L, std::vector<std::vector<LogRec>>(maxBatch));
    CK(cudaStreamSynchronize(st));
}
F3DBatch::~F3DBatch()
{
    for (auto g : p_->gstream) cudaStreamDestroy(g);
    for (auto e : p_->evGroup) cudaEventDestroy(e);
    if (p_->evMain) cudaEventDestroy(p_->evMain);
    if (p_->hflag) cudaFreeHost(p_->hflag);
    for (auto d : p_->refLv) pool_free(d);
    for (auto d : p_->maskLv) if (d) pool_free(d);
    delete p_;
}
GridGeom F3DBatch::finalGrid() const { return p_->gridL[p_->P.levelToPerform - 1]; }
const std::vector<LogRec> &F3DBatch::trace(int level, int tpl) const { return p_->traces[level][tpl]; }
const std::vector<EvalRec> &F3DBatch::evtrace(int level, int tpl) const { static std::vector<EvalRec> e; if (p_->evtrace.size() <= (size_t)level) return e; return p_->evtrace[level][tpl]; }

struct LevelWork {
    int B = 0;
    ImgGeom R, F;
    GridGeom G;
    GridArgs ga;
    size_t N = 0, Nf = 0, nn = 0, ndof = 0, scr = 0;
    float *cur = nullptr, *best = nullptr, *grad = nullptr, *arr1 = nullptr, *arr2 = nullptr, *scratch = nullptr;
    float *warped = nullptr, *wgrad = nullptr, *vgrad = nullptr, *vtmp = nullptr;
    PenaltyOut *pen = nullptr;
    double *lncc = nullptr;
    TState *S = nullptr;
    LogRec *log = nullptr;
    EvalRec *ev = nullptr;
    float *dbg = nullptr;
    unsigned char *cflag = nullptr;
    int *actGrad = nullptr, *trialOK = nullptr, *actWarp = nullptr;
    float *scale = nullptr;
    int *flag = nullptr;
    LnccCtx lc;
    GradCtx gc;
    PenaltyBufs pb;
    AxisTab *tab[3] = {nullptr, nullptr, nullptr};
    WarpArgs wa;
    float beW, leW, jlW;
    float simWf;
    double simW;
    const int *mask = nullptr;
};

__global__ void alldone_k(const TState *S, int B, int *flag)
{
    int ok = 1;
    for (int b = 0; b < B; b++) {
        const TState &s = S[b];
        bool done = (s.mode == MODE_DONE) || (s.mode == MODE_GRAD && (s.startLen == 0.f || s.iter >= s.maxit));
        if (!done) ok = 0;
    }
    *flag = ok;
}

// a view of templates [b0, b0+Bg) of a level: all per-template buffers are offset, shared tables are kept
static LevelWork make_view(const LevelWork &W, int b0, int Bg, int *flag)
{
    LevelWork V = W;
    V.B = Bg;
    const size_t nd = W.ndof;
    V.cur += b0 * nd; V.best += b0 * nd; V.grad += b0 * nd; V.arr1 += b0 * nd; V.arr2 += b0 * nd;
    V.scratch += (size_t)b0 * W.scr;
    V.warped += (size_t)b0 * W.N; V.wgrad += (size_t)b0 * 3 * W.N; V.vgrad += (size_t)b0 * 3 * W.N; V.vtmp += (size_t)b0 * 3 * W.N;
    V.pen += b0; V.lncc += b0; V.S += b0; V.log += (size_t)b0 * LOG_CAP;
    if (V.ev) V.ev += (size_t)b0 * EVLOG_CAP;
    if (V.dbg) V.dbg += (size_t)b0 * 4 * nd;
    V.actGrad += b0; V.trialOK += b0; V.actWarp += b0; V.scale += b0;
    V.lc.B = Bg;
    V.lc.bufA += (size_t)b0 * 6 * W.N; V.lc.bufB += (size_t)b0 * 6 * W.N; V.lc.bufC += (size_t)b0 * 6 * W.N;
    V.lc.partial += (size_t)b0 * W.lc.partStride * 2; V.lc.acc += (size_t)b0 * 4;
    V.gc.B = Bg;
    V.pb.partial += (size_t)b0 * W.pb.nblk * 3; V.pb.fold += b0; V.pb.fix += b0;
    V.flag = flag;
    if (V.wa.cflag) V.wa.cflag += (size_t)b0 * W.wa.ncell;
    return V;
}

static void enqueue_step(LevelWork &W, cudaStream_t st, const float *flo)
{
    ctl_begin(W.S, W.log, W.actGrad, W.B, st);
    // ---- gradient phase (templates with actGrad)
    warp(W.wa, W.B, W.cur, W.ndof, flo, W.Nf, W.warped, W.wgrad, nullptr, W.actGrad, st);
    lncc_gradient(W.lc, W.warped, W.wgrad, W.vgrad, W.actGrad, st);
    float *res = gradconv_apply(W.gc, W.vgrad, W.vtmp, W.actGrad, st);
    vox2node(W.R, W.G, W.F.ijk, W.simWf, res, W.grad, W.B, W.actGrad, st);
    if (W.beW > 0) be_gradient(W.ga, W.cur, W.grad, W.ndof, W.scratch, W.scr, W.beW, W.B, W.actGrad, st);
    if (W.jlW > 0) jac_gradient(W.ga, W.cur, W.grad, W.ndof, W.scratch, W.scr, W.jlW, W.B, W.actGrad, st);
    if (W.leW > 0) le_gradient(W.ga, W.cur, W.grad, W.ndof, W.scratch, W.scr, W.leW, W.B, W.actGrad, st);
    ctl_cg(W.S, W.log, W.ev, W.dbg, W.grad, W.arr1, W.arr2, W.cur, W.best, W.actGrad, W.trialOK, W.scale, W.ndof, W.B, st);
    // ---- trial phase (templates with trialOK)
    penalty_value(W.ga, W.pb, W.cur, W.best, W.grad, W.ndof, W.scratch, W.scr, W.scale, W.jlW, W.beW, W.leW, W.pen, W.B, W.trialOK, st);
    ctl_mid(W.pen, W.trialOK, W.actWarp, W.B, st);
    warp(W.wa, W.B, W.cur, W.ndof, flo, W.Nf, W.warped, nullptr, nullptr, W.actWarp, st);
    lncc_value(W.lc, W.warped, W.lncc, W.actWarp, st);
    ctl_after(W.S, W.log, W.ev, W.cur, W.best, W.lncc, W.pen, W.simW, W.trialOK, W.trialOK, W.scale, W.ndof, W.B, nullptr, st);
    alldone_k<<<1, 1, 0, st>>>(W.S, W.B, W.flag);
}

void F3DBatch::run(const float *const *d_tpl, const ImgGeom &fg, int n, const M44 *aff, float *const *d_out, F3DResult *res,
                   F3DTiming *tm)
{
    Impl &I = *p_;
    cudaStream_t st = I.st;
    auto T0 = std::chrono::steady_clock::now();
    int L = I.P.levelToPerform;
    bool trace = getenv("F3D_TRACE") != nullptr;
    // ---- template pyramids, packed per level
    std::vector<float *> floPack(L, nullptr);
    std::vector<ImgGeom> floG(L);
    for (int i = 0; i < n; i++) {
        std::vector<float *> lv; std::vector<ImgGeom> gl;
        build_pyramid(d_tpl[i], fg, I.P.levelNumber, L, lv, gl, st);
        for (int l = 0; l < L; l++) {
            size_t nv = gl[l].nvox();
            if (!floPack[l]) { floPack[l] = dalloc<float>(nv * n); floG[l] = gl[l]; }
            rescale01(lv[l], nv, st);
            CK(cudaMemcpyAsync(floPack[l] + (size_t)i * nv, lv[l], nv * sizeof(float), cudaMemcpyDeviceToDevice, st));
            CK(cudaStreamSynchronize(st));
            pool_free(lv[l]);
        }
    }
    // ---- initial grids from the affines (level 0)
    GridGeom g0 = I.gridL[0];
    size_t nn0 = g0.nnodes();
    float *prevCp = dalloc<float>(3 * nn0 * n);
    for (int i = 0; i < n; i++) {
        auto h = grid_init_affine(g0, aff[i]);
        CK(cudaMemcpy(prevCp + (size_t)i * 3 * nn0, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    auto T1 = std::chrono::steady_clock::now();
    if (tm) tm->setupMs = std::chrono::duration<double, std::milli>(T1 - T0).count();
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    float *finalCp = nullptr;
    GridGeom prevG = g0;
    for (int l = 0; l < L; l++) {
        LevelWork W;
        W.B = n; W.R = I.refG[l]; W.F = floG[l]; W.G = I.gridL[l];
        W.N = W.R.nvox(); W.Nf = W.F.nvox(); W.nn = W.G.nnodes(); W.ndof = 3 * W.nn;
        W.scr = penalty_scratch_floats(W.G);
        W.ga = make_grid_args(W.G, W.R);
        W.mask = I.maskLv[l];
        level_weights(I.P, l, W.beW, W.leW);
        W.jlW = I.P.jlWeight;
        W.simW = I.simWeight; W.simWf = (float)I.simWeight;
        // control point arrays
        W.cur = dalloc<float>(W.ndof * n); W.best = dalloc<float>(W.ndof * n); W.grad = dalloc_zero<float>(W.ndof * n);
        W.arr1 = dalloc_zero<float>(W.ndof * n); W.arr2 = dalloc_zero<float>(W.ndof * n);
        W.scratch = dalloc_zero<float>(W.scr * n);
        if (l == 0) {
            CK(cudaMemcpyAsync(W.cur, prevCp, W.ndof * n * sizeof(float), cudaMemcpyDeviceToDevice, st));
        } else {
            grid_refine(prevCp, prevG, W.cur, W.G, n, st);
        }
        pool_free(prevCp);
        W.warped = dalloc<float>(W.N * n);
        W.wgrad = dalloc<float>(3 * W.N * n); W.vgrad = dalloc<float>(3 * W.N * n); W.vtmp = dalloc<float>(3 * W.N * n);
        W.pen = dalloc_zero<PenaltyOut>(n); W.lncc = dalloc_zero<double>(n);
        W.S = dalloc_zero<TState>(n); W.log = dalloc_zero<LogRec>((size_t)n * LOG_CAP);
        if (getenv("F3D_TRACE_EVALS")) { W.ev = dalloc_zero<EvalRec>((size_t)n * EVLOG_CAP); W.dbg = dalloc_zero<float>(W.ndof * 4 * n); }
        W.actGrad = dalloc_zero<int>(n); W.trialOK = dalloc_zero<int>(n); W.actWarp = dalloc_zero<int>(n);
        W.scale = dalloc_zero<float>(n); W.flag = dalloc_zero<int>(1);
        // warp tables
        memset(&W.wa, 0, sizeof W.wa);
        for (int a = 0; a < 3; a++) { W.wa.rn[a] = W.R.n[a]; W.wa.gn[a] = W.G.n[a]; W.wa.fn[a] = W.F.n[a]; }
        for (int r = 0; r < 3; r++) for (int c = 0; c < 4; c++) W.wa.fijk[r * 4 + c] = W.F.ijk.m[r][c];
        float gvs[3];
        for (int a = 0; a < 3; a++) gvs[a] = W.G.d[a] / W.R.d[a];
        bool lut = (gvs[0] == 5.f);
        std::vector<AxisTab> htab[3];
        for (int a = 0; a < 3; a++) {
            htab[a] = lut ? build_tab_lut(W.R.n[a], W.G.n[a]) : build_tab_generic(W.R.n[a], gvs[a]);
            W.tab[a] = h2d(htab[a].data(), htab[a].size());
            W.wa.tab[a] = W.tab[a];
        }
        warp_prepare(W.wa, htab);
        const int fmask = (I.P.numerics == 1) ? (I.P.fastLevelMask ? I.P.fastLevelMask : 6) : I.P.fastLevelMask;
        const bool fastL = ((fmask >> l) & 1) != 0;
        W.wa.fast = (I.P.numerics == 1) && fastL;
        W.ga.fast = W.wa.fast;
        W.cflag = dalloc<unsigned char>(W.wa.ncell * n);
        if (!getenv("F3D_NOCELLSKIP")) W.wa.cflag = W.cflag;
        W.wa.mask = W.mask;
        penalty_bufs_init(W.pb, W.G, n);
        lncc_init(W.lc, W.R.n, W.R.d, I.P.lnccSigmaMM, I.refLv[l], W.mask, n, !fastL);
        float ns[3] = {W.G.d[0], W.G.d[1], W.G.d[2]};
        gradconv_init(W.gc, W.R.n, W.R.d, ns, n);
        W.gc.exact = !W.wa.fast;
        gradconv_prune(W.gc, W.R, W.G);
        // optimiser init: bestDOF = copy of the grid, then the initial objective evaluation
        CK(cudaMemcpyAsync(W.best, W.cur, W.ndof * n * sizeof(float), cudaMemcpyDeviceToDevice, st));
        int maxit = I.P.maxIterations;
        {
            double m = (double)maxit * pow(2.0, (double)(L - 1));
            maxit = (int)m;
            for (int k = 0; k < l; k++) maxit /= 2;
        }
        float maxStep = W.R.d[0] > W.R.d[1] ? W.R.d[0] : W.R.d[1];
        maxStep = (W.R.d[2] > maxStep) ? W.R.d[2] : maxStep;
        CK(cudaEventRecord(e0, st));
        {
            penalty_value(W.ga, W.pb, W.cur, W.best, W.grad, W.ndof, W.scratch, W.scr, nullptr, W.jlW, W.beW, W.leW, W.pen, n, nullptr, st);
            // all templates are evaluated (even NaN penalty ones; their objective is NaN anyway)
            CK(cudaMemsetAsync(W.actWarp, 0x01, n * sizeof(int), st));
            // memset 0x01 per byte gives 0x01010101 (non-zero) -> active
            warp(W.wa, n, W.cur, W.ndof, floPack[l], W.Nf, W.warped, nullptr, nullptr, W.actWarp, st);
            lncc_value(W.lc, W.warped, W.lncc, W.actWarp, st);
            ctl_init_level(W.S, W.ev, W.lncc, W.pen, W.simW, maxStep, maxit, n, st);
        }
        if (trace) {
            CK(cudaStreamSynchronize(st));
            auto hs = d2h(W.S, n);
            for (int b = 0; b < n && b < 3; b++) printf("[L%d t%d] initial objective %.9g = sim %.9g - be %.9g - le %.9g - jac %.9g\n", l, b, hs[b].bestObj, hs[b].bestSim, hs[b].bestBE, hs[b].bestLE, hs[b].bestJac);
        }
        // ---- iterate: the batch is split into G groups of templates, each with its own stream (kernels of different groups overlap:
        // the double-precision / latency-bound penalty kernels of one group run next to the LNCC kernels of another one)
        int steps = 0;
        {
            int G = I.P.groups > 0 ? I.P.groups : 1;
            if (getenv("F3D_GROUPS")) G = atoi(getenv("F3D_GROUPS"));
            if (W.ev || W.dbg || n < 2) G = 1;
            if (G > n) G = n;
            I.ensure_streams(G);
            std::vector<LevelWork> V(G);
            std::vector<int *> dflag(G);
            std::vector<int> bs(G), nb(G);
            for (int g = 0; g < G; g++) {
                bs[g] = (int)((long)n * g / G); nb[g] = (int)((long)n * (g + 1) / G) - bs[g];
                dflag[g] = (g == 0) ? W.flag : dalloc_zero<int>(1);
                V[g] = make_view(W, bs[g], nb[g], dflag[g]);
            }
            CK(cudaEventRecord(I.evMain, st));
            std::vector<int> gsteps(G, 0), active(G, 1);
            int nactive = G;
            const int K = (W.N > 200000) ? 2 : 8;
            // CUDA graphs: one macro step (about 45 kernel launches) is captured once per group and replayed (host launch cost: the 8 groups
            // together issue ~160k launches at the small levels, which makes the host the bottleneck there). Same kernels, same order, same results.
            static const bool nograph = getenv("F3D_NOGRAPH") != nullptr;
            const bool useGraph = !nograph && !W.ev && !W.dbg;
            const int KG = 2;   // macro steps per graph
            std::vector<cudaGraphExec_t> gexec(G, nullptr);
            if (useGraph) {
                for (int g = 0; g < G; g++) {
                    cudaStream_t gs = I.gstream[g];
                    CK(cudaStreamWaitEvent(gs, I.evMain, 0));
                    cudaGraph_t gr;
                    CK(cudaStreamBeginCapture(gs, cudaStreamCaptureModeThreadLocal));
                    for (int k = 0; k < KG; k++) enqueue_step(V[g], gs, floPack[l] + (size_t)bs[g] * W.Nf);
                    CK(cudaStreamEndCapture(gs, &gr));
                    CK(cudaGraphInstantiate(&gexec[g], gr, 0));
                    CK(cudaGraphDestroy(gr));
                }
            }
            auto enqueue_group = [&](int g) {
                cudaStream_t gs = I.gstream[g];
                if (useGraph) { for (int k = 0; k < K / KG; k++) { CK(cudaGraphLaunch(gexec[g], gs)); gsteps[g] += KG; } }
                else for (int k = 0; k < K; k++) { enqueue_step(V[g], gs, floPack[l] + (size_t)bs[g] * W.Nf); gsteps[g]++; }
                CK(cudaMemcpyAsync(I.hflag + g, dflag[g], sizeof(int), cudaMemcpyDeviceToHost, gs));
                CK(cudaEventRecord(I.evGroup[g], gs));
            };
            for (int g = 0; g < G; g++) { CK(cudaStreamWaitEvent(I.gstream[g], I.evMain, 0)); I.hflag[g] = 0; enqueue_group(g); }
            while (nactive > 0) {
                bool any = false;
                for (int g = 0; g < G; g++) {
                    if (!active[g]) continue;
                    if (cudaEventQuery(I.evGroup[g]) != cudaSuccess) continue;
                    any = true;
                    if (I.hflag[g] || gsteps[g] > 6000) { if (gsteps[g] > 6000) fprintf(stderr, "F3D: step limit reached\n"); active[g] = 0; nactive--; }
                    else enqueue_group(g);
                }
                if (!any) std::this_thread::yield();
            }
            for (int g = 0; g < G; g++) { CK(cudaStreamWaitEvent(st, I.evGroup[g], 0)); steps = std::max(steps, gsteps[g]); }
            CK(cudaStreamSynchronize(st));
            for (int g = 0; g < G; g++) if (gexec[g]) CK(cudaGraphExecDestroy(gexec[g]));
            for (int g = 1; g < G; g++) pool_free(dflag[g]);
        }
        CK(cudaEventRecord(e1, st));
        CK(cudaEventSynchronize(e1));
        { cudaError_t le = cudaGetLastError(); if (le != cudaSuccess) { fprintf(stderr, "F3D: CUDA error after level %d: %s (a kernel launch failed)\n", l, cudaGetErrorString(le)); abort(); } }
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (tm) { tm->levelMs[l] = ms; tm->macroSteps[l] = steps; }
        // ---- results of the level
        auto hs = d2h(W.S, n);
        auto hlog = d2h(W.log, (size_t)n * LOG_CAP);
        if (W.dbg) {
            auto hd = d2h(W.dbg, W.ndof * 4 * n);
            char fn[128]; snprintf(fn, sizeof fn, "/tmp/gpu_grad_L%d.bin", l);
            FILE *f = fopen(fn, "wb"); fwrite(hd.data(), 4, W.ndof * 4, f); fclose(f);
            pool_free(W.dbg);
        }
        if (W.ev) {
            auto he = d2h(W.ev, (size_t)n * EVLOG_CAP);
            for (int b = 0; b < n; b++) { I.evtrace.resize(L); I.evtrace[l].resize(n); I.evtrace[l][b].assign(he.begin() + (size_t)b * EVLOG_CAP, he.begin() + (size_t)b * EVLOG_CAP + (hs[b].nev < EVLOG_CAP ? hs[b].nev : EVLOG_CAP)); }
        }
        for (int b = 0; b < n; b++) {
            if (res) {
                res[b].iterations[l] = hs[b].iter; res[b].gradEvals[l] = hs[b].nevalGrad; res[b].valueEvals[l] = hs[b].nevalValue;
                res[b].finalObjective[l] = hs[b].bestObj; res[b].finalLncc[l] = hs[b].bestSim;
            }
            auto &tr = I.traces[l][b];
            tr.assign(hlog.begin() + (size_t)b * LOG_CAP, hlog.begin() + (size_t)b * LOG_CAP + (hs[b].nlog < LOG_CAP ? hs[b].nlog : LOG_CAP));
        }
        // ---- CorrectTransformation: voxel based Jacobian penalty (approximation switched off), up to 20 correction passes
        auto Tc0 = std::chrono::steady_clock::now();
        if (I.P.jlWeight > 0) {
            const int Bc = std::min(n, getenv("F3D_VJ_CHUNK") ? atoi(getenv("F3D_VJ_CHUNK")) : 10);
            VoxJacCtx vc; voxjac_init(vc, W.ga, W.G, W.R, Bc);
            int nfold = 0, nfail = 0;
            for (int b0 = 0; b0 < n; b0 += Bc) {
                int nt = std::min(Bc, n - b0), f1 = 0, f2 = 0;
                voxjac_fix_batch(vc, W.cur + (size_t)b0 * W.ndof, W.best + (size_t)b0 * W.ndof, nt, st, &f1, &f2);
                nfold += f1; nfail += f2;
            }
            CK(cudaStreamSynchronize(st));
            if (trace) printf("[L%d] voxel-level folding correction: %d templates corrected, %d failed (restored)\n", l, nfold, nfail);
            voxjac_free(vc);
        }
        if (tm) tm->correctMs[l] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - Tc0).count();
        prevCp = W.cur; prevG = W.G; W.cur = nullptr;
        finalCp = prevCp;
        // free level data
        if (W.cflag) pool_free(W.cflag);
        pool_free(W.best); pool_free(W.grad); pool_free(W.arr1); pool_free(W.arr2); pool_free(W.scratch);
        pool_free(W.warped); pool_free(W.wgrad); pool_free(W.vgrad); pool_free(W.vtmp); pool_free(W.pen); pool_free(W.lncc);
        pool_free(W.S); pool_free(W.log); if (W.ev) pool_free(W.ev); pool_free(W.actGrad); pool_free(W.trialOK); pool_free(W.actWarp); pool_free(W.scale); pool_free(W.flag);
        for (int a = 0; a < 3; a++) pool_free(W.tab[a]);
        lncc_free(W.lc); gradconv_free(W.gc); penalty_bufs_free(W.pb);
    }
    for (int i = 0; i < n; i++)
        CK(cudaMemcpyAsync(d_out[i], finalCp + (size_t)i * prevG.nnodes() * 3, prevG.nnodes() * 3 * sizeof(float), cudaMemcpyDeviceToDevice, st));
    CK(cudaStreamSynchronize(st));
    pool_free(finalCp);
    for (auto d : floPack) pool_free(d);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    auto T2 = std::chrono::steady_clock::now();
    if (tm) tm->totalMs = std::chrono::duration<double, std::milli>(T2 - T0).count();
}

}  // namespace f3d
