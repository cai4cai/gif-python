// SPDX-License-Identifier: BSD-3-Clause
// Device-resident control logic: gradient normalisation, conjugate gradient, line search bookkeeping (reg_optimiser)
#include "f3d_internal.h"
#include "f3d_state.h"
#include <cmath>

namespace f3d {

__device__ double bsum(double v, double *sh)
{
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    double r = 0.0;
    if (threadIdx.x < 32) {
        r = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.0;
        for (int o = 16; o > 0; o >>= 1) r += __shfl_down_sync(0xffffffffu, r, o);
        if (threadIdx.x == 0) sh[0] = r;
    }
    __syncthreads();
    r = sh[0];
    __syncthreads();
    return r;
}
__device__ float bmax(float v, float *sh)
{
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    float r = 0.f;
    if (threadIdx.x < 32) {
        r = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.f;
        for (int o = 16; o > 0; o >>= 1) r = fmaxf(r, __shfl_down_sync(0xffffffffu, r, o));
        if (threadIdx.x == 0) sh[0] = r;
    }
    __syncthreads();
    r = sh[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ bool ls_cond(const TState &s) { return s.curLen > s.smallStep && s.lineIter < 12 && s.iter < s.maxit; }

// end of a line search: startLength = added ; restore best DOF ; record log
__device__ void finish_ls(TState *s, LogRec *log, float *cur, const float *best, size_t ndof)
{
    for (size_t i = threadIdx.x; i < ndof; i += blockDim.x) cur[i] = best[i];
    __syncthreads();
    if (threadIdx.x == 0) {
        s->startLen = s->added;
        s->mode = MODE_GRAD;
        if (s->nlog < LOG_CAP) {
            LogRec &r = log[s->nlog];
            r.iter = s->iter; r.added = s->added; r.bestObj = s->bestObj; r.sim = s->bestSim; r.be = s->bestBE; r.le = s->bestLE; r.jac = s->bestJac;
        }
        s->nlog++;
    }
    __syncthreads();
}

__global__ void ctl_init_k(TState *S, EvalRec *ev, const double *lncc, const PenaltyOut *pen, double simW, float maxStep, int maxit, int B)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    TState &s = S[b];
    double sim = (simW > 0) ? simW * lncc[b] : 0.0;
    s.curObj = s.bestObj = sim - pen[b].be - pen[b].le - pen[b].jac;
    s.bestSim = sim; s.bestBE = pen[b].be; s.bestLE = pen[b].le; s.bestJac = pen[b].jac;
    s.maxStep = maxStep;
    s.smallStep = maxStep / (float)100.0;
    s.startLen = maxStep;
    s.curLen = 0.f; s.added = 0.f;
    s.lineIter = 0; s.iter = 0; s.maxit = maxit; s.mode = MODE_GRAD; s.firstcall = 1; s.nlog = 0;
    s.nevalValue = 1; s.nevalGrad = 0; s.nev = 0;
    if (ev) { EvalRec &r = ev[(size_t)b * EVLOG_CAP]; r.kind = 0; r.scale = 0.f; r.obj = s.bestObj; r.sim = sim; r.be = pen[b].be; r.le = pen[b].le; r.jac = pen[b].jac; s.nev = 1; }
}
void ctl_init_level(TState *S, EvalRec *ev, const double *lncc, const PenaltyOut *pen, double simW, float maxStep, int maxit, int B, cudaStream_t st)
{
    ctl_init_k<<<cdiv(B, 32), 32, 0, st>>>(S, ev, lncc, pen, simW, maxStep, maxit, B);
}

__global__ void ctl_begin_k(TState *S, LogRec *log, int *actGrad, int B)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    TState &s = S[b];
    int a = 0;
    if (s.mode == MODE_GRAD) {
        if (s.startLen == 0.f || s.iter >= s.maxit) s.mode = MODE_DONE;
        else if (s.bestObj != s.bestObj) {
            // The best objective value is NaN (folded start): `curObj > bestObj` can never be true, so every trial of the line
            // search is rejected and the grid is left unchanged. Fast-forward the iteration bookkeeping exactly as NiftyReg
            // would run it (one gradient evaluation, then the line search until the step is below the smallest size).
            s.iter++; s.nevalGrad++;
            float cs = s.startLen > s.maxStep ? s.maxStep : s.startLen;
            int li = 0;
            while (cs > s.smallStep && li < 12 && s.iter < s.maxit) {
                cs = (float)((double)cs * 0.5);
                s.iter++; li++; s.nevalValue++;
            }
            s.curLen = cs; s.lineIter = li; s.added = 0.f; s.startLen = 0.f;
            if (s.nlog < LOG_CAP) {
                LogRec &r = log[(size_t)b * LOG_CAP + s.nlog];
                r.iter = s.iter; r.added = 0.f; r.bestObj = s.bestObj; r.sim = s.bestSim; r.be = s.bestBE; r.le = s.bestLE; r.jac = s.bestJac;
            }
            s.nlog++;
            s.mode = MODE_DONE;
        } else a = 1;
    }
    actGrad[b] = a;
}
void ctl_begin(TState *S, LogRec *log, int *actGrad, int B, cudaStream_t st)
{
    ctl_begin_k<<<cdiv(B, 32), 32, 0, st>>>(S, log, actGrad, B);
}

// gradient normalisation + conjugate gradient + line-search initialisation (one block per template)
__global__ void ctl_cg_k(TState *S, LogRec *log, EvalRec *ev, float *dbg, float *grad, float *arr1, float *arr2, float *cur, const float *best,
                         const int *actGrad, int *trialOK, float *scale, size_t ndof)
{
    __shared__ double shd[32];
    __shared__ float shf[32];
    int b = blockIdx.x;
    if (!actGrad[b]) return;
    TState *s = &S[b];
    size_t nn = ndof / 3;
    float *g = grad + (size_t)b * ndof, *a1 = arr1 + (size_t)b * ndof, *a2 = arr2 + (size_t)b * ndof;
    float *cp = cur + (size_t)b * ndof;
    const float *bp = best + (size_t)b * ndof;
    if (dbg && s->nevalGrad < 4) {
        float *d = dbg + ((size_t)b * 4 + s->nevalGrad) * ndof;
        for (size_t i = threadIdx.x; i < ndof; i += blockDim.x) d[i] = g[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) { s->iter++; s->nevalGrad++; }
    __syncthreads();
    // --- NormaliseGradient
    float mx = 0.f;
    for (size_t i = threadIdx.x; i < nn; i += blockDim.x) {
        float vx = g[i], vy = g[nn + i], vz = g[2 * nn + i];
        float len = sqrtf(vx * vx + vy * vy + vz * vz);
        mx = (len > mx) ? len : mx;
    }
    mx = bmax(mx, shf);
    if (threadIdx.x == 0 && ev && s->nev < EVLOG_CAP) { EvalRec &r = ev[(size_t)b * EVLOG_CAP + s->nev]; r.kind = 1; r.scale = 0.f; r.sim = mx; r.obj = r.be = r.le = r.jac = 0; s->nev++; }
    for (size_t i = threadIdx.x; i < nn; i += blockDim.x) {
        float vx = g[i], vy = g[nn + i], vz = g[2 * nn + i];
        g[i] = vx / mx; g[nn + i] = vy / mx; g[2 * nn + i] = vz / mx;
    }
    __syncthreads();
    // --- currentSize = min(currentSize, maxStepSize)
    float cs = s->startLen;
    cs = cs > s->maxStep ? s->maxStep : cs;
    // --- reg_conjugateGradient::UpdateGradientValues
    if (s->firstcall) {
        for (size_t i = threadIdx.x; i < ndof; i += blockDim.x) { float v = -g[i]; a1[i] = v; a2[i] = v; }
        __syncthreads();
        if (threadIdx.x == 0) s->firstcall = 0;
    } else {
        double gg = 0.0, dgg = 0.0;
        for (size_t i = threadIdx.x; i < ndof; i += blockDim.x) {
            gg += (double)(a2[i] * a1[i]);
            dgg += (double)((g[i] + a1[i]) * g[i]);
        }
        gg = bsum(gg, shd);
        dgg = bsum(dgg, shd);
        double gam = dgg / gg;
        for (size_t i = threadIdx.x; i < ndof; i += blockDim.x) {
            a1[i] = -g[i];
            a2[i] = (float)((double)a1[i] + gam * (double)a2[i]);
            g[i] = -a2[i];
        }
        __syncthreads();
    }
    // --- line search init (reg_optimiser::Optimise)
    if (threadIdx.x == 0) { s->lineIter = 0; s->added = 0.f; s->curLen = cs; s->mode = MODE_LS; }
    __syncthreads();
    bool ok = ls_cond(*s);
    if (threadIdx.x == 0) { trialOK[b] = ok ? 1 : 0; scale[b] = -s->curLen; }
    __syncthreads();
    if (!ok) finish_ls(s, log + (size_t)b * LOG_CAP, cp, bp, ndof);
}
void ctl_cg(TState *S, LogRec *log, EvalRec *ev, float *dbg, float *grad, float *arr1, float *arr2, float *cur, const float *best, const int *actGrad,
            int *trialOK, float *scale, size_t ndof, int B, cudaStream_t st)
{
    ctl_cg_k<<<B, 1024, 0, st>>>(S, log, ev, dbg, grad, arr1, arr2, cur, best, actGrad, trialOK, scale, ndof);
}

__global__ void ctl_mid_k(const PenaltyOut *pen, const int *trialOK, int *actWarp, int B)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    actWarp[b] = (trialOK[b] && pen[b].jac == pen[b].jac) ? 1 : 0;
}
void ctl_mid(const PenaltyOut *pen, const int *trialOK, int *actWarp, int B, cudaStream_t st)
{
    ctl_mid_k<<<cdiv(B, 32), 32, 0, st>>>(pen, trialOK, actWarp, B);
}

// objective evaluation outcome of one trial
__global__ void ctl_after_k(TState *S, LogRec *log, EvalRec *ev, float *cur, float *best, const double *lncc, const PenaltyOut *pen,
                            double simW, int *trialOK, float *scale, size_t ndof)
{
    int b = blockIdx.x;
    if (!trialOK[b]) return;
    TState *s = &S[b];
    float *cp = cur + (size_t)b * ndof, *bp = best + (size_t)b * ndof;
    __shared__ int accept;
    if (threadIdx.x == 0) {
        double sim = (simW > 0) ? simW * lncc[b] : 0.0;
        const PenaltyOut &p = pen[b];
        double obj = sim - p.be - p.le - p.jac;
        s->curObj = obj;
        if (ev && s->nev < EVLOG_CAP) { EvalRec &r = ev[(size_t)b * EVLOG_CAP + s->nev]; r.kind = 0; r.scale = scale[b]; r.obj = obj; r.sim = sim; r.be = p.be; r.le = p.le; r.jac = p.jac; s->nev++; }
        s->nevalValue++;
        accept = (obj > s->bestObj) ? 1 : 0;
        if (accept) {
            s->bestObj = obj; s->bestSim = sim; s->bestBE = p.be; s->bestLE = p.le; s->bestJac = p.jac;
            s->added += s->curLen;
            s->curLen *= 1.1f;
            s->curLen = (s->curLen < s->maxStep) ? s->curLen : s->maxStep;
        } else {
            s->curLen = (float)((double)s->curLen * 0.5);
        }
        s->iter++;
        s->lineIter++;
    }
    __syncthreads();
    if (accept) {
        for (size_t i = threadIdx.x; i < ndof; i += blockDim.x) bp[i] = cp[i];
    }
    __syncthreads();
    bool ok = ls_cond(*s);
    if (threadIdx.x == 0) { trialOK[b] = ok ? 1 : 0; scale[b] = -s->curLen; }
    __syncthreads();
    if (!ok) finish_ls(s, log + (size_t)b * LOG_CAP, cp, bp, ndof);
}
void ctl_after(TState *S, LogRec *log, EvalRec *ev, float *cur, float *best, const double *lncc, const PenaltyOut *pen, double simW,
               const int *trialOK, int *trialOKnext, float *scale, size_t ndof, int B, int *doneCount, cudaStream_t st)
{
    ctl_after_k<<<B, 1024, 0, st>>>(S, log, ev, cur, best, lncc, pen, simW, trialOKnext, scale, ndof);
}

}  // namespace f3d
