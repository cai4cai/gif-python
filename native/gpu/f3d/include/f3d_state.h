// SPDX-License-Identifier: BSD-3-Clause
// Per-template optimiser state (device resident) -- mirrors reg_optimiser / reg_conjugateGradient / reg_f3d bookkeeping.
#pragma once
#include <cstddef>
namespace f3d {

enum { MODE_GRAD = 0, MODE_LS = 1, MODE_DONE = 2 };

struct LogRec {
    int iter;
    float added;
    double bestObj, sim, be, le, jac;
};
struct EvalRec {
    int kind;      // 0 = objective evaluation, 1 = gradient normalisation (sim holds the max gradient length)
    float scale;
    double obj, sim, be, le, jac;
};
struct TState {
    double bestObj, curObj;
    double bestSim, bestBE, bestLE, bestJac;
    float startLen, curLen, added, maxStep, smallStep;
    int lineIter, iter, maxit, mode, firstcall, nlog, nevalValue, nevalGrad, nev;
};
static const int LOG_CAP = 1200;
static const int EVLOG_CAP = 4000;

}  // namespace f3d
