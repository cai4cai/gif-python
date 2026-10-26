// SPDX-License-Identifier: BSD-3-Clause
// Registration settings used by the tests and by the reference programs (ref_f3d, ref_f3d_dbg); the two sides must agree.
// The values are an arbitrary non-default test configuration that exercises the bending-energy and Jacobian penalties and a
// 3-level run of a 4-level pyramid.
#pragma once
// The two penalty weights can be overridden by the environment variables F3D_TEST_BE and F3D_TEST_JL (read by both sides).
#include <cstdlib>
namespace testcfg {
inline float envf(const char *name, float def) { const char *e = getenv(name); return e ? (float)atof(e) : def; }
inline float beWeight_() { return envf("F3D_TEST_BE", 0.005f); }
inline float jlWeight_() { return envf("F3D_TEST_JL", 0.0001f); }
constexpr int levelNumber = 4;
constexpr int levelToPerform = 3;
constexpr int maxIterations = 250;
constexpr float lnccSigmaMM = 5.0f;
constexpr float spacingVox = -5.0f;
}  // namespace testcfg
