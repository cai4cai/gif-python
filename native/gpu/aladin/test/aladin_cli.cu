// SPDX-License-Identifier: BSD-3-Clause
// Test / benchmark driver: ./aladin_cli <out_dir> <reps> <target.hdr> <tmpl1.hdr> ... <tmplN.hdr>
// Registers all templates to the target on the GPU, prints timings, writes <out_dir>/<template>.txt
// (4x4 matrix, text, reference world -> floating world).  Inputs are the raw files written by tools/prep_data.py.
// Environment: ALADIN_LP (levels to perform), ALADIN_EXACT, ALADIN_FAST, ALADIN_STEP, ALADIN_PERTURB/ALADIN_SEED.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include "../aladin.h"
using namespace aladin;

static bool load(const std::string& hdr, Image& im, float** dev, std::string& name) {
  FILE* f = fopen(hdr.c_str(), "r");
  if (!f) return false;
  if (fscanf(f, "%d %d %d", &im.nx, &im.ny, &im.nz) != 3) return false;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) if (fscanf(f, "%f", &im.sform.m[i][j]) != 1) return false;
  fclose(f);
  std::string raw = hdr.substr(0, hdr.size() - 4) + ".f32";
  size_t nv = (size_t)im.nx * im.ny * im.nz;
  std::vector<float> h(nv);
  FILE* g = fopen(raw.c_str(), "rb");
  if (!g || fread(h.data(), 4, nv, g) != nv) return false;
  fclose(g);
  cudaMalloc(dev, nv * 4);
  cudaMemcpy(*dev, h.data(), nv * 4, cudaMemcpyHostToDevice);
  im.dev = *dev;
  size_t p = hdr.find_last_of('/');
  name = hdr.substr(p == std::string::npos ? 0 : p + 1);
  name = name.substr(0, name.size() - 4);
  return true;
}
__global__ void perturb_kernel(float* p, size_t n, float eps, int seed) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) p[i] *= (1.0f + eps * sinf((float)(i % 100003) * 0.7f + (float)seed * 1.3f));
}
static void write_mat(const std::string& fn, const Mat44& m) {
  FILE* f = fopen(fn.c_str(), "w");
  for (int i = 0; i < 4; ++i) fprintf(f, "%.7f %.7f %.7f %.7f\n", m.m[i][0], m.m[i][1], m.m[i][2], m.m[i][3]);
  fclose(f);
}
int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: %s out_dir reps target.hdr t1.hdr ...\n", argv[0]); return 1; }
  std::string out = argv[1];
  int reps = atoi(argv[2]);
  size_t mem_free0 = 0, mem_total = 0;
  cudaFree(0);
  cudaMemGetInfo(&mem_free0, &mem_total);
  Image tg; float* dtg; std::string tname;
  if (!load(argv[3], tg, &dtg, tname)) { fprintf(stderr, "cannot load target\n"); return 1; }
  int n = argc - 4;
  std::vector<Image> tm(n); std::vector<float*> dt(n); std::vector<std::string> names(n);
  for (int i = 0; i < n; ++i) if (!load(argv[4 + i], tm[i], &dt[i], names[i])) { fprintf(stderr, "cannot load %s\n", argv[4 + i]); return 1; }
  cudaFree(0);
  if (getenv("ALADIN_PERTURB")) {  // debug: multiply templates by (1 + eps*sin(..)) to measure the sensitivity of the discrete algorithm
    float eps = atof(getenv("ALADIN_PERTURB"));
    int seed = getenv("ALADIN_SEED") ? atoi(getenv("ALADIN_SEED")) : 0;
    for (int i = 0; i < n; ++i) {
      size_t nv = (size_t)tm[i].nx * tm[i].ny * tm[i].nz;
      perturb_kernel<<<(unsigned)((nv + 255) / 256), 256>>>(dt[i], nv, eps, seed + i);
    }
    cudaDeviceSynchronize();
  }
  std::vector<Mat44> res(n);
  std::vector<double> tot, reg, tsetup, tgsetup;
  std::vector<std::vector<double>> lev(8);
  Params p;
  if (getenv("ALADIN_LP")) p.levels_to_perform = atoi(getenv("ALADIN_LP"));
  if (getenv("ALADIN_EXACT")) p.exact = true;
  if (getenv("ALADIN_FAST")) p.fast_solver = true;
  if (getenv("ALADIN_STEP")) p.block_step = atoi(getenv("ALADIN_STEP"));
  for (int r = 0; r < reps; ++r) {
    AladinBatch b(tg, p);
    TemplateSet ts(tm.data(), n, p);
    b.run(ts, res.data());
    if (r == 0) {
      size_t f1 = 0, tt = 0;
      cudaMemGetInfo(&f1, &tt);
      printf("device memory in use with target + %d templates + pyramids + work buffers + allocator pool: %.0f MB (inputs alone: %.0f MB)\n", n, (double)(mem_free0 - f1) / 1e6, (double)n * tm[0].nx * tm[0].ny * tm[0].nz * 4 / 1e6 + (double)tg.nx * tg.ny * tg.nz * 4 / 1e6);
    }
    const Timings& t = b.timings();
    tgsetup.push_back(t.target_setup); tsetup.push_back(t.template_setup); reg.push_back(t.registration);
    for (int l = 0; l < t.levels; ++l) lev[l].push_back(t.level[l]);
    tot.push_back(t.total());
    std::string st;
    for (int i = 0; i < n; ++i) if (b.status()[i]) st += " " + std::to_string(i);
    printf("rep %d: target_setup %.2f ms, template_setup %.2f ms, registration %.2f ms (levels", r, t.target_setup, t.template_setup, t.registration);
    for (int l = 0; l < t.levels; ++l) printf(" %.2f", t.level[l]);
    printf(") total %.2f ms%s\n", t.total(), st.empty() ? "" : (" FAILED:" + st).c_str());
  }
  auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
  auto mn = [](std::vector<double> v) { return *std::min_element(v.begin(), v.end()); };
  printf("SUMMARY n=%d reps=%d  min/median ms: target_setup %.2f/%.2f template_setup %.2f/%.2f registration %.2f/%.2f total %.2f/%.2f\n", n, reps,
         mn(tgsetup), med(tgsetup), mn(tsetup), med(tsetup), mn(reg), med(reg), mn(tot), med(tot));
  for (int l = 0; l < p.levels_to_perform; ++l) printf("  level %d: min %.2f median %.2f ms\n", l, mn(lev[l]), med(lev[l]));
  for (int i = 0; i < n; ++i) write_mat(out + "/" + names[i] + ".txt", res[i]);
  return 0;
}
