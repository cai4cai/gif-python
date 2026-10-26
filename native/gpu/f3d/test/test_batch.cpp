// SPDX-License-Identifier: BSD-3-Clause
// Batched registration of N templates to one target (settings of test_config.h); optionally compares the final control point
// grids with grids written by NiftyReg (test/ref_f3d).
// usage: test_batch target.nii[.gz] affine.txt reps cpu_grid_pattern tpl1.nii [tpl2.nii ...]
//   cpu_grid_pattern: printf pattern with one %s (template basename without extension), e.g. /tmp/ref/%s_cpp.nii, or "-" for none
// environment: F3D_FASTMASK, F3D_NUMERICS (see F3DParams), F3D_SAVE=<dir> (raw grids as gpu_cpp_<i>.bin)
#include "f3d.h"
#include "test_common.h"
#include <chrono>
#include <string>
static std::string base_of(const std::string &p)
{
    std::string b = p.substr(p.find_last_of('/') + 1);
    return b.substr(0, b.find(".nii"));
}
int main(int argc, char **argv)
{
    if (argc < 6) { fprintf(stderr, "usage: test_batch target affine reps cpu_grid_pattern|- tpl1 [tpl2 ...]\n"); return 1; }
    nifti_image *ref = load(argv[1]);
    mat44 aff; reg_tool_ReadAffineFile(&aff, argv[2]);
    int reps = atoi(argv[3]);
    std::string pattern = argv[4];
    int n = argc - 5;
    size_t memFree0 = 0, memTot = 0;
    cudaMemGetInfo(&memFree0, &memTot);
    ImgGeom rg = geom_of(ref);
    float *dref = h2d((float *)ref->data, ref->nvox);
    std::vector<unsigned char> ones(ref->nvox, 1);
    unsigned char *dmask = h2d(ones.data(), ones.size());
    F3DParams P = test_params();
    if (getenv("F3D_FASTMASK")) P.fastLevelMask = atoi(getenv("F3D_FASTMASK"));
    if (getenv("F3D_NUMERICS")) P.numerics = atoi(getenv("F3D_NUMERICS"));
    F3DBatch batch(dref, rg, dmask, P, n);
    GridGeom fgrid = batch.finalGrid();
    std::vector<float *> dt(n), dout(n);
    ImgGeom fg;
    std::vector<nifti_image *> cpu(n, nullptr);
    for (int i = 0; i < n; i++) {
        nifti_image *flo = load(argv[5 + i]);
        fg = geom_of(flo);
        dt[i] = h2d((float *)flo->data, flo->nvox);
        dout[i] = dalloc<float>(3 * fgrid.nnodes());
        nifti_image_free(flo);
        if (pattern != "-") {
            char fn[1024];
            snprintf(fn, sizeof fn, pattern.c_str(), base_of(argv[5 + i]).c_str());
            cpu[i] = nifti_image_read(fn, true);
            if (!cpu[i]) { fprintf(stderr, "cannot read %s\n", fn); return 1; }
        }
    }
    std::vector<M44> affs(n, to_m44(aff));
    std::vector<F3DResult> res(n);
    F3DTiming tm;
    std::vector<const float *> cdt(dt.begin(), dt.end());
    for (int r = 0; r < reps; r++) {
        batch.run(cdt.data(), fg, n, affs.data(), dout.data(), res.data(), &tm);
        printf("run %d: total %.1f ms (setup %.1f) levels:", r, tm.totalMs, tm.setupMs);
        for (int l = 0; l < 3; l++) printf(" L%d %.1f ms/%d steps (+%.1f ms correction)", l, tm.levelMs[l], tm.macroSteps[l], tm.correctMs[l]);
        printf("\n");
    }
    { size_t f1 = 0; cudaMemGetInfo(&f1, &memTot); printf("device memory taken by this process (incl. caller buffers and the pool): %.0f MB\n", ((double)memFree0 - (double)f1) / 1048576.0); }
    double sumMean = 0, maxMax = 0;
    printf("tpl | iterations L0 L1 L2 | final objective L0 L1 L2 | final LNCC(sim) L2 | diff vs NiftyReg grid: mean max (mm)\n");
    for (int b = 0; b < n; b++) {
        auto h = d2h(dout[b], 3 * fgrid.nnodes());
        double mean = 0, mx = 0;
        if (cpu[b])
            for (size_t i = 0; i < h.size(); i++) { double d = fabs((double)h[i] - ((float *)cpu[b]->data)[i]); mean += d; mx = d > mx ? d : mx; }
        mean /= h.size(); sumMean += mean; maxMax = mx > maxMax ? mx : maxMax;
        printf("%3d | %4d %4d %4d | %.6f %.6f %.6f | %.6f | %.4f %.3f\n", b + 1, res[b].iterations[0], res[b].iterations[1], res[b].iterations[2],
               res[b].finalObjective[0], res[b].finalObjective[1], res[b].finalObjective[2], res[b].finalLncc[2], mean, mx);
        if (getenv("F3D_SAVE")) { char fn[256]; snprintf(fn, sizeof fn, "%s/gpu_cpp_%d.bin", getenv("F3D_SAVE"), b + 1); FILE *f = fopen(fn, "wb"); fwrite(h.data(), 4, h.size(), f); fclose(f); }
    }
    if (pattern != "-") printf("mean of mean diffs %.4f mm, max of max %.3f mm\n", sumMean / n, maxMax);
    return 0;
}
