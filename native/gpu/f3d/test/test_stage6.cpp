// SPDX-License-Identifier: BSD-3-Clause
// Stage 6/7/8 test: full multi-level GPU registration of a target to one or more templates; compare to NiftyReg outputs.
// usage: test_stage6 target.nii.gz affine.txt cpu_cpp.nii tpl1.nii [tpl2.nii ...]
#include "f3d.h"
#include "test_common.h"
#include <chrono>
int main(int argc, char **argv)
{
    if (argc < 5) { fprintf(stderr, "usage: test_stage6 target affine cpu_cpp tpl...\n"); return 1; }
    nifti_image *ref = load(argv[1]);
    mat44 aff; reg_tool_ReadAffineFile(&aff, argv[2]);
    nifti_image *cpu = nifti_image_read(argv[3], true);
    int n = argc - 4;
    ImgGeom rg = geom_of(ref);
    float *dref = h2d((float *)ref->data, ref->nvox);
    std::vector<unsigned char> ones(ref->nvox, 1);
    unsigned char *dmask = h2d(ones.data(), ones.size());
    F3DParams P = test_params();
    F3DBatch batch(dref, rg, dmask, P, n);
    GridGeom fgrid = batch.finalGrid();
    std::vector<float *> dt(n), dout(n);
    ImgGeom fg;
    for (int i = 0; i < n; i++) {
        nifti_image *flo = load(argv[4 + i]);
        fg = geom_of(flo);
        dt[i] = h2d((float *)flo->data, flo->nvox);
        dout[i] = dalloc<float>(3 * fgrid.nnodes());
        nifti_image_free(flo);
    }
    std::vector<M44> affs(n, to_m44(aff));
    std::vector<F3DResult> res(n);
    F3DTiming tm;
    std::vector<const float *> cdt(dt.begin(), dt.end());
    batch.run(cdt.data(), fg, n, affs.data(), dout.data(), res.data(), &tm);
    printf("total %.1f ms (setup %.1f)  levels:", tm.totalMs, tm.setupMs);
    for (int l = 0; l < 3; l++) printf(" L%d %.1f ms/%d steps", l, tm.levelMs[l], tm.macroSteps[l]);
    printf("\n");
    for (int b = 0; b < n && b < 3; b++)
        for (int l = 0; l < 3; l++) {
            printf("--- template %d level %d: iter=%d grad=%d value=%d final obj %.9g sim %.9g\n", b, l, res[b].iterations[l], res[b].gradEvals[l], res[b].valueEvals[l], res[b].finalObjective[l], res[b].finalLncc[l]);
            for (auto &r : batch.trace(l, b))
                printf("[%d] Current objective function: %g = (wSIM)%g - (wBE)%.2e - (wLE)%.2e - (wJAC)%.2e [+ %g mm]\n", r.iter, r.bestObj, r.sim, r.be, r.le, r.jac, r.added);
        }
    if (getenv("F3D_TRACE_EVALS")) {
        FILE *f = fopen("/tmp/gpu_evals.log", "w");
        int k = 0;
        for (int l = 0; l < 3; l++) for (auto &r : batch.evtrace(l, 0)) {
            if (r.kind == 1) fprintf(f, "NORM maxlen=%.9g\n", r.sim);
            else { if (r.scale != 0.f) fprintf(f, "UPD scale=%.9g\n", r.scale); fprintf(f, "EVAL %d obj=%.17g sim=%.17g be=%.17g le=%.17g jac=%.17g\n", k++, r.obj, r.sim, r.be, r.le, r.jac); }
        }
        fclose(f);
    }
    auto h = d2h(dout[0], 3 * fgrid.nnodes());
    printf("final grid dims gpu %dx%dx%d cpu %dx%dx%d\n", fgrid.n[0], fgrid.n[1], fgrid.n[2], cpu->nx, cpu->ny, cpu->nz);
    cmp("final cpp vs CPU (mm)", h.data(), (float *)cpu->data, 3 * fgrid.nnodes());
    return 0;
}
