// SPDX-License-Identifier: BSD-3-Clause
// Command line driver for the GPU F3D batch registration.
//   f3d_run <target.nii[.gz]> <affine.txt> <out_dir> <template1.nii> [template2.nii ...]
// Registers every template (floating) to the target (reference) from the initial affine in affine.txt with LNCC (5 mm),
// settings of F3DParams (override with F3D_BE, F3D_JL, F3D_LEVELS, F3D_LEVELS_PERFORM, F3D_MAXIT in the environment) and
// writes <out_dir>/<template basename>_cpp.nii (control point positions, NiftyReg format).
#include "f3d.h"
#include "f3d_internal.h"
#include "_reg_tools.h"
#include "_reg_ReadWriteMatrix.h"
#include "_reg_localTrans.h"
#include <cstdio>
#include <cstdlib>
#include <string>
using namespace f3d;

static nifti_image *load(const char *fn)
{
    nifti_image *im = nifti_image_read(fn, true);
    if (!im) { fprintf(stderr, "cannot read %s\n", fn); exit(1); }
    nifti_free_extensions(im);
    im->scl_inter = 0; im->scl_slope = 1;
    if (im->datatype != NIFTI_TYPE_FLOAT32) reg_tools_changeDatatype<float>(im);
    nifti_update_dims_from_array(im);
    return im;
}
static M44 to_m44(const mat44 &m) { M44 r; for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) r.m[i][j] = m.m[i][j]; return r; }
static ImgGeom geom_of(nifti_image *im)
{
    if (im->sform_code <= 0) { fprintf(stderr, "images need an sform\n"); exit(1); }
    return make_geom(im->nx, im->ny, im->nz, im->dx, im->dy, im->dz, to_m44(im->sto_xyz));
}
int main(int argc, char **argv)
{
    if (argc < 5) { fprintf(stderr, "usage: %s target affine out_dir tpl1 [tpl2 ...]\n", argv[0]); return 1; }
    nifti_image *ref = load(argv[1]);
    mat44 aff; reg_tool_ReadAffineFile(&aff, argv[2]);
    std::string outdir = argv[3];
    int n = argc - 4;
    ImgGeom rg = geom_of(ref);
    float *dref = h2d((float *)ref->data, ref->nvox);
    std::vector<unsigned char> ones(ref->nvox, 1);
    unsigned char *dmask = h2d(ones.data(), ones.size());
    F3DParams P;
    if (getenv("F3D_BE")) P.beWeight = (float)atof(getenv("F3D_BE"));
    if (getenv("F3D_JL")) P.jlWeight = (float)atof(getenv("F3D_JL"));
    if (getenv("F3D_LEVELS")) P.levelNumber = atoi(getenv("F3D_LEVELS"));
    if (getenv("F3D_LEVELS_PERFORM")) P.levelToPerform = atoi(getenv("F3D_LEVELS_PERFORM"));
    if (getenv("F3D_MAXIT")) P.maxIterations = atoi(getenv("F3D_MAXIT"));
    F3DBatch batch(dref, rg, dmask, P, n);
    GridGeom fgrid = batch.finalGrid();
    std::vector<float *> dt(n), dout(n);
    std::vector<nifti_image *> tpl(n);
    ImgGeom fg;
    for (int i = 0; i < n; i++) {
        tpl[i] = load(argv[4 + i]);
        fg = geom_of(tpl[i]);
        dt[i] = h2d((float *)tpl[i]->data, tpl[i]->nvox);
        dout[i] = dalloc<float>(3 * fgrid.nnodes());
    }
    std::vector<M44> affs(n, to_m44(aff));
    std::vector<F3DResult> res(n);
    F3DTiming tm;
    std::vector<const float *> cdt(dt.begin(), dt.end());
    batch.run(cdt.data(), fg, n, affs.data(), dout.data(), res.data(), &tm);
    printf("registered %d templates in %.1f ms (L0 %.1f, L1 %.1f, L2 %.1f ms; end-of-level correction %.1f ms)\n", n, tm.totalMs, tm.levelMs[0],
           tm.levelMs[1], tm.levelMs[2], tm.correctMs[0] + tm.correctMs[1] + tm.correctMs[2]);
    for (int i = 0; i < n; i++) {
        auto h = d2h(dout[i], 3 * fgrid.nnodes());
        int dim[8] = {5, fgrid.n[0], fgrid.n[1], fgrid.n[2], 1, 3, 1, 1};
        nifti_image *cpp = nifti_make_new_nim(dim, NIFTI_TYPE_FLOAT32, true);
        memcpy(cpp->data, h.data(), h.size() * sizeof(float));
        cpp->pixdim[1] = cpp->dx = fgrid.d[0]; cpp->pixdim[2] = cpp->dy = fgrid.d[1]; cpp->pixdim[3] = cpp->dz = fgrid.d[2];
        cpp->sform_code = ref->sform_code; cpp->qform_code = ref->qform_code;
        for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) cpp->sto_xyz.m[r][c] = fgrid.xyz.m[r][c];
        cpp->sto_ijk = nifti_mat44_inverse(cpp->sto_xyz);
        // qform of the grid: reference orientation, origin shifted by one node (as NiftyReg does)
        cpp->quatern_b = ref->quatern_b; cpp->quatern_c = ref->quatern_c; cpp->quatern_d = ref->quatern_d; cpp->qfac = ref->qfac;
        cpp->qoffset_x = fgrid.xyz.m[0][3]; cpp->qoffset_y = fgrid.xyz.m[1][3]; cpp->qoffset_z = fgrid.xyz.m[2][3];
        cpp->qto_xyz = nifti_quatern_to_mat44(cpp->quatern_b, cpp->quatern_c, cpp->quatern_d, cpp->qoffset_x, cpp->qoffset_y, cpp->qoffset_z,
                                              cpp->dx, cpp->dy, cpp->dz, cpp->qfac);
        cpp->qto_ijk = nifti_mat44_inverse(cpp->qto_xyz);
        cpp->intent_code = NIFTI_INTENT_VECTOR;
        memset(cpp->intent_name, 0, 16); strcpy(cpp->intent_name, "NREG_TRANS");
        cpp->intent_p1 = CUB_SPLINE_GRID;
        std::string base = argv[4 + i];
        base = base.substr(base.find_last_of('/') + 1);
        base = base.substr(0, base.find(".nii"));
        std::string fn = outdir + "/" + base + "_cpp.nii";
        nifti_set_filenames(cpp, fn.c_str(), 0, 0);
        nifti_image_write(cpp);
        nifti_image_free(cpp);
        printf("  %s: iterations L0/L1/L2 = %d/%d/%d, final objective (last level with a finite value) %.6f\n", fn.c_str(), res[i].iterations[0],
               res[i].iterations[1], res[i].iterations[2], res[i].finalObjective[2] == res[i].finalObjective[2] ? res[i].finalObjective[2] : res[i].finalObjective[1]);
    }
    return 0;
}
