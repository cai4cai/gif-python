// SPDX-License-Identifier: BSD-3-Clause
// Host replica of reg_spline_approxBendingEnergyGradient3D<float> (verbatim arithmetic) to find the exact FMA structure.
#include "test_common.h"
#include "f3d_basis_host.h"
static void replica(nifti_image *splineControlPoint, nifti_image *gradientImage, float weight, int variant)
{
    typedef float DTYPE;
    size_t nodeNumber = (size_t)splineControlPoint->nx * splineControlPoint->ny * splineControlPoint->nz;
    int a, b, c, x, y, z, X, Y, Z, index, i;
    DTYPE *splinePtrX = static_cast<DTYPE *>(splineControlPoint->data);
    DTYPE *splinePtrY = &splinePtrX[nodeNumber];
    DTYPE *splinePtrZ = &splinePtrY[nodeNumber];
    DTYPE basisXX[27], basisYY[27], basisZZ[27], basisXY[27], basisYZ[27], basisXZ[27];
    set_second_order_bspline_basis_values(basisXX, basisYY, basisZZ, basisXY, basisYZ, basisXZ);
    DTYPE splineCoeffX, splineCoeffY, splineCoeffZ;
    DTYPE XX_x, YY_x, ZZ_x, XY_x, YZ_x, XZ_x;
    DTYPE XX_y, YY_y, ZZ_y, XY_y, YZ_y, XZ_y;
    DTYPE XX_z, YY_z, ZZ_z, XY_z, YZ_z, XZ_z;
    DTYPE *derivativeValues = (DTYPE *)calloc(18 * nodeNumber, sizeof(DTYPE));
    DTYPE *derivativeValuesPtr;
    reg_getDisplacementFromDeformation(splineControlPoint);
    for (z = 0; z < splineControlPoint->nz; z++) {
        derivativeValuesPtr = &derivativeValues[18 * z * splineControlPoint->ny * splineControlPoint->nx];
        for (y = 0; y < splineControlPoint->ny; y++) {
            for (x = 0; x < splineControlPoint->nx; x++) {
                XX_x = 0.0, YY_x = 0.0, ZZ_x = 0.0; XY_x = 0.0, YZ_x = 0.0, XZ_x = 0.0;
                XX_y = 0.0, YY_y = 0.0, ZZ_y = 0.0; XY_y = 0.0, YZ_y = 0.0, XZ_y = 0.0;
                XX_z = 0.0, YY_z = 0.0, ZZ_z = 0.0; XY_z = 0.0, YZ_z = 0.0, XZ_z = 0.0;
                i = 0;
                for (c = -1; c < 2; c++) {
                    for (b = -1; b < 2; b++) {
                        for (a = -1; a < 2; a++) {
                            if (-1 < (x + a) && -1 < (y + b) && -1 < (z + c) && (x + a) < splineControlPoint->nx && (y + b) < splineControlPoint->ny && (z + c) < splineControlPoint->nz) {
                                index = ((z + c) * splineControlPoint->ny + y + b) * splineControlPoint->nx + x + a;
                                splineCoeffX = splinePtrX[index]; splineCoeffY = splinePtrY[index]; splineCoeffZ = splinePtrZ[index];
                                XX_x += basisXX[i] * splineCoeffX; YY_x += basisYY[i] * splineCoeffX; ZZ_x += basisZZ[i] * splineCoeffX;
                                XY_x += basisXY[i] * splineCoeffX; YZ_x += basisYZ[i] * splineCoeffX; XZ_x += basisXZ[i] * splineCoeffX;
                                XX_y += basisXX[i] * splineCoeffY; YY_y += basisYY[i] * splineCoeffY; ZZ_y += basisZZ[i] * splineCoeffY;
                                XY_y += basisXY[i] * splineCoeffY; YZ_y += basisYZ[i] * splineCoeffY; XZ_y += basisXZ[i] * splineCoeffY;
                                XX_z += basisXX[i] * splineCoeffZ; YY_z += basisYY[i] * splineCoeffZ; ZZ_z += basisZZ[i] * splineCoeffZ;
                                XY_z += basisXY[i] * splineCoeffZ; YZ_z += basisYZ[i] * splineCoeffZ; XZ_z += basisXZ[i] * splineCoeffZ;
                            }
                            ++i;
                        }
                    }
                }
                *derivativeValuesPtr++ = XX_x; *derivativeValuesPtr++ = XX_y; *derivativeValuesPtr++ = XX_z;
                *derivativeValuesPtr++ = YY_x; *derivativeValuesPtr++ = YY_y; *derivativeValuesPtr++ = YY_z;
                *derivativeValuesPtr++ = ZZ_x; *derivativeValuesPtr++ = ZZ_y; *derivativeValuesPtr++ = ZZ_z;
                *derivativeValuesPtr++ = (DTYPE)(2.0 * XY_x); *derivativeValuesPtr++ = (DTYPE)(2.0 * XY_y); *derivativeValuesPtr++ = (DTYPE)(2.0 * XY_z);
                *derivativeValuesPtr++ = (DTYPE)(2.0 * YZ_x); *derivativeValuesPtr++ = (DTYPE)(2.0 * YZ_y); *derivativeValuesPtr++ = (DTYPE)(2.0 * YZ_z);
                *derivativeValuesPtr++ = (DTYPE)(2.0 * XZ_x); *derivativeValuesPtr++ = (DTYPE)(2.0 * XZ_y); *derivativeValuesPtr++ = (DTYPE)(2.0 * XZ_z);
            }
        }
    }
    DTYPE *gradientXPtr = static_cast<DTYPE *>(gradientImage->data);
    DTYPE *gradientYPtr = &gradientXPtr[nodeNumber];
    DTYPE *gradientZPtr = &gradientYPtr[nodeNumber];
    DTYPE approxRatio = (DTYPE)weight / (DTYPE)nodeNumber;
    DTYPE gradientValue[3];
    for (z = 0; z < splineControlPoint->nz; z++) {
        index = z * splineControlPoint->nx * splineControlPoint->ny;
        for (y = 0; y < splineControlPoint->ny; y++) {
            for (x = 0; x < splineControlPoint->nx; x++) {
                gradientValue[0] = gradientValue[1] = gradientValue[2] = 0.0;
                a = 0;
                for (Z = z - 1; Z < z + 2; Z++) {
                    for (Y = y - 1; Y < y + 2; Y++) {
                        for (X = x - 1; X < x + 2; X++) {
                            if (-1 < X && -1 < Y && -1 < Z && X < splineControlPoint->nx && Y < splineControlPoint->ny && Z < splineControlPoint->nz) {
                                derivativeValuesPtr = &derivativeValues[18 * ((Z * splineControlPoint->ny + Y) * splineControlPoint->nx + X)];
                                gradientValue[0] += (*derivativeValuesPtr++) * basisXX[a]; gradientValue[1] += (*derivativeValuesPtr++) * basisXX[a]; gradientValue[2] += (*derivativeValuesPtr++) * basisXX[a];
                                gradientValue[0] += (*derivativeValuesPtr++) * basisYY[a]; gradientValue[1] += (*derivativeValuesPtr++) * basisYY[a]; gradientValue[2] += (*derivativeValuesPtr++) * basisYY[a];
                                gradientValue[0] += (*derivativeValuesPtr++) * basisZZ[a]; gradientValue[1] += (*derivativeValuesPtr++) * basisZZ[a]; gradientValue[2] += (*derivativeValuesPtr++) * basisZZ[a];
                                gradientValue[0] += (*derivativeValuesPtr++) * basisXY[a]; gradientValue[1] += (*derivativeValuesPtr++) * basisXY[a]; gradientValue[2] += (*derivativeValuesPtr++) * basisXY[a];
                                gradientValue[0] += (*derivativeValuesPtr++) * basisYZ[a]; gradientValue[1] += (*derivativeValuesPtr++) * basisYZ[a]; gradientValue[2] += (*derivativeValuesPtr++) * basisYZ[a];
                                gradientValue[0] += (*derivativeValuesPtr++) * basisXZ[a]; gradientValue[1] += (*derivativeValuesPtr++) * basisXZ[a]; gradientValue[2] += (*derivativeValuesPtr++) * basisXZ[a];
                            }
                            a++;
                        }
                    }
                }
                gradientXPtr[index] += approxRatio * gradientValue[0];
                gradientYPtr[index] += approxRatio * gradientValue[1];
                gradientZPtr[index] += approxRatio * gradientValue[2];
                index++;
            }
        }
    }
    free(derivativeValues);
    reg_getDeformationFromDisplacement(splineControlPoint);
}
int main(int argc, char **argv)
{
    nifti_image *ref = load(argv[1]);
    mat44 aff; reg_tool_ReadAffineFile(&aff, argv[3]);
    ImgGeom rg = geom_of(ref);
    nifti_image *rp[3]; reg_createImagePyramid<float>(ref, rp, 4, 3);
    int l = 1; float sp[3]; level_spacing(rg, l, sp);
    nifti_image *grid = nullptr; reg_createControlPointGrid<float>(&grid, rp[l], sp); reg_affine_getDeformationField(&aff, grid);
    std::mt19937 rng(5); std::normal_distribution<float> nd(0.f, atof(argv[4]));
    float *gp = (float *)grid->data; for (size_t i = 0; i < grid->nvox; i++) gp[i] += nd(rng);
    const float W = 0.005f / 256.f;
    nifti_image *g1 = nifti_copy_nim_info(grid); g1->data = malloc(grid->nvox * 4); memcpy(g1->data, grid->data, grid->nvox * 4);
    nifti_image *g2 = nifti_copy_nim_info(grid); g2->data = malloc(grid->nvox * 4); memcpy(g2->data, grid->data, grid->nvox * 4);
    nifti_image *o1 = nifti_copy_nim_info(grid); o1->data = calloc(grid->nvox, 4);
    nifti_image *o2 = nifti_copy_nim_info(grid); o2->data = calloc(grid->nvox, 4);
    reg_spline_approxBendingEnergyGradient(g1, o1, W);
    replica(g2, o2, W, 0);
    size_t nd2 = 0; double mx = 0;
    for (size_t i = 0; i < grid->nvox; i++) { double d = fabs(((float *)o1->data)[i] - ((float *)o2->data)[i]); nd2 += d > 0; mx = d > mx ? d : mx; }
    printf("replica vs library: ndiff %zu of %zu maxabs %g\n", nd2, grid->nvox, mx);
    return 0;
}
