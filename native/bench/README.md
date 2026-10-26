# bench - registration quality scores

`reg_scores.cpp` computes NCC, LNCC (gaussian window, 5 mm) and NMI of warped floating images against a reference, plus Jacobian statistics of
a control point grid, using NiftyReg's own measure classes (`reg_lncc`, `reg_nmi`, `reg_measure`) and warping (`reg_spline_getDeformationField`,
`reg_resampleImage`).  Usage is documented at the top of the source; build with `make NIFTYREG_PREFIX=<prefix>`.
