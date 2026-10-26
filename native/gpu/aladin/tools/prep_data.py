#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Convert NIfTI volumes to the raw format read by aladin_cli: <out>/<name>.f32 (float32, x fastest)
and <out>/<name>.hdr ("nx ny nz" + 4 sform rows).  Uses the sform exactly as NiftyReg (sto_xyz)."""
import sys, os, nibabel as nib, numpy as np

def conv(path, outdir):
    name = os.path.basename(path)
    for ext in ('.nii.gz', '.nii'):
        if name.endswith(ext):
            name = name[:-len(ext)]
    im = nib.load(path)
    h = im.header
    data = np.asarray(im.dataobj)  # raw values (scl_slope is NaN/1 in this data)
    sl, ic = h['scl_slope'], h['scl_inter']
    if np.isfinite(sl) and sl != 0 and not (sl == 1 and (not np.isfinite(ic) or ic == 0)):
        data = data * sl + (ic if np.isfinite(ic) else 0)
    data = data.astype(np.float32)
    assert data.ndim == 3
    S = h.get_sform(coded=True)
    assert S[1] > 0, 'no sform'
    S = S[0]
    data.ravel(order='F').tofile(os.path.join(outdir, name + '.f32'))
    with open(os.path.join(outdir, name + '.hdr'), 'w') as f:
        f.write('%d %d %d\n' % data.shape)
        for r in range(4):
            f.write(' '.join('%.9g' % np.float32(S[r, c]) for c in range(4)) + '\n')
    return name

if __name__ == '__main__':
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    for p in sys.argv[2:]:
        print(conv(p, outdir))
