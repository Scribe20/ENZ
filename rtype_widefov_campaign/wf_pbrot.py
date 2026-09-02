"""PB rotation law vs incident angle (spec section 36): rotate the
hard-binary geometry alpha = 0..180 step 15 and evaluate the cross-
circular phase and amplitude at theta = 0, 30, 60, 75 (phi = 0, the
incidence plane). Fit phase = phase0 + s*alpha (ideal |s| = 2).

usage: python wf_pbrot.py <tag|rectangle|oldA|oldB> [...]
Writes results/pb_rotation_vs_angle.csv (appending, idempotent per tag).
"""
import csv
import math
import sys

import numpy as np
import scipy.ndimage as ndi
import torch

import wf_core as wf
import rt_core as rc
from wf_anglemap import load_geometry

R = wf.HERE / 'results'
THETAS = [0.0, 30.0, 45.0, 60.0, 75.0]


def analyze(name):
    rho, P, H, e = load_geometry(name)
    out = R / 'pb_rotation_vs_angle.csv'
    done = set()
    if out.exists():
        with open(out) as f:
            done = {(r['tag'], float(r['theta'])) for r in
                    csv.DictReader(f)}
    rows = []
    n = rho.shape[0]
    ax = (np.arange(n) + 0.5) / n * P - P / 2
    X, Y = np.meshgrid(ax, ax, indexing='ij')
    for th in THETAS:
        if (name, float(th)) in done:
            continue
        recs = []
        for al in range(0, 181, 15):
            rr = ndi.rotate(rho.numpy(), al, reshape=False, order=0,
                            mode='constant', cval=0.0)
            rr = torch.tensor((rr > 0.5).astype(np.float32))
            outside = float(rr.numpy()[(X ** 2 + Y ** 2)
                                       > rc.r_design(P) ** 2].sum()) \
                if name not in ('rectangle',) else -1.0
            with torch.no_grad():
                Rj, Tj = wf.jones_angle(rr, P, H, th, 0.0, order=(9, 9))
            Rc = rc.circular(Rj)
            amp2 = 0.5 * float(torch.abs(Rc[0, 1]) ** 2
                               + torch.abs(Rc[1, 0]) ** 2)
            ph = math.degrees(float(torch.angle(Rc[0, 1])))
            recs.append((al, ph, amp2))
            rows.append({'tag': name, 'theta': th, 'alpha': al,
                         'phase_deg': ph, 'R_cross': amp2,
                         'pixels_outside': outside})
        al_a = np.array([r[0] for r in recs], dtype=float)
        ph_u = np.degrees(np.unwrap(np.radians([r[1] for r in recs])))
        cf = np.polyfit(al_a, ph_u, 1)
        resid = ph_u - np.polyval(cf, al_a)
        amps = np.array([r[2] for r in recs])
        rows.append({'tag': name, 'theta': th, 'alpha': -1,
                     'phase_deg': cf[0], 'R_cross': float(np.std(resid)),
                     'pixels_outside': float(amps.min())})
        print(f'{name} th={th:.0f}: slope={cf[0]:+.3f} rms='
              f'{np.std(resid):.1f} deg Rc range {amps.min():.3f}-'
              f'{amps.max():.3f}', flush=True)
    new = not out.exists()
    with open(out, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'PBROT_DONE {name}', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    for nm in sys.argv[1:]:
        analyze(nm)
