"""Complex-amplitude Fourier-order convergence at normal AND oblique
incidence (spec section 48): Re/Im of the circular cross amplitude, the
power metrics, and the lossless-closure residual at orders 9/11/13/15,
at (theta,phi) = (0,0), (60,0), (60,45), (80,0).

usage: python wf_convergence.py <name> [...]
Writes results/convergence.csv (appending, idempotent per name).
"""
import csv
import sys

import torch

import wf_core as wf
import rt_core as rc
from wf_anglemap import load_geometry

R = wf.HERE / 'results'
POINTS = [(0.0, 0.0), (60.0, 0.0), (60.0, 45.0), (80.0, 0.0)]


def analyze(name):
    rho, P, H, e = load_geometry(name)
    out = R / 'convergence.csv'
    done = set()
    if out.exists():
        with open(out) as f:
            done = {r['tag'] for r in csv.DictReader(f)}
    if name in done:
        print(f'{name}: convergence already done', flush=True)
        return
    rows = []
    for th, ph in POINTS:
        for o in (9, 11, 13, 15):
            with torch.no_grad():
                Rj, Tj = wf.jones_angle(rho, P, H, th, ph, order=(o, o))
            s = wf.angle_scores(Rj, Tj)
            rcx = rc.circular(Rj)[0, 1]
            rows.append({'tag': name, 'theta': th, 'phi': ph, 'order': o,
                         'r_cross_re': float(rcx.real),
                         'r_cross_im': float(rcx.imag),
                         'R_cross': float(s['Rc']),
                         'R_co': float(s['co']), 'T_tot': float(s['Tt']),
                         'A': float(s['A'])})
            print(f'{name} ({th:.0f},{ph:.0f}) o{o}: '
                  f'r_cross={rcx.real:+.4f}{rcx.imag:+.4f}j '
                  f'Rc={float(s["Rc"]):.4f}', flush=True)
    new = not out.exists()
    with open(out, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'CONV_DONE {name}', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    for nm in sys.argv[1:]:
        analyze(nm)
