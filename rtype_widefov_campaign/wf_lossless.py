"""Lossless (k=0) counterfactual scan, spec section 42: R_cross(theta)
at phi = 0 for frozen finalists + rectangle, real vs k=0 material.
Diagnostic only - never used in optimization.

usage: python wf_lossless.py <name> [...]   (plain names; ':lossless'
appended internally for the counterfactual rows)
Writes results/lossless_scan.csv
"""
import csv
import sys

import numpy as np
import torch

import wf_core as wf
import rt_core as rc
from wf_anglemap import load_geometry

R = wf.HERE / 'results'


def scan(name):
    rows = []
    for suffix in ('', ':lossless'):
        rho, P, H, e = load_geometry(name + suffix)
        eps = rc.eps_asi() if e is None else e
        for th in np.arange(0.0, 85.1, 5.0):
            import torcwa, math
            sim = torcwa.rcwa(freq=1.0 / wf.LAM0, order=[9, 9],
                              L=[float(P), float(P)], dtype=rc.SIM_DTYPE,
                              device=rc.DEVICE)
            sim.add_input_layer(eps=rc.EPS_GLASS)
            sim.set_incident_angle(inc_ang=wf.glass_angle(float(th)),
                                   azi_ang=0.0)
            sim.add_layer(thickness=float(H), eps=rho * (eps - 1.0) + 1.0)
            sim.solve_global_smatrix()
            with torch.no_grad():
                Rj, Tj = wf.jones_dev(sim)
            s = wf.angle_scores(Rj, Tj)
            rows.append({'tag': name, 'variant': 'lossless' if suffix
                         else 'real', 'theta': float(th),
                         'R_cross': float(s['Rc']),
                         'R_co': float(s['co']), 'T_tot': float(s['Tt']),
                         'A': float(s['A'])})
        print(f'{name}{suffix}: scan done', flush=True)
    return rows


if __name__ == '__main__':
    torch.set_num_threads(2)
    rows = []
    for nm in sys.argv[1:]:
        rows += scan(nm)
    out = R / 'lossless_scan.csv'
    new = not out.exists()
    with open(out, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print('LOSSLESS_DONE', flush=True)
