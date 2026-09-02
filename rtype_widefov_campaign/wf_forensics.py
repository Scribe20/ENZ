"""Angle-resolved exact current-multipole forensics (spec section 38):
family fractions ED/MD/EQ/MQ + Cartesian purities at theta =
0/20/40/60/75, phi = 0 (x-source = p, y-source = s exactly) and phi=45
(p/s composed as transverse amplitude combos), for finalists +
rectangle + old champions.

usage: python wf_forensics.py <name> [...]
Writes results/angle_multipoles.csv (appending, idempotent per name).
"""
import csv
import math
import sys

import numpy as np
import torch

import wf_core as wf
import rt_core as rc
from wf_anglemap import load_geometry

sys.path.insert(0, str(wf.HERE.parent / 'ed_eq_causality_campaign'))
import ed_eq_core as core                                   # noqa: E402

R = wf.HERE / 'results'
THETAS = [0.0, 20.0, 40.0, 60.0, 75.0]


def families_at(rho, P, H, th, ph, pol, order=(9, 9), n_xy=48, nz=7):
    """Exact 4-family fractions under oblique p/s illumination."""
    sim = wf.build_sim_angle(rho, P, H, th, ph, order=order)
    phr = math.radians(ph)
    if pol == 'p':
        amp = [math.cos(phr), math.sin(phr)]
    else:
        amp = [-math.sin(phr), math.cos(phr)]
    sim.source_planewave(amplitude=amp, direction='backward')
    with torch.no_grad():
        x_ax, z_ax, E, _ = core.fields_3d(sim, float(P), float(H),
                                          n_xy, nz)
    e = rc.eps_asi()
    n = rho.shape[0]
    idx = (torch.floor(x_ax / P * n).long()) % n
    eps3 = (rho[idx][:, idx] * (e - 1.0) + 1.0)[:, :, None] \
        .expand(n_xy, n_xy, nz)
    mo = core.torch_moments(E, eps3, x_ax, z_ax, wf.LAM0)
    Cp, Cm, CQe, CQm = core.family_weights4(mo)
    tot = float(Cp + Cm + CQe + CQm)
    k = mo['k']
    cE = k ** 4 / (6 * math.pi * core.EPS0 ** 2)
    out = {'f_ED': float(Cp) / tot, 'f_MD': float(Cm) / tot,
           'f_EQ': float(CQe) / tot, 'f_MQ': float(CQm) / tot}
    for nm in ('px', 'py', 'pz'):
        out[f'{nm}_in_ED'] = float(cE * torch.abs(mo[nm]) ** 2
                                   / (Cp + 1e-300))
    for nm in ('mx', 'my', 'mz'):
        out[f'{nm}_in_MD'] = float(cE / core.C0 ** 2
                                   * torch.abs(mo[nm]) ** 2
                                   / (Cm + 1e-300))
    return out


def analyze(name):
    rho, P, H, e = load_geometry(name)
    out = R / 'angle_multipoles.csv'
    done = set()
    if out.exists():
        with open(out) as f:
            done = {r['tag'] for r in csv.DictReader(f)}
    if name in done:
        print(f'{name}: forensics already done', flush=True)
        return
    rows = []
    for th in THETAS:
        for ph in (0.0, 45.0):
            for pol in ('p', 's'):
                fam = families_at(rho, P, H, th, ph, pol)
                dom = max(('f_ED', 'f_MD', 'f_EQ', 'f_MQ'),
                          key=lambda k: fam[k])
                rows.append({'tag': name, 'theta': th, 'phi': ph,
                             'pol': pol, **fam, 'dominant': dom[2:]})
            print(f'{name} th={th:.0f} phi={ph:.0f}: '
                  f'p->{rows[-2]["dominant"]}'
                  f'({rows[-2]["f_" + rows[-2]["dominant"]]:.2f}) '
                  f's->{rows[-1]["dominant"]}'
                  f'({rows[-1]["f_" + rows[-1]["dominant"]]:.2f})',
                  flush=True)
    new = not out.exists()
    with open(out, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'FORENSICS_DONE {name}', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    for nm in sys.argv[1:]:
        analyze(nm)
