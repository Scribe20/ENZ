"""Full fine theta/phi angle maps (spec sections 34-35, 43): theta =
0..85 step 5 (+88 diagnostic), phi = 0..90 step 15, all channels, at
order [9,9], hard-binary geometries.

usage: python wf_anglemap.py <name> [...]
  name = a run tag (searched in refinement/coarse/finalists of THIS
  campaign), or one of the specials:
    rectangle          paper rectangle P226/H170
    oldA               previous normal-incidence champion A_P271_H200
    oldB               previous normal-incidence champion B_P271_H215
    bare               unpatterned cell (glass/air interface only)
    film<fill>_P<P>_H<H>  uniform a-Si film of eff. eps at given fill
  add ':lossless' suffix to any name for the k=0 counterfactual.
Appends to results/full_angle_maps.csv (idempotent per (tag,theta,phi)).
"""
import csv
import json
import math
import sys

import numpy as np
import torch

import wf_core as wf
import rt_core as rc
from wf_preflight import rect_rho, P_R, H_R

R = wf.HERE / 'results'
OLD = wf.HERE.parent / 'rtype_freeform_campaign'
THETAS = list(np.arange(0.0, 85.1, 5.0)) + [88.0]
PHIS = list(np.arange(0.0, 90.1, 15.0))


def load_geometry(name):
    """Returns (rho tensor, P, H, eps_override or None)."""
    lossless = name.endswith(':lossless')
    base = name.split(':')[0]
    e = None
    if lossless:
        e = complex(rc.EPS_ASI_633.real, 0.0)
    if base == 'rectangle':
        return rect_rho(), P_R, H_R, e
    if base == 'oldA' or base == 'oldB':
        tag = ('A_P271_H200_s11_g96_o9' if base == 'oldA'
               else 'B_P271_H215_s11_g96_o9')
        rho = torch.tensor(np.load(OLD / 'refinement' / tag
                                   / 'rho_binary.npy'))
        rec = json.loads((OLD / 'refinement' / tag / 'final.json')
                         .read_text())
        return rho, rec['P'], rec['H'], e
    if base == 'bare':
        return torch.zeros(32, 32), P_R, H_R, e
    if base.startswith('film'):
        head, Ps, Hs = base.split('_')
        fill = float(head[4:])
        # uniform layer with the fill-weighted eps (simple-baseline film)
        eps_a = rc.EPS_ASI_633 if e is None else e
        eps_f = fill * eps_a + (1 - fill) * 1.0
        return torch.ones(32, 32), float(Ps[1:]), float(Hs[1:]), eps_f
    for stage in ('refinement', 'coarse', 'finalists'):
        p = wf.HERE / stage / base
        if (p / 'final.json').exists():
            rec = json.loads((p / 'final.json').read_text())
            rho = torch.tensor(np.load(p / 'rho_binary.npy'))
            return rho, rec['P'], rec['H'], e
    raise FileNotFoundError(name)


def map_one(name):
    rho, P, H, e = load_geometry(name)
    out = R / 'full_angle_maps.csv'
    done = set()
    if out.exists():
        with open(out) as f:
            done = {(r['tag'], float(r['theta']), float(r['phi']))
                    for r in csv.DictReader(f)}
    fields = None
    for th in THETAS:
        for ph in PHIS:
            if (name, float(th), float(ph)) in done:
                continue
            eps = rc.eps_asi() if e is None else e
            import torcwa
            sim = torcwa.rcwa(freq=1.0 / wf.LAM0, order=[9, 9],
                              L=[float(P), float(P)], dtype=rc.SIM_DTYPE,
                              device=rc.DEVICE)
            sim.add_input_layer(eps=rc.EPS_GLASS)
            sim.set_incident_angle(inc_ang=wf.glass_angle(float(th)),
                                   azi_ang=math.radians(float(ph)))
            sim.add_layer(thickness=float(H),
                          eps=rho * (eps - 1.0) + 1.0)
            sim.solve_global_smatrix()
            with torch.no_grad():
                Rj, Tj = wf.jones_dev(sim)
            s = wf.angle_scores(Rj, Tj)
            m = rc.device_metrics(Rj, Tj)
            row = {'tag': name, 'theta': float(th), 'phi': float(ph),
                   'R_cross': float(s['Rc']), 'R_co': float(s['co']),
                   'T_cross': float(s['Tc']), 'T_tot': float(s['Tt']),
                   'A': float(s['A']),
                   'R_cross_12': float(torch.abs(
                       rc.circular(Rj)[0, 1]) ** 2),
                   'R_cross_21': float(torch.abs(
                       rc.circular(Rj)[1, 0]) ** 2),
                   'abs_rx': m['abs_rx'], 'abs_ry': m['abs_ry'],
                   'dphi_r_deg': m['dphi_r_deg'],
                   'rxx_re': m['rxx_re'], 'rxx_im': m['rxx_im'],
                   'ryy_re': m['ryy_re'], 'ryy_im': m['ryy_im'],
                   'rxy_abs': m['abs_rxy'], 'ryx_abs': m['abs_ryx']}
            if fields is None:
                fields = list(row.keys())
            new = not out.exists()
            with open(out, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fields)
                if new:
                    w.writeheader()
                w.writerow(row)
        print(f'{name} theta={th:.0f} done', flush=True)
    print(f'ANGLEMAP_DONE {name}', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    for nm in sys.argv[1:]:
        map_one(nm)
