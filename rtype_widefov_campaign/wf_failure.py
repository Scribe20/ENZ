"""Angular failure-budget decomposition (spec section 39).

For a diagonal Jones matrix (D2 structure, phi = 0/90) the deficit is
EXACT:
    R_cross = (|rx|^2 + |ry|^2 - 2|rx||ry| cos dphi)/4
    1 - R_cross = T_loss + A_loss + R_co(+ mixing residual)
    R_co = imbalance + retardance
        imbalance  = (|rx| - |ry|)^2 / 4          (amplitude failure A)
        retardance = |rx||ry| (1 + cos dphi)/2    (phase failure B)
Off-diagonal mixing (F) = (|rxy|^2+|ryx|^2)/2 from the map. Diffraction
(E) is identically zero (specular-only domain, exact preflight). Modal
change (C) and absorption growth (D) come from angle_multipoles.csv and
the A column; dispersion (G) is discussed via the forensics.

usage: python wf_failure.py <tag> [...]   (tags present in the fine map)
Writes results/failure_budget.csv and prints the table.
"""
import sys

import numpy as np
import pandas as pd

import wf_core as wf

R = wf.HERE / 'results'
THETAS = [0.0, 20.0, 40.0, 60.0, 75.0, 85.0]


def budget(name):
    df = pd.read_csv(R / 'full_angle_maps.csv')
    df = df[(df.tag == name) & (df.phi == 0.0)]
    try:
        mp = pd.read_csv(R / 'angle_multipoles.csv')
        mp = mp[(mp.tag == name) & (mp.phi == 0.0)]
    except FileNotFoundError:
        mp = None
    rows = []
    for th in THETAS:
        sub = df[df.theta == th]
        if not len(sub):
            continue
        r = sub.iloc[0]
        rx, ry = r.abs_rx, r.abs_ry
        dphi = np.radians(r.dphi_r_deg)
        imb = (rx - ry) ** 2 / 4
        ret = rx * ry * (1 + np.cos(dphi)) / 2
        mix = (r.rxy_abs ** 2 + r.ryx_abs ** 2) / 2
        row = {'tag': name, 'theta': th, 'R_cross': r.R_cross,
               'T_loss': r.T_tot, 'A_loss': r.A,
               'imbalance_loss': imb, 'retardance_loss': ret,
               'mixing': mix,
               'closure': r.R_cross + r.T_tot + r.A + imb + ret + mix}
        if mp is not None and len(mp[mp.theta == th]):
            m2 = mp[(mp.theta == th) & (mp.pol == 'p')].iloc[0]
            row['dom_p'] = m2.dominant
            row['f_dom_p'] = m2[f'f_{m2.dominant}']
        rows.append(row)
    return rows


if __name__ == '__main__':
    allr = []
    for nm in sys.argv[1:]:
        rs = budget(nm)
        allr += rs
        print(f'\n== {nm} (phi=0) ==')
        print(f'{"th":>4} {"Rc":>6} {"T":>6} {"A":>6} {"imbal":>6} '
              f'{"retard":>7} {"mix":>6} {"closure":>7}')
        for r in rs:
            print(f'{r["theta"]:4.0f} {r["R_cross"]:6.3f} '
                  f'{r["T_loss"]:6.3f} {r["A_loss"]:6.3f} '
                  f'{r["imbalance_loss"]:6.3f} '
                  f'{r["retardance_loss"]:7.3f} {r["mixing"]:6.3f} '
                  f'{r["closure"]:7.3f}')
    pd.DataFrame(allr).to_csv(R / 'failure_budget.csv', index=False)
    print('\nFAILURE_BUDGET_DONE', flush=True)
