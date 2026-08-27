"""MQ-panel spectra for the audit's 5-panel multipole figures.

Recomputes the complete 4-family radiation weights (incl. the magnetic
quadrupole, absent from the Stage-A qualify scans) on a 2-nm lambda grid
for the audit's key candidates, at the qualify settings (order [9,9],
48x48x9 moment grid, canonical origin, main material scenario).
Per-point CSV checkpointing; safe to re-run (skips computed points).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

import ed_eq_core as core

HERE = Path(__file__).resolve().parent
PILOT = HERE / 'results' / 'pilot'
AUD = HERE / 'results' / 'audit'
LAM0 = 1332.5
LAMS = np.arange(LAM0 - 40.0, LAM0 + 40.0 + 0.01, 2.0)
ORDER_Q = [9, 9]
N_XY, NZ = 48, 9

CANDS = ['P0750_H0250_seed011', 'P0550_H0250_seed011',
         'P0750_H0350_seed011', 'P0650_H0350_seed029',
         'P0750_H0350_seed029']

COLS = ['lam_nm', 'C_ED', 'C_MD', 'C_EQ', 'C_MQ',
        'f_ED', 'f_MD', 'f_EQ', 'f_MQ',
        'C_Qmxx', 'C_Qmyy', 'C_Qmzz', 'C_Qmxy', 'C_Qmxz', 'C_Qmyz']


def run_candidate(name):
    AUD.mkdir(parents=True, exist_ok=True)
    out = AUD / f'mqspec_{name}.csv'
    done = set()
    if out.exists():
        for line in out.read_text().splitlines()[1:]:
            if line.strip():
                done.add(round(float(line.split(',')[0]), 3))
    else:
        out.write_text(','.join(COLS) + '\n')
    cfg = json.loads((PILOT / name / 'config.json').read_text())
    rho = torch.tensor(np.load(PILOT / name / 'rho_binary.npy'),
                       dtype=torch.float32)
    P, h = float(cfg['P']), float(cfg['h'])
    nmask = rho.shape[0]
    for lam in LAMS:
        key = round(float(lam), 3)
        if key in done:
            continue
        eps_si = core.si_eps(float(lam))
        sim = core.build_sim(rho, P, h, float(lam), ORDER_Q, eps_si=eps_si)
        x_ax, z_ax, E, _ = core.fields_3d(sim, P, h, N_XY, NZ)
        idx = (torch.floor(x_ax / P * nmask).long()) % nmask
        eps3 = (rho[idx][:, idx] * (eps_si - 1.0) + 1.0)[:, :, None] \
            .expand(N_XY, N_XY, NZ)
        mo = core.torch_moments(E, eps3, x_ax, z_ax, float(lam))
        Cp, Cm, CQe, CQm = (float(v) for v in core.family_weights4(mo))
        tot = Cp + Cm + CQe + CQm
        k = float(mo['k'])
        cE = k ** 4 / (6 * np.pi * core.EPS0 ** 2)
        row = [key, Cp, Cm, CQe, CQm,
               Cp / tot, Cm / tot, CQe / tot, CQm / tot]
        for t in ('Qmxx', 'Qmyy', 'Qmzz', 'Qmxy', 'Qmxz', 'Qmyz'):
            w = 1 if t.endswith(('xx', 'yy', 'zz')) else 2
            row.append(cE / 120 * (k / core.C0) ** 2 * w
                       * abs(complex(mo[t])) ** 2)
        with out.open('a') as f:
            f.write(','.join(f'{v:.10g}' for v in row) + '\n')
        print(f'{name} {key} f_MQ={row[8]:.4f}', flush=True)
    print(f'MQSPEC_CAND_DONE {name}', flush=True)


if __name__ == '__main__':
    names = sys.argv[1:] or CANDS
    for n in names:
        run_candidate(n)
    print('MQSPEC_DONE', flush=True)
