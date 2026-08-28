"""Phase 1: freeze and revalidate the P0550_H0250_seed011 reference.

Recomputes the audited ledger-v2 lam0 row with the exact audited settings
(order [9,9], LOSSLESS material as in ed_eq_audit.decompose_at defaults,
48x48x9 grid, canonical origin) and compares field by field. Records
SHA256 checksums of the frozen inputs and a snapshot JSON.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import stage_core as sc
from ed_eq_audit import decompose_at, family_row, classify

NAME = 'P0550_H0250_seed011'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    rho, P, h = sc.load_ref(NAME)
    mo, ch = decompose_at(rho, P, h, sc.LAM0)
    fr = family_row(mo, sc.LAM0)
    fr['class'] = classify(fr)

    import csv
    led = None
    with open(sc.CAMP / 'results' / 'candidate_ledger_v2.csv') as f:
        for r in csv.DictReader(f):
            if r['run_id'] == NAME:
                led = r
    assert led is not None

    checks = {}
    for k in ('f_ED', 'f_MD', 'f_EQ', 'f_MQ', 'px_given_ED', 'my_given_MD',
              'Qxz_given_EQ', 'ED_EQ_balance', 'Cpx_total_fraction',
              'CQxz_total_fraction', 'CT_diag_over_CED'):
        new, old = float(fr[k]), float(led[k])
        checks[k] = {'audited': old, 'revalidated': new,
                     'abs_diff': abs(new - old)}
    worst = max(v['abs_diff'] for v in checks.values())
    match = worst < 1e-9
    print(f'revalidation worst |diff| = {worst:.3e}  exact_match={match}')
    for k, v in checks.items():
        print(f"  {k:22s} audited {v['audited']:.9f}  now {v['revalidated']:.9f}")

    mat = sc.core.si_eps(sc.LAM0)
    snapshot = {
        'name': NAME, 'P_nm': P, 'h_nm': h, 'lam0_nm': sc.LAM0,
        'order': sc.ORDER, 'grid': [sc.N_XY, sc.N_XY, sc.NZ],
        'origin': 'cell-center (canonical)',
        'material': {'eps_si_at_lam0': [mat.real, mat.imag],
                     'source': 'Franta 2013 a-Si (refractiveindex.info)'},
        'substrate_eps': [float(sc.core.SUBSTRATE_EPS.real),
                          float(sc.core.SUBSTRATE_EPS.imag)],
        'illumination': 'x-pol planewave, normal incidence from substrate,'
                        ' exp(-j w t)',
        'sha256': {
            'rho_binary.npy': sha(sc.PILOT / NAME / 'rho_binary.npy'),
            'config.json': sha(sc.PILOT / NAME / 'config.json'),
            'ed_eq_core.py': sha(sc.CAMP / 'ed_eq_core.py'),
            'material_model.py': sha(sc.CAMP / 'material_model.py'),
        },
        'rho_stats': {'shape': list(rho.shape),
                      'values': sorted(set(np.unique(rho.numpy()).tolist())),
                      'fill': float(rho.mean())},
        'revalidation': {'worst_abs_diff': worst, 'exact_match': bool(match),
                         'fields': checks},
        'class': fr['class'],
        'toroidal_policy': 'diagnostic only, never a 5th family',
        'bookkeeping': 'C_total_exact = C_ED + C_MD + C_EQ + C_MQ '
                       '(complete partition; deprecated 3-family never used)',
    }
    sc.RESULTS.mkdir(parents=True, exist_ok=True)
    out = sc.RESULTS / 'p0550_frozen_snapshot.json'
    out.write_text(json.dumps(snapshot, indent=1))
    print('snapshot ->', out)
    print('class =', fr['class'])
    print('PHASE1_DONE')


if __name__ == '__main__':
    main()
