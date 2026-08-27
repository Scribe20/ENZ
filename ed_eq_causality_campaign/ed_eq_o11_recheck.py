"""Order-[11,11] recheck of the two Q refits that failed ONLY the
energy-conservation gate at [9,9] (P0550_H0150_seed011, worst 0.041;
P0750_H0250_seed029, worst 0.0075). If the higher order heals |T+R-1|
inside the fit window, the Q value can be reported as RESOLVED at [11,11];
otherwise it stays UNRESOLVED. Appends to results/q_validation_v2_o11.csv.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ed_eq_audit import adaptive_qrefine, PILOT, RESULTS

TARGETS = [('P0550_H0150_seed011', 1330.67),
           ('P0750_H0250_seed029', 1312.04)]
OUT = RESULTS / 'q_validation_v2_o11.csv'
FIELDS = ['tag', 'lam_pole', 'Q_pole', 'Q_unc', 'fwhm_nm',
          'samples_in_fwhm', 'rms_rel', 'window_stability',
          'energy_res_worst', 'order', 'Q_RESOLVED']


def main():
    done = set()
    if OUT.exists():
        done = {r['tag'] for r in csv.DictReader(open(OUT))}
    else:
        with open(OUT, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    for name, seed in TARGETS:
        if name in done:
            continue
        cfg = json.loads((PILOT / name / 'config.json').read_text())
        rho = torch.tensor(np.load(PILOT / name / 'rho_binary.npy'),
                           dtype=torch.float32)
        rec, pts = adaptive_qrefine(rho, float(cfg['P']), float(cfg['h']),
                                    seed, name + '_o11', max_rounds=6,
                                    order=[11, 11])
        rec['tag'] = name
        print('o11 recheck:', rec, flush=True)
        with open(OUT, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(
                {k: rec.get(k) for k in FIELDS})
        keys = sorted(pts)
        np.savez(RESULTS / 'audit' / f'qrefine_{name}_o11.npz',
                 lam=np.array(keys),
                 t=np.array([pts[k]['t'] for k in keys]),
                 r=np.array([pts[k]['r'] for k in keys]),
                 en=np.array([pts[k]['en'] for k in keys]))
    print('O11_RECHECK_DONE', flush=True)


if __name__ == '__main__':
    main()
