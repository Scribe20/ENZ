"""Phase 10: thickness (beta = h) phase-sweep family around P0550.

Chosen by the Phase-9 sensitivity analysis: d(dphi)/dh = -0.81 deg/nm with
|d bal/dh| ~ 2e-4 /nm (merit 47.7, best of all candidate knobs).

Grid: coarse map h in 205..295 step 10 x lam in 1290..1420 step 5,
plus fine h scan 200..300 step 2.5 at lam0. Sharded by h; per-point
checkpoint; full observable rows (scan_point_full).
usage: python phase_sweep.py <shard> <nshards>
"""
import sys

import numpy as np
import torch

import stage_core as sc

NAME = 'P0550_H0250_seed011'
H_COARSE = np.arange(205.0, 295.0 + 0.1, 10.0)
LAM_COARSE = np.arange(1290.0, 1420.0 + 0.1, 5.0)
H_FINE = np.arange(200.0, 300.0 + 0.1, 2.5)


def main():
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    rho, P, h0 = sc.load_ref(NAME)
    jobs = [(h, lam) for h in H_COARSE for lam in LAM_COARSE]
    jobs += [(h, 1332.5) for h in H_FINE]
    jobs = jobs[shard::nsh]
    out = sc.RESULTS / f'phase_sweep_shard{shard}.csv'
    done = set()
    import csv
    if out.exists():
        with open(out) as f:
            done = {(round(float(r['h_nm']), 2), round(float(r['lam_nm']), 3))
                    for r in csv.DictReader(f)}
    fields = None
    for h, lam in jobs:
        key = (round(float(h), 2), round(float(lam), 3))
        if key in done:
            continue
        row = sc.scan_point_full(rho, P, h0, float(lam), h_override=float(h))
        row = {'h_nm': float(h), **row}
        if fields is None:
            fields = list(row.keys())
        sc.append_row(out, row, fieldnames=fields)
        print(f's{shard} h={h:.1f} lam={lam:.1f} '
              f'B={row["ED_EQ_balance"]:.3f} R={row["R"]:.3f}', flush=True)
    print(f'PHASE_SWEEP_SHARD_DONE {shard}', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
