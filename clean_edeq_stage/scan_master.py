"""Master observable scan (Phases 2-3 data): sharded + checkpointed.

usage: python scan_master.py <cand> <shard> <nshards>
  cand: p0550 | p0750
Writes results/scan_<cand>_shard<k>.csv (one full row per wavelength,
resume by lam key). Merge with merge_master.py.
"""
import sys

import numpy as np
import torch

import stage_core as sc

CANDS = {'p0550': ('P0550_H0250_seed011', np.arange(1260.0, 1420.0 + 0.1, 1.0)),
         'p0750': ('P0750_H0250_seed011', np.arange(1300.0, 1360.0 + 0.1, 1.0))}


def main():
    cand, shard, nsh = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    name, lams = CANDS[cand]
    lams = lams[shard::nsh]
    rho, P, h = sc.load_ref(name)
    out = sc.RESULTS / f'scan_{cand}_shard{shard}.csv'
    done = sc.done_keys(out)
    fields = None
    for lam in lams:
        key = round(float(lam), 3)
        if key in done:
            continue
        row = sc.scan_point_full(rho, P, h, float(lam))
        if fields is None:
            fields = list(row.keys())
        sc.append_row(out, row, fieldnames=fields)
        print(f'{cand} s{shard} {key} f_ED={row["f_ED"]:.3f} '
              f'f_EQ={row["f_EQ"]:.3f} R={row["R"]:.3f}', flush=True)
    print(f'SCAN_SHARD_DONE {cand} {shard}', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
