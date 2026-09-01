"""AR-audit key points: full-field rows for Argand diagrams, order matrix
with fractions/phase, grid matrix, and [17,17] spot checks.

Needs results/ar_rt_all.csv (merged) to locate lambda_Rmin per thickness.
Writes results/ar_keypoints.csv (full rows + tags) and
results/ar_grid_matrix.csv.
"""
import csv

import numpy as np
import pandas as pd
import torch

import stage_core as sc
from ed_eq_audit import decompose_at, family_row
from ar_audit import H_STAR, HS, build_case, rt_both

NAME = 'P0550_H0250_seed011'
LAM0 = 1332.5


def lam_rmin(df, h):
    d = df[(df.case == 'p0550') & (df.order == 9)
           & (np.isclose(df.h_nm, h))].sort_values('lam_nm')
    i = d.f_R.idxmin()
    return float(d.loc[i, 'lam_nm']), float(d.loc[i, 'f_R'])


def main():
    df = pd.read_csv(sc.RESULTS / 'ar_rt_all.csv')
    rho, P, h0 = sc.load_ref(NAME)
    out = sc.RESULTS / 'ar_keypoints.csv'
    done = set()
    if out.exists():
        with open(out) as f:
            done = {r['tag'] for r in csv.DictReader(f)}
    jobs = []
    lam_star, _ = lam_rmin(df, H_STAR)
    for h in HS:
        lr, _ = lam_rmin(df, h)
        jobs.append((f'h{h:g}_lamRmin', h, lr, [9, 9]))
        jobs.append((f'h{h:g}_lam0', h, LAM0, [9, 9]))
    jobs.append(('h221_lam0', 221.0, LAM0, [9, 9]))
    for o in (11, 13, 15):
        jobs.append((f'hstar_lamRmin_o{o}', H_STAR, lam_star, [o, o]))
        jobs.append((f'hstar_lam0_o{o}', H_STAR, LAM0, [o, o]))
    fields = None
    for tag, h, lam, order in jobs:
        if tag in done:
            continue
        row = sc.scan_point_full(rho, P, h0, float(lam), order=order,
                                 h_override=float(h))
        row = {'tag': tag, 'h_nm': float(h), 'order': order[0], **row}
        if fields is None:
            fields = list(row.keys())
        sc.append_row(out, row, fieldnames=fields)
        print(f'{tag}: R={row["R"]:.4f} f_ED={row["f_ED"]:.3f} '
              f'f_EQ={row["f_EQ"]:.3f}', flush=True)
    # [17,17] R/T spot checks
    spot = sc.RESULTS / 'ar_o17_spot.csv'
    if not spot.exists():
        rows = []
        for lam in (LAM0, lam_star):
            with torch.no_grad():
                sim = build_case('p0550', H_STAR, lam, [17, 17])
                rr = rt_both(sim)
            rows.append({'h_nm': H_STAR, 'lam_nm': lam, 'order': 17,
                         **{k: rr[k] for k in ('f_R', 'f_T', 'b_R', 'b_T')}})
            print(f'o17 spot lam={lam}: R={rr["f_R"]:.4f}', flush=True)
        pd.DataFrame(rows).to_csv(spot, index=False)
    # grid matrix at (H_STAR, lam_star)
    gout = sc.RESULTS / 'ar_grid_matrix.csv'
    if not gout.exists():
        rows = []
        for (nxy, nz) in [(32, 5), (48, 9), (64, 15), (96, 21)]:
            mo, ch = decompose_at(rho, P, H_STAR, lam_star, lossless=False,
                                  n_xy=nxy, nz=nz)
            fr = family_row(mo, lam_star)
            q = complex(mo['px']) / complex(mo['Qxz'])
            rows.append({'n_xy': nxy, 'nz': nz,
                         'R': ch['R'], 'T': ch['T'],
                         'f_ED': float(fr['f_ED']), 'f_EQ': float(fr['f_EQ']),
                         'f_MD': float(fr['f_MD']), 'f_MQ': float(fr['f_MQ']),
                         'px_given_ED': float(fr['px_given_ED']),
                         'Qxz_given_EQ': float(fr['Qxz_given_EQ']),
                         'my_total_frac': float(fr['Cmy_total_fraction']),
                         'arg_px_over_Qxz_deg': float(np.degrees(
                             np.angle(q)))})
            print(f'grid {nxy}x{nz}: f_ED={rows[-1]["f_ED"]:.4f} '
                  f'f_EQ={rows[-1]["f_EQ"]:.4f}', flush=True)
        pd.DataFrame(rows).to_csv(gout, index=False)
    print('AR_KEYPTS_DONE', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
