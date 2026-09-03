"""Pareto archive over every perfect-R run (spec sec 26): maximize F,
minimize T, co, A (theta=0), non-dominated set + per-branch ledgers.

usage: python pr_pareto.py
Writes results/perfectR_pareto.csv, theta0_ceiling_real.csv,
theta0_ceiling_lossless.csv, device_{D2,C2,full}_results.csv
"""
import pandas as pd

import pr_core as pr

RES = pr.HERE / 'results'


def nondominated(df, cols_min, col_max):
    keep = []
    vals = df[[col_max] + cols_min].values
    for i, v in enumerate(vals):
        dom = False
        for j, u in enumerate(vals):
            if j == i:
                continue
            if (u[0] >= v[0] and all(u[k] <= v[k] for k in range(1, len(v)))
                    and (u[0] > v[0] or any(u[k] < v[k]
                                            for k in range(1, len(v))))):
                dom = True
                break
        keep.append(not dom)
    return df[keep]


def main():
    frames = []
    for stage in ('ceiling_theta0', 'ceiling_largeP', 'device_C2',
                  'device_full', 'lossless_ceiling'):
        p = RES / f'{stage}_ledger.csv'
        if p.exists():
            d = pd.read_csv(p)
            d['stage'] = stage
            frames.append(d)
    df = pd.concat(frames).drop_duplicates('tag', keep='last')
    real = df[~df.lossless]
    real[real.stage.isin(['ceiling_theta0', 'ceiling_largeP'])] \
        .sort_values('F', ascending=False) \
        .to_csv(RES / 'theta0_ceiling_real.csv', index=False)
    df[df.lossless].sort_values('F', ascending=False) \
        .to_csv(RES / 'theta0_ceiling_lossless.csv', index=False)
    for br, nm in (('D2', 'D2'), ('C2', 'C2'), ('FULL', 'full')):
        real[real.branch == br].sort_values('F', ascending=False) \
            .to_csv(RES / f'device_{nm}_results.csv', index=False)
    pf = nondominated(real.reset_index(drop=True), ['T', 'co', 'A'], 'F')
    pf.sort_values('F', ascending=False) \
        .to_csv(RES / 'perfectR_pareto.csv', index=False)
    print(f'{len(real)} real runs, {len(pf)} Pareto-optimal; top F:')
    print(real.sort_values('F', ascending=False).head(10)[
        ['tag', 'stage', 'F', 'T', 'co', 'A', 'phase_err_deg', 'n_islands',
         'min_si_nm', 'min_gap_nm']].to_string(index=False))


if __name__ == '__main__':
    main()
