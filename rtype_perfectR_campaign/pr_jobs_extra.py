"""Stage-I extra seed families at the top basins (spec secs 8, 20) and
the lossless-from-scratch ceiling jobs (spec sec 27) + C2/FULL branch
jobs (spec sec 10-11) in the same basins.

usage: python pr_jobs_extra.py <n_basins>
Reads results/ceiling_theta0_ledger.csv (Stage-I base set), ranks P/H
points by best F subject to T <= 0.12, and emits:
  extra_jobs.txt     7 more distinct families per top basin (D2)
  c2full_jobs.txt    C2 (3 seeds) + FULL (2 seeds) per top basin
  lossless_jobs.txt  k=0 from scratch, 3 seeds per top basin
"""
import sys

import pandas as pd

import pr_core as pr

S = pr.HERE / 'mining' / 'seeds'


def main(nb):
    df = pd.read_csv(pr.HERE / 'results' / 'ceiling_theta0_ledger.csv')
    ok = df[df['T'] <= 0.12]
    best = ok.groupby(['P', 'H'])['F'].max().sort_values(ascending=False)
    basins = list(best.head(nb).index)
    print('top basins (P,H,F):', [(p, h, round(best[(p, h)], 3))
                                  for p, h in basins])
    extra, c2f, lossless = [], [], []
    for P, H in basins:
        fams = [f'warm:{S}/rect.npy', f'warm:{S}/oldA.npy',
                f'warm:{S}/p258.npy', f'warm:{S}/lowT.npy',
                f'warm:{S}/mirror.npy', f'mix:{S}/newA.npy+{S}/oldB.npy',
                f'mix:{S}/mirror.npy+{S}/oldB.npy']
        for s in fams:
            extra.append(f'D2 {P:.0f} {H:.0f} {s} 150 ceiling_theta0')
        for s in ('rand21', 'multi9', f'warm:{S}/oldB.npy'):
            c2f.append(f'C2 {P:.0f} {H:.0f} {s} 150 device_C2')
        for s in ('rand31', 'multi13'):
            c2f.append(f'FULL {P:.0f} {H:.0f} {s} 150 device_full')
        for s in ('rand11', 'multi7', f'warm:{S}/oldB.npy'):
            lossless.append(f'D2 {P:.0f} {H:.0f} {s} 150 lossless_ceiling L')
    (pr.HERE / 'extra_jobs.txt').write_text('\n'.join(extra) + '\n')
    (pr.HERE / 'c2full_jobs.txt').write_text('\n'.join(c2f) + '\n')
    (pr.HERE / 'lossless_jobs.txt').write_text('\n'.join(lossless) + '\n')
    print(len(extra), 'extra;', len(c2f), 'C2/FULL;', len(lossless),
          'lossless jobs')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
