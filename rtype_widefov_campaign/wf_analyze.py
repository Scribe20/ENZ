"""Robust P/H maps + basin selection + refinement/seed job generation
(spec sections 29-33).

usage: python wf_analyze.py
Reads results/widefov_master_ledger.csv (coarse stage rows), writes
results/coarse_methodA.csv / coarse_methodB.csv, figures/wf_heatmap_*.png,
refine_jobs.txt, seed_jobs.txt and prints the chosen basins.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import wf_core as wf
import rt_core as rc

R = wf.HERE / 'results'
F = wf.HERE / 'figures'
F.mkdir(exist_ok=True)

METRICS = [('Rc_mean', 'mean R_cross (pool)'),
           ('Rc_min', 'worst-angle R_cross'),
           ('Rc_tail25', 'lower-tail R_cross'),
           ('Rc_omega', 'solid-angle-weighted R_cross'),
           ('co_mean', 'mean co-pol leakage'),
           ('T_max', 'worst-angle transmission'),
           ('R_cross0', 'theta=0 R_cross (old score)')]


def robust_score(r):
    return (r.Rc_omega + 0.5 * r.Rc_tail25 + 0.5 * r.Rc_min
            - 0.3 * r.co_mean - 0.2 * r.T_mean)


def fab_ok(r):
    return (r.n_components == 1 and r.min_si_linewidth_nm >= 30.0
            and r.min_air_gap_nm >= 30.0)


def main():
    df = pd.read_csv(R / 'widefov_master_ledger.csv')
    df = df[df.stage == 'coarse'].drop_duplicates('tag', keep='last') \
        .reset_index(drop=True)
    df['score'] = df.apply(robust_score, axis=1)
    df['fab_ok'] = df.apply(fab_ok, axis=1)
    for m in ('A', 'B'):
        sub = df[df.method == m]
        sub.to_csv(R / f'coarse_method{m}.csv', index=False)
        fig, axs = plt.subplots(2, 4, figsize=(19, 8.5))
        for ax, (col, title) in zip(axs.ravel(), METRICS):
            piv = sub.pivot_table(index='H', columns='P', values=col)
            im = ax.imshow(piv.values, origin='lower', cmap='viridis',
                           aspect='auto',
                           extent=[min(wf.PERIODS) - 6.5,
                                   max(wf.PERIODS) + 6.5,
                                   min(wf.HEIGHTS) - 7.5,
                                   max(wf.HEIGHTS) + 7.5])
            for _, r in sub.iterrows():
                ax.text(r.P, r.H, f'{r[col]:.2f}', ha='center',
                        va='center', fontsize=7,
                        color='w' if r[col] < piv.values.max() * 0.7
                        else 'k')
            ax.set_title(title, fontsize=9)
            ax.set_xlabel('P (nm)')
            ax.set_ylabel('H (nm)')
            plt.colorbar(im, ax=ax, shrink=0.85)
        ax = axs.ravel()[-1]
        piv = sub.pivot_table(index='H', columns='P', values='score')
        im = ax.imshow(piv.values, origin='lower', cmap='magma',
                       aspect='auto',
                       extent=[min(wf.PERIODS) - 6.5, max(wf.PERIODS) + 6.5,
                               min(wf.HEIGHTS) - 7.5, max(wf.HEIGHTS) + 7.5])
        for _, r in sub.iterrows():
            mark = '' if r.fab_ok else ' !'
            ax.text(r.P, r.H, f'{r.score:.2f}{mark}', ha='center',
                    va='center', fontsize=7, color='w')
        ax.set_title('ROBUST SCORE (! = fab issue)', fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.85)
        fig.suptitle(f'Method {m}: angle-aware coarse P/H maps '
                     '(full 21-point pool, hard-binary, [9,9])',
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(F / f'wf_heatmap_{m}.png', dpi=160)
        plt.close(fig)

    # basin selection: top-2 per method by robust score (fab-valid first)
    refine, seeds = [], []
    for m in ('A', 'B'):
        sub = df[df.method == m].sort_values('score', ascending=False)
        pref = sub[sub.fab_ok]
        pick = pd.concat([pref, sub[~sub.fab_ok]]).head(2)
        for rank, (_, r) in enumerate(pick.iterrows()):
            print(f'Method {m} basin {rank}: P{r.P:.0f}/H{r.H:.0f} '
                  f'score={r.score:.3f} omega={r.Rc_omega:.3f} '
                  f'min={r.Rc_min:.3f} co={r.co_mean:.3f} '
                  f'fab_ok={r.fab_ok}', flush=True)
            dP = [0.0, -6.0, 6.0]
            dH = [0.0, -10.0, 10.0]
            pts = ([(r.P + d, r.H) for d in dP]
                   + [(r.P, r.H + d) for d in dH[1:]])
            for P2, H2 in pts:
                # one-sided boundary extension only along the robust
                # gradient; never below H=140 or above H=210 (sec 9/32)
                if H2 < 132 or H2 > 210 or P2 < 192 or P2 > 260:
                    continue
                refine.append(f'{m} {P2:.0f} {H2:.0f} 11 60 refinement '
                              f'{r.tag}')
            if rank == 0:
                for s in (23, 47):
                    seeds.append(f'{m} {r.P:.0f} {r.H:.0f} {s} 80 coarse')
    (wf.HERE / 'refine_jobs.txt').write_text('\n'.join(refine) + '\n')
    (wf.HERE / 'seed_jobs.txt').write_text('\n'.join(seeds) + '\n')
    print(f'{len(refine)} refine jobs, {len(seeds)} seed jobs',
          flush=True)


if __name__ == '__main__':
    main()
