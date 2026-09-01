"""Coarse-campaign analysis: P/H heatmaps, Pareto ranking, basin
selection, refinement job list. usage: python rt_analyze.py [stage]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rt_core as rc

R = rc.HERE / 'results'
F = rc.HERE / 'figures'


def load(stage='coarse'):
    rows = []
    for p in sorted((rc.HERE / stage).glob('*/final.json')):
        rows.append(json.loads(p.read_text()))
    df = pd.DataFrame(rows)
    df['T_tot_mean'] = 0.5 * (df.T_total_x + df.T_total_y)
    df['A_mean'] = 0.5 * (df.A_x + df.A_y)
    df['score'] = (df.R_cross - 0.5 * df.R_co - 0.5 * df.T_tot_mean
                   - 0.3 * df.pb_phase_err_deg / 180.0)
    df['fab_ok'] = ((df.n_components == 1)
                    & (df.min_si_linewidth_nm >= 24)
                    & (df.min_air_gap_nm >= 24)
                    & (df.edge_clearance_nm > 0))
    return df


def domfam(row, pol):
    fams = {k: row[f'f_{k}_{pol}'] for k in ('ED', 'MD', 'EQ', 'MQ')}
    return max(fams, key=fams.get)


def heatmaps(df, stage='coarse'):
    F.mkdir(exist_ok=True, parents=True)
    for method in ('A', 'B'):
        d = df[df.method == method]
        cols = ['R_cross', 'T_tot_mean', 'R_co', 'pb_phase_err_deg',
                'score']
        fig, axs = plt.subplots(1, 5, figsize=(20, 3.6))
        for ax, c in zip(axs, cols):
            piv = d.pivot_table(index='H', columns='P', values=c)
            im = ax.imshow(piv.to_numpy(), origin='lower', aspect='auto',
                           extent=[min(rc.PERIODS) - 9, max(rc.PERIODS) + 9,
                                   min(rc.HEIGHTS) - 15,
                                   max(rc.HEIGHTS) + 15],
                           cmap='viridis' if c in ('R_cross', 'score')
                           else 'viridis_r')
            plt.colorbar(im, ax=ax)
            ax.set_title(f'{method}: {c}', fontsize=9)
            ax.set_xlabel('P (nm)')
        axs[0].set_ylabel('H (nm)')
        fig.tight_layout()
        fig.savefig(F / f'heatmap_{stage}_{method}.png', dpi=150)
        plt.close(fig)
        d.sort_values('P').to_csv(R / f'{stage}_method{method}.csv',
                                  index=False)
    # dominant-family maps for Method B
    d = df[df.method == 'B'].copy()
    d['dom_x'] = d.apply(lambda r: domfam(r, 'x'), axis=1)
    d['dom_y'] = d.apply(lambda r: domfam(r, 'y'), axis=1)
    fams = ['ED', 'MD', 'EQ', 'MQ']
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, col in zip(axs, ('dom_x', 'dom_y')):
        piv = d.pivot_table(index='H', columns='P', values=col,
                            aggfunc=lambda s: fams.index(s.iloc[0]))
        im = ax.imshow(piv.to_numpy(), origin='lower', aspect='auto',
                       vmin=0, vmax=3, cmap='tab10',
                       extent=[min(rc.PERIODS) - 9, max(rc.PERIODS) + 9,
                               min(rc.HEIGHTS) - 15, max(rc.HEIGHTS) + 15])
        ax.set_title(f'Method B dominant family, {col[-1]}-pol', fontsize=9)
        ax.set_xlabel('P (nm)')
    cb = plt.colorbar(im, ax=axs, ticks=[0, 1, 2, 3])
    cb.ax.set_yticklabels(fams)
    fig.savefig(F / f'heatmap_{stage}_B_domfam.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def basins(df, n=2):
    out = {}
    for method in ('A', 'B'):
        d = df[(df.method == method)].sort_values('score', ascending=False)
        d_ok = d[d.fab_ok]
        pool = d_ok if len(d_ok) >= n else d
        picks, seen = [], set()
        for _, r in pool.iterrows():
            key = (r.P, r.H)
            if any(abs(r.P - p) <= 18 and abs(r.H - h) <= 30
                   for p, h in seen):
                continue
            picks.append(r)
            seen.add(key)
            if len(picks) >= n:
                break
        out[method] = picks
    return out


def main(stage='coarse'):
    df = load(stage)
    print(f'{len(df)} runs loaded')
    heatmaps(df, stage)
    pd.set_option('display.width', 220)
    for method in ('A', 'B'):
        d = df[df.method == method].sort_values('score', ascending=False)
        print(f'\n== Method {method} top 8 ==')
        print(d[['tag', 'R_cross', 'R_co', 'T_tot_mean', 'A_mean',
                 'pb_phase_err_deg', 'abs_rx', 'abs_ry', 'n_components',
                 'min_si_linewidth_nm', 'min_air_gap_nm', 'fab_ok',
                 'score']].head(8).to_string(index=False,
                float_format=lambda v: f'{v:.3f}'))
    bs = basins(df)
    jobs = []
    for method, picks in bs.items():
        for r in picks:
            print(f'\nbasin {method}: P={r.P:.0f} H={r.H:.0f} '
                  f'score={r.score:.3f} Rc={r.R_cross:.3f}')
            for dP in (-9, 0, 9):
                for dH in (-15, 0, 15):
                    jobs.append(f'{method} {r.P+dP:.0f} {r.H+dH:.0f} 11 '
                                f'100 96 9 refinement')
    (rc.HERE / 'configs' / 'refine_jobs.txt').write_text(
        '\n'.join(dict.fromkeys(jobs)) + '\n')
    print(f'\n{len(set(jobs))} refinement jobs written')
    print('RT_ANALYZE_DONE')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'coarse')
