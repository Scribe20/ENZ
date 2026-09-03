"""Required figures FIG 1-12 (spec sec 37) from the campaign CSVs.
usage: python pr_figs.py [newA_label newB_label ...finalist labels]
Figures that need data not yet present are skipped with a message.
"""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

import pr_core as pr

RES = pr.HERE / 'results'
FIG = pr.HERE / 'figures'


def opt(path):
    return pd.read_csv(path) if path.exists() else None


def fig1(mine):
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    sc = axs[0].scatter(mine['T'], mine.F, c=mine.A, cmap='viridis', s=28,
                        edgecolors='k', linewidths=0.3)
    axs[0].set_xlabel('T_tot (theta=0)'); axs[0].set_ylabel('F_ideal')
    plt.colorbar(sc, ax=axs[0], label='absorption A')
    axs[1].scatter(mine.co, mine.F, c=mine.Rtot, cmap='plasma', s=28,
                   edgecolors='k', linewidths=0.3)
    axs[1].set_xlabel('R_co (theta=0)'); axs[1].set_ylabel('F_ideal')
    for ax in axs:
        ax.grid(alpha=0.25)
    for _, r in mine.sort_values('F', ascending=False).head(4).iterrows():
        axs[0].annotate(r.tag[:18], (r['T'], r.F), fontsize=6.5)
    fig.suptitle('FIG 1: historical partial-solution map (184 geometries, '
                 'corrected conventions)', fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / 'pr_FIG1_historical_pareto.png', dpi=160)
    plt.close(fig)


def fig2(ceil):
    ok = ceil[ceil['T'] <= 0.12]
    best = ok.groupby(['P', 'H']).F.max().reset_index()
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (d, ttl) in zip(axs, ((ceil.groupby(['P', 'H']).F.max()
                                    .reset_index(), 'best F (any T)'),
                                   (best, 'best F with T<=0.12'))):
        piv = d.pivot_table(index='H', columns='P', values='F')
        im = ax.imshow(piv.values, origin='lower', cmap='magma', aspect='auto',
                       extent=[piv.columns.min() - 6, piv.columns.max() + 6,
                               piv.index.min() - 15, piv.index.max() + 15])
        for _, r in d.iterrows():
            ax.text(r.P, r.H, f'{r.F:.2f}', ha='center', va='center',
                    fontsize=7, color='w')
        ax.set_title(f'FIG 2: theta=0 ideal-matrix fidelity, {ttl}', fontsize=9)
        ax.set_xlabel('P (nm)'); ax.set_ylabel('H (nm)')
        plt.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout(); fig.savefig(FIG / 'pr_FIG2_fidelity_landscape.png', dpi=160)
    plt.close(fig)


def fig3(real, loss):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for d, c, lab in ((real, 'tab:red', 'real a-Si'),
                      (loss, 'tab:blue', 'lossless-optimized (k=0)')):
        if d is None or not len(d):
            continue
        b = d.groupby('H').F.max()
        ax.plot(b.index, b.values, 'o-', c=c, label=f'{lab}: best F vs H')
    ax.axhline(1.0, color='gray', ls='--', lw=1)
    ax.set_xlabel('H (nm)'); ax.set_ylabel('best F_ideal (theta=0, any P)')
    ax.grid(alpha=0.25); ax.legend()
    ax.set_title('FIG 3: real vs lossless-optimized ceiling', fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / 'pr_FIG3_real_vs_lossless.png', dpi=160)
    plt.close(fig)


def fig456(real):
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2))
    cols = {'D2': 'tab:blue', 'C2': 'tab:green', 'FULL': 'tab:red'}
    for br, c in cols.items():
        d = real[real.branch == br]
        if not len(d):
            continue
        axs[0].scatter(d['T'], d.F, c=c, s=22, label=br, alpha=0.8)
        axs[1].scatter(d['T'], d.F, c=c, s=22, alpha=0.8)
        axs[2].scatter(d.A, d.F, c=c, s=22, alpha=0.8)
    axs[0].set_title('FIG 4: D2 vs C2 vs FULL (F vs T)', fontsize=9)
    axs[1].set_title('FIG 5: F_ideal vs transmission', fontsize=9)
    axs[2].set_title('FIG 6: F_ideal vs absorption', fontsize=9)
    axs[0].legend()
    for ax, xl in zip(axs, ('T_tot', 'T_tot', 'A')):
        ax.set_xlabel(xl); ax.set_ylabel('F_ideal'); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(FIG / 'pr_FIG456_branches_T_A.png', dpi=160)
    plt.close(fig)


def fig7(labels_paths):
    n = len(labels_paths)
    fig, axs = plt.subplots(1, n, figsize=(3.4 * n, 3.6))
    axs = np.atleast_1d(axs)
    for ax, (lab, path, P, H) in zip(axs, labels_paths):
        b = np.load(path) > 0.5
        ax.imshow(b.T, origin='lower', cmap='gray_r',
                  extent=[-P / 2, P / 2, -P / 2, P / 2], interpolation='nearest')
        ax.add_patch(Circle((0, 0), pr.r_design(P), fill=False, ec='tab:red',
                            ls='--', lw=1.0))
        ax.set_title(f'{lab}\nP={P:.0f} H={H:.0f}', fontsize=8.5)
    fig.suptitle('FIG 7: best geometries (red = 15-nm rotation-safe envelope)',
                 fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / 'pr_FIG7_geometries.png', dpi=200)
    plt.close(fig)


def fig8(conv):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for lab, d in conv.groupby('label'):
        d = d.sort_values('order')
        for k, mk in ((0, 'o'), (1, 's')):
            ax.plot(d[f'evR{k}_re'], d[f'evR{k}_im'], mk + '-', ms=5,
                    label=f'{lab} eig{k} (orders 9->15)')
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), 'k--', lw=0.7)
    ax.set_aspect('equal'); ax.grid(alpha=0.25); ax.legend(fontsize=7)
    ax.set_title('FIG 8: complex reflection eigenvalues (unit circle = ideal)',
                 fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / 'pr_FIG8_reflection_eigen.png', dpi=160)
    plt.close(fig)


def fig10_11(cont):
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5))
    for tag, d in cont.groupby('tag'):
        u = d[d.alpha == 0].groupby('theta').F.agg(['mean', 'min'])
        axs[0].plot(u.index, u['mean'], 'o-', label=f'{tag[:26]} mean')
        axs[0].plot(u.index, u['min'], 'x--', label=f'{tag[:26]} min over phi')
        r = d[d.alpha != 0].pivot_table(index='theta', columns='alpha', values='F')
        for al in r.columns:
            axs[1].plot(r.index, r[al], 's-', label=f'{tag[:20]} alpha={al:.0f}')
    axs[0].set_title('FIG 10: PB matrix fidelity vs incident angle', fontsize=9)
    axs[1].set_title('FIG 11: physically-rotated fidelity F(U_alpha)', fontsize=9)
    for ax in axs:
        ax.set_xlabel('theta (deg)'); ax.set_ylabel('F'); ax.grid(alpha=0.25)
        ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(FIG / 'pr_FIG10_11_pb_vs_angle.png', dpi=160)
    plt.close(fig)


def fig12(mine, real):
    fig, ax = plt.subplots(figsize=(8, 5))
    top = real.sort_values('F', ascending=False).head(6)
    labs = [t[:20] for t in top.tag]
    parts = np.array([top.F, top['T'], top.co, top.A]).T
    bottom = np.zeros(len(top))
    for j, (nm, c) in enumerate((('F_ideal (useful)', 'tab:green'),
                                 ('T_tot', 'tab:blue'), ('R_co', 'tab:orange'),
                                 ('A', 'tab:red'))):
        ax.bar(labs, parts[:, j], bottom=bottom, color=c, label=nm)
        bottom += parts[:, j]
    ax.set_ylabel('power budget (theta=0)'); ax.legend(fontsize=8)
    ax.set_title('FIG 12: real-material failure budget of the top candidates',
                 fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=25, ha='right', fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / 'pr_FIG12_failure_budget.png', dpi=160)
    plt.close(fig)


def main(fin):
    mine = opt(RES / 'perfect_r_workspace_candidates.csv')
    real = opt(RES / 'theta0_ceiling_real.csv')
    loss = opt(RES / 'theta0_ceiling_lossless.csv')
    if mine is not None:
        fig1(mine)
    if real is not None:
        fig2(real[real.stage == 'ceiling_theta0'])
        fig3(real, loss)
        fig456(real)
        fig12(mine, real)
    conv = opt(RES / 'convergence.csv')
    if conv is not None:
        fig8(conv)
    frames = [opt(RES / 'cont55_ledger.csv')]
    pools = []
    for p in (pr.HERE / 'continuation').glob('*cont55*/pool_final.csv'):
        d = pd.read_csv(p); d['tag'] = p.parent.name; pools.append(d)
    if pools:
        fig10_11(pd.concat(pools))
    if fin:
        fig7([(f[0], f[1], float(f[2]), float(f[3]))
              for f in [x.split(',') for x in fin]])
    print('PR_FIGS_DONE')


if __name__ == '__main__':
    main(sys.argv[1:])
