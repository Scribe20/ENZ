"""5-panel multipole-family spectra figures (audit deliverable).

Panels: (1) ED component radiation weights, (2) MD, (3) EQ, (4) MQ
(from the audit's mqspec scans - absent from Stage-A data), (5) complete
4-family fractions. Sources: results/qualify/<cand>/spectra_main.csv
(1-nm grid, complex moments) and results/audit/mqspec_<cand>.csv
(2-nm grid, exact Qm). Vertical line: lam0.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
QUAL = HERE / 'results' / 'qualify'
AUD = HERE / 'results' / 'audit'
FIG = HERE / 'results' / 'figures'
LAM0 = 1332.5
EPS0 = 8.8541878128e-12
C0 = 299792458.0

CANDS = ['P0750_H0250_seed011', 'P0550_H0250_seed011',
         'P0750_H0350_seed011', 'P0650_H0350_seed029',
         'P0750_H0350_seed029']
TITLES = {'P0750_H0250_seed011': 'champion (re-classified my/Qxz hybrid)',
          'P0550_H0250_seed011': 'cleanest balanced ED-EQ',
          'P0750_H0350_seed011': 'second clean balanced ED-EQ',
          'P0650_H0350_seed029': 'MQ-codominant exception',
          'P0750_H0350_seed029': 'needle candidate'}


def weights_from_qualify(df):
    lam = df['lam_nm'].to_numpy()
    k = 2 * math.pi / (lam * 1e-9)
    cE = k ** 4 / (6 * math.pi * EPS0 ** 2)
    out = {'lam': lam}
    for t in ('px', 'py', 'pz'):
        out[t] = cE * (df[f'{t}_re'] ** 2 + df[f'{t}_im'] ** 2).to_numpy()
    for t in ('mx', 'my', 'mz'):
        out[t] = cE / C0 ** 2 * (df[f'{t}_re'] ** 2 + df[f'{t}_im'] ** 2).to_numpy()
    for t in ('Qxx', 'Qyy', 'Qzz', 'Qxy', 'Qxz', 'Qyz'):
        w = 1 if t in ('Qxx', 'Qyy', 'Qzz') else 2
        out[t] = cE / 120 * k ** 2 * w * \
            (df[f'{t}_re'] ** 2 + df[f'{t}_im'] ** 2).to_numpy()
    return out


def one_figure(name):
    df = pd.read_csv(QUAL / name / 'spectra_main.csv')
    W = weights_from_qualify(df)
    mq = pd.read_csv(AUD / f'mqspec_{name}.csv').sort_values('lam_nm')
    fig, axs = plt.subplots(5, 1, figsize=(7.2, 13.5), sharex=True)
    specs = [(0, 'ED', ('px', 'py', 'pz')),
             (1, 'MD', ('mx', 'my', 'mz')),
             (2, 'EQ', ('Qxx', 'Qyy', 'Qzz', 'Qxy', 'Qxz', 'Qyz'))]
    for i, fam, comps in specs:
        for t in comps:
            lw = 2.0 if t in ('px', 'my', 'Qxz') else 1.0
            axs[i].semilogy(W['lam'], np.maximum(W[t], 1e-30), lw=lw, label=t)
        axs[i].set_ylabel(f'{fam} weight (SI)')
        axs[i].legend(fontsize=7, ncol=3, loc='lower right')
    for t in ('Qmxx', 'Qmyy', 'Qmzz', 'Qmxy', 'Qmxz', 'Qmyz'):
        axs[3].semilogy(mq['lam_nm'], np.maximum(mq[f'C_{t}'], 1e-30),
                        lw=1.2, label=t)
    axs[3].set_ylabel('MQ weight (SI)')
    axs[3].legend(fontsize=7, ncol=3, loc='lower right')
    for f, c in (('f_ED', 'tab:blue'), ('f_MD', 'tab:green'),
                 ('f_EQ', 'tab:red'), ('f_MQ', 'tab:purple')):
        axs[4].plot(mq['lam_nm'], mq[f], color=c, lw=1.8, label=f)
    axs[4].set_ylim(0, 1)
    axs[4].set_ylabel('family fraction')
    axs[4].set_xlabel('wavelength (nm)')
    axs[4].legend(fontsize=8, ncol=4)
    for ax in axs:
        ax.axvline(LAM0, color='k', ls=':', lw=0.8)
        ax.grid(alpha=0.25)
    axs[0].set_title(f'{name} — {TITLES.get(name, "")}', fontsize=10)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f'family_spectra_{name}.png'
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print('wrote', out, flush=True)


if __name__ == '__main__':
    for n in CANDS:
        p = AUD / f'mqspec_{n}.csv'
        if not p.exists() or len(pd.read_csv(p)) < 41:
            print(f'skip {n}: mqspec incomplete', flush=True)
            continue
        one_figure(n)
    print('FAMILY_FIGS_DONE', flush=True)
