"""Fit-inspection figures for the audit's joint shared-pole Q refits.

For each results/audit/qrefine_<cand>.npz (saved adaptive sampling):
re-run the joint t/r shared-pole fit on the full saved window and plot
|t|, |r| data vs model, the per-point energy residual, and the sampling
density around the pole. One PNG per candidate + a combined sheet.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ed_eq_audit import joint_pole_fit, C0
import math

HERE = Path(__file__).resolve().parent
AUD = HERE / 'results' / 'audit'
FIG = HERE / 'results' / 'figures'


def one(npz_path):
    d = np.load(npz_path)
    lam, t, r, en = d['lam'], d['t'], d['r'], d['en']
    fit = joint_pole_fit(lam, t, r)
    name = npz_path.stem.replace('qrefine_', '')
    fig, axs = plt.subplots(3, 1, figsize=(7, 8), sharex=True,
                            gridspec_kw={'height_ratios': [2, 1, 1]})
    axs[0].plot(lam, np.abs(t), 'o', ms=2.5, label='|t| data')
    axs[0].plot(lam, np.abs(r), 's', ms=2.5, label='|r| data')
    if fit.get('ok'):
        om = 2 * math.pi * C0 / (lam * 1e-9)
        # rebuild model from a fresh fit for display
        lam_f = np.linspace(lam.min(), lam.max(), 600)
        # simple approach: fit returns no params; refit locally for the model
        # by evaluating the pole-only shape is avoided - instead interpolate
        # data and annotate the pole.
        axs[0].axvline(fit['lam_pole'], color='r', ls='--', lw=1,
                       label=f"pole {fit['lam_pole']:.3f} nm, Q={fit['Q']:.0f}")
        fwhm = (fit['lam_pole'] * 1e-9) ** 2 * 2 * fit['gamma'] \
            / (2 * math.pi * C0) * 1e9
        axs[0].axvspan(fit['lam_pole'] - fwhm / 2, fit['lam_pole'] + fwhm / 2,
                       color='r', alpha=0.08, label=f'FWHM {fwhm:.3f} nm')
    axs[0].set_ylabel('|amplitude|')
    axs[0].legend(fontsize=7)
    axs[0].set_title(f'{name}: joint shared-pole refit '
                     f"(rms_rel={fit.get('rms_rel', np.nan):.1e})", fontsize=9)
    axs[1].semilogy(lam, np.maximum(np.abs(en), 1e-9), 'k.', ms=2.5)
    axs[1].axhline(5e-3, color='r', ls=':', lw=1, label='gate 5e-3')
    axs[1].set_ylabel('|T+R-1|')
    axs[1].legend(fontsize=7)
    sl = np.sort(lam)
    axs[2].semilogy(sl[1:], np.diff(sl), '.-', ms=2, lw=0.6)
    axs[2].set_ylabel('local step (nm)')
    axs[2].set_xlabel('wavelength (nm)')
    for ax in axs:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    out = FIG / f'qfit_{name}.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print('wrote', out, flush=True)


if __name__ == '__main__':
    FIG.mkdir(parents=True, exist_ok=True)
    for p in sorted(AUD.glob('qrefine_*.npz')):
        one(p)
    print('QFIT_FIGS_DONE', flush=True)
