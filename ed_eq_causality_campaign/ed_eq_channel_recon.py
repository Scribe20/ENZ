"""Formal odd/even channel reconstruction for the champion (audit section 20).

For every wavelength of the alpha = 1.000 detuning scan (lossless
material, order [9,9]):
  measured scattered amplitudes   t_sc = t_full - t_bare,
                                  r_sc = r_full - r_bare
      (bare = same substrate/air stack with an EMPTY patterned layer),
  exact induced-current channel integrals E_up, E_dn (stored),
  parity split               even = (E_up+E_dn)/2, odd = (E_up-E_dn)/2,
  multipole ladder terms     even_px (0th: p_x), odd_m (1st: m_y),
                             odd_Q (1st: Qe_xz)   (stored),
  residuals of the ladder truncation, the m_y/Q_xz ratio and relative
  phase, and the empirical complex conversion g = E_up / t_sc that fixes
  the TORCWA reference-plane convention.
Writes results/audit/channel_recon_champion.csv and
results/figures/channel_recon_champion.png.
"""
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import ed_eq_core as core

HERE = Path(__file__).resolve().parent
SRC = HERE / 'results' / 'detuning' / 'alpha_1.000.csv'
AUD = HERE / 'results' / 'audit'
FIG = HERE / 'results' / 'figures'
P, H = 750.0, 250.0
ORDER = [9, 9]
LAM0 = 1332.5


def main():
    rows = list(csv.DictReader(open(SRC)))
    out = []
    for i, r in enumerate(rows):
        lam = float(r['lam_nm'])
        bare = core.bare_stack_amplitudes(P, H, lam, ORDER)
        tb, rb = complex(bare['txx']), complex(bare['rxx'])
        t = float(r['txx_re']) + 1j * float(r['txx_im'])
        rr = float(r['rxx_re']) + 1j * float(r['rxx_im'])
        Eu = float(r['E_up_re']) + 1j * float(r['E_up_im'])
        Ed = float(r['E_dn_re']) + 1j * float(r['E_dn_im'])
        ev_px = float(r['even_px_re']) + 1j * float(r['even_px_im'])
        od_Q = float(r['odd_Q_re']) + 1j * float(r['odd_Q_im'])
        od_m = float(r['odd_m_re']) + 1j * float(r['odd_m_im'])
        t_sc, r_sc = t - tb, rr - rb
        even_meas, odd_meas = (Eu + Ed) / 2, (Eu - Ed) / 2
        g = Eu / t_sc if abs(t_sc) > 1e-12 else complex('nan')
        out.append({
            'lam_nm': lam,
            't_sc_re': t_sc.real, 't_sc_im': t_sc.imag,
            'r_sc_re': r_sc.real, 'r_sc_im': r_sc.imag,
            'E_up_abs': abs(Eu), 'E_dn_abs': abs(Ed),
            'even_meas_abs': abs(even_meas), 'odd_meas_abs': abs(odd_meas),
            'even_px_abs': abs(ev_px),
            'odd_m_abs': abs(od_m), 'odd_Q_abs': abs(od_Q),
            'odd_pred_abs': abs(od_m + od_Q),
            'even_resid_rel': abs(even_meas - ev_px) / (abs(even_meas) + 1e-300),
            'odd_resid_rel': abs(odd_meas - od_m - od_Q) / (abs(odd_meas) + 1e-300),
            'my_over_Qxz_abs': abs(od_m) / (abs(od_Q) + 1e-300),
            'my_Qxz_phase_deg': math.degrees(np.angle(od_m / od_Q))
            if abs(od_Q) > 0 else float('nan'),
            'g_abs': abs(g), 'g_arg_deg': math.degrees(np.angle(g)),
            'net_odd_over_parts': abs(od_m + od_Q)
            / (abs(od_m) + abs(od_Q) + 1e-300),
        })
        if i % 20 == 0:
            print(f'{lam:.2f} nm done', flush=True)
    out.sort(key=lambda d: d['lam_nm'])
    AUD.mkdir(parents=True, exist_ok=True)
    with open(AUD / 'channel_recon_champion.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        for d in out:
            w.writerow(d)

    lam = np.array([d['lam_nm'] for d in out])
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 9.5), sharex=True)
    for kk, lab, lw in (('even_meas_abs', '|even| exact integral', 2),
                        ('even_px_abs', '|px ladder term|', 1),
                        ('odd_meas_abs', '|odd| exact integral', 2),
                        ('odd_pred_abs', '|my+Qxz ladder|', 1)):
        axs[0].semilogy(lam, [d[kk] for d in out], lw=lw, label=lab)
    axs[0].set_ylabel('channel amplitude')
    axs[0].legend(fontsize=7)
    for kk, lab in (('odd_m_abs', '|m_y term|'), ('odd_Q_abs', '|Q_xz term|'),
                    ('odd_pred_abs', '|sum|')):
        axs[1].semilogy(lam, [d[kk] for d in out], label=lab)
    ax1b = axs[1].twinx()
    ax1b.plot(lam, [d['my_Qxz_phase_deg'] for d in out], 'k:', lw=1,
              label='rel. phase (deg)')
    ax1b.set_ylabel('arg(m_y/Q_xz) (deg)')
    axs[1].set_ylabel('odd-channel parts')
    axs[1].legend(fontsize=7, loc='upper left')
    axs[2].plot(lam, [d['even_resid_rel'] for d in out], label='even resid')
    axs[2].plot(lam, [d['odd_resid_rel'] for d in out], label='odd resid')
    axs[2].set_yscale('log')
    axs[2].set_ylabel('ladder truncation resid (rel)')
    axs[2].set_xlabel('wavelength (nm)')
    axs[2].legend(fontsize=8)
    for ax in axs:
        ax.axvline(LAM0, color='k', ls=':', lw=0.8)
        ax.grid(alpha=0.25)
    axs[0].set_title('P0750_H0250_seed011 odd/even channel reconstruction '
                     '(alpha=1.000, lossless)', fontsize=10)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / 'channel_recon_champion.png', dpi=160)
    print('CHANNEL_RECON_DONE', flush=True)


if __name__ == '__main__':
    main()
