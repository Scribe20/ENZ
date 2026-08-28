"""Phase 20: publication figure set for the clean-ED-EQ stage.

All panels reproducible from machine-readable CSV/npz in results/.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import stage_core as sc

R = sc.RESULTS
F = R / 'figures'
LAM0 = 1332.5
CLEAN = (1314, 1387)


def cx(d, s):
    return d[s + '_re'].to_numpy() + 1j * d[s + '_im'].to_numpy()


def weights(d):
    lam = d['lam_nm'].to_numpy()
    k = 2 * math.pi / (lam * 1e-9)
    cE = k ** 4 / (6 * math.pi * sc.EPS0 ** 2)
    W = {}
    for t in ('px', 'py', 'pz'):
        W[t] = cE * np.abs(cx(d, t)) ** 2
    for t in ('mx', 'my', 'mz'):
        W[t] = cE / sc.C0 ** 2 * np.abs(cx(d, t)) ** 2
    for t in ('Qxx', 'Qyy', 'Qzz', 'Qxy', 'Qxz', 'Qyz'):
        w = 1 if t[-1] == t[-2] else 2
        W[t] = cE / 120 * k ** 2 * w * np.abs(cx(d, t)) ** 2
    for t in ('Qmxx', 'Qmyy', 'Qmzz', 'Qmxy', 'Qmxz', 'Qmyz'):
        w = 1 if t[-1] == t[-2] else 2
        W[t] = cE / 120 * (k / sc.C0) ** 2 * w * np.abs(cx(d, t)) ** 2
    return lam, W


def shade(ax):
    ax.axvspan(*CLEAN, color='tab:green', alpha=0.07)
    ax.axvline(LAM0, color='k', ls=':', lw=0.8)
    ax.grid(alpha=0.25)


def main():
    F.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(R / 'p0550_multipole_spectra.csv')
    ch = pd.read_csv(R / 'p0550_channel_amplitudes.csv')
    md = pd.read_csv(R / 'p0550_models_tr.csv')
    lam, W = weights(d)

    # A: topology + induced-current proxy
    g = np.load(R / 'audit' / 'gradient_maps.npz')
    rho, Ex = g['rho'], g['Ex']
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.2))
    axs[0].imshow(rho.T, origin='lower', cmap='gray_r',
                  extent=[0, 550, 0, 550])
    axs[0].set_title('binary topology (P = 550 nm)', fontsize=10)
    J = np.abs(Ex[:, :, Ex.shape[2] // 2]) * (rho > 0.5)
    im = axs[1].imshow(J.T, origin='lower', cmap='magma',
                       extent=[0, 550, 0, 550])
    axs[1].set_title('|J_x| proxy, mid-slab, 1332.5 nm', fontsize=10)
    plt.colorbar(im, ax=axs[1], shrink=0.8)
    for a in axs:
        a.set_xlabel('x (nm)'); a.set_ylabel('y (nm)')
    fig.tight_layout(); fig.savefig(F / 'figA_topology_current.png', dpi=160)

    # B: family spectra; C: p components; D: EQ components
    figs = [('figB_families.png',
             [('ED', ['px', 'py', 'pz']), ('MD', ['mx', 'my', 'mz']),
              ('EQ', ['Qxx', 'Qyy', 'Qzz', 'Qxy', 'Qxz', 'Qyz']),
              ('MQ', ['Qmxx', 'Qmyy', 'Qmzz', 'Qmxy', 'Qmxz', 'Qmyz'])])]
    fig, axs = plt.subplots(4, 1, figsize=(7, 10), sharex=True)
    for ax, (fam, comps) in zip(axs, figs[0][1]):
        tot = sum(W[c] for c in comps)
        ax.semilogy(lam, tot, 'k-', lw=2, label=f'C_{fam}')
        for c in comps:
            ax.semilogy(lam, np.maximum(W[c], 1e-30), lw=1, alpha=0.8, label=c)
        ax.set_ylabel(f'{fam} weight'); ax.legend(fontsize=6, ncol=4)
        shade(ax)
    axs[-1].set_xlabel('wavelength (nm)')
    fig.tight_layout(); fig.savefig(F / 'figB_families.png', dpi=160)

    fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for c, lw in (('px', 2), ('py', 1), ('pz', 1)):
        axs[0].semilogy(lam, np.maximum(W[c], 1e-30), lw=lw, label=c)
    axs[0].set_ylabel('ED components'); axs[0].legend(fontsize=8)
    for c in ('Qxx', 'Qyy', 'Qzz', 'Qxy', 'Qxz', 'Qyz'):
        axs[1].semilogy(lam, np.maximum(W[c], 1e-30),
                        lw=2 if c == 'Qxz' else 1, label=c)
    axs[1].set_ylabel('EQ components'); axs[1].legend(fontsize=7, ncol=2)
    axs[1].set_xlabel('wavelength (nm)')
    for a in axs:
        shade(a)
    fig.tight_layout(); fig.savefig(F / 'figCD_px_EQ.png', dpi=160)

    # E: fractions + balance; F: |A|; G: phases; I: R/T; J: directionality
    fig, axs = plt.subplots(5, 1, figsize=(7.2, 13), sharex=True)
    axs[0].plot(lam, d['f_ED'], label='f_ED')
    axs[0].plot(lam, d['f_EQ'], label='f_EQ')
    axs[0].plot(lam, d['f_MD'], label='f_MD')
    axs[0].plot(lam, d['ED_EQ_balance'], 'k--', lw=1.5, label='B_ED_EQ')
    axs[0].set_ylabel('fractions / balance'); axs[0].set_ylim(0, 1)
    axs[0].legend(fontsize=7, ncol=4)
    axs[1].plot(ch['lam_nm'], ch['A_ED_top_abs'], label='|A_ED|')
    axs[1].plot(ch['lam_nm'], ch['A_EQ_top_abs'], label='|A_EQ|')
    axs[1].plot(ch['lam_nm'], ch['A_MD_top_abs'], label='|A_MD|')
    axs[1].set_ylabel('channel |A| (t-units)'); axs[1].legend(fontsize=7)
    axs[2].plot(ch['lam_nm'], ch['dphi_top_deg'], label='dphi top')
    axs[2].plot(ch['lam_nm'], ch['dphi_bot_deg'], label='dphi bottom')
    axs[2].axhline(180, color='r', ls=':', lw=1)
    axs[2].axhline(0, color='gray', ls=':', lw=1)
    axs[2].set_ylabel('ED-EQ phase (deg)'); axs[2].legend(fontsize=7)
    axs[3].plot(lam, d['T'], label='T')
    axs[3].plot(lam, d['R'], label='R')
    axs[3].set_ylabel('R, T'); axs[3].legend(fontsize=8)
    axs[4].plot(ch['lam_nm'], ch['eta_dir'], label='eta_dir')
    axs[4].plot(ch['lam_nm'], ch['xi_bot'], label='xi_bot')
    axs[4].plot(ch['lam_nm'], ch['xi_top'], label='xi_top')
    axs[4].set_ylabel('directionality'); axs[4].legend(fontsize=7)
    axs[4].set_xlabel('wavelength (nm)')
    for a in axs:
        shade(a)
    fig.tight_layout(); fig.savefig(F / 'figEFGIJ_channels.png', dpi=160)

    # H: reconstructions
    fig, axs = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)
    for q, ax in (('T', axs[0]), ('R', axs[1])):
        ax.plot(md['lam_nm'], md[f'{q}_full'], 'k-', lw=2.2, label='full TORCWA')
        for mname, lab in (('A_EDonly', 'ED only'), ('B_ED_EQ', 'ED+EQ'),
                           ('C_ED_EQ_MD', 'ED+EQ+MD'),
                           ('D_plus_2nd', '+2nd order')):
            ax.plot(md['lam_nm'], md[f'{q}_{mname}'], lw=1.1, label=lab)
        ax.set_ylabel(q); ax.legend(fontsize=7); shade(ax)
    axs[1].set_xlabel('wavelength (nm)')
    fig.tight_layout(); fig.savefig(F / 'figH_reconstruction.png', dpi=160)

    # K: thickness trajectory
    sw = pd.read_csv(R / 'p0550_phase_sweep.csv')
    ev, oQ = cx(sw, 'even_px'), cx(sw, 'odd_Q')
    Eu, Ed = cx(sw, 'E_up'), cx(sw, 'E_dn')
    sw['dphi_bot'] = (np.degrees(np.angle(ev / oQ)) + 180)
    sw['dphi_bot'] = np.where(sw['dphi_bot'] > 180, sw['dphi_bot'] - 360,
                              sw['dphi_bot'])
    fine = sw[sw.lam_nm == 1332.5].sort_values('h_nm')
    fig, axs = plt.subplots(3, 1, figsize=(7, 8.5), sharex=True)
    axs[0].plot(fine.h_nm, fine.dphi_bot, 'o-', ms=3)
    axs[0].axhline(180, color='r', ls=':'); axs[0].axhline(-180, color='r', ls=':')
    axs[0].set_ylabel('dphi_bot (deg)')
    axs[1].plot(fine.h_nm, fine.ED_EQ_balance, 'o-', ms=3, label='B_ED_EQ')
    axs[1].plot(fine.h_nm, fine.f_ED + fine.f_EQ, 's-', ms=3, label='f_ED+f_EQ')
    axs[1].plot(fine.h_nm, fine.Qxz_given_EQ, '^-', ms=3, label='Qxz|EQ')
    axs[1].set_ylim(0, 1.05); axs[1].legend(fontsize=7)
    axs[1].set_ylabel('composition')
    axs[2].semilogy(fine.h_nm, fine.R, 'o-', ms=3, label='R')
    axs[2].semilogy(fine.h_nm, fine['T'], 's-', ms=3, label='T')
    axs[2].legend(fontsize=8); axs[2].set_ylabel('R, T at 1332.5 nm')
    axs[2].set_xlabel('slab thickness h (nm)')
    for a in axs:
        a.axvline(250, color='k', ls=':', lw=0.8)
        a.axvline(221, color='tab:red', ls='--', lw=0.8)
        a.grid(alpha=0.25)
    axs[0].set_title('thickness phase knob at 1332.5 nm '
                     '(dotted: frozen h=250; dashed: 180-deg crossing)',
                     fontsize=9)
    fig.tight_layout(); fig.savefig(F / 'figK_phase_knob.png', dpi=160)

    # coarse map
    piv = sw[sw.lam_nm % 5 == 0].pivot_table(index='h_nm', columns='lam_nm',
                                             values='R')
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    im = ax.imshow(piv.to_numpy(), origin='lower', aspect='auto',
                   cmap='viridis_r',
                   extent=[piv.columns.min(), piv.columns.max(),
                           piv.index.min(), piv.index.max()],
                   vmin=0, vmax=0.4)
    plt.colorbar(im, label='R')
    ax.set_xlabel('wavelength (nm)'); ax.set_ylabel('h (nm)')
    ax.set_title('reflection map of the thickness family', fontsize=10)
    fig.tight_layout(); fig.savefig(F / 'figK2_R_map.png', dpi=160)

    # L: robustness (if available)
    rp = [R / f'robustness_shard{i}.csv' for i in range(4)]
    if all(p.exists() for p in rp):
        rob = pd.concat([pd.read_csv(p) for p in rp])
        rob.to_csv(R / 'p0550_robustness.csv', index=False)
        piv_b = rob.pivot_table(index='tag', columns='lam_nm',
                                values='ED_EQ_balance')
        piv_r = rob.pivot_table(index='tag', columns='lam_nm', values='R')
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        for ax, piv2, ttl in ((axs[0], piv_b, 'B_ED_EQ'),
                              (axs[1], piv_r, 'R')):
            for tag, rowv in piv2.iterrows():
                ax.plot(rowv.index, rowv.values, 'o-', ms=2.5, lw=1,
                        label=tag)
            ax.set_title(ttl, fontsize=10); ax.grid(alpha=0.25)
            ax.set_xlabel('wavelength (nm)')
        axs[0].legend(fontsize=6, ncol=2)
        fig.tight_layout(); fig.savefig(F / 'figL_robustness.png', dpi=160)

    # comparison figure: composition vs interference
    d7 = pd.read_csv(R / 'p0750_multipole_spectra.csv')
    c7 = pd.read_csv(R / 'p0750_channel_amplitudes.csv')
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    for j, (dd, cc, nm) in enumerate((
            (d, ch, 'P0550 (composition reference)'),
            (d7, c7, 'P0750 (interference/dark reference)'))):
        axs[0, j].stackplot(dd['lam_nm'], dd['f_ED'], dd['f_EQ'], dd['f_MD'],
                            dd['f_MQ'],
                            labels=['f_ED', 'f_EQ', 'f_MD', 'f_MQ'],
                            colors=['tab:blue', 'tab:red', 'tab:green',
                                    'tab:purple'], alpha=0.85)
        axs[0, j].set_title(nm, fontsize=10); axs[0, j].set_ylim(0, 1)
        axs[0, j].legend(fontsize=6, loc='lower right')
        axs[1, j].plot(cc['lam_nm'], cc['xi_bot'], label='xi_bot')
        axs[1, j].plot(cc['lam_nm'], cc['eta_dir'], label='eta_dir')
        axs[1, j].plot(dd['lam_nm'], dd['R'], 'k--', lw=1, label='R')
        axs[1, j].legend(fontsize=7); axs[1, j].grid(alpha=0.25)
        axs[1, j].set_xlabel('wavelength (nm)')
    axs[0, 0].set_ylabel('family fraction')
    axs[1, 0].set_ylabel('interference metrics')
    fig.suptitle('multipolar COMPOSITION (top) vs INTERFERENCE (bottom)',
                 fontsize=11)
    fig.tight_layout(); fig.savefig(F / 'fig_comparison_0550_0750.png', dpi=160)
    print('FIGS_DONE')


if __name__ == '__main__':
    main()
