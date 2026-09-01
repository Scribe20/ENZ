"""Port-state audit figures 1-7 (Phase V). All from machine-readable CSVs."""
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import stage_core as sc
from ar_figs import draw_argand

R = sc.RESULTS
F = R / 'figures'
LAM0 = 1332.5


def cx(d, s):
    return d[s + '_re'].to_numpy() + 1j * d[s + '_im'].to_numpy()


def tpiece(row, key):
    return complex(row[f't_{key}_re'], row[f't_{key}_im'])


def draw_t_argand(ax, row, title):
    tbg = complex(row['tbg_re'], row['tbg_im'])
    tfull = complex(row['t_full_re'], row['t_full_im'])
    pieces = {k: tpiece(row, k) for k in ('ED', 'MD', 'EQ', '2nd')}
    cum = [('bg', tbg)]
    z = tbg
    for name in ('ED', 'MD', 'EQ', '2nd'):
        z += pieces[name]
        cum.append((f'+{name}', z))
    colors = ['#666', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple']
    prev = 0j
    for (name, z), c in zip(cum, colors):
        ax.annotate('', xy=(z.real, z.imag), xytext=(prev.real, prev.imag),
                    arrowprops=dict(arrowstyle='->', color=c, lw=1.8))
        ax.text(((prev + z) / 2).real, ((prev + z) / 2).imag, name,
                fontsize=7.5, color=c)
        prev = z
    ax.annotate('', xy=(tfull.real, tfull.imag),
                xytext=(prev.real, prev.imag),
                arrowprops=dict(arrowstyle='->', color='k', lw=1.1,
                                linestyle=':'))
    ax.plot([tfull.real], [tfull.imag], 'k*', ms=11,
            label=f'full t, |t|={abs(tfull):.3f}')
    ax.plot([0], [0], 'ko', ms=6, mfc='none', label='t = 0')
    pts = [c for _, c in cum] + [tfull, 0j]
    lim = 1.1 * max(max(abs(p.real), abs(p.imag)) for p in pts)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    ax.set_aspect('equal'); ax.grid(alpha=0.2)
    ax.set_xlabel('Re t'); ax.set_ylabel('Im t')
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc='lower left')
    axi = ax.inset_axes([0.63, 0.63, 0.35, 0.35])
    prev = 0j
    for (name, z), c in zip(cum, colors):
        axi.annotate('', xy=(z.real, z.imag), xytext=(prev.real, prev.imag),
                     arrowprops=dict(arrowstyle='->', color=c, lw=1.2))
        prev = z
    axi.annotate('', xy=(tfull.real, tfull.imag),
                 xytext=(prev.real, prev.imag),
                 arrowprops=dict(arrowstyle='->', color='k', lw=0.9,
                                 linestyle=':'))
    axi.plot([tfull.real], [tfull.imag], 'k*', ms=8)
    axi.plot([0], [0], 'ko', ms=4, mfc='none')
    zl = max(4 * abs(tfull), 0.15)
    axi.set_xlim(-zl, zl); axi.set_ylim(-zl, zl)
    axi.set_aspect('equal'); axi.tick_params(labelsize=5)
    axi.axhline(0, color='gray', lw=0.4); axi.axvline(0, color='gray', lw=0.4)
    axi.set_title('zoom at origin', fontsize=6)


def main():
    ph = pd.read_csv(R / 'p0550_h_phase_unwrapped.csv')
    mp = pd.read_csv(R / 'p0550_h_multipoles.csv')
    rt = pd.read_csv(R / 'p0550_h_complex_rt.csv')
    summ = json.load(open(R / 'port_state_summary.json'))
    lo, hi = summ['mode_identity_interval']['h_range']

    # FIG 1
    fig, axs = plt.subplots(3, 2, figsize=(11.5, 11), sharex=True)
    a = axs[0, 0]
    a.plot(ph.h_nm, ph.dphi_bot_wrapped_deg, 'o-', ms=2.5)
    a.axhline(180, color='r', ls=':'); a.axhline(-180, color='r', ls=':')
    a.set_title('A  wrapped ED-EQ bottom phase (jump = wrapping artifact)',
                fontsize=9); a.set_ylabel('deg')
    a = axs[0, 1]
    a.plot(ph.h_nm, ph.dphi_bot_unwrapped_deg, 'o-', ms=2.5,
           label='bottom (unwrapped)')
    a.plot(ph.h_nm, ph.dphi_top_unwrapped_deg, 's-', ms=2, lw=0.8,
           label='top (unwrapped)')
    a.set_title('B  unwrapped: strictly monotonic, -1.34 deg/nm in knob '
                'region', fontsize=9)
    a.legend(fontsize=7); a.set_ylabel('deg')
    a = axs[1, 0]
    for c, col in (('f_ED', 'tab:blue'), ('f_EQ', 'tab:red'),
                   ('f_MD', 'tab:green'), ('f_MQ', 'tab:purple')):
        a.plot(mp.h_nm, mp[c], color=col, label=c)
    a.plot(mp.h_nm, mp.ED_EQ_balance, 'k--', lw=1, label='B_ED_EQ')
    a.set_ylim(0, 1); a.legend(fontsize=6, ncol=3)
    a.set_title('C  composition vs h', fontsize=9)
    a = axs[1, 1]
    a.semilogy(rt.h_nm, rt.R, label='R')
    a.semilogy(rt.h_nm, rt['T'], label='T')
    a.semilogy(rt.h_nm, np.abs(rt.A), label='|A|', lw=0.8)
    a.legend(fontsize=7); a.set_title('D  R, T, A vs h', fontsize=9)
    a = axs[2, 0]
    a.plot(ph.h_nm, ph.phi_t_unwrapped_deg, label='arg(t) unwrapped')
    a.plot(ph.h_nm, ph.phi_r_unwrapped_deg, label='arg(r) unwrapped')
    a.legend(fontsize=7)
    a.set_title('E  device phases (corr with dphi: -0.98 / -0.99)',
                fontsize=9)
    a.set_xlabel('h (nm)'); a.set_ylabel('deg')
    a = axs[2, 1]
    m = (ph.h_nm >= lo) & (ph.h_nm <= hi)
    a.plot(ph.h_nm[m], ph.phi_t_unwrapped_deg[m]
           - ph.phi_t_unwrapped_deg[m].iloc[0], 'k-o', ms=2.5)
    cov = summ['phase_coverage_over_identity_interval']['arg_t_deg']
    a.set_title(f'F  arg(t) coverage over identity interval '
                f'h={lo:g}-{hi:g}: {cov:.0f} deg', fontsize=9)
    a.set_xlabel('h (nm)'); a.set_ylabel('relative arg(t) (deg)')
    for row_ in axs:
        for a in row_:
            a.grid(alpha=0.25)
            a.axvspan(lo, hi, color='tab:green', alpha=0.06)
    fig.suptitle('FIGURE 1 - P0550 thickness phase-knob audit (1332.5 nm)',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(F / 'ps_fig1_phase_knob_audit.png', dpi=180)
    fig.savefig(F / 'ps_fig1_phase_knob_audit.pdf')
    plt.close(fig)

    # FIG 2: r-plane Argand h=221/227.2/250
    kp = pd.read_csv(R / 'ar_keypoints.csv')
    rows = {r['tag']: r for _, r in kp.iterrows()}
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.6))
    draw_argand(axs[0], rows['h221_lam0'], 'h = 221 nm (internal ED-EQ '
                'cancellation:\nonly background left)')
    draw_argand(axs[1], rows['h227.2_lam0'], 'h = 227.2 nm (residual '
                'opposes bg:\ntransparent state)')
    draw_argand(axs[2], rows['h250_lam0'], 'h = 250 nm (frozen: partial '
                'interference)')
    fig.suptitle('FIGURE 2 - P0550 complex-r construction at 1332.5 nm',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(F / 'ps_fig2_r_argand.png', dpi=180)
    fig.savefig(F / 'ps_fig2_r_argand.pdf')
    plt.close(fig)

    # FIG 3: P0750 high-res spectrum
    p7 = pd.read_csv(R / 'p0750_highres_rt.csv').sort_values('lam_nm')
    p7 = p7[(p7.lam_nm >= 1320) & (p7.lam_nm <= 1350)]
    Txx = p7.txx_re ** 2 + p7.txx_im ** 2
    Tyx = p7.tyx_re ** 2 + p7.tyx_im ** 2
    fig, axs = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True)
    a = axs[0]
    a.semilogy(p7.lam_nm, p7['T'], 'k-', lw=1.8, label='T (total)')
    a.semilogy(p7.lam_nm, Txx, 'b--', lw=1.2, label='T co-pol |txx|^2')
    a.semilogy(p7.lam_nm, Tyx, 'c:', lw=1.2, label='T cross-pol |tyx|^2')
    a.semilogy(p7.lam_nm, p7.R, 'r-', lw=1.4, label='R (total)')
    a.semilogy(p7.lam_nm, np.abs(1 - p7['T'] - p7.R), color='gray', lw=0.9,
               label='|A|')
    a.axvline(1334.0, color='k', ls=':', lw=0.8)
    a.set_ylim(1e-5, 1.5); a.legend(fontsize=7, ncol=2)
    a.set_title('FIGURE 3 - P0750 resonance at 0.2-nm sampling: '
                'T_min = 0.011 (co-pol 1.8e-4), R = 0.988, A = 9e-4',
                fontsize=9)
    a.set_ylabel('power')
    a = axs[1]
    for c, col in (('f_ED', 'tab:blue'), ('f_MD', 'tab:green'),
                   ('f_EQ', 'tab:red'), ('f_MQ', 'tab:purple')):
        a.plot(p7.lam_nm, p7[c], color=col, label=c)
    a.axvline(1334.0, color='k', ls=':', lw=0.8)
    a.set_ylim(0, 1); a.legend(fontsize=7, ncol=4)
    a.set_ylabel('family fraction'); a.set_xlabel('wavelength (nm)')
    for a in axs:
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(F / 'ps_fig3_p0750_highres.png', dpi=180)
    fig.savefig(F / 'ps_fig3_p0750_highres.pdf')
    plt.close(fig)

    # FIG 4: t-plane Argand at Tmin and +-1.6 nm
    ta = pd.read_csv(R / 'p0750_t_argand.csv')
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.6))
    for a, lam in zip(axs, (1332.4, 1334.0, 1335.6)):
        row = ta.iloc[(ta.lam_nm - lam).abs().idxmin()]
        draw_t_argand(a, row, f'{row["lam_nm"]:.1f} nm')
    fig.suptitle('FIGURE 4 - P0750 complex-t construction across the '
                 'resonance (t = 0 marked; ladder to 2nd order)',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(F / 'ps_fig4_t_argand.png', dpi=180)
    fig.savefig(F / 'ps_fig4_t_argand.pdf')
    plt.close(fig)

    # FIG 5: duality
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 5))
    draw_argand(axs[0], rows['h227.2_lam0'],
                'TRANSPARENT: P0550 h=227.2, r_total -> 0.14\n'
                'T = 0.98, R = 0.020')
    row = ta.iloc[(ta.lam_nm - 1334.0).abs().idxmin()]
    draw_t_argand(axs[1], row,
                  'REFLECTIVE: P0750 at 1334.0 nm, t_xx -> 0.014\n'
                  'T = 0.011, R = 0.988')
    fig.suptitle('FIGURE 5 - transparent / reflective duality in one '
                 'convention (left: r-plane; right: t-plane)', fontsize=10)
    fig.tight_layout()
    fig.savefig(F / 'ps_fig5_duality.png', dpi=180)
    fig.savefig(F / 'ps_fig5_duality.pdf')
    plt.close(fig)

    # FIG 6: same-composition / different-function
    pr = pd.read_csv(R / 'same_composition_opposite_function_pairs.csv')
    fam = mp.merge(rt[['h_nm', 'R', 'T']], on='h_nm')
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    a = axs[0]
    # all-pairs cloud from the family
    hs = fam.h_nm.to_numpy()
    fED, fMD = fam.f_ED.to_numpy(), fam.f_MD.to_numpy()
    fEQ, fMQ = fam.f_EQ.to_numpy(), fam.f_MQ.to_numpy()
    Rv = fam.R.to_numpy()
    D, dR = [], []
    for i in range(len(hs)):
        for j in range(i + 1, len(hs)):
            D.append(math.sqrt((fED[i] - fED[j]) ** 2 + (fMD[i] - fMD[j]) ** 2
                               + (fEQ[i] - fEQ[j]) ** 2
                               + (fMQ[i] - fMQ[j]) ** 2))
            dR.append(abs(Rv[i] - Rv[j]))
    a.scatter(D, dR, s=6, alpha=0.35, color='tab:blue')
    top = pr.head(5)
    a.scatter(top.D_comp, top.dR, s=45, color='tab:red', zorder=5,
              label='top decoupling pairs')
    for _, rr in top.head(2).iterrows():
        a.annotate(f'h={rr.h1:g} vs {rr.h2:g}', (rr.D_comp, rr.dR),
                   fontsize=7, xytext=(6, 4), textcoords='offset points')
    a.set_xlabel('composition distance D_comp')
    a.set_ylabel('|R(h1) - R(h2)|')
    a.legend(fontsize=7); a.grid(alpha=0.25)
    a.set_title('A  same composition, different function (P0550 h pairs)',
                fontsize=9)
    a = axs[1]
    scpl = a.scatter(fam.h_nm, fam.R, c=fam.ED_EQ_balance, cmap='viridis',
                     s=20)
    plt.colorbar(scpl, ax=a, label='B_ED_EQ')
    a.set_xlabel('h (nm)'); a.set_ylabel('R'); a.grid(alpha=0.25)
    a.set_title('B  R vs h colored by balance: R moves 14x while balance '
                'stays high', fontsize=9)
    fig.suptitle('FIGURE 6 - composition-function decoupling', fontsize=10)
    fig.tight_layout()
    fig.savefig(F / 'ps_fig6_decoupling.png', dpi=180)
    fig.savefig(F / 'ps_fig6_decoupling.pdf')
    plt.close(fig)

    # FIG 7: composition-port map + phase-port map
    cm = pd.read_csv(R / 'composition_port_map.csv')
    cm = cm[np.isfinite(cm['T'])]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8))
    a = axs[0]
    x = cm.f_ED - cm.f_EQ
    y = cm.f_MD + cm.f_MQ
    mk = {'P0550_hfam': 'o', 'P0750_lam': 's'}
    for s_, m in mk.items():
        d = cm[cm.set == s_]
        scpl = a.scatter(d.f_ED - d.f_EQ, d.f_MD + d.f_MQ, c=d['T'],
                         cmap='RdYlBu', vmin=0, vmax=1, s=26, marker=m,
                         edgecolors='k', linewidths=0.3, label=s_)
    plt.colorbar(scpl, ax=a, label='T')
    a.set_xlabel('f_ED - f_EQ'); a.set_ylabel('f_MD + f_MQ')
    a.legend(fontsize=7); a.grid(alpha=0.25)
    a.set_title('A  composition -> port map: T spans 0.01-0.98 at similar '
                'compositions', fontsize=9)
    a = axs[1]
    dphi_dist = np.abs(((ph.dphi_bot_wrapped_deg + 180) % 360) - 180)
    a.scatter(dphi_dist, rt.R, c=mp.ED_EQ_balance, cmap='viridis', s=18)
    a.set_xlabel('|wrapped dphi_bot| distance from 180 (deg)')
    a.set_ylabel('R'); a.grid(alpha=0.25)
    a.set_title('B  phase -> port map (corr -0.93): phase predicts R; '
                'balance does not (corr 0.17)', fontsize=9)
    fig.suptitle('FIGURE 7 - composition vs phase as port-state predictors',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(F / 'ps_fig7_maps.png', dpi=180)
    fig.savefig(F / 'ps_fig7_maps.pdf')
    plt.close(fig)
    print('PS_FIGS_DONE')


if __name__ == '__main__':
    main()
