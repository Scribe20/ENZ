"""Required figures (spec section 51). F1/F2 come from wf_analyze,
F9 from wf_argand. This script renders F3-F8, F10, F12 from the CSVs
and F11 geometry comparison.

usage: python wf_figs.py <newA_tag> <newB_tag>
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
import torch

import wf_core as wf
import rt_core as rc
from wf_anglemap import load_geometry

R = wf.HERE / 'results'
F = wf.HERE / 'figures'
COMPARE = ['rectangle', 'oldA', 'oldB']   # + new tags appended in main
COLORS = {'rectangle': '#444444', 'oldA': 'tab:orange',
          'oldB': 'tab:red', 'newA': 'tab:blue', 'newB': 'tab:green'}


def label_of(tag, newA, newB):
    if tag == newA:
        return 'newA'
    if tag == newB:
        return 'newB'
    return tag


def main(newA, newB):
    amap = pd.read_csv(R / 'full_angle_maps.csv')
    acc = pd.read_csv(R / 'acceptance_metrics.csv')
    led = pd.read_csv(R / 'widefov_master_ledger.csv') \
        .drop_duplicates('tag', keep='last')
    tags = COMPARE + [newA, newB]

    # F3: normal-incidence vs robust metric for ALL campaign candidates
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for m, mk in (('A', 'o'), ('B', 's')):
        sub = led[led.method == m]
        sc = ax.scatter(sub.R_cross0, sub.Rc_omega, c=sub.P, marker=mk,
                        cmap='plasma', vmin=195, vmax=258, s=55,
                        edgecolors='k', linewidths=0.4,
                        label=f'Method {m}')
    rrow = acc[acc.tag == 'rectangle'].iloc[0]
    ax.plot(rrow.R_cross0, rrow.Rc_omega_085, '*', ms=18, c='k',
            label='paper rectangle')
    for old in ('oldA', 'oldB'):
        if len(acc[acc.tag == old]):
            orow = acc[acc.tag == old].iloc[0]
            ax.plot(orow.R_cross0, orow.Rc_omega_085, 'X', ms=12,
                    c=COLORS[old], label=f'{old} (theta0-optimized)')
    plt.colorbar(sc, ax=ax, label='P (nm)')
    ax.set_xlabel('R_cross(theta=0)')
    ax.set_ylabel('solid-angle-weighted <R_cross> (pool/0-85)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_title('F3: peak vs wide-angle robust metric', fontsize=10)
    fig.tight_layout()
    fig.savefig(F / 'wf_F3_peak_vs_robust.png', dpi=170)
    plt.close(fig)

    # F4/F5: R_cross(theta), R_co(theta) comparison (min/mean over phi)
    for col, nm in (('R_cross', 'F4'), ('R_co', 'F5')):
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for tag in tags:
            sub = amap[(amap.tag == tag) & (amap.theta <= 85)]
            if not len(sub):
                continue
            lab = label_of(tag, newA, newB)
            g = sub.groupby('theta')[col]
            ax.plot(g.mean().index, g.mean().values, '-',
                    c=COLORS[lab], label=lab, lw=2)
            ax.fill_between(g.mean().index, g.min().values,
                            g.max().values, color=COLORS[lab], alpha=0.15)
        ax.set_xlabel('theta_air (deg)')
        ax.set_ylabel(col)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        ax.set_title(f'{nm}: {col} vs angle (band = phi spread)',
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(F / f'wf_{nm}_{col}_vs_theta.png', dpi=170)
        plt.close(fig)

    # F6: theta-phi maps
    fig, axs = plt.subplots(1, len(tags), figsize=(3.6 * len(tags), 4.4),
                            sharey=True)
    for ax, tag in zip(axs, tags):
        sub = amap[(amap.tag == tag) & (amap.theta <= 85)]
        if not len(sub):
            continue
        piv = sub.pivot_table(index='phi', columns='theta',
                              values='R_cross')
        im = ax.imshow(piv.values, origin='lower', aspect='auto',
                       cmap='inferno', vmin=0, vmax=0.85,
                       extent=[0, 85, 0, 90])
        ax.set_title(label_of(tag, newA, newB), fontsize=9)
        ax.set_xlabel('theta (deg)')
    axs[0].set_ylabel('phi (deg)')
    fig.colorbar(im, ax=axs, shrink=0.8, label='R_cross')
    fig.suptitle('F6: R_cross(theta, phi)', fontsize=11)
    fig.savefig(F / 'wf_F6_thetaphi_maps.png', dpi=170,
                bbox_inches='tight')
    plt.close(fig)

    # F7: |rx|, |ry|, dphi vs theta (phi=0)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for tag in tags:
        sub = amap[(amap.tag == tag) & (amap.phi == 0)
                   & (amap.theta <= 85)].sort_values('theta')
        if not len(sub):
            continue
        lab = label_of(tag, newA, newB)
        axs[0].plot(sub.theta, sub.abs_rx, c=COLORS[lab], label=lab)
        axs[1].plot(sub.theta, sub.abs_ry, c=COLORS[lab])
        derr = np.radians(sub.dphi_r_deg - 180.0)
        err = np.abs(np.degrees(np.arctan2(np.sin(derr), np.cos(derr))))
        axs[2].plot(sub.theta, err, c=COLORS[lab])
    axs[0].set_title('|r_x| (p-like)')
    axs[1].set_title('|r_y| (s-like)')
    axs[2].set_title('|dphi - pi| (deg)')
    for a in axs:
        a.set_xlabel('theta (deg)')
        a.grid(alpha=0.25)
    axs[0].legend(fontsize=8)
    fig.suptitle('F7: half-wave condition vs angle (phi=0)', fontsize=11)
    fig.tight_layout()
    fig.savefig(F / 'wf_F7_halfwave_vs_theta.png', dpi=170)
    plt.close(fig)

    # F8: angle-resolved multipole fractions
    try:
        mp = pd.read_csv(R / 'angle_multipoles.csv')
        fig, axs = plt.subplots(2, len(tags),
                                figsize=(3.6 * len(tags), 7.5),
                                sharey=True)
        for j, tag in enumerate(tags):
            for i, pol in enumerate(('p', 's')):
                sub = mp[(mp.tag == tag) & (mp.phi == 0)
                         & (mp.pol == pol)].sort_values('theta')
                if not len(sub):
                    continue
                ax = axs[i][j]
                bot = np.zeros(len(sub))
                for fam, c in (('f_ED', 'tab:blue'), ('f_MD', 'tab:green'),
                               ('f_EQ', 'tab:red'), ('f_MQ', 'tab:purple')):
                    ax.bar(sub.theta, sub[fam], bottom=bot, width=12,
                           color=c, label=fam if (i == 0 and j == 0)
                           else None)
                    bot += sub[fam].values
                ax.set_title(f'{label_of(tag, newA, newB)} {pol}-pol',
                             fontsize=8.5)
                ax.set_xlabel('theta')
        fig.legend(fontsize=8, loc='upper right')
        fig.suptitle('F8: exact multipole family fractions vs angle '
                     '(phi=0)', fontsize=11)
        fig.tight_layout()
        fig.savefig(F / 'wf_F8_multipoles_vs_theta.png', dpi=170)
        plt.close(fig)
    except FileNotFoundError:
        print('F8 skipped (no angle_multipoles.csv yet)')

    # F10: PB rotation law vs incident angle
    try:
        pb = pd.read_csv(R / 'pb_rotation_vs_angle.csv')
        fits = pb[pb.alpha == -1]
        fig, axs = plt.subplots(1, 2, figsize=(12.5, 5))
        for tag in tags:
            sub = fits[fits.tag == tag]
            if not len(sub):
                continue
            lab = label_of(tag, newA, newB)
            axs[0].plot(sub.theta, sub.phase_deg, 'o-', c=COLORS[lab],
                        label=lab)
            axs[1].plot(sub.theta, sub.R_cross, 's-', c=COLORS[lab])
        axs[0].axhline(-2, color='gray', ls='--', lw=1)
        axs[0].set_ylabel('fitted PB slope (deg/deg)')
        axs[1].set_ylabel('RMS phase residual (deg)')
        for a in axs:
            a.set_xlabel('incident theta (deg)')
            a.grid(alpha=0.25)
        axs[0].legend(fontsize=8)
        fig.suptitle('F10: geometric-phase rotation law vs incident angle',
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(F / 'wf_F10_pb_vs_theta.png', dpi=170)
        plt.close(fig)
    except FileNotFoundError:
        print('F10 skipped (no pb_rotation_vs_angle.csv yet)')

    # F11: geometry comparison
    fig, axs = plt.subplots(1, len(tags), figsize=(3.4 * len(tags), 3.8))
    for ax, tag in zip(axs, tags):
        rho, P, H, _ = load_geometry(tag)
        b = rho.numpy() > 0.5
        ax.imshow(b.T, origin='lower', cmap='gray_r',
                  extent=[-P / 2, P / 2, -P / 2, P / 2],
                  interpolation='nearest')
        ax.add_patch(Circle((0, 0), rc.r_design(P), fill=False,
                            ec='tab:red', ls='--', lw=1.0))
        ax.set_title(f'{label_of(tag, newA, newB)}\nP={P:.0f} H={H:.0f}',
                     fontsize=8.5)
        ax.set_xlabel('x (nm)')
    fig.suptitle('F11: geometries (red: rotation-safe envelope; the '
                 'paper rectangle predates the envelope rule)',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(F / 'wf_F11_geometries.png', dpi=200)
    plt.close(fig)

    # F12: angular Pareto frontier from acceptance metrics
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    for _, r in acc.iterrows():
        lab = label_of(r.tag, newA, newB)
        c = COLORS.get(lab, 'tab:gray')
        axs[0].plot(r.R_cross0, r.Rc_omega_085, 'o', c=c, ms=10)
        axs[0].annotate(lab, (r.R_cross0, r.Rc_omega_085), fontsize=7.5,
                        xytext=(4, 4), textcoords='offset points')
        axs[1].plot(r.theta_20, r.R_cross0, 's', c=c, ms=10)
        axs[1].annotate(lab, (r.theta_20, r.R_cross0), fontsize=7.5,
                        xytext=(4, 4), textcoords='offset points')
    axs[0].set_xlabel('R_cross(0)')
    axs[0].set_ylabel('solid-angle <R_cross> 0-85')
    axs[1].set_xlabel('theta_20 acceptance (deg)')
    axs[1].set_ylabel('R_cross(0)')
    for a in axs:
        a.grid(alpha=0.25)
    fig.suptitle('F12: peak-efficiency vs FOV Pareto view', fontsize=11)
    fig.tight_layout()
    fig.savefig(F / 'wf_F12_pareto.png', dpi=170)
    plt.close(fig)
    print('WF_FIGS_DONE', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    main(sys.argv[1], sys.argv[2])
