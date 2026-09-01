"""Section 30: geometry output for top finalists (both methods).

usage: python rt_geometry_figs.py <tag> [<tag> ...]
Per tag: top view, contour, 3D, x-z / y-z cross sections, current
overlay at 633 nm - PNG (300 dpi) + PDF, fully labeled.
"""
import json
import math
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Rectangle
import numpy as np
import torch

import rt_core as rc
from rt_qualify import load_tag

F = rc.HERE / 'figures'
ASI, SIL, AIRC = '#7A6A54', '#B9CDD8', '#F4F6F8'


def suite(tag):
    rec, rho = load_tag(tag)
    P, H = rec['P'], rec['H']
    b = rho.numpy() > 0.5
    n = b.shape[0]
    px = P / n
    ext = [-P / 2, P / 2, -P / 2, P / 2]
    label = (f"{rec['method']} | P={P:.0f} H={H:.0f} | pad="
             f"{rc.padding(P):.1f} r_des={rc.r_design(P):.1f} | fill="
             f"{rec['fill']:.2f} | Rc={rec['R_cross']:.3f} "
             f"err={rec['pb_phase_err_deg']:.0f}deg")

    fig, axs = plt.subplots(1, 2, figsize=(11, 5.4))
    axs[0].imshow(b.T, origin='lower', cmap='gray_r', extent=ext,
                  interpolation='nearest')
    axs[0].add_patch(Circle((0, 0), rc.r_design(P), fill=False,
                            ec='tab:red', ls='--', lw=1.2))
    axs[0].set_title('hard-binary top view + rotation-safe envelope',
                     fontsize=9)
    xg = (np.arange(n) + 0.5) * px - P / 2
    axs[1].contourf(xg, xg, b.T.astype(float), levels=[0.5, 1.5],
                    colors=[ASI], alpha=0.6)
    axs[1].contour(xg, xg, b.T.astype(float), levels=[0.5], colors='k',
                   linewidths=1.3)
    axs[1].add_patch(Circle((0, 0), rc.r_design(P), fill=False,
                            ec='tab:red', ls='--', lw=1.0))
    axs[1].add_patch(Rectangle((-P / 2 + 12, -P / 2 + 12), 50, 7, fc='k'))
    axs[1].text(-P / 2 + 37, -P / 2 + 26, '50 nm', ha='center', fontsize=8)
    axs[1].set_title('material contour', fontsize=9)
    axs[1].set_aspect('equal')
    for a in axs:
        a.set_xlabel('x (nm)')
        a.set_ylabel('y (nm)')
    fig.suptitle(label, fontsize=9)
    fig.tight_layout()
    fig.savefig(F / f'geom_{tag}_topview.png', dpi=300)
    fig.savefig(F / f'geom_{tag}_topview.pdf')
    plt.close(fig)

    # 3D
    ds = max(1, n // 48)
    rb = b[::ds, ::ds]
    m = rb.shape[0]
    fig = plt.figure(figsize=(7.4, 6))
    ax = fig.add_subplot(111, projection='3d')
    nz_sub, nz_si = 2, 3
    vox = np.zeros((m, m, nz_sub + nz_si), dtype=bool)
    col = np.empty(vox.shape, dtype=object)
    vox[:, :, :nz_sub] = True
    col[:, :, :nz_sub] = SIL
    for z in range(nz_sub, nz_sub + nz_si):
        vox[:, :, z] = rb
        col[:, :, z] = ASI
    ax.voxels(vox, facecolors=col, edgecolor=None, shade=True)
    ax.set_box_aspect((1, 1, 0.55))
    ax.set_xticks([0, m]); ax.set_xticklabels(['0', f'{P:.0f}'])
    ax.set_yticks([0, m]); ax.set_yticklabels(['0', f'{P:.0f}'])
    ax.set_zticks([])
    ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)')
    ax.quiver(m * 0.5, -m * 0.28, nz_sub + nz_si + 1.4, 0, 0, -2.4,
              color='crimson', arrow_length_ratio=0.25, lw=2)
    ax.text(m * 0.5, -m * 0.42, nz_sub + nz_si + 1.6, 'k (from air)',
            color='crimson', fontsize=9)
    ax.text(-m * 0.2, m * 0.5, nz_sub + nz_si + 1.6, 'E_x', color='navy',
            fontsize=9)
    ax.set_title(label, fontsize=8)
    fig.savefig(F / f'geom_{tag}_3d.png', dpi=300)
    fig.savefig(F / f'geom_{tag}_3d.pdf')
    plt.close(fig)

    # cross sections
    def cross(ax, line, axis):
        ax.add_patch(Rectangle((-P / 2, H), P, 70, fc=AIRC, ec='none'))
        ax.add_patch(Rectangle((-P / 2, -80), P, 80, fc=SIL, ec='none'))
        start = None
        mask = line > 0.5
        for i in range(n + 1):
            on = i < n and mask[i]
            if on and start is None:
                start = i
            if not on and start is not None:
                ax.add_patch(Rectangle((start * px - P / 2, 0),
                                       (i - start) * px, H, fc=ASI,
                                       ec='k', lw=0.4))
                start = None
        ax.set_xlim(-P / 2, P / 2)
        ax.set_ylim(-80, H + 70)
        ax.add_patch(FancyArrow(P * 0.38, H + 62, 0, -40, width=3,
                                head_width=12, head_length=14,
                                fc='crimson', ec='crimson'))
        ax.text(P * 0.30, H + 40, 'k', color='crimson', fontsize=9)
        ax.text(-P / 2 + 6, -70, 'glass n=1.457', fontsize=8)
        ax.text(-P / 2 + 6, H + 45, 'air', fontsize=8)
        ax.annotate('', xy=(-P / 2 + 14, H), xytext=(-P / 2 + 14, 0),
                    arrowprops=dict(arrowstyle='<->', lw=0.9))
        ax.text(-P / 2 + 20, H / 2, f'H={H:.0f}', fontsize=8)
        ax.set_xlabel(f'{axis} (nm)')
        ax.set_ylabel('z (nm)')

    mid = n // 2
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    cross(axs[0], b[:, mid], 'x')
    axs[0].set_title('x-z at y=0', fontsize=9)
    cross(axs[1], b[mid, :], 'y')
    axs[1].set_title('y-z at x=0', fontsize=9)
    fig.suptitle(label, fontsize=9)
    fig.tight_layout()
    fig.savefig(F / f'geom_{tag}_xz_yz.png', dpi=300)
    fig.savefig(F / f'geom_{tag}_xz_yz.pdf')
    plt.close(fig)

    # current overlay (|chi E| mid-slab, x-pol device illumination)
    import torcwa
    e = rc.eps_asi()
    sim = torcwa.rcwa(freq=1.0 / rc.LAM0, order=[9, 9], L=[P, P],
                      dtype=rc.SIM_DTYPE, device=rc.DEVICE)
    sim.add_input_layer(eps=rc.EPS_GLASS)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    sim.add_layer(thickness=H, eps=rho * (e - 1.0) + 1.0)
    sim.solve_global_smatrix()
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    for a, amp, pol in ((axs[0], [1.0, 0.0], 'x'), (axs[1], [0.0, 1.0],
                                                    'y')):
        sim.source_planewave(amplitude=amp, direction='backward')
        x_ax = torch.linspace(0.0, P, 96)
        with torch.no_grad():
            [Ex, Ey, Ez], _ = sim.field_xy(0, x_ax, x_ax, z_prop=H / 2)
        Em = (torch.abs(Ex) ** 2 + torch.abs(Ey) ** 2
              + torch.abs(Ez) ** 2).sqrt().numpy()
        nb = np.kron(b, np.ones((1, 1)))
        idx = (np.floor(np.arange(96) / 96 * n)).astype(int)
        maskf = b[np.ix_(idx, idx)]
        im = a.imshow((Em * maskf).T, origin='lower', cmap='magma',
                      extent=ext)
        a.contour(xg, xg, b.T.astype(float), levels=[0.5], colors='cyan',
                  linewidths=0.8)
        plt.colorbar(im, ax=a, shrink=0.8)
        a.set_title(f'|E| in a-Si, mid-slab, {pol}-pol', fontsize=9)
        a.set_xlabel('x (nm)')
    fig.suptitle(label, fontsize=9)
    fig.tight_layout()
    fig.savefig(F / f'geom_{tag}_current.png', dpi=300)
    plt.close(fig)
    print(f'geometry suite: {tag}', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    for tag in sys.argv[1:]:
        suite(tag)
    print('RT_GEOM_DONE')
