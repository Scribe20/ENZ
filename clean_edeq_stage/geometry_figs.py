"""Sections 1-3 of the AR audit: publication-quality geometry export.

Plots the EXACT hard-binary 110x110 density (no smoothing, nearest-
neighbor rendering); contour panel uses the 0.5 level of the binary
array, which traces pixel boundaries. Saves PNG (300 dpi) + PDF, the
exact array copy, and SHA256.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle
import numpy as np

import stage_core as sc

F = sc.RESULTS / 'figures'
G = sc.RESULTS / 'geometry'
P = 550.0
H_LIST = [250.0, 235.0, 227.2, 225.0]
ASI = '#7A6A54'      # a-Si
SIL = '#B9CDD8'      # silica
AIRC = '#F4F6F8'


def save(fig, stem):
    fig.savefig(F / f'{stem}.png', dpi=300)
    fig.savefig(F / f'{stem}.pdf')
    plt.close(fig)
    print('wrote', stem, flush=True)


def main():
    F.mkdir(parents=True, exist_ok=True)
    G.mkdir(parents=True, exist_ok=True)
    rho_t, _, _ = sc.load_ref()
    rho = rho_t.numpy()
    n = rho.shape[0]
    px = P / n
    # ---- section 3: verification + export
    src = sc.PILOT / 'P0550_H0250_seed011' / 'rho_binary.npy'
    np.save(G / 'p0550_rho_binary_exact.npy', rho)
    meta = {
        'source': str(src),
        'sha256_source': hashlib.sha256(src.read_bytes()).hexdigest(),
        'sha256_export': hashlib.sha256(
            (G / 'p0550_rho_binary_exact.npy').read_bytes()).hexdigest(),
        'shape': list(rho.shape), 'min': float(rho.min()),
        'max': float(rho.max()),
        'unique_values': np.unique(rho).tolist(),
        'binary_fraction': float(np.mean((rho == 0) | (rho == 1))),
        'threshold_used': 'none needed - array is exactly {0,1}; 0.5 used '
                          'only for contour tracing',
        'fill_fraction': float(rho.mean()),
        'pixel_nm': px, 'period_nm': P,
        'note': 'this exact array produced every Stage-B and AR-audit '
                'spectrum; no filtered/projected intermediate is plotted',
    }
    (G / 'p0550_geometry_verification.json').write_text(
        json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ('min', 'max', 'binary_fraction',
                                           'fill_fraction')}, indent=1))

    ext = [0, P, 0, P]

    # ---- A: top view
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.imshow(rho.T, origin='lower', cmap='gray_r', extent=ext,
              interpolation='nearest')
    ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)')
    ax.set_title('P0550_H0250_seed011 - hard-binary top view\n'
                 'dark = a-Si (fill 0.618), white = air; P = 550 nm',
                 fontsize=10)
    ax.annotate('', xy=(P, -28), xytext=(0, -28),
                arrowprops=dict(arrowstyle='<->', lw=1),
                annotation_clip=False)
    ax.text(P / 2, -55, 'P = 550 nm', ha='center', fontsize=9, clip_on=False)
    fig.tight_layout()
    save(fig, 'p0550_geometry_topview')

    # ---- B: contour rendering + scale bar
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.imshow(rho.T, origin='lower', cmap='gray_r', extent=ext, alpha=0.12,
              interpolation='nearest')
    xg = (np.arange(n) + 0.5) * px
    ax.contour(xg, xg, rho.T, levels=[0.5], colors='k', linewidths=1.4)
    ax.contourf(xg, xg, rho.T, levels=[0.5, 1.5], colors=[ASI], alpha=0.55)
    ax.add_patch(Rectangle((30, 30), 100, 12, fc='k'))
    ax.text(80, 55, '100 nm', ha='center', fontsize=9)
    ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)')
    ax.set_title('material boundary (0.5 contour of the binary array)',
                 fontsize=10)
    fig.tight_layout()
    save(fig, 'p0550_geometry_contour')

    # ---- C: 3D perspective (voxel-exact extrusion)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    h = 250.0
    ds = 2                        # 55x55 voxel columns for tractable render
    rb = rho[::ds, ::ds] > 0.5
    m = rb.shape[0]
    fig = plt.figure(figsize=(7.6, 6.2))
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
    zc = nz_sub + nz_si
    ax.set_box_aspect((1, 1, 0.55))
    ax.set_xticks([0, m]); ax.set_xticklabels(['0', '550'])
    ax.set_yticks([0, m]); ax.set_yticklabels(['0', '550'])
    ax.set_zticks([0, nz_sub, zc])
    ax.set_zticklabels(['', 'silica top', f'z = {h:.0f} nm'])
    ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)')
    ax.quiver(m * 0.5, -m * 0.28, -1.2, 0, 0, 2.6, color='crimson',
              arrow_length_ratio=0.25, lw=2)
    ax.text(m * 0.5, -m * 0.42, 0.4, 'k (from silica)', color='crimson',
            fontsize=9)
    ax.quiver(-m * 0.18, m * 0.5, zc + 0.8, 2.2, 0, 0, color='navy',
              arrow_length_ratio=0.3, lw=2)
    ax.text(-m * 0.20, m * 0.5, zc + 1.9, 'E (x-pol)', color='navy',
            fontsize=9)
    ax.set_title('freeform a-Si meta-atom on silica (unit cell, h = 250 nm; '
                 'x2 voxel downsample for rendering only)', fontsize=9)
    save(fig, 'p0550_geometry_3d')

    # ---- D: x-z and y-z cross sections (+ thickness comparison)
    def cross(ax, line, axis, h):
        # air band
        ax.add_patch(Rectangle((0, h), P, 90, fc=AIRC, ec='none'))
        # substrate
        ax.add_patch(Rectangle((0, -110), P, 110, fc=SIL, ec='none'))
        mask = line > 0.5
        start = None
        for i in range(n + 1):
            on = i < n and mask[i]
            if on and start is None:
                start = i
            if not on and start is not None:
                ax.add_patch(Rectangle((start * px, 0), (i - start) * px, h,
                                       fc=ASI, ec='k', lw=0.4))
                start = None
        ax.set_xlim(0, P); ax.set_ylim(-110, h + 90)
        ax.axhline(0, color='k', lw=0.5); ax.axhline(h, color='k', lw=0.5)
        ax.annotate('', xy=(P * 0.06, h), xytext=(P * 0.06, 0),
                    arrowprops=dict(arrowstyle='<->', lw=0.9))
        ax.text(P * 0.09, h / 2, f'h = {h:g} nm', fontsize=8, va='center')
        ax.add_patch(FancyArrow(P * 0.88, -95, 0, 55, width=4,
                                head_width=16, head_length=18, fc='crimson',
                                ec='crimson'))
        ax.text(P * 0.88, -105, 'k', color='crimson', fontsize=9,
                ha='center')
        ax.text(P * 0.02, -100, 'silica (n=1.46)', fontsize=8)
        ax.text(P * 0.02, h + 60, 'air', fontsize=8)
        ax.set_xlabel(f'{axis} (nm)'); ax.set_ylabel('z (nm)')

    mid = n // 2
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    cross(ax, rho[:, mid], 'x', 250.0)
    ax.set_title('x-z cross section at y = P/2 (a-Si columns where the '
                 'binary map = 1)', fontsize=9)
    fig.tight_layout(); save(fig, 'p0550_geometry_xz')

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    cross(ax, rho[mid, :], 'y', 250.0)
    ax.set_title('y-z cross section at x = P/2', fontsize=9)
    fig.tight_layout(); save(fig, 'p0550_geometry_yz')

    fig, axs = plt.subplots(len(H_LIST), 1, figsize=(6.4, 10.6), sharex=True)
    for ax, h in zip(axs, H_LIST):
        cross(ax, rho[:, mid], 'x', h)
        ax.set_title(f'h = {h:g} nm', fontsize=9)
    fig.suptitle('thickness family (identical lateral topology)', fontsize=10)
    fig.tight_layout(); save(fig, 'p0550_geometry_thickness_family')

    # ---- E: current overlay at lam0 (from saved dense fields, h = 250)
    gm = np.load(sc.RESULTS / 'audit' / 'gradient_maps.npz')
    Ex = gm['Ex']
    nf = Ex.shape[0]
    idx = (np.floor(np.arange(nf) / nf * n)).astype(int) % n
    rho_f = rho[np.ix_(idx, idx)]
    J = np.abs(Ex[:, :, Ex.shape[2] // 2]) * (rho_f > 0.5)
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    im = ax.imshow(J.T, origin='lower', cmap='magma', extent=ext)
    ax.contour(xg, xg, rho.T, levels=[0.5], colors='cyan', linewidths=0.9)
    plt.colorbar(im, label='|J_x| proxy (|chi||E_x|, mid-slab)')
    ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)')
    ax.set_title('induced current at 1332.5 nm on the geometry outline '
                 '(h = 250)', fontsize=9)
    fig.tight_layout(); save(fig, 'p0550_geometry_current_overlay')
    print('GEOMETRY_FIGS_DONE')


if __name__ == '__main__':
    main()
