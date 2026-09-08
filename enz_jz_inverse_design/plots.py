"""Figure helpers for the JZ F_z campaign (matplotlib, Agg)."""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


def history_plot(result_json, out_png, title=None):
    r = json.load(open(result_json)) if not isinstance(result_json, dict) else result_json
    h = r["history"]
    it = np.arange(len(h["Fz"]))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(it, h["Fz"], label="F_z (objective)")
    ax[0].plot(it, h["Fx"], label="F_x"); ax[0].plot(it, h["Fy"], label="F_y")
    ax[0].plot(it, h["Ftot"], "k--", label="F_tot = F_x+F_y+F_z")
    ax[0].plot(it, h["A"], "r:", label="A = 1-R-T")
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("absorbed fraction"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].semilogy(it, np.abs(h["identity_resid"]), label="|F_tot - A|")
    ax[1].semilogy(it, h["grad_norm"], label="|grad|")
    ax[1].set_xlabel("iteration"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[2].plot(it, h["binarization"], label="binarization (0 = binary)")
    ax[2].plot(it, h["s_flip"], label="S_flip (lr)"); ax[2].plot(it, h["s_flipud"], label="S_flip (ud)")
    ax2 = ax[2].twinx(); ax2.semilogy(it, h["beta_proj"], "g--", label="beta_proj"); ax2.set_ylabel("beta_proj")
    ax[2].set_xlabel("iteration"); ax[2].legend(fontsize=8, loc="upper left"); ax[2].grid(alpha=.3)
    fig.suptitle(title or f"{r['tag']}: P={r['P']:.0f} h={r['h']:.0f} pad={r['pad_frac']:.2f} seed={r['seed']} order {r['order']}")
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


def geometry_plot(rho_files, out_png, titles=None, P=None):
    n = len(rho_files)
    fig, axs = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for i, f in enumerate(rho_files):
        a = np.load(f) if isinstance(f, (str, Path)) else np.asarray(f)
        ext = [0, P, 0, P] if P else None
        axs[0, i].imshow(a.T, origin="lower", cmap="gray_r", vmin=0, vmax=1, extent=ext)
        axs[0, i].set_title(titles[i] if titles else str(f), fontsize=9)
        axs[0, i].set_xlabel("x (nm)" if P else "x px"); axs[0, i].set_ylabel("y (nm)" if P else "y px")
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


def loss_maps_plot(maps, rho_hard, P, h, out_png, title=""):
    """F_x, F_y, F_z z-integrated loss-density maps in the ITO + |Ez|^2 mid-plane + xz cut."""
    dens = maps["loss_density_xyz"]                   # (3, nx, nx), units 1/nm^2 (fraction per area)
    E = maps["E_ito"]
    nz = E.shape[0]
    Ez2_mid = np.abs(E[nz // 2, 2]) ** 2
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    ext = [0, P, 0, P]
    for i, (lab, ax) in enumerate(zip(("F_x", "F_y", "F_z"), axs[0])):
        im = ax.imshow((dens[i] * P ** 2).T, origin="lower", extent=ext, cmap="magma")
        ax.contour(np.linspace(0, P, rho_hard.shape[0]), np.linspace(0, P, rho_hard.shape[1]), np.asarray(rho_hard).T, levels=[0.5], colors="cyan", linewidths=0.6)
        ax.set_title(f"{lab} density x P^2 (integral = {lab} = {dens[i].sum() * (P/dens.shape[1])**2:.4f})", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    im = axs[1, 0].imshow(Ez2_mid.T, origin="lower", extent=ext, cmap="viridis")
    axs[1, 0].contour(np.linspace(0, P, rho_hard.shape[0]), np.linspace(0, P, rho_hard.shape[1]), np.asarray(rho_hard).T, levels=[0.5], colors="w", linewidths=0.6)
    axs[1, 0].set_title(f"|Ez/E_inc|^2 at ITO mid-plane (max {Ez2_mid.max():.1f})", fontsize=9); fig.colorbar(im, ax=axs[1, 0], shrink=0.8)
    Exz = maps["E_xz"]; z = maps["z_xz"]; x = maps["x"]
    im = axs[1, 1].imshow((np.abs(Exz[2]) ** 2).T, origin="lower", extent=[x[0], x[-1], z[0], z[-1]], aspect="auto", cmap="viridis")
    for zz in (0, h, h + config.D_ITO_NM):
        axs[1, 1].axhline(zz, color="w", lw=0.5)
    axs[1, 1].set_title("|Ez|^2, xz cut at y = P/2 (air / a-Si / ITO / glass)", fontsize=9); axs[1, 1].set_xlabel("x (nm)"); axs[1, 1].set_ylabel("z (nm, 0 = a-Si top)")
    fig.colorbar(im, ax=axs[1, 1], shrink=0.8)
    Etot = np.sqrt(sum(np.abs(Exz[i]) ** 2 for i in range(3)))
    im = axs[1, 2].imshow(Etot.T, origin="lower", extent=[x[0], x[-1], z[0], z[-1]], aspect="auto", cmap="inferno")
    for zz in (0, h, h + config.D_ITO_NM):
        axs[1, 2].axhline(zz, color="w", lw=0.5)
    axs[1, 2].set_title("|E|/|E_inc|, xz cut", fontsize=9); axs[1, 2].set_xlabel("x (nm)")
    fig.colorbar(im, ax=axs[1, 2], shrink=0.8)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


def spectra_plot(specs, out_png, lam_ze, title="", keys=("Fz", "Fx", "Fy", "A", "R", "T")):
    """specs: dict label -> spec_arrays dict."""
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    for lab, sp in specs.items():
        if not np.all(np.isnan(sp["Fz"])):
            axs[0].plot(sp["lam"], sp["Fz"], label=f"F_z {lab}")
        axs[1].plot(sp["lam"], sp["A"], label=f"A {lab}")
        axs[2].plot(sp["lam"], sp["R"], "--", label=f"R {lab}"); axs[2].plot(sp["lam"], sp["T"], ":", label=f"T {lab}")
    for ax, yl in zip(axs, ("F_z", "A = 1-R-T", "R, T")):
        ax.axvline(lam_ze, color="k", lw=0.6, ls="--"); ax.set_xlabel("wavelength (nm)"); ax.set_ylabel(yl); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


def landscape_plot(rows, out_png, value="Fz_hard", title="Stage 1 landscape (best of seeds)"):
    """rows: list of dicts with P, h, pad_frac, seed, value.  One heatmap per pad."""
    pads = sorted(set(r["pad_frac"] for r in rows))
    Ps = sorted(set(r["P"] for r in rows)); hs = sorted(set(r["h"] for r in rows))
    fig, axs = plt.subplots(1, len(pads), figsize=(5.2 * len(pads), 4.5), squeeze=False)
    vmax = max(r[value] for r in rows)
    for ax, pad in zip(axs[0], pads):
        Z = np.full((len(hs), len(Ps)), np.nan)
        for i, h in enumerate(hs):
            for j, P in enumerate(Ps):
                v = [r[value] for r in rows if r["pad_frac"] == pad and r["P"] == P and r["h"] == h]
                if v:
                    Z[i, j] = max(v)
        im = ax.imshow(Z, origin="lower", aspect="auto", vmin=0, vmax=vmax, cmap="viridis")
        ax.set_xticks(range(len(Ps))); ax.set_xticklabels([f"{p:.0f}" for p in Ps]); ax.set_yticks(range(len(hs))); ax.set_yticklabels([f"{h:.0f}" for h in hs])
        ax.set_xlabel("P (nm)"); ax.set_ylabel("h (nm)"); ax.set_title(f"pad = {pad:.2f} P")
        for i in range(len(hs)):
            for j in range(len(Ps)):
                if not np.isnan(Z[i, j]):
                    ax.text(j, i, f"{Z[i,j]:.3f}", ha="center", va="center", fontsize=7, color="w" if Z[i, j] < 0.6 * vmax else "k")
        fig.colorbar(im, ax=ax, shrink=0.8, label=value)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)
