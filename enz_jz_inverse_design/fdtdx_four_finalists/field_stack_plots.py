"""Plot-only post-processing of the existing FDTDX raw field data of the four frozen finalists.

No simulation is run: everything is read from the stored raw outputs of the campaign
(`<design>/prod/fields_normalized.npz`, `<design>/prod/Ez2_slices_aSi_<lam>nm.npz`,
`<design>/rho_hard_binary.npy`, `<design>/prod/run_meta.json`).

    <python> field_stack_plots.py            # all four designs, lambda_ZE, tag 'prod'

Outputs go to `field_plots/` (see field_plots/README.md).

Conventions (taken from the stored data, not assumed):
  * z is the FDTDX domain coordinate: z = 0 is the bottom edge of the simulation domain, which sits
    inside the glass (the lowest 16 cells are the glass-side PML).  +z runs glass -> ITO -> a-Si:H -> air.
    The plane wave is launched above the a-Si:H and propagates in -z, i.e. it enters from the air,
    crosses a-Si:H, then ITO, then leaves into the glass.
  * `E_over_Einc[l, c, ix, iy, iz]`, c = (Ex, Ey, Ez), complex amplitude in the exp(-i w t) convention,
    normalized by the incident Ex phasor (reference run).  Squared magnitudes are |E_c/E_inc|^2.
  * D_z is formed from the stored field as  D_z / (eps0 E_inc) = eps_zz(x, y, z) * (E_z / E_inc),
    with eps_zz the same fitted permittivity model the simulation used (materials/material_models.json).
    Inside the a-Si:H layer the pattern interfaces are vertical, so E_z is the interface-parallel
    component and the cell-averaged eps_zz is the arithmetic mean  f*eps_aSi + (1-f)*1  with f the
    exact area fill fraction of the cell (same `area_fraction` rule the simulation used).
  * Normalization is identical for all four designs (same source, same reference run), so all curves
    and all colour scales are directly comparable.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
OUT = HERE / "field_plots"
MODELS = json.load(open(HERE / "materials" / "material_models.json"))
PROV = json.load(open(HERE / "config" / "geometry_provenance.json"))

C0 = 299792458.0
DESIGNS = ["final3", "final1", "final0", "final2"]
LAM_ZE = MODELS["lambda_ZE_data_nm"]
TAG = "prod"

# region colours / labels, in +z order
REGION_STYLE = {
    "glass": dict(color="#7fb3d5", label="glass (SiO$_2$)"),
    "ITO":   dict(color="#e8a33d", label="ITO"),
    "aSi":   dict(color="#9b7fc9", label="a-Si:H"),
    "air":   dict(color="#d8d8d8", label="air"),
}


# --------------------------------------------------------------------------- material models
def eps_model(name, lam_m):
    """Complex permittivity of the fitted model at wavelength(s) lam_m [m], exp(-i w t).

    Verbatim from fx_sim.eps_model (copied so that this plotting script does not need fdtdx/jax)."""
    m = MODELS[name]
    w = 2 * np.pi * C0 / np.asarray(lam_m)
    if name == "ITO":
        return (m["eps_inf"]
                - m["wp_rad_s"] ** 2 / (w ** 2 + 1j * m["gamma_d_rad_s"] * w)
                + m["delta_eps_L"] * m["w0_L_rad_s"] ** 2
                / (m["w0_L_rad_s"] ** 2 - w ** 2 - 1j * m["gamma_L_rad_s"] * w))
    return (m["eps_inf"] + m["delta_eps"] * m["w0_rad_s"] ** 2
            / (m["w0_rad_s"] ** 2 - w ** 2 - 1j * max(m["gamma_rad_s"], 1e9) * w))


def area_fraction(rho, n):
    """Exact area fraction of the 128x128 binary design inside each of n x n equal cells of the same
    period.  Verbatim from fx_sim.area_fraction."""
    m = rho.shape[0]
    e_pix = np.arange(m + 1) / m
    e_cell = np.arange(n + 1) / n
    W = np.zeros((n, m))
    for c in range(n):
        lo = np.maximum(e_cell[c], e_pix[:-1])
        hi = np.minimum(e_cell[c + 1], e_pix[1:])
        W[c] = np.clip(hi - lo, 0.0, None) * n
    return W @ rho.astype(float) @ W.T


# --------------------------------------------------------------------------- loading
def load_design(design, tag=TAG):
    """Read the raw stored FDTDX volume block and build everything derived from it."""
    ddir = HERE / design / tag
    d = np.load(ddir / "fields_normalized.npz")
    meta = json.load(open(ddir / "run_meta.json"))
    prov = PROV[design]

    lam_nm = d["lam_nm"]
    E = d["E_over_Einc"]                                   # (Nlam, 3, nx, ny, nz) complex
    z_nm = d["z_m"] * 1e9
    dz_nm = d["dz_m"] * 1e9
    x_nm = d["x_m"] * 1e9
    y_nm = d["y_m"] * 1e9
    a0, a1 = [int(v) for v in d["z_asi_range_idx"]]
    i0, i1 = [int(v) for v in d["z_ito_range_idx"]]
    imid = int(d["z_ito_mid_idx"])
    nz = len(z_nm)

    # per-cell region label of the stored volume block, in +z order
    region = np.array(["glass"] * nz, dtype=object)
    region[i0:i1] = "ITO"
    region[a0:a1] = "aSi"
    region[a1:] = "air"

    # cell edges from the stored centres and widths (they tile the block contiguously)
    edges = np.empty(nz + 1)
    edges[:-1] = z_nm - dz_nm / 2
    edges[-1] = z_nm[-1] + dz_nm[-1] / 2

    # eps_zz(x, y, z) at every recorded wavelength, on the 64 x 64 x nz volume grid
    rho = np.load(HERE / design / "rho_hard_binary.npy")
    nxy = E.shape[2]
    fill = area_fraction(rho, nxy)                          # (nx, ny), exact area fraction of a-Si:H
    lam_m = lam_nm * 1e-9
    eps_ito = eps_model("ITO", lam_m)
    eps_asi = eps_model("aSiH", lam_m)
    eps_gl = eps_model("glass", lam_m)
    eps_zz = np.ones((len(lam_nm), nxy, nxy, nz), dtype=np.complex128)
    for il in range(len(lam_nm)):
        eps_zz[il, :, :, :i0] = eps_gl[il]
        eps_zz[il, :, :, i0:i1] = eps_ito[il]
        # a-Si:H layer: E_z is parallel to the (vertical) pattern interfaces -> arithmetic cell average
        eps_zz[il, :, :, a0:a1] = (fill * eps_asi[il] + (1.0 - fill) * 1.0)[:, :, None]
        eps_zz[il, :, :, a1:] = 1.0

    return dict(design=design, tag=tag, dir=ddir, meta=meta, prov=prov, npz=d,
                lam_nm=lam_nm, E=E, z_nm=z_nm, dz_nm=dz_nm, edges=edges, x_nm=x_nm, y_nm=y_nm,
                a0=a0, a1=a1, i0=i0, i1=i1, imid=imid, nz=nz, region=region,
                eps_zz=eps_zz, fill=fill, rho=rho,
                P_nm=prov["P_nm"], h_nm=prov["h_nm"], pad=prov["pad_frac"])


def profiles(R, il):
    """xy-averaged |E|^2, |Ez|^2 and |Dz|^2 vs z at wavelength index il, from the raw complex volume."""
    E = R["E"][il].astype(np.complex128)                    # (3, nx, ny, nz); float64 so that the
    I2 = np.abs(E) ** 2                                     # xy means do not accumulate in float32
    Dz = R["eps_zz"][il] * E[2]                             # D_z / (eps0 E_inc)
    return dict(E2=I2.sum(0).mean((0, 1)),
                Ez2=I2[2].mean((0, 1)),
                Ex2=I2[0].mean((0, 1)),
                Ey2=I2[1].mean((0, 1)),
                Dz2=(np.abs(Dz) ** 2).mean((0, 1)))


def region_spans(R):
    """[(name, z_lo, z_hi), ...] of the contiguous material regions of the stored volume block."""
    out, k = [], 0
    while k < R["nz"]:
        j = k
        while j + 1 < R["nz"] and R["region"][j + 1] == R["region"][k]:
            j += 1
        out.append((R["region"][k], R["edges"][k], R["edges"][j + 1]))
        k = j + 1
    return out


def shade_regions(ax, R, horizontal=True):
    for name, lo, hi in region_spans(R):
        st = REGION_STYLE[name]
        if horizontal:
            ax.axhspan(lo, hi, color=st["color"], alpha=0.20, lw=0, zorder=0)
            ax.axhline(lo, color="0.35", lw=0.6, ls="--", zorder=1)
        else:
            ax.axvspan(lo, hi, color=st["color"], alpha=0.20, lw=0, zorder=0)
            ax.axvline(lo, color="0.35", lw=0.6, ls="--", zorder=1)


def title_of(R):
    return (f"{R['design']}  (P = {R['P_nm']:.0f} nm, h = {R['h_nm']:.0f} nm, "
            f"pad = {100 * R['pad']:.0f} %, fill = {R['prov']['fill_fraction']:.3f})")


# --------------------------------------------------------------------------- A: z profiles
def fig_zprofile(R, il, outdir):
    lam = R["lam_nm"][il]
    p = profiles(R, il)
    z = R["z_nm"]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 8.2), sharey=True)
    panels = [(p["E2"], r"$\langle |E/E_{\rm inc}|^2\rangle_{xy}$", "#1f4e79"),
              (p["Ez2"], r"$\langle |E_z/E_{\rm inc}|^2\rangle_{xy}$", "#a13d2d"),
              (p["Dz2"], r"$\langle |D_z/(\varepsilon_0 E_{\rm inc})|^2\rangle_{xy}$", "#2d6a4f")]
    for ax, (y, lab, col) in zip(axes, panels):
        shade_regions(ax, R)
        ax.plot(y, z, color=col, lw=1.8, marker="o", ms=2.6, zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel(lab, fontsize=11)
        ax.grid(alpha=0.25, which="both", lw=0.5)
        ax.axhline(R["z_nm"][R["imid"]], color="#e8a33d", lw=1.0, ls=":", zorder=2)
    axes[0].set_ylabel("z  [nm]   (FDTDX domain coordinate, +z: glass $\\to$ ITO $\\to$ a-Si:H $\\to$ air)",
                       fontsize=10)
    axes[0].set_ylim(R["edges"][0], R["edges"][-1])

    # region names written inside the bands of the last panel
    for name, lo, hi in region_spans(R):
        axes[2].text(0.97, 0.5 * (lo + hi), REGION_STYLE[name]["label"], transform=
                     axes[2].get_yaxis_transform(), ha="right", va="center", fontsize=9,
                     color="0.15", zorder=4,
                     bbox=dict(fc="white", ec="none", alpha=0.65, pad=1.5))

    # secondary axis: z measured from the a-Si:H / ITO interface (bottom of the a-Si:H layer)
    z_ref = R["edges"][R["a0"]]
    sec = axes[2].secondary_yaxis("right", functions=(lambda v: v - z_ref, lambda v: v + z_ref))
    sec.set_ylabel("z $-$ z(a-Si:H bottom)  [nm]", fontsize=9)

    fig.suptitle(f"{title_of(R)}\nFDTDX raw volume fields, tag '{R['tag']}' (with ITO), "
                 f"$\\lambda$ = {lam:.3f} nm" + ("  ($\\lambda_{ZE}$)" if abs(lam - LAM_ZE) < 1e-2 else ""),
                 fontsize=12)
    fig.text(0.5, 0.012,
             "xy averages over the full 64x64 unit cell of the stored complex phasors; "
             "illumination enters from the air and propagates in $-z$.  Dotted line: ITO mid-plane.",
             ha="center", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=[0, 0.028, 1, 0.93])
    f = outdir / f"zprofile_{R['design']}_{lam:.1f}nm.png"
    fig.savefig(f, dpi=190)
    plt.close(fig)
    return f


def csv_zprofile(R, il, outdir):
    lam = R["lam_nm"][il]
    p = profiles(R, il)
    f = outdir / f"zprofile_{R['design']}_{lam:.1f}nm.csv"
    with open(f, "w") as fh:
        fh.write("# xy-averaged squared field magnitudes vs z from "
                 f"{R['design']}/{R['tag']}/fields_normalized.npz, lambda = {lam:.4f} nm\n")
        fh.write("# z_nm is the FDTDX domain coordinate (z=0 at the domain bottom, inside the glass PML); "
                 "+z: glass -> ITO -> a-Si:H -> air\n")
        fh.write("# E normalized to the incident Ex phasor; Dz in units of eps0*E_inc\n")
        fh.write("z_nm,dz_nm,region,E2_xyavg,Ex2_xyavg,Ey2_xyavg,Ez2_xyavg,Dz2_xyavg\n")
        for k in range(R["nz"]):
            fh.write(f"{R['z_nm'][k]:.4f},{R['dz_nm'][k]:.4f},{R['region'][k]},"
                     f"{p['E2'][k]:.8e},{p['Ex2'][k]:.8e},{p['Ey2'][k]:.8e},"
                     f"{p['Ez2'][k]:.8e},{p['Dz2'][k]:.8e}\n")
    return f


def fig_overlay(Rs, il_of, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 7.6))
    cols = ["#1f4e79", "#a13d2d", "#2d6a4f", "#8250b8"]
    keys = [("E2", r"$\langle |E/E_{\rm inc}|^2\rangle_{xy}$"),
            ("Ez2", r"$\langle |E_z/E_{\rm inc}|^2\rangle_{xy}$"),
            ("Dz2", r"$\langle |D_z/(\varepsilon_0 E_{\rm inc})|^2\rangle_{xy}$")]
    for ax, (k, lab) in zip(axes, keys):
        for R, c in zip(Rs, cols):
            p = profiles(R, il_of[R["design"]])
            # common origin for the comparison: z measured from the ITO / a-Si:H interface
            ax.plot(p[k], R["z_nm"] - R["edges"][R["a0"]], color=c, lw=1.5,
                    label=f"{R['design']} (h={R['h_nm']:.0f} nm, pad {100*R['pad']:.0f}%)")
        ax.set_xscale("log")
        ax.set_xlabel(lab, fontsize=11)
        ax.grid(alpha=0.25, which="both", lw=0.5)
        ax.axhline(0.0, color="0.3", lw=0.8, ls="--")
    # the glass / ITO bands are common to all four runs (identical below the a-Si:H); the a-Si:H top
    # differs per design, so only the sub-a-Si:H part of the stack is shaded here
    R0 = Rs[0]
    zr = R0["edges"][R0["a0"]]
    for ax in axes:
        for name, lo, hi in region_spans(R0):
            if name in ("glass", "ITO"):
                ax.axhspan(lo - zr, hi - zr, color=REGION_STYLE[name]["color"], alpha=0.20, lw=0, zorder=0)
        ax.text(0.985, -0.5 * (R0["edges"][R0["a0"]] - R0["edges"][R0["i0"]]), "ITO",
                transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8, color="0.15")
    axes[0].set_ylabel("z $-$ z(a-Si:H bottom)  [nm]   (0 = ITO / a-Si:H interface)", fontsize=10)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Four finalists, FDTDX raw volume fields (with ITO), "
                 f"$\\lambda_{{ZE}}$ = {LAM_ZE:.3f} nm\n"
                 "z origin shifted to the ITO / a-Si:H interface so the stacks are aligned "
                 "(a-Si:H heights differ)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    f = outdir / "overlay_zprofiles_all_candidates.png"
    fig.savefig(f, dpi=190)
    plt.close(fig)
    return f


# --------------------------------------------------------------------------- B: Ez xy maps
def _map_panel(ax, img, P, norm, cmap="inferno"):
    im = ax.imshow(img.T, origin="lower", extent=[0, P, 0, P], norm=norm, cmap=cmap,
                   interpolation="nearest", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    return im


def fig_slices_fullstack(R, il, outdir, n_asi=12, n_air=2):
    """|Ez/E_inc|^2 xy maps at a representative set of z planes spanning the whole stored stack."""
    lam = R["lam_nm"][il]
    Ez2 = np.abs(R["E"][il, 2]) ** 2                        # (nx, ny, nz)
    a0, a1, i0, i1 = R["a0"], R["a1"], R["i0"], R["i1"]

    idx = list(range(0, i0))                                # every stored glass cell
    idx += list(range(i0, i1))                              # every ITO cell
    idx += list(np.unique(np.linspace(a0, a1 - 1, n_asi).round().astype(int)))
    idx += list(range(a1, min(a1 + n_air, R["nz"])))
    # panels ordered along the propagation direction: air -> a-Si:H -> ITO -> glass
    idx = sorted(set(idx), reverse=True)

    sub = Ez2[:, :, idx]
    vmax = float(sub.max())
    norm = LogNorm(vmin=vmax / 1e4, vmax=vmax)

    ncol = 5
    nrow = int(np.ceil(len(idx) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.35 * ncol + 1.6, 2.62 * nrow + 1.5))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    im = None
    for n, k in enumerate(idx):
        ax = axes.ravel()[n]
        ax.axis("on")
        im = _map_panel(ax, Ez2[:, :, k], R["P_nm"], norm)
        reg = REGION_STYLE[R["region"][k]]
        ax.set_title(f"z = {R['z_nm'][k]:.1f} nm\n{reg['label']}  "
                     f"($\\Delta z_{{\\rm a\\text{{-}}Si}}$ = {R['z_nm'][k] - R['edges'][a0]:+.0f} nm)",
                     fontsize=8.0)
        for s in ax.spines.values():
            s.set_color(reg["color"]); s.set_linewidth(2.4)
    axes[-1, 0].axis("on")
    axes[-1, 0].set_xticks([0, R["P_nm"] / 2, R["P_nm"]]); axes[-1, 0].set_yticks([0, R["P_nm"] / 2, R["P_nm"]])
    axes[-1, 0].set_xlabel("x [nm]", fontsize=8); axes[-1, 0].set_ylabel("y [nm]", fontsize=8)
    axes[-1, 0].tick_params(labelsize=7)

    cb = fig.colorbar(im, ax=axes, fraction=0.021, pad=0.015)
    cb.set_label(r"$|E_z/E_{\rm inc}|^2$  (common log scale)", fontsize=10)
    fig.suptitle(f"{title_of(R)}\n$|E_z/E_{{\\rm inc}}|^2$ xy maps through the full stored stack, "
                 f"$\\lambda$ = {lam:.3f} nm — panels ordered along the propagation direction "
                 "(air $\\to$ a-Si:H $\\to$ ITO $\\to$ glass)", fontsize=12)
    f = outdir / f"Ez2_xy_slices_fullstack_{R['design']}_{lam:.1f}nm.png"
    fig.savefig(f, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return f, len(idx)


def fig_slices_asi_all(R, il, outdir):
    """|Ez/E_inc|^2 on every stored a-Si:H z cell, read from the campaign's Ez2_slices file."""
    lam = R["lam_nm"][il]
    fn = R["dir"] / f"Ez2_slices_aSi_{lam:.1f}nm.npz"
    d = np.load(fn)
    A = d["Ez2_over_Einc2"]                                 # (nx, ny, n_asi)
    zb = d["z_from_aSi_bottom_m"] * 1e9
    n = A.shape[2]
    order = list(range(n))[::-1]                            # top of the a-Si:H first
    vmax = float(A.max())
    norm = LogNorm(vmin=vmax / 1e4, vmax=vmax)
    ncol = 7
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.85 * ncol + 1.6, 2.05 * nrow + 1.4))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    im = None
    for m, k in enumerate(order):
        ax = axes.ravel()[m]
        ax.axis("on")
        im = _map_panel(ax, A[:, :, k], R["P_nm"], norm)
        ax.set_title(f"{zb[k]:.0f} nm  ({zb[k] / R['h_nm']:.2f} h)", fontsize=7.4)
        for s in ax.spines.values():
            s.set_color(REGION_STYLE["aSi"]["color"]); s.set_linewidth(1.8)
    a0 = axes.ravel()[0]
    a0.set_xticks([0, R["P_nm"]]); a0.set_yticks([0, R["P_nm"]])
    a0.set_xlabel("x [nm]", fontsize=7); a0.set_ylabel("y [nm]", fontsize=7)
    a0.tick_params(labelsize=6)
    cb = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.012)
    cb.set_label(r"$|E_z/E_{\rm inc}|^2$  (common log scale)", fontsize=10)
    fig.suptitle(f"{title_of(R)}\nAll {n} stored a-Si:H z cells, $|E_z/E_{{\\rm inc}}|^2$, "
                 f"$\\lambda$ = {lam:.3f} nm — panel label: height above the a-Si:H bottom "
                 "(ITO interface); reading order is downward, from the a-Si:H top to the ITO",
                 fontsize=12)
    f = outdir / f"Ez2_xy_slices_aSi_all_{R['design']}_{lam:.1f}nm.png"
    fig.savefig(f, dpi=165, bbox_inches="tight")
    plt.close(fig)
    return f, n, fn


def fig_fullheight_planes(R, il, outdir):
    """Supplementary: the stored full-height xz / yz cuts, which reach far deeper into the glass than
    the 3D volume block (the volume detector recorded only 4 glass cells)."""
    lam = R["lam_nm"][il]
    d = R["npz"]
    zp = d["z_plane_m"] * 1e9
    xz = np.abs(d["xz_plane"][il, 2]) ** 2                  # (nx, nzp) |Ez|^2, cut at y = P/2
    yz = np.abs(d["yz_plane"][il, 2]) ** 2                  # (ny, nzp) |Ez|^2, cut at x = P/2
    vmax = float(max(xz.max(), yz.max()))
    norm = LogNorm(vmin=vmax / 1e4, vmax=vmax)

    # the z grid of the plane detectors is NON-uniform (25.8 nm in the glass, 4.6 nm in the ITO,
    # 12.85 nm in the a-Si:H), so the maps must be drawn on the true cell edges, not on a linear extent
    ze_all = np.asarray(R["meta"]["z_edges_m"]) * 1e9
    idx = R["meta"]["idx"]
    z_edges_plane = ze_all[idx["z_plane0"]:idx["z_plane1"] + 1]
    x_edges = np.linspace(0.0, R["P_nm"], xz.shape[0] + 1)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 7.4), sharey=True,
                             gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
    for ax, M, lab in ((axes[0], xz, "xz cut (y = P/2)"), (axes[1], yz, "yz cut (x = P/2)")):
        im = ax.pcolormesh(x_edges, z_edges_plane, M.T, norm=norm, cmap="inferno", shading="flat")
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("x [nm]" if M is xz else "y [nm]", fontsize=9)
    axes[2].semilogx(xz.mean(0), zp, color="#a13d2d", lw=1.4, label="xz cut, $\\langle\\cdot\\rangle_x$")
    axes[2].semilogx(yz.mean(0), zp, color="#1f4e79", lw=1.4, ls="--", label="yz cut, $\\langle\\cdot\\rangle_y$")
    axes[2].set_xlabel(r"line-averaged $|E_z/E_{\rm inc}|^2$", fontsize=10)
    axes[2].grid(alpha=0.25, which="both", lw=0.5)
    axes[2].legend(fontsize=8, loc="upper right")

    # material boundaries of the full domain, from run_meta
    ze = ze_all
    bounds = [("glass", ze[0], ze[idx["z_ito0"]]), ("ITO", ze[idx["z_ito0"]], ze[idx["z_asi0"]]),
              ("aSi", ze[idx["z_asi0"]], ze[idx["z_asi1"]]), ("air", ze[idx["z_asi1"]], ze[-1])]
    for ax in axes:
        for name, lo, hi in bounds:
            ax.axhline(lo, color="w" if ax is not axes[2] else "0.35", lw=0.9, ls="--", alpha=0.8)
        ax.set_ylim(z_edges_plane[0], z_edges_plane[-1])
    z_lo, z_hi = z_edges_plane[0], z_edges_plane[-1]
    for name, lo, hi in bounds:
        axes[2].axhspan(max(lo, z_lo), min(hi, z_hi), color=REGION_STYLE[name]["color"],
                        alpha=0.18, lw=0, zorder=0)
        mid = 0.5 * (max(lo, z_lo) + min(hi, z_hi))
        axes[2].text(0.98, mid, REGION_STYLE[name]["label"], transform=axes[2].get_yaxis_transform(),
                     ha="right", va="center", fontsize=8.5,
                     bbox=dict(fc="white", ec="none", alpha=0.65, pad=1.5))
    axes[0].set_ylabel("z [nm]  (FDTDX domain coordinate)", fontsize=10)
    cb = fig.colorbar(im, ax=axes[:2], fraction=0.028, pad=0.015)
    cb.set_label(r"$|E_z/E_{\rm inc}|^2$", fontsize=10)
    fig.suptitle(f"{title_of(R)}\nSupplementary: stored full-height xz / yz field cuts, "
                 f"$\\lambda$ = {lam:.3f} nm — these are the only stored data that reach deep into "
                 "the glass\n(the 3D volume detector stored only 4 glass cells below the ITO)",
                 fontsize=11.5)
    f = outdir / f"Ez2_xz_yz_fullheight_{R['design']}_{lam:.1f}nm.png"
    fig.savefig(f, dpi=175, bbox_inches="tight")
    plt.close(fig)
    return f


# --------------------------------------------------------------------------- C: geometry
def fig_geometry(R, outdir):
    rho = R["rho"]
    P = R["P_nm"]
    fig, ax = plt.subplots(figsize=(5.0, 5.4))
    ax.imshow(rho.T, origin="lower", extent=[0, P, 0, P], cmap="binary", vmin=0, vmax=1,
              interpolation="nearest", aspect="equal")
    ax.set_xlabel("x [nm]"); ax.set_ylabel("y [nm]")
    ax.set_xticks([0, P / 4, P / 2, 3 * P / 4, P]); ax.set_yticks([0, P / 4, P / 2, 3 * P / 4, P])
    ax.set_title(f"{R['design']}\nP = {P:.0f} nm, h = {R['h_nm']:.0f} nm, pad = {100 * R['pad']:.0f} %, "
                 f"fill = {R['prov']['fill_fraction']:.4f}\n"
                 f"{rho.shape[0]}x{rho.shape[1]} hard-binary mask, "
                 f"pixel = {R['prov']['pixel_nm']:.4f} nm (black = a-Si:H)", fontsize=9.5)
    fig.tight_layout()
    f = outdir / f"geometry_{R['design']}.png"
    fig.savefig(f, dpi=200)
    plt.close(fig)
    return f


def fig_geometry_all(Rs, outdir):
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 5.2))
    for ax, R in zip(axes, Rs):
        P = R["P_nm"]
        ax.imshow(R["rho"].T, origin="lower", extent=[0, P, 0, P], cmap="binary", vmin=0, vmax=1,
                  interpolation="nearest", aspect="equal")
        ax.set_title(f"{R['design']}\nP {P:.0f} / h {R['h_nm']:.0f} nm, pad {100 * R['pad']:.0f} %\n"
                     f"fill {R['prov']['fill_fraction']:.4f}", fontsize=10)
        ax.set_xlabel("x [nm]", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("y [nm]", fontsize=9)
    fig.suptitle("Four finalists: hard-binary a-Si:H masks (128x128, black = a-Si:H)", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    f = outdir / "geometry_all_four.png"
    fig.savefig(f, dpi=190)
    plt.close(fig)
    return f


# --------------------------------------------------------------------------- driver
def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "geometry").mkdir(exist_ok=True)
    manifest = {"lambda_ZE_nm": LAM_ZE, "tag": TAG, "designs": {}}
    Rs, il_of = [], {}
    for design in DESIGNS:
        R = load_design(design)
        il = int(np.argmin(np.abs(R["lam_nm"] - LAM_ZE)))
        Rs.append(R); il_of[design] = il
        odir = OUT / design
        odir.mkdir(exist_ok=True)

        fA = fig_zprofile(R, il, odir)
        fC = csv_zprofile(R, il, odir)
        fB1, n1 = fig_slices_fullstack(R, il, odir)
        fB2, n2, srcB2 = fig_slices_asi_all(R, il, odir)
        fB3 = fig_fullheight_planes(R, il, odir)
        fG = fig_geometry(R, OUT / "geometry")

        manifest["designs"][design] = {
            "P_nm": R["P_nm"], "h_nm": R["h_nm"], "pad_frac": R["pad"],
            "fill_fraction": R["prov"]["fill_fraction"],
            "lambda_used_nm": float(R["lam_nm"][il]),
            "raw_inputs": [str(R["dir"] / "fields_normalized.npz"),
                           str(srcB2), str(R["dir"] / "run_meta.json"),
                           str(HERE / design / "rho_hard_binary.npy")],
            "volume_block": {
                "shape_lam_c_x_y_z": list(R["E"].shape),
                "n_glass_cells": R["i0"], "n_ito_cells": R["i1"] - R["i0"],
                "n_asi_cells": R["a1"] - R["a0"], "n_air_cells": R["nz"] - R["a1"],
                "z_nm_min": float(R["edges"][0]), "z_nm_max": float(R["edges"][-1]),
                "glass_depth_stored_nm": float(R["edges"][R["i0"]] - R["edges"][0])},
            "outputs": {"zprofile_png": str(fA), "zprofile_csv": str(fC),
                        "slices_fullstack_png": str(fB1), "n_fullstack_panels": n1,
                        "slices_asi_all_png": str(fB2), "n_asi_panels": n2,
                        "fullheight_planes_png": str(fB3), "geometry_png": str(fG)}}
        print(f"[{design}] {fA.name} | {fB1.name} ({n1} panels) | {fB2.name} ({n2} panels) | "
              f"{fB3.name} | {fG.name}", flush=True)

    fO = fig_overlay(Rs, il_of, OUT)
    fGA = fig_geometry_all(Rs, OUT / "geometry")
    manifest["overlay_png"] = str(fO)
    manifest["geometry_all_png"] = str(fGA)
    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=1)
    print("wrote", OUT / "manifest.json")


if __name__ == "__main__":
    sys.exit(main())
