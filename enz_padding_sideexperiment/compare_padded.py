"""Padded (85-nm air ring) vs unpadded baseline: full comparison suite.

Both geometries are FINAL HARD-BINARY designs evaluated with identical
machinery, normalization, target, source, grid, and Fourier order.
Baseline = the authoritative unpadded 850-nm campaign result
(enz_inverse_design/outputs, seed 333); padded = this side experiment.

Produces: headline_comparison.csv, spectra npz + figures, locality metrics,
field/current/overlap maps, Fourier-order convergence, with/without-ITO
spectra for the padded winner.

Run:  python compare_padded.py     (after run_padded.py)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / "enz_inverse_design"
sys.path.insert(0, str(PKG))

import config                                  # noqa: E402
import target_mode                             # noqa: E402
import torcwa_forward as fwd                   # noqa: E402
import objective as obj                        # noqa: E402
from validate_with_without_ito import (build_sim, power_RT,   # noqa: E402
                                       spectrum)

C_BLUE, C_ORANGE, C_AQUA, C_VIOLET, C_MAGENTA = \
    "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": "#e3e2df", "grid.linewidth": 0.6, "font.size": 10,
    "lines.linewidth": 2.0, "legend.frameon": False})

OUT = HERE / "outputs"
FIG = OUT / "figures"


def _save(fig, name):
    fig.savefig(FIG / name, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"figure written: {FIG/name}")


# ---------------------------------------------------------------------------
# shared problem context
# ---------------------------------------------------------------------------
def context():
    torch.set_num_threads(config.N_THREADS)
    tgt = target_mode.load_target_npz()
    lam = float(tgt["wavelength_nm"])
    config.ITO_THICKNESS_NM = float(tgt["ito_thickness_nm"])
    config.N_GLASS = float(tgt["glass_index"])
    if config.EPS_ASI is None:
        config.EPS_ASI = fwd.eps_asi_of_lambda(lam)
    eps_ito = complex(float(tgt["eps_ito_real"]), float(tgt["eps_ito_imag"]))
    x, y = fwd.grid_axes()
    zp = target_mode.ito_z_slices(config.ITO_THICKNESS_NM,
                                  config.Z_SAMPLES_ITO)
    Tp, dV = target_mode.build_target_field(tgt, x.cpu().numpy(),
                                            y.cpu().numpy(), zp, "+x")
    Tm, _ = target_mode.build_target_field(tgt, x.cpu().numpy(),
                                           y.cpu().numpy(), zp, "-x")
    with torch.no_grad():
        Ez_ref = fwd.ez_in_ito(
            fwd.build_solved_sim(None, lam, eps_ito, config.N_GLASS),
            x, y, zp).detach()
    return dict(tgt=tgt, lam=lam, eps_ito=eps_ito, x=x, y=y, zp=zp,
                Tp=Tp, Tm=Tm, dV=dV, Ez_ref=Ez_ref, p_inc=fwd.p_inc_cell())


def e_in_ito(sim, ctx):
    """All E components at the ITO z-slices: (3, Nz, Nx, Ny)."""
    comps = [[], [], []]
    for zpv in ctx["zp"]:
        E, _ = sim.field_xy(1, ctx["x"], ctx["y"], float(zpv))
        for c in range(3):
            comps[c].append(E[c])
    return [torch.stack(c, dim=0) for c in comps]


def optical_metrics(rho_t, ctx, order=None):
    """Everything in the headline table that comes from one forward solve."""
    o = order or config.FOURIER_ORDER
    old = config.FOURIER_ORDER
    config.FOURIER_ORDER = o
    try:
        with torch.no_grad():
            sim = fwd.build_solved_sim(rho_t, ctx["lam"], ctx["eps_ito"],
                                       config.N_GLASS)
            Ex, Ey, Ez = e_in_ito(sim, ctx)
            Ez_scat = Ez - ctx["Ez_ref"]
            F, d = obj.enz_objective(ctx["Tp"], Ez_scat, ctx["dV"],
                                     ctx["p_inc"], target_minus=ctx["Tm"],
                                     direction="bidir")
            I_scat = float(torch.sum(torch.abs(Ez_scat) ** 2).real * ctx["dV"])
            eta_pm = (float(torch.abs(d["a_plus"]) ** 2)
                      + float(torch.abs(d["a_minus"]) ** 2)) / I_scat
            Iz = float(torch.sum(torch.abs(Ez) ** 2).real * ctx["dV"])
            It = float(torch.sum(torch.abs(Ex) ** 2
                                 + torch.abs(Ey) ** 2).real * ctx["dV"])
            # a-Si field (mid-height) vs ITO field intensity means
            Easi, _ = sim.field_xy(0, ctx["x"], ctx["y"],
                                   config.ASI_THICKNESS_NM / 2)
            asi_I = float(np.mean(sum(np.abs(c.cpu().numpy()) ** 2
                                      for c in Easi)))
            ito_I = float(torch.mean(torch.abs(Ex) ** 2 + torch.abs(Ey) ** 2
                                     + torch.abs(Ez) ** 2).real)
            R, T = power_RT(build_sim(rho_t, ctx["lam"], True, order=o))
            # ITO |Ez|^2 xy-distribution (for locality metrics)
            Iz_xy = torch.sum(torch.abs(Ez) ** 2, dim=0).cpu().numpy()
        return dict(
            F_QNM=float(F),
            a_plus2=float(torch.abs(d["a_plus"]) ** 2),
            a_minus2=float(torch.abs(d["a_minus"]) ** 2),
            a_plus=complex(d["a_plus"]), a_minus=complex(d["a_minus"]),
            eta_pm=eta_pm,
            F_Ez=Iz / ctx["p_inc"], F_Et=It / ctx["p_inc"],
            eta_z=Iz / (Iz + It),
            asi_over_ito_intensity=asi_I / ito_I,
            T=R and T, R=R, A=1 - R - T,
            Iz_xy=Iz_xy,
        )
    finally:
        config.FOURIER_ORDER = old


# ---------------------------------------------------------------------------
# locality metrics
# ---------------------------------------------------------------------------
def periodic_components(rho_bin):
    """Connected components with periodic wrap-around."""
    lab, n = ndimage.label(rho_bin)
    if n == 0:
        return 0
    parent = list(range(n + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    nx, ny = rho_bin.shape
    for i in range(nx):
        a, b = lab[i, 0], lab[i, ny - 1]
        if a and b:
            union(a, b)
    for j in range(ny):
        a, b = lab[0, j], lab[nx - 1, j]
        if a and b:
            union(a, b)
    return len({find(k) for k in range(1, n + 1)})


def min_scale_by_morphology(mask_bin, dx, what="feature", max_r=10,
                            loss_thresh=0.02):
    """Smallest disk radius whose opening (feature) / closing (gap)
    changes the pattern by more than loss_thresh of the Si area."""
    area = mask_bin.sum()
    if area == 0:
        return np.nan
    for r in range(1, max_r + 1):
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        disk = (xx ** 2 + yy ** 2) <= r ** 2
        if what == "feature":
            changed = area - ndimage.binary_opening(mask_bin, disk).sum()
        else:
            changed = ndimage.binary_closing(mask_bin, disk).sum() - area
        if changed > loss_thresh * area:
            return 2 * r * dx
    return 2 * max_r * dx    # lower bound; ">= value"


def locality_metrics(rho_bin, Iz_xy, mask=None):
    nx, ny = rho_bin.shape
    dx = config.PX_NM / nx
    boundary = np.concatenate([rho_bin[0, :], rho_bin[-1, :],
                               rho_bin[:, 0], rho_bin[:, -1]])
    # participation ratio of the ITO |Ez|^2 xy distribution (1 = uniform)
    p = Iz_xy.ravel()
    PR = float((p.sum() ** 2) / (p.size * (p ** 2).sum()))
    # fraction of ITO |Ez|^2 inside the central 680x680 window
    i0, i1 = int(85 / dx), int((config.PX_NM - 85) / dx) + 1
    frac_central = float(Iz_xy[i0:i1, i0:i1].sum() / Iz_xy.sum())
    return dict(
        fill_fraction=float(rho_bin.mean()),
        boundary_contact_fraction=float(boundary.mean()),
        connected_components=periodic_components(rho_bin),
        min_feature_nm=min_scale_by_morphology(rho_bin > 0.5, dx, "feature"),
        min_gap_nm=min_scale_by_morphology(rho_bin > 0.5, dx, "gap"),
        Ez2_participation_ratio=PR,
        Ez2_fraction_central_680=frac_central,
    )


# ---------------------------------------------------------------------------
# spectra of F_QNM / ITO Ez^2 / T,R,A
# ---------------------------------------------------------------------------
def fqnm_spectrum(rho_t, ctx, lams):
    out = {"lam": [], "F": [], "Iz": [], "T": [], "R": []}
    with torch.no_grad():
        for lam in lams:
            eps_ito = fwd.eps_ito_of_lambda(lam)
            eps_asi = fwd.eps_asi_of_lambda(lam)
            sim_ref = fwd.build_solved_sim(None, lam, eps_ito, config.N_GLASS)
            Ez_ref = fwd.ez_in_ito(sim_ref, ctx["x"], ctx["y"], ctx["zp"])
            sim = fwd.build_solved_sim(rho_t, lam, eps_ito, config.N_GLASS,
                                       eps_asi=eps_asi)
            Ez = fwd.ez_in_ito(sim, ctx["x"], ctx["y"], ctx["zp"])
            F, _ = obj.enz_objective(ctx["Tp"], Ez - Ez_ref, ctx["dV"],
                                     ctx["p_inc"], target_minus=ctx["Tm"],
                                     direction="bidir")
            R, T = power_RT(build_sim(rho_t, lam, True))
            out["lam"].append(lam); out["F"].append(float(F))
            out["Iz"].append(float(torch.sum(torch.abs(Ez) ** 2).real
                                   * ctx["dV"] / ctx["p_inc"]))
            out["T"].append(T); out["R"].append(R)
    return {k: np.array(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# field / current / overlap visualization
# ---------------------------------------------------------------------------
def maps_figure(rho_bin, rho_t, ctx, tag):
    lam = ctx["lam"]
    sim = build_sim(rho_t, lam, True)
    nplot = 180
    xs = torch.as_tensor((np.arange(nplot) + 0.5) / nplot * config.PX_NM,
                         dtype=config.GEO_DTYPE)
    ys = torch.as_tensor((np.arange(nplot) + 0.5) / nplot * config.PY_NM,
                         dtype=config.GEO_DTYPE)
    with torch.no_grad():
        # ITO mid-plane Ez(x,y)
        E_ito, _ = sim.field_xy(1, xs, ys, float(config.ITO_THICKNESS_NM / 2))
        Ez_ito = E_ito[2].cpu().numpy()
        # a-Si mid-plane in-plane E -> induced polarization current map
        E_asi, _ = sim.field_xy(0, xs, ys, float(config.ASI_THICKNESS_NM / 2))
        Jx = (complex(config.EPS_ASI) - 1) * E_asi[0].cpu().numpy()
        Jy = (complex(config.EPS_ASI) - 1) * E_asi[1].cpu().numpy()
        # x-z cut
        zax = torch.linspace(-120.0, config.ASI_THICKNESS_NM
                             + config.ITO_THICKNESS_NM + 120.0, 241,
                             dtype=config.GEO_DTYPE)
        Exz, _ = sim.field_xz(xs, zax, config.PY_NM / 2)
        Emag = np.sqrt(sum(np.abs(c.cpu().numpy()) ** 2 for c in Exz))
        Ez_xz = Exz[2].cpu().numpy()

    # upsample geometry mask for current plot masking
    zoom = nplot / rho_bin.shape[0]
    geo = ndimage.zoom(rho_bin, zoom, order=0)
    Jmag = np.hypot(np.abs(Jx), np.abs(Jy)) * geo

    fig, axs = plt.subplots(2, 3, figsize=(15.5, 8.6))
    ext = [0, config.PX_NM, 0, config.PY_NM]
    ax = axs[0, 0]
    ax.imshow(rho_bin.T, origin="lower", cmap="Greys", extent=ext)
    ax.set_title(f"{tag}: final binary a-Si (top view)")
    ax = axs[0, 1]
    im = ax.imshow(np.abs(Ez_ito).T, origin="lower", cmap="Blues", extent=ext)
    fig.colorbar(im, ax=ax); ax.set_title(r"$|E_z|$ mid-ITO $(x,y)$")
    ax = axs[0, 2]
    im = ax.imshow(np.angle(Ez_ito).T, origin="lower", cmap="twilight",
                   extent=ext)
    fig.colorbar(im, ax=ax); ax.set_title(r"arg $E_z$ mid-ITO")
    ax = axs[1, 0]
    im = ax.imshow(Jmag.T, origin="lower", cmap="Blues", extent=ext)
    st = slice(4, None, 12)
    ax.quiver(np.outer((np.arange(nplot) + 0.5) / nplot * config.PX_NM,
                       np.ones(nplot))[st, st],
              np.outer(np.ones(nplot),
                       (np.arange(nplot) + 0.5) / nplot * config.PY_NM)[st, st],
              (Jx.real * geo)[st, st], (Jy.real * geo)[st, st],
              color=INK, scale=np.abs(Jmag).max() * 22, width=0.003)
    fig.colorbar(im, ax=ax)
    ax.set_title(r"a-Si polarization current $|J_\parallel|$, Re vectors")
    extz = [0, config.PX_NM, -120, config.ASI_THICKNESS_NM
            + config.ITO_THICKNESS_NM + 120]
    ax = axs[1, 1]
    im = ax.imshow(Emag.T, origin="lower", cmap="Blues", aspect="auto",
                   extent=extz)
    ax.axhline(0, color=INK, lw=0.7)
    ax.axhline(config.ASI_THICKNESS_NM, color=INK, lw=0.7)
    ax.axhline(config.ASI_THICKNESS_NM + config.ITO_THICKNESS_NM,
               color=C_ORANGE, lw=1.0)
    fig.colorbar(im, ax=ax); ax.set_title(r"$|E|$ x-z (y = P/2)")
    ax = axs[1, 2]
    v = np.nanpercentile(np.abs(Ez_xz.real), 99.5)
    im = ax.imshow(Ez_xz.real.T, origin="lower", cmap="RdBu_r", aspect="auto",
                   extent=extz, vmin=-v, vmax=v)
    ax.axhline(0, color=INK, lw=0.7)
    ax.axhline(config.ASI_THICKNESS_NM, color=INK, lw=0.7)
    ax.axhline(config.ASI_THICKNESS_NM + config.ITO_THICKNESS_NM,
               color=C_ORANGE, lw=1.0)
    fig.colorbar(im, ax=ax); ax.set_title(r"Re $E_z$ x-z")
    for ax in axs.ravel():
        ax.grid(False)
    fig.suptitle(f"{tag} at lambda = {lam:.1f} nm (orange line: ITO bottom)",
                 y=1.0)
    _save(fig, f"maps_{tag}.png")


def overlap_profile_figure(rhos, ctx):
    """Driven +-G Ez(z) vs the QNM target profile + overlap integrand."""
    tgt = ctx["tgt"]
    d = config.ITO_THICKNESS_NM
    nz = 15
    zs = (np.arange(nz) + 0.5) * d / nz
    prof_t = np.interp(d - zs, tgt["z_nm"], tgt["Ez"].real) \
        + 1j * np.interp(d - zs, tgt["z_nm"], tgt["Ez"].imag)
    fig, axs = plt.subplots(1, 3, figsize=(13.6, 3.8))
    n = config.NX_DESIGN
    for tag, rho_t, col in rhos:
        sim = fwd.build_solved_sim(rho_t, ctx["lam"], ctx["eps_ito"],
                                   config.N_GLASS)
        ap, am = [], []
        with torch.no_grad():
            for zpv in zs:
                E, _ = sim.field_xy(1, ctx["x"], ctx["y"], float(zpv))
                F2 = np.fft.fft2(E[2].cpu().numpy()) / n ** 2
                ap.append(F2[1 % n, 0]); am.append(F2[-1 % n, 0])
        ap, am = np.array(ap), np.array(am)
        axs[0].plot(zs, np.abs(ap), color=col, label=f"{tag} $|E_z^{{+G}}|$")
        axs[0].plot(zs, np.abs(am), color=col, ls="--",
                    label=f"{tag} $|E_z^{{-G}}|$")
        axs[1].plot(zs, (np.conj(prof_t) * ap).real, color=col, label=tag)
        axs[1].plot(zs, (np.conj(prof_t) * am).real, color=col, ls="--")
    axs[2].plot(zs, np.abs(prof_t), color=INK, label=r"QNM $|E_z(z)|$")
    axs[0].set_title("driven $\\pm G_{10}$ harmonic $|E_z(z)|$ in ITO")
    axs[1].set_title(r"overlap integrand Re$[E_z^{QNM*} E_z^{\pm G}](z)$")
    axs[2].set_title("target QNM profile (arb. norm)")
    for ax in axs:
        ax.set_xlabel("z into ITO from a-Si side (nm)")
        ax.legend(fontsize=8)
    _save(fig, "overlap_profiles.png")


# ---------------------------------------------------------------------------
def main():
    FIG.mkdir(exist_ok=True, parents=True)
    ctx = context()

    rho_base = np.load(PKG / "outputs" / "geometries" / "rho_hard_binary.npy")
    rho_pad = np.load(OUT / "geometries" / "rho_hard_binary.npy")
    tb = torch.as_tensor(rho_base, dtype=config.GEO_DTYPE)
    tp = torch.as_tensor(rho_pad, dtype=config.GEO_DTYPE)

    rows = {}
    for tag, rb, rt in (("unpadded", rho_base, tb), ("padded85", rho_pad, tp)):
        m = optical_metrics(rt, ctx)
        loc = locality_metrics(rb, m.pop("Iz_xy"))
        m.pop("a_plus"), m.pop("a_minus")
        rows[tag] = {**m, **loc}
        print(f"[{tag}] F_QNM = {m['F_QNM']:.4e}, eta_pm = {m['eta_pm']:.3f}, "
              f"F_Ez = {m['F_Ez']:.3e}, eta_z = {m['eta_z']:.3f}, "
              f"T/R/A = {m['T']:.3f}/{m['R']:.3f}/{m['A']:.3f}")
        print(f"[{tag}] locality: {loc}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "headline_comparison.csv")
    print(f"[saved] {OUT/'headline_comparison.csv'}")

    # convergence: order refinement (rebuild ctx reference per order)
    conv = {}
    for od in ([7, 7], [9, 9], [11, 11]):
        old = config.FOURIER_ORDER
        config.FOURIER_ORDER = od
        with torch.no_grad():
            Ez_ref_o = fwd.ez_in_ito(
                fwd.build_solved_sim(None, ctx["lam"], ctx["eps_ito"],
                                     config.N_GLASS),
                ctx["x"], ctx["y"], ctx["zp"]).detach()
        ctx_o = {**ctx, "Ez_ref": Ez_ref_o}
        config.FOURIER_ORDER = old
        for tag, rt in (("unpadded", tb), ("padded85", tp)):
            m = optical_metrics(rt, ctx_o, order=od)
            conv[f"{tag}_{od}"] = {k: m[k] for k in
                                   ("F_QNM", "T", "R", "A", "F_Ez")}
            print(f"[conv] {tag} order {od}: F = {m['F_QNM']:.4e}, "
                  f"T = {m['T']:.4f}, A = {m['A']:.4f}")
    with open(OUT / "histories" / "convergence.json", "w") as f:
        json.dump(conv, f, indent=1, default=float)

    # spectra around the target (both geometries, matched)
    lams = np.arange(1350.0, 1550.5, 2.0)
    print("[spectra] F_QNM(lambda), padded ...")
    spP = fqnm_spectrum(tp, ctx, lams)
    print("[spectra] F_QNM(lambda), unpadded ...")
    spU = fqnm_spectrum(tb, ctx, lams)
    np.savez(OUT / "histories" / "fqnm_spectra.npz",
             lam=lams, F_padded=spP["F"], F_unpadded=spU["F"],
             Iz_padded=spP["Iz"], Iz_unpadded=spU["Iz"],
             T_padded=spP["T"], T_unpadded=spU["T"],
             R_padded=spP["R"], R_unpadded=spU["R"])

    fig, axs = plt.subplots(1, 3, figsize=(14.6, 4.0))
    axs[0].semilogy(lams, spU["F"], color=C_BLUE, label="unpadded")
    axs[0].semilogy(lams, spP["F"], color=C_ORANGE, label="padded 85 nm")
    axs[0].set_ylabel(r"$F_{QNM}(\lambda)$")
    axs[1].semilogy(lams, spU["Iz"], color=C_BLUE)
    axs[1].semilogy(lams, spP["Iz"], color=C_ORANGE)
    axs[1].set_ylabel(r"$\int_{ITO}|E_z|^2 dV / P_{inc}$ (nm)")
    axs[2].plot(lams, 1 - spU["T"] - spU["R"], color=C_BLUE)
    axs[2].plot(lams, 1 - spP["T"] - spP["R"], color=C_ORANGE)
    axs[2].set_ylabel("A (ITO absorption)")
    for ax in axs:
        ax.axvline(ctx["lam"], color=C_AQUA, ls="--", lw=1.0)
        ax.set_xlabel("wavelength (nm)")
    axs[0].legend(fontsize=9)
    fig.suptitle("padded vs unpadded around the QNM target "
                 "(dashed: 1433.5 nm)", y=1.02)
    _save(fig, "fqnm_spectra.png")

    # with/without-ITO for the padded winner (Karimi-style control)
    lam_b = np.arange(1200.0, 1700.5, 2.0)
    print("[spectra] padded, no ITO ...")
    spA = spectrum(tp, lam_b, False, tag="pad-noITO")
    print("[spectra] padded, with ITO ...")
    spB = spectrum(tp, lam_b, True, tag="pad-ITO")
    np.savez(OUT / "histories" / "padded_with_without_ito.npz",
             lam=lam_b, T_no=spA["T"], R_no=spA["R"],
             T_ito=spB["T"], R_ito=spB["R"])
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(lam_b, spA["T"], color=C_BLUE, label="padded design / glass")
    ax.plot(lam_b, spB["T"], color=C_ORANGE,
            label="padded design / 23-nm ITO / glass")
    ax.axvline(1419.59, color=C_VIOLET, ls="--", lw=1.0)
    ax.axvline(ctx["lam"], color=C_AQUA, ls="--", lw=1.0)
    ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("T"); ax.set_ylim(0, 1.02)
    ax.set_title("padded winner: with vs without ITO (same frozen geometry)")
    ax.legend(loc="lower left", fontsize=9)
    _save(fig, "padded_with_without_ITO.png")
    print(f"[closure] max|A| padded no-ITO = "
          f"{np.abs(spA['A']).max():.2e} (lossless -> ~0)")

    # field/current maps + overlap profiles
    maps_figure(rho_base, tb, ctx, "unpadded")
    maps_figure(rho_pad, tp, ctx, "padded85")
    overlap_profile_figure([("unpadded", tb, C_BLUE),
                            ("padded85", tp, C_ORANGE)], ctx)

    # geometry side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(9.8, 4.6), sharey=True)
    for ax, r, t in ((axs[0], rho_base, "unpadded"),
                     (axs[1], rho_pad, "padded 85 nm")):
        ax.imshow(r.T, origin="lower", cmap="Greys",
                  extent=[0, config.PX_NM, 0, config.PY_NM])
        ax.set_title(t); ax.set_xlabel("x (nm)"); ax.grid(False)
    axs[1].add_patch(plt.Rectangle((85, 85), config.PX_NM - 170,
                                   config.PY_NM - 170, fill=False,
                                   ec=C_ORANGE, lw=1.4, ls="--"))
    axs[0].set_ylabel("y (nm)")
    _save(fig, "geometry_comparison.png")

    return rows


if __name__ == "__main__":
    main()
