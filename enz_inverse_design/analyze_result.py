"""Post-optimization analysis: Figures 1-8 + numerical convergence checks.

Run after optimize_enz_overlap.py:  python analyze_result.py
(Heavy spectral scans can be reduced with --quick.)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

C_BLUE, C_ORANGE, C_AQUA, C_VIOLET, C_MAGENTA = \
    "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": "#e3e2df", "grid.linewidth": 0.6, "font.size": 10.5,
    "lines.linewidth": 2.0, "legend.frameon": False})

OUT = Path(config.OUT_DIR)
FIG = OUT / "figures"


def _save(fig, name):
    fig.savefig(FIG / name, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"figure written: {FIG/name}")


def fig1_convergence(hist):
    it = np.arange(len(hist["F"]))
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    axs[0].semilogy(it, hist["F"], color=C_BLUE)
    axs[0].set_xlabel("iteration"); axs[0].set_ylabel(r"$F_{\rm ENZ}$")
    axs[0].set_title("ENZ-overlap FoM")
    axs[1].semilogy(it, hist["grad_norm"], color=C_ORANGE)
    axs[1].set_xlabel("iteration"); axs[1].set_ylabel(r"$\|\nabla_\rho F\|$")
    axs[1].set_title("gradient norm")
    axs[2].plot(it, hist["binarization"], color=C_AQUA)
    axs[2].set_xlabel("iteration"); axs[2].set_ylabel("gray metric")
    axs[2].set_title("binarization (0 = binary)")
    _save(fig, "fig1_convergence.png")


def fig2_geometry():
    r0 = np.load(OUT / "geometries" / "rho_initial.npy")
    r1 = np.load(OUT / "geometries" / "rho_proj_final.npy")
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.6), sharey=True)
    for ax, r, t in ((axs[0], r0, "initial (filtered) rho"),
                     (axs[1], r1, "final projected rho")):
        im = ax.imshow(r.T, origin="lower", cmap="Greys", vmin=0, vmax=1,
                       extent=[0, config.PX_NM, 0, config.PY_NM])
        ax.set_title(t); ax.set_xlabel("x (nm)"); ax.grid(False)
    axs[0].set_ylabel("y (nm)")
    fig.colorbar(im, ax=axs, label=r"$\rho$ (a-Si fill)", shrink=0.85)
    _save(fig, "fig2_geometry.png")


def _midslice(arr):        # (Nz, Nx, Ny) -> mid-z slice (Nx, Ny)
    return arr[arr.shape[0] // 2]


def fig3_target():
    T = np.load(OUT / "fields" / "Ez_target_plus.npy")
    Tm = _midslice(T)
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.2), sharey=True)
    v = np.abs(Tm.real).max()
    im0 = axs[0].imshow(Tm.real.T, origin="lower", cmap="RdBu_r",
                        vmin=-v, vmax=v, extent=[0, config.PX_NM, 0, config.PY_NM])
    axs[0].set_title(r"Re $E_z^{\rm target}$ (mid-ITO)")
    fig.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(np.abs(Tm).T, origin="lower", cmap="Blues",
                        extent=[0, config.PX_NM, 0, config.PY_NM])
    axs[1].set_title(r"$|E_z^{\rm target}|$")
    fig.colorbar(im1, ax=axs[1])
    for ax in axs:
        ax.set_xlabel("x (nm)"); ax.grid(False)
    axs[0].set_ylabel("y (nm)")
    _save(fig, "fig3_target_field.png")


def _field_pair(name, tag):
    E = np.load(OUT / "fields" / name)
    Em = _midslice(E)
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.2), sharey=True)
    v = np.abs(Em.real).max()
    im0 = axs[0].imshow(Em.real.T, origin="lower", cmap="RdBu_r", vmin=-v,
                        vmax=v, extent=[0, config.PX_NM, 0, config.PY_NM])
    axs[0].set_title(rf"Re $E_z^{{\rm scat}}$ ({tag}, mid-ITO)")
    fig.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(np.abs(Em).T, origin="lower", cmap="Blues",
                        extent=[0, config.PX_NM, 0, config.PY_NM])
    axs[1].set_title(rf"$|E_z^{{\rm scat}}|$ ({tag})")
    fig.colorbar(im1, ax=axs[1])
    for ax in axs:
        ax.set_xlabel("x (nm)"); ax.grid(False)
    axs[0].set_ylabel("y (nm)")
    _save(fig, f"fig{'4' if tag=='initial' else '5'}_Ez_scat_{tag}.png")


def fig6_comparison():
    T = _midslice(np.load(OUT / "fields" / "Ez_target_plus.npy"))
    E = _midslice(np.load(OUT / "fields" / "Ez_scat_final.npy"))
    fig, axs = plt.subplots(2, 2, figsize=(10.6, 8.4), sharex=True, sharey=True)
    panels = [(np.abs(T), "target $|E_z|$", "Blues"),
              (np.abs(E), "optimized $|E_z^{scat}|$", "Blues"),
              (np.angle(T), "target phase", "twilight"),
              (np.angle(E), "optimized phase", "twilight")]
    for ax, (v, t, cm) in zip(axs.ravel(), panels):
        im = ax.imshow(v.T, origin="lower", cmap=cm,
                       extent=[0, config.PX_NM, 0, config.PY_NM])
        ax.set_title(t); ax.grid(False)
        fig.colorbar(im, ax=ax)
    for ax in axs[1]:
        ax.set_xlabel("x (nm)")
    for ax in axs[:, 0]:
        ax.set_ylabel("y (nm)")
    fig.suptitle("mid-ITO slice: target vs optimized scattered field", y=0.995)
    _save(fig, "fig6_complex_comparison.png")


def spectra(lams, rho_proj, tgt, quick=False):
    """T/R/A and F_ENZ vs wavelength for the FIXED final geometry.

    The target profile is kept at the design-wavelength ENZ target
    (fixed-target diagnostic, stated on the figure); eps_ITO follows the
    Phase-1 CSV dispersion; eps_aSi is the constant config value.
    """
    import target_mode
    import torcwa_forward as fwd
    import objective as obj
    torch.set_num_threads(config.N_THREADS)
    x_axis, y_axis = fwd.grid_axes()
    z_prop = target_mode.ito_z_slices(config.ITO_THICKNESS_NM,
                                      config.Z_SAMPLES_ITO)
    T_plus, dV = target_mode.build_target_field(
        tgt, x_axis.cpu().numpy(), y_axis.cpu().numpy(), z_prop, "+x")
    T_minus, _ = target_mode.build_target_field(
        tgt, x_axis.cpu().numpy(), y_axis.cpu().numpy(), z_prop, "-x")
    p_inc = fwd.p_inc_cell()
    rho_t = torch.as_tensor(rho_proj, dtype=config.GEO_DTYPE,
                            device=config.DEVICE)
    out = {"lam": [], "T": [], "R": [], "A": [], "F": []}
    with torch.no_grad():
        for lam in lams:
            eps_ito = fwd.eps_ito_of_lambda(lam)
            eps_asi = fwd.eps_asi_of_lambda(lam)      # dispersive a-Si
            sim_ref = fwd.build_solved_sim(None, lam, eps_ito, config.N_GLASS)
            Ez_ref = fwd.ez_in_ito(sim_ref, x_axis, y_axis, z_prop)
            sim = fwd.build_solved_sim(rho_t, lam, eps_ito, config.N_GLASS,
                                       eps_asi=eps_asi)
            Ez_scat = fwd.ez_in_ito(sim, x_axis, y_axis, z_prop) - Ez_ref
            F, _ = obj.enz_objective(T_plus, Ez_scat, dV, p_inc,
                                     target_minus=T_minus,
                                     direction=config.TARGET_DIRECTION)
            R, T = fwd.specular_RT(sim)
            out["lam"].append(lam); out["F"].append(float(F))
            out["R"].append(float(R)); out["T"].append(float(T))
            out["A"].append(float(1 - R - T))
            print(f"  lambda {lam:6.0f}: T={float(T):.3f} R={float(R):.3f} "
                  f"A={float(1-R-T):.3f} F={float(F):.3e}")
    return {k: np.array(v) for k, v in out.items()}


def fig78_spectra(sp, lam0):
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(sp["lam"], sp["T"], color=C_BLUE, label="T")
    ax.plot(sp["lam"], sp["R"], color=C_ORANGE, label="R")
    ax.plot(sp["lam"], sp["A"], color=C_AQUA, label="A = 1-T-R")
    ax.axvline(lam0, color=C_VIOLET, ls="--", lw=1.2)
    ax.text(lam0 + 4, 0.9, "target $\\lambda_E$", color=C_VIOLET, fontsize=9)
    ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("power fraction")
    ax.set_ylim(0, 1); ax.legend()
    ax.set_title("optimized structure: spectral response "
                 "(not by itself evidence of strong coupling)")
    _save(fig, "fig7_spectral_response.png")

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.semilogy(sp["lam"], sp["F"], color=C_MAGENTA)
    ax.axvline(lam0, color=C_VIOLET, ls="--", lw=1.2)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel(r"$F_{\rm ENZ}(\lambda)$ (fixed $\lambda_E$ target)")
    ax.set_title("ENZ-overlap spectrum of the final geometry")
    _save(fig, "fig8_overlap_spectrum.png")


def convergence_checks(tgt, rho_proj):
    """Final-F sensitivity to Fourier order and z-sampling."""
    import target_mode
    import torcwa_forward as fwd
    import objective as obj
    lam = float(tgt["wavelength_nm"])
    eps_ito = complex(float(tgt["eps_ito_real"]), float(tgt["eps_ito_imag"]))
    rho_t = torch.as_tensor(rho_proj, dtype=config.GEO_DTYPE,
                            device=config.DEVICE)
    results = {}
    base_order = list(config.FOURIER_ORDER)
    base_nz = config.Z_SAMPLES_ITO
    with torch.no_grad():
        for tag, order, nz in (("baseline", base_order, base_nz),
                               ("order+2", [base_order[0] + 2,
                                            base_order[1] + 2], base_nz),
                               ("nz x2", base_order, 2 * base_nz + 1)):
            config.FOURIER_ORDER = order
            config.Z_SAMPLES_ITO = nz
            import importlib
            x_axis, y_axis = fwd.grid_axes()
            z_prop = target_mode.ito_z_slices(config.ITO_THICKNESS_NM, nz)
            T_plus, dV = target_mode.build_target_field(
                tgt, x_axis.cpu().numpy(), y_axis.cpu().numpy(), z_prop, "+x")
            sim_ref = fwd.build_solved_sim(None, lam, eps_ito, config.N_GLASS)
            Ez_ref = fwd.ez_in_ito(sim_ref, x_axis, y_axis, z_prop)
            sim = fwd.build_solved_sim(rho_t, lam, eps_ito, config.N_GLASS)
            Ez_scat = fwd.ez_in_ito(sim, x_axis, y_axis, z_prop) - Ez_ref
            F, _ = obj.enz_objective(T_plus, Ez_scat, dV, fwd.p_inc_cell(),
                                     direction="+x")
            results[tag] = float(F)
            print(f"[convergence] {tag:9s} order={order} nz={nz}: "
                  f"F = {float(F):.5e}")
    config.FOURIER_ORDER = base_order
    config.Z_SAMPLES_ITO = base_nz
    base = results["baseline"]
    for tag, v in results.items():
        if tag != "baseline":
            print(f"[convergence] {tag}: relative change {abs(v-base)/base:.2%}")
    return results


def main(quick=False):
    import target_mode
    with open(OUT / "histories" / "history.json") as f:
        hist = json.load(f)
    tgt = target_mode.load_target_npz()
    config.ITO_THICKNESS_NM = float(tgt["ito_thickness_nm"])
    config.N_GLASS = float(tgt["glass_index"])
    lam0 = float(tgt["wavelength_nm"])
    rho_final = np.load(OUT / "geometries" / "rho_proj_final.npy")

    fig1_convergence(hist)
    fig2_geometry()
    fig3_target()
    _field_pair("Ez_scat_initial.npy", "initial")
    _field_pair("Ez_scat_final.npy", "final")
    fig6_comparison()

    lams = np.arange(1380.0, 1661.0, 40.0) if quick \
        else np.arange(1360.0, 1681.0, 10.0)
    sp = spectra(lams, rho_final, tgt, quick=quick)
    np.savez(OUT / "histories" / "spectra.npz", **sp)
    fig78_spectra(sp, lam0)

    convergence_checks(tgt, rho_final)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    main(quick=args.quick)
