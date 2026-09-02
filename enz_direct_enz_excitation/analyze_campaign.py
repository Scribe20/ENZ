"""Post-optimization analysis for the direct-ENZ-excitation campaign.

Compares OLD (padded-85nm QNM-target winner) vs NEW (padded-85nm F_ENZ
winner) under identical machinery; produces the headline table, order
convergence, F_ENZ/eta_z/T/R/A spectra, with/without-ITO control, field
maps, post-hoc QNM-overlap and Fourier-content diagnostics, and figures.

Run:  python analyze_campaign.py     (after run_campaign.py)
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

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / "enz_inverse_design"
PAD = HERE.parent / "enz_padding_sideexperiment"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PAD))

import config                        # noqa: E402
config.OBJECTIVE = "ito_ez_volume"
import target_mode                   # noqa: E402
import torcwa_forward as fwd         # noqa: E402
import objective as obj              # noqa: E402
import compare_padded as cp          # noqa: E402  (reused analysis machinery)
from validate_with_without_ito import build_sim, power_RT, spectrum  # noqa: E402

OUT = HERE / "outputs"
FIG = OUT / "figures"
# route the reused figure writers into THIS campaign's folders
cp.OUT, cp.FIG = OUT, FIG

C_BLUE, C_ORANGE, C_AQUA, C_VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": "#e3e2df", "grid.linewidth": 0.6, "font.size": 10,
    "lines.linewidth": 2.0, "legend.frameon": False})


def _save(fig, name):
    FIG.mkdir(exist_ok=True, parents=True)
    fig.savefig(FIG / name, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"figure written: {FIG/name}")


V_FACTOR = None   # F_ENZ = F_Ez(B_ITO) * p_inc / V_ITO, set in main


def full_point_metrics(rho_t, ctx, order=None):
    m = cp.optical_metrics(rho_t, ctx, order=order)
    m["F_ENZ"] = m["F_Ez"] * V_FACTOR
    return m


def enz_spectrum(rho_t, ctx, lams):
    """F_ENZ, eta_z, T, R, A vs wavelength (dispersive materials)."""
    out = {k: [] for k in ("lam", "F_ENZ", "eta_z", "T", "R", "A")}
    with torch.no_grad():
        for lam in lams:
            eps_ito = fwd.eps_ito_of_lambda(lam)
            eps_asi = fwd.eps_asi_of_lambda(lam)
            sim = fwd.build_solved_sim(rho_t, lam, eps_ito, config.N_GLASS,
                                       eps_asi=eps_asi)
            Ex, Ey, Ez = cp.e_in_ito(sim, ctx)
            Iz = float(torch.sum(torch.abs(Ez) ** 2).real * ctx["dV"])
            It = float(torch.sum(torch.abs(Ex) ** 2
                                 + torch.abs(Ey) ** 2).real * ctx["dV"])
            R, T = power_RT(build_sim(rho_t, lam, True))
            v_ito = config.PX_NM * config.PY_NM * config.ITO_THICKNESS_NM
            out["lam"].append(lam)
            out["F_ENZ"].append(Iz / v_ito)
            out["eta_z"].append(Iz / (Iz + It))
            out["T"].append(T); out["R"].append(R); out["A"].append(1 - R - T)
    return {k: np.array(v) for k, v in out.items()}


def fourier_content(rho_t, ctx):
    """|Ez| harmonic amplitudes (rms over z) in the ITO at lambda_E."""
    sim = fwd.build_solved_sim(rho_t, ctx["lam"], ctx["eps_ito"],
                               config.N_GLASS)
    n = 96
    xs = torch.as_tensor((np.arange(n) + 0.5) / n * config.PX_NM,
                         dtype=config.GEO_DTYPE)
    ys = torch.as_tensor((np.arange(n) + 0.5) / n * config.PY_NM,
                         dtype=config.GEO_DTYPE)
    acc = 0.0
    with torch.no_grad():
        for zpv in ctx["zp"]:
            E, _ = sim.field_xy(1, xs, ys, float(zpv))
            F2 = np.fft.fft2(E[2].cpu().numpy()) / n ** 2
            acc = acc + np.abs(F2) ** 2
    acc = np.sqrt(acc / len(ctx["zp"]))
    chans = {}
    for m in range(-4, 5):
        for nn in range(-4, 5):
            v = float(acc[m % n, nn % n])
            if v > 1e-8:
                chans[f"({m},{nn})"] = v
    return dict(sorted(chans.items(), key=lambda kv: -kv[1])[:12])


def main():
    global V_FACTOR
    ctx = cp.context()
    p_inc = ctx["p_inc"]
    v_ito = config.PX_NM * config.PY_NM * config.ITO_THICKNESS_NM
    V_FACTOR = p_inc / v_ito

    rho_old = np.load(PAD / "outputs" / "geometries" / "rho_hard_binary.npy")
    rho_new = np.load(OUT / "geometries" / "rho_hard_binary.npy")
    to = torch.as_tensor(rho_old, dtype=config.GEO_DTYPE)
    tn = torch.as_tensor(rho_new, dtype=config.GEO_DTYPE)

    # soft/projected/hard convention report for the NEW winner
    rho_soft = np.load(OUT / "geometries" / "rho_proj_final.npy")
    ts = torch.as_tensor(rho_soft, dtype=config.GEO_DTYPE)
    f_soft = full_point_metrics(ts, ctx)["F_ENZ"]

    rows = {}
    for tag, rb, rt in (("old_qnm_target", rho_old, to),
                        ("new_f_enz", rho_new, tn)):
        m = full_point_metrics(rt, ctx)
        loc = cp.locality_metrics(rb, m.pop("Iz_xy"))
        m.pop("a_plus"), m.pop("a_minus")
        rows[tag] = {**m, **loc}
        print(f"[{tag}] F_ENZ = {m['F_ENZ']:.4f}, eta_z = {m['eta_z']:.3f}, "
              f"F_QNM(diag) = {m['F_QNM']:.2f}, eta_pm = {m['eta_pm']:.3f}, "
              f"T/R/A = {m['T']:.3f}/{m['R']:.3f}/{m['A']:.3f}")
        print(f"[{tag}] locality: {loc}")
    print(f"[convention] NEW winner: soft-final F_ENZ = {f_soft:.4f}, "
          f"hard-binary = {rows['new_f_enz']['F_ENZ']:.4f}")

    # order convergence
    conv = {}
    for od in ([7, 7], [9, 9], [11, 11]):
        old_o = config.FOURIER_ORDER
        config.FOURIER_ORDER = od
        with torch.no_grad():
            Ez_ref_o = fwd.ez_in_ito(fwd.build_solved_sim(
                None, ctx["lam"], ctx["eps_ito"], config.N_GLASS),
                ctx["x"], ctx["y"], ctx["zp"]).detach()
        config.FOURIER_ORDER = old_o
        ctx_o = {**ctx, "Ez_ref": Ez_ref_o}
        for tag, rt in (("old", to), ("new", tn)):
            m = full_point_metrics(rt, ctx_o, order=od)
            conv[f"{tag}_{od}"] = {k: m[k] for k in
                                   ("F_ENZ", "eta_z", "T", "R", "A")}
            print(f"[conv] {tag} {od}: F_ENZ = {m['F_ENZ']:.4f}, "
                  f"T = {m['T']:.4f}, A = {m['A']:.4f}")
    with open(OUT / "histories" / "convergence.json", "w") as f:
        json.dump(conv, f, indent=1, default=float)

    pd.DataFrame(rows).to_csv(OUT / "headline_comparison.csv")
    print(f"[saved] {OUT/'headline_comparison.csv'}")

    # spectra around ENZ
    lams = np.arange(1350.0, 1550.5, 2.0)
    print("[spectra] new winner ..."); spN = enz_spectrum(tn, ctx, lams)
    print("[spectra] old winner ..."); spO = enz_spectrum(to, ctx, lams)
    np.savez(OUT / "histories" / "enz_spectra.npz", lam=lams,
             **{f"new_{k}": v for k, v in spN.items() if k != "lam"},
             **{f"old_{k}": v for k, v in spO.items() if k != "lam"})
    fig, axs = plt.subplots(1, 3, figsize=(14.6, 4.0))
    axs[0].plot(lams, spO["F_ENZ"], color=C_BLUE, label="old (QNM target)")
    axs[0].plot(lams, spN["F_ENZ"], color=C_ORANGE, label="new (F_ENZ)")
    axs[0].set_ylabel(r"$F_{ENZ}(\lambda) = \langle|E_z/E_{inc}|^2\rangle$")
    axs[1].plot(lams, spO["eta_z"], color=C_BLUE)
    axs[1].plot(lams, spN["eta_z"], color=C_ORANGE)
    axs[1].set_ylabel(r"$\eta_z$")
    axs[2].plot(lams, spO["A"], color=C_BLUE)
    axs[2].plot(lams, spN["A"], color=C_ORANGE)
    axs[2].set_ylabel("A (= ITO absorption)")
    for ax in axs:
        ax.axvline(ctx["lam"], color=C_AQUA, ls="--", lw=1.0)
        ax.set_xlabel("wavelength (nm)")
    axs[0].legend(fontsize=9)
    fig.suptitle("old vs new objective, same padded design class "
                 "(dashed: 1433.5 nm)", y=1.02)
    _save(fig, "enz_spectra.png")

    # with/without ITO for the new winner
    lam_b = np.arange(1200.0, 1700.5, 2.0)
    print("[spectra] new winner, no ITO ...")
    spA = spectrum(tn, lam_b, False, tag="new-noITO")
    print("[spectra] new winner, with ITO ...")
    spB = spectrum(tn, lam_b, True, tag="new-ITO")
    np.savez(OUT / "histories" / "with_without_ito.npz", lam=lam_b,
             T_no=spA["T"], R_no=spA["R"], T_ito=spB["T"], R_ito=spB["R"])
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(lam_b, spA["T"], color=C_BLUE, label="new winner / glass")
    ax.plot(lam_b, spB["T"], color=C_ORANGE, label="new winner / ITO / glass")
    ax.axvline(1419.59, color=C_VIOLET, ls="--", lw=1.0)
    ax.axvline(ctx["lam"], color=C_AQUA, ls="--", lw=1.0)
    ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("T"); ax.set_ylim(0, 1.02)
    ax.set_title("new F_ENZ winner: with vs without ITO (frozen geometry)")
    ax.legend(loc="lower left", fontsize=9)
    _save(fig, "with_without_ITO.png")
    print(f"[closure] max|A| no-ITO = {np.abs(spA['A']).max():.2e}")

    # geometry + history + Ez maps
    fig, axs = plt.subplots(1, 2, figsize=(9.8, 4.6), sharey=True)
    for ax, r, t in ((axs[0], rho_old, "OLD: QNM-target winner"),
                     (axs[1], rho_new, "NEW: F_ENZ winner")):
        ax.imshow(r.T, origin="lower", cmap="Greys",
                  extent=[0, config.PX_NM, 0, config.PY_NM])
        ax.add_patch(plt.Rectangle((85, 85), config.PX_NM - 170,
                                   config.PY_NM - 170, fill=False,
                                   ec=C_ORANGE, lw=1.2, ls="--"))
        ax.set_title(t); ax.set_xlabel("x (nm)"); ax.grid(False)
    axs[0].set_ylabel("y (nm)")
    _save(fig, "geometry_old_vs_new.png")

    with open(OUT / "histories" / "history.json") as f:
        hist = json.load(f)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.semilogy(hist["F"], color=C_ORANGE)
    ax.set_xlabel("iteration"); ax.set_ylabel(r"$F_{ENZ}$")
    ax.axhline(rows["old_qnm_target"]["F_ENZ"], color=C_BLUE, ls="--",
               label="old winner (evaluated under F_ENZ)")
    ax.legend(fontsize=9)
    ax.set_title("direct-F_ENZ optimization history")
    _save(fig, "history.png")

    # ITO |Ez/Einc|^2 maps, same scale
    maps = {}
    for tag, rt in (("old", to), ("new", tn)):
        sim = fwd.build_solved_sim(rt, ctx["lam"], ctx["eps_ito"],
                                   config.N_GLASS)
        with torch.no_grad():
            _, _, Ez = cp.e_in_ito(sim, ctx)
        maps[tag] = torch.mean(torch.abs(Ez) ** 2, dim=0).cpu().numpy()
    vmax = max(m.max() for m in maps.values())
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)
    for ax, tag in zip(axs, ("old", "new")):
        im = ax.imshow(maps[tag].T, origin="lower", cmap="Blues", vmin=0,
                       vmax=vmax, extent=[0, config.PX_NM, 0, config.PY_NM])
        ax.set_title(f"{tag}: z-avg $|E_z/E_{{inc}}|^2$ in ITO")
        ax.set_xlabel("x (nm)"); ax.grid(False)
    axs[0].set_ylabel("y (nm)")
    fig.colorbar(im, ax=axs, shrink=0.8)
    _save(fig, "ito_ez_maps.png")

    # x-z field map + current maps (reused writer)
    cp.maps_figure(rho_new, tn, ctx, "new_f_enz_winner")

    # Fourier content of the driven ITO Ez
    fc = {"old": fourier_content(to, ctx), "new": fourier_content(tn, ctx)}
    with open(OUT / "histories" / "fourier_content.json", "w") as f:
        json.dump(fc, f, indent=1)
    print("[fourier] dominant |Ez| harmonics (rms over ITO z):")
    for tag in ("old", "new"):
        print(f"  {tag}: " + ", ".join(f"{k}:{v:.3f}"
                                       for k, v in list(fc[tag].items())[:6]))
    return rows


if __name__ == "__main__":
    main()
