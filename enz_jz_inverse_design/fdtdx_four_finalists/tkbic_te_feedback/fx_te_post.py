"""Post-processing of the Te FDTDX runs (runs/ref, runs/Te<K>): R, T, A = 1 - R - T spectra, the electromagnetic
sanity checks, the T/R/A lookup table for the thermal feedback, and the two spectral figures.

    <python> fx_te_post.py

Formulas are those of the campaign post-processor ../fx_post.py (plane_flux, dt scaling of pulse-mode phasors,
E / E_inc normalization, ITO volume-loss integral); copied here so that this study stays isolated from it.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CAMP = HERE.parent
sys.path.insert(0, str(HERE))
from ito_te_models import model_eps, eps_target, eps_cold_model, BAND      # noqa: E402

RUNS, OUT = HERE / "runs", HERE / "outputs"
C0 = 299792458.0
TE_MODELS = json.load(open(OUT / "ito_te_models.json"))
TE_LIST = [float(t) for t in TE_MODELS["Te_K"]]
COLD_MODELS = json.load(open(CAMP / "materials" / "material_models.json"))


# ----------------------------------------------------------------------------- loading (fx_post conventions)
class _ScaledPhasors:
    """FDTDX pulse-mode phasors are stride-weighted sums over time steps; multiplied by dt on access (fx_post)."""
    def __init__(self, z, dt):
        self.z, self.dt = z, dt

    def __getitem__(self, k):
        a = self.z[k]
        return a.astype(np.complex128) * self.dt if k.endswith("/phasor") else a


def load_run(tag):
    d = RUNS / tag
    if not (d / "phasors.npz").exists():
        return None
    meta = json.load(open(d / "run_meta.json"))
    return dict(z=_ScaledPhasors(np.load(d / "phasors.npz"), meta["dt_s"]), meta=meta, dir=d)


def plane_flux(ph, dxy):
    """ph: (Nlam, 4, nx, ny, 1) with Ex, Ey, Hx, Hy -> S_z(lam) = sum 1/2 Re(Ex Hy* - Ey Hx*) dA (fx_post.plane_flux)."""
    Ex, Ey, Hx, Hy = ph[:, 0, :, :, 0], ph[:, 1, :, :, 0], ph[:, 2, :, :, 0], ph[:, 3, :, :, 0]
    return 0.5 * np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx)).sum(axis=(1, 2)) * dxy ** 2


def spectra_of(run, ref):
    lam = run["z"]["lam_spec_m"] * 1e9
    assert np.allclose(lam, ref["z"]["lam_spec_m"] * 1e9)
    dxy = run["meta"]["dxy_m"]
    P_inc = -plane_flux(ref["z"]["T_plane/phasor"], dxy)
    leak = plane_flux(ref["z"]["R_plane/phasor"], dxy)
    T = -plane_flux(run["z"]["T_plane/phasor"], dxy) / P_inc
    R = (plane_flux(run["z"]["R_plane/phasor"], dxy) - leak) / P_inc
    return dict(lam=lam, R=R, T=T, A=1 - R - T, leak_over_Pinc=leak / P_inc)


def ito_loss_check(run, ref, Te):
    """Absorbed fraction from the volume fields at the single field wavelength: (w/c) Im(eps) sum |E/E_inc|^2 dV / P^2
    per material (fx_post do_fields formula), to verify A = 1 - R - T is the ITO absorption."""
    m = run["meta"]; idx = m["idx"]; ze = np.array(m["z_edges_m"]); dz = np.diff(ze)
    lamf = float(run["z"]["lam_field_m"][0]) * 1e9
    assert abs(float(ref["z"]["lam_field_m"][0]) * 1e9 - lamf) < 1e-6
    E0 = ref["z"]["inc_field_plane/phasor"][0, 0, :, :, 0]
    E0m = E0.mean()
    vol = run["z"]["vol_fields/phasor"][0] / E0m                         # (3, nx, ny, nz_vol)
    I2 = np.abs(vol) ** 2
    zv0 = idx["z_vol0"]
    i0, i1 = idx["z_ito0"] - zv0, idx["z_asi0"] - zv0
    a0, a1 = idx["z_asi0"] - zv0, idx["z_asi1"] - zv0
    g0 = 0
    dz_v = dz[zv0:idx["z_vol1"]]
    w = 2 * np.pi * C0 / (lamf * 1e-9); dxy = m["dxy_m"]; P = m["P_m"]
    eps_ito = model_eps(TE_MODELS["models"][f"{Te:.0f}"], lamf)           # the eps FDTDX actually used
    from ito_te_models import eps_drude_lorentz
    ca, cg = COLD_MODELS["aSiH"], COLD_MODELS["glass"]
    eps_a = eps_drude_lorentz(lamf, ca["eps_inf"], 0.0, 1e12, ca["delta_eps"], ca["w0_rad_s"], max(ca["gamma_rad_s"], 1e9))
    eps_g = eps_drude_lorentz(lamf, cg["eps_inf"], 0.0, 1e12, cg["delta_eps"], cg["w0_rad_s"], max(cg["gamma_rad_s"], 1e9))
    sys.path.insert(0, str(CAMP))
    import importlib
    fill = importlib.import_module("field_stack_plots").area_fraction(np.load(CAMP / "final3" / "rho_hard_binary.npy"), m["nxy"])

    def loss(eps, zsl, weight=None):
        dV = dxy ** 2 * dz_v[zsl]
        wgt = 1.0 if weight is None else weight[:, :, None]
        return float((w / C0) * eps.imag * np.sum(I2[:, :, :, zsl].sum(0) * wgt * dV[None, None, :]) / P ** 2)
    F_ito = loss(eps_ito, slice(i0, i1))
    F_asi = loss(eps_a, slice(a0, a1), fill)
    F_glass = loss(eps_g, slice(g0, i0))
    return dict(lambda_nm=lamf, F_ito=F_ito, F_aSiH_volblock=F_asi, F_glass_volblock=F_glass,
                eps_ito_used=[float(eps_ito.real), float(eps_ito.imag)], Im_eps_aSiH=float(eps_a.imag), Im_eps_glass=float(eps_g.imag),
                E_inc_nonuniformity=float(E0.std() / abs(E0m)))


def decay_of(run):
    r = {}
    for k in ("probe_asi", "probe_ito"):
        f = run["z"][f"{k}/fields"]; a = np.abs(f).max(axis=1); pk = a.max()
        r[f"{k}_end_over_peak"] = float(a[-50:].max() / pk)
        r[f"{k}_peak_time_fs"] = float(np.argmax(a) * run["meta"]["dt_s"] * 1e15)
    return r


# ----------------------------------------------------------------------------- main
def main():
    ref = load_run("ref"); assert ref is not None, "reference run missing"
    lam_band = np.arange(BAND[0], BAND[1] + 0.01, 1.0)
    fam = dict(lambda_nm=lam_band, Te_K=[], R=[], T=[], A=[])
    full = dict(lambda_nm=None, R=[], T=[], A=[])
    checks = dict(runs={}, missing=[])
    for Te in TE_LIST:
        run = load_run(f"Te{Te:.0f}")
        if run is None:
            checks["missing"].append(Te); continue
        s = spectra_of(run, ref)
        sel = np.isin(np.round(s["lam"], 6), np.round(lam_band, 6))
        assert sel.sum() == len(lam_band)
        fam["Te_K"].append(Te)
        for k in ("R", "T", "A"):
            fam[k].append(s[k][sel]); full[k].append(s[k])
        full["lambda_nm"] = s["lam"]
        lc = ito_loss_check(run, ref, Te)
        A_at = float(np.interp(lc["lambda_nm"], s["lam"], s["A"]))
        rt = dict(time_fs=run["meta"]["time_s"] * 1e15, n_steps=run["meta"]["n_steps_run"], finite=run["meta"]["finite"],
                  max_abs_E_final=run["meta"]["max_abs_E_final"], decay=decay_of(run),
                  closure=dict(max_abs_1_minus_RTA=float(np.abs(1 - s["R"] - s["T"] - s["A"]).max()),
                               min_R_band=float(s["R"][sel].min()), min_T_band=float(s["T"][sel].min()), min_A_band=float(s["A"][sel].min()),
                               max_A_band=float(s["A"][sel].max()), max_leak_over_Pinc=float(np.abs(s["leak_over_Pinc"]).max())),
                  ito_loss_check=dict(lc, A_from_flux=A_at, A_minus_F_ito=A_at - lc["F_ito"],
                                      nonITO_over_A=(lc["F_aSiH_volblock"] + lc["F_glass_volblock"]) / max(A_at, 1e-12)))
        checks["runs"][f"{Te:.0f}"] = rt
        print(f"[post] Te={Te:6.0f} K  {rt['time_fs']:.0f} fs  decay a-Si {rt['decay']['probe_asi_end_over_peak']:.1e} ITO {rt['decay']['probe_ito_end_over_peak']:.1e} | "
              f"band: T {s['T'][sel].min():.3f}..{s['T'][sel].max():.3f}  A {s['A'][sel].min():.3f}..{s['A'][sel].max():.3f} | "
              f"A(1255)={A_at:.4f} vs ITO loss integral {lc['F_ito']:.4f} (a-Si {lc['F_aSiH_volblock']:.1e}, glass {lc['F_glass_volblock']:.1e})", flush=True)
    for k in ("R", "T", "A"):
        fam[k] = np.array(fam[k]); full[k] = np.array(full[k])
    fam["Te_K"] = np.array(fam["Te_K"])

    # --- 300 K vs the existing final3 FDTDX result (final3/prod/spectra_prod.csv, 300 fs, same grid)
    if 300.0 in fam["Te_K"]:
        old = np.genfromtxt(CAMP / "final3" / "prod" / "spectra_prod.csv", delimiter=",", names=True)
        lam_o = old["lambda_nm"]; i300 = list(fam["Te_K"]).index(300.0)
        common = np.isin(np.round(full["lambda_nm"], 6), np.round(lam_o, 6)); io = np.isin(np.round(lam_o, 6), np.round(full["lambda_nm"], 6))
        d = {k: full[k][i300][common] - old[k][io] for k in ("R", "T", "A")}
        lam_c = full["lambda_nm"][common]; inb = (lam_c >= BAND[0]) & (lam_c <= BAND[1])
        checks["check_300K_vs_existing_final3_prod"] = {
            "note": "new 300 K run (400 fs, Te-refit ITO poles) minus existing final3/prod (300 fs, campaign ITO poles), same grid/source/detectors",
            **{f"max_abs_d{k}_1200_1400": float(np.abs(d[k]).max()) for k in d},
            **{f"max_abs_d{k}_1230_1280": float(np.abs(d[k][inb]).max()) for k in d},
            **{f"rms_d{k}_1230_1280": float(np.sqrt(np.mean(d[k][inb] ** 2))) for k in d},
            "A_1255_new": float(np.interp(1255.0, full["lambda_nm"], full["A"][i300])), "A_1255_existing": float(np.interp(1255.0, lam_o, old["A"]))}
        checks["old_300K"] = dict(lambda_nm=lam_o.tolist(), R=old["R"].tolist(), T=old["T"].tolist(), A=old["A"].tolist())
    # --- eps checks (the fit report, copied for the summary)
    checks["ito_fit_check"] = TE_MODELS["fit_check"]
    checks["lossless_other_media"] = dict(aSiH_gamma_rad_s=COLD_MODELS["aSiH"]["gamma_rad_s"], glass_gamma_rad_s=COLD_MODELS["glass"]["gamma_rad_s"],
                                          note="a-Si:H and glass are lossless Lorentz poles in the campaign model (ADE damping floor 1e9 rad/s); ITO is the only lossy medium")
    # --- save lookup tables
    np.savez(OUT / "final3_Te_family_fdtdx.npz", lambda_nm=fam["lambda_nm"], Te_K=fam["Te_K"], R=fam["R"], T=fam["T"], A=fam["A"],
             lambda_full_nm=full["lambda_nm"], R_full=full["R"], T_full=full["T"], A_full=full["A"],
             note="final3 (P825/h525/pad12%), FDTDX, ITO eps(lam,Te) = eps_meas + delta-Drude(Te) refitted per Te; A = 1 - R - T; rows = Te_K, cols = lambda_nm")
    with open(OUT / "final3_Te_family_fdtdx.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["Te_K", "lambda_nm", "R", "T", "A"])
        for i, Te in enumerate(fam["Te_K"]):
            for j, l in enumerate(fam["lambda_nm"]):
                w.writerow([f"{Te:.0f}", f"{l:.1f}", f"{fam['R'][i, j]:.6f}", f"{fam['T'][i, j]:.6f}", f"{fam['A'][i, j]:.6f}"])
    json.dump(checks, open(OUT / "em_checks.json", "w"), indent=1)
    make_figures(fam, full, checks)
    print(json.dumps({k: v for k, v in checks.items() if k not in ("old_300K", "ito_fit_check")}, indent=1)[:6000])


def make_figures(fam, full, checks):
    Te = fam["Te_K"]; lam = fam["lambda_nm"]
    cols = plt.cm.plasma(np.linspace(0.05, 0.9, len(Te)))
    # Fig 1: T(lambda; Te)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=130)
    for i, (t, c) in enumerate(zip(Te, cols)):
        ax.plot(lam, fam["T"][i], "-", color=c, lw=1.6, label=f"{t:.0f} K")
    if "old_300K" in checks:
        o = checks["old_300K"]; lo = np.array(o["lambda_nm"]); m = (lo >= lam[0]) & (lo <= lam[-1])
        ax.plot(lo[m], np.array(o["T"])[m], "k:", lw=1.2, label="300 K, existing final3/prod")
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("T"); ax.set_xlim(lam[0], lam[-1]); ax.grid(alpha=0.3)
    ax.set_title(r"final3 (P825 / h525 / pad 12 %), FDTDX: $T(\lambda; T_e)$, ITO $\varepsilon(\lambda,T_e)$ = measured + Kane delta-Drude", fontsize=9.5)
    ax.legend(fontsize=7.5, ncol=2, title="$T_e$", title_fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig1_T_lambda_Te_final3.png"); plt.close(fig)
    # Fig 2: 2D map, interpolated linearly in Te between the 9 computed rows (the lookup the feedback uses)
    Te_f = np.linspace(Te[0], Te[-1], 400)
    Tmap = np.array([np.interp(Te_f, Te, fam["T"][:, j]) for j in range(len(lam))]).T
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=130)
    im = ax.pcolormesh(lam, Te_f, Tmap, cmap="viridis", shading="auto", vmin=0, vmax=max(0.5, float(fam["T"].max())))
    for t in Te:
        ax.axhline(t, color="w", lw=0.5, alpha=0.6)
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel(r"$T_e$ (K)")
    ax.set_title(r"final3, FDTDX: $T(\lambda, T_e)$ (linear interpolation between the 9 computed $T_e$ rows, white lines)", fontsize=9)
    plt.colorbar(im, ax=ax, label="T")
    fig.tight_layout(); fig.savefig(OUT / "fig2_T_map_lambda_Te_final3.png"); plt.close(fig)
    # diagnostic: R, T, A on the full grid for all Te + 300 K comparison
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), dpi=120)
    for k, ax in zip(("R", "T", "A"), axs):
        for i, (t, c) in enumerate(zip(Te, cols)):
            ax.plot(full["lambda_nm"], full[k][i], "-", color=c, lw=1.1, label=f"{t:.0f} K")
        if "old_300K" in checks:
            o = checks["old_300K"]; ax.plot(o["lambda_nm"], o[k], "k:", lw=1.2, label="300 K existing")
        ax.axvspan(*BAND, color="0.9", zorder=0); ax.set_xlabel("λ [nm]"); ax.set_ylabel(k); ax.grid(alpha=0.3)
    axs[0].legend(fontsize=6.5, ncol=2)
    fig.suptitle("final3 FDTDX Te family, full recorded band (diagnostic)", fontsize=10); fig.tight_layout()
    fig.savefig(OUT / "diag_RTA_full_band.png"); plt.close(fig)


if __name__ == "__main__":
    main()
