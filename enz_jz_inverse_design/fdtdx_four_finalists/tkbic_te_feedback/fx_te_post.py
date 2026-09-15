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

    @property
    def files(self):
        return self.z.files


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
    out = dict(lam=lam, R=R, T=T, A=1 - R - T, leak_over_Pinc=leak / P_inc)
    if "ito_top_plane/phasor" in run["z"].files:
        # ITO absorption measured directly: net downward power entering the ITO from above minus the power leaving below
        # (S_z > 0 means +z; the wave travels in -z), normalized by the incident power.  Only lossless cells lie between
        # the two planes and the ITO, so this is A_ITO on the full spectral grid (the ZIP's Lumerical-family 'A_ito').
        S_top = plane_flux(run["z"]["ito_top_plane/phasor"], dxy)
        S_bot = plane_flux(run["z"]["ito_bot_plane/phasor"], dxy)
        out["A_ito"] = (S_bot - S_top) / P_inc
        out["A_ito_minus_A"] = out["A_ito"] - out["A"]
    return out


def ito_loss_check(run, ref, Te):
    """Absorbed fraction from the volume fields at the single field wavelength: (w/c) Im(eps) sum |E/E_inc|^2 dV / P^2
    per material (fx_post do_fields formula), to verify A = 1 - R - T is the ITO absorption."""
    m = run["meta"]; idx = m["idx"]; ze = np.array(m["z_edges_m"]); dz = np.diff(ze)
    lam_f = np.asarray(run["z"]["lam_field_m"]) * 1e9
    lam_s = np.asarray(ref["z"]["lam_spec_m"]) * 1e9
    return [_ito_loss_at(run, ref, Te, il, float(l), lam_s, m, idx, dz) for il, l in enumerate(lam_f)]


def _ito_loss_at(run, ref, Te, il, lamf, lam_s, m, idx, dz):
    # E_inc: the reference Ex phasor at the same plane (ITO mid-plane position, air) - the spectral-grid detector of the
    # reference run sits at that plane and contains every field wavelength (they are on the 1-nm grid)
    js = int(np.argmin(np.abs(lam_s - lamf))); assert abs(lam_s[js] - lamf) < 1e-6, (lamf, lam_s[js])
    E0 = ref["z"]["inc_ito_plane/phasor"][js, 0, :, :, 0]
    E0m = E0.mean()
    vol = run["z"]["vol_fields/phasor"][il] / E0m                        # (3, nx, ny, nz_vol)
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
    fam = dict(lambda_nm=lam_band, Te_K=[], R=[], T=[], A_rt=[], A_ito=[])
    full = dict(lambda_nm=None, R=[], T=[], A_rt=[], A_ito=[])
    checks = dict(runs={}, missing=[])
    for Te in TE_LIST:
        run = load_run(f"Te{Te:.0f}")
        if run is None:
            checks["missing"].append(Te); continue
        s = spectra_of(run, ref)
        sel = np.isin(np.round(s["lam"], 6), np.round(lam_band, 6))
        assert sel.sum() == len(lam_band)
        fam["Te_K"].append(Te)
        assert "A_ito" in s, "ITO flux planes missing"
        for k, sk in (("R", "R"), ("T", "T"), ("A_rt", "A"), ("A_ito", "A_ito")):
            fam[k].append(s[sk][sel]); full[k].append(s[sk])
        full["lambda_nm"] = s["lam"]
        lcs = ito_loss_check(run, ref, Te)
        for lc in lcs:
            lc["A_rt_from_flux"] = float(np.interp(lc["lambda_nm"], s["lam"], s["A"]))
            lc["A_ito_from_flux_planes"] = float(np.interp(lc["lambda_nm"], s["lam"], s["A_ito"]))
            lc["A_ito_planes_minus_F_ito_volume"] = lc["A_ito_from_flux_planes"] - lc["F_ito"]
            lc["A_rt_minus_F_ito_volume"] = lc["A_rt_from_flux"] - lc["F_ito"]
            lc["nonITO_over_A"] = (lc["F_aSiH_volblock"] + lc["F_glass_volblock"]) / max(lc["A_ito_from_flux_planes"], 1e-12)
        lc = lcs[0]; A_at = lc["A_rt_from_flux"]
        inb = (s["lam"] >= BAND[0]) & (s["lam"] <= BAND[1]); cut = (s["lam"] >= 1250) & (s["lam"] <= 1265)
        rt = dict(time_fs=run["meta"]["time_s"] * 1e15, n_steps=run["meta"]["n_steps_run"], finite=run["meta"]["finite"],
                  max_abs_E_final=run["meta"]["max_abs_E_final"], decay=decay_of(run),
                  closure=dict(min_R_band=float(s["R"][sel].min()), min_T_band=float(s["T"][sel].min()), max_T_band=float(s["T"][sel].max()), min_A_rt_band=float(s["A"][sel].min()),
                               max_A_rt_band=float(s["A"][sel].max()), min_A_ito_band=float(s["A_ito"][sel].min()), max_A_ito_band=float(s["A_ito"][sel].max()),
                               max_leak_over_Pinc=float(np.abs(s["leak_over_Pinc"]).max()),
                               A_ito_minus_A_rt=dict(max_abs_band=float(np.abs(s["A_ito_minus_A"][inb]).max()), rms_band=float(np.sqrt(np.mean(s["A_ito_minus_A"][inb] ** 2))),
                                                     max_abs_1250_1265=float(np.abs(s["A_ito_minus_A"][cut]).max()), max_abs_band_outside_1250_1265=float(np.abs(s["A_ito_minus_A"][inb & ~cut]).max()),
                                                     max_abs_1200_1400=float(np.abs(s["A_ito_minus_A"]).max()))),
                  ito_loss_check=lcs)
        checks["runs"][f"{Te:.0f}"] = rt
        print(f"[post] Te={Te:6.0f} K  {rt['time_fs']:.0f} fs  decay a-Si {rt['decay']['probe_asi_end_over_peak']:.1e} ITO {rt['decay']['probe_ito_end_over_peak']:.1e} | "
              f"band: T {s['T'][sel].min():.3f}..{s['T'][sel].max():.3f}  A_ito {s['A_ito'][sel].min():.3f}..{s['A_ito'][sel].max():.3f} | "
              f"|A_ito-(1-R-T)| band max {rt['closure']['A_ito_minus_A_rt']['max_abs_band']:.4f} (outside 1250-1265: {rt['closure']['A_ito_minus_A_rt']['max_abs_band_outside_1250_1265']:.4f}) | "
              + " ".join(f"{l['lambda_nm']:.0f}nm: A_rt {l['A_rt_from_flux']:.4f} A_ito {l['A_ito_from_flux_planes']:.4f} F_vol {l['F_ito']:.4f}" for l in lcs), flush=True)
    for k in ("R", "T", "A_rt", "A_ito"):
        fam[k] = np.array(fam[k]); full[k] = np.array(full[k])
    fam["Te_K"] = np.array(fam["Te_K"])
    checks["extras"] = extra_checks(ref, full)

    # --- 300 K vs the existing final3 FDTDX result (final3/prod/spectra_prod.csv, 300 fs, same grid)
    if 300.0 in fam["Te_K"]:
        old = np.genfromtxt(CAMP / "final3" / "prod" / "spectra_prod.csv", delimiter=",", names=True)
        lam_o = old["lambda_nm"]; i300 = list(fam["Te_K"]).index(300.0)
        common = np.isin(np.round(full["lambda_nm"], 6), np.round(lam_o, 6)); io = np.isin(np.round(lam_o, 6), np.round(full["lambda_nm"], 6))
        d = {k: full[kk][i300][common] - old[k][io] for k, kk in (("R", "R"), ("T", "T"), ("A", "A_rt"))}
        lam_c = full["lambda_nm"][common]; inb = (lam_c >= BAND[0]) & (lam_c <= BAND[1])
        checks["check_300K_vs_existing_final3_prod"] = {
            "note": "new 300 K run (400 fs, Te-refit ITO poles) minus existing final3/prod (300 fs, campaign ITO poles), same grid/source/detectors",
            **{f"max_abs_d{k}_1200_1400": float(np.abs(d[k]).max()) for k in d},
            **{f"max_abs_d{k}_1230_1280": float(np.abs(d[k][inb]).max()) for k in d},
            **{f"rms_d{k}_1230_1280": float(np.sqrt(np.mean(d[k][inb] ** 2))) for k in d},
            "A_1255_new_rt": float(np.interp(1255.0, full["lambda_nm"], full["A_rt"][i300])), "A_1255_existing": float(np.interp(1255.0, lam_o, old["A"]))}
        checks["old_300K"] = dict(lambda_nm=lam_o.tolist(), R=old["R"].tolist(), T=old["T"].tolist(), A=old["A"].tolist())
    # --- eps checks (the fit report, copied for the summary)
    checks["ito_fit_check"] = TE_MODELS["fit_check"]
    checks["lossless_other_media"] = dict(aSiH_gamma_rad_s=COLD_MODELS["aSiH"]["gamma_rad_s"], glass_gamma_rad_s=COLD_MODELS["glass"]["gamma_rad_s"],
                                          note="a-Si:H and glass are lossless Lorentz poles in the campaign model (ADE damping floor 1e9 rad/s); ITO is the only lossy medium")
    # --- save lookup tables
    np.savez(OUT / "final3_Te_family_fdtdx.npz", lambda_nm=fam["lambda_nm"], Te_K=fam["Te_K"], R=fam["R"], T=fam["T"], A=fam["A_ito"], A_rt=fam["A_rt"], A_ito=fam["A_ito"],
             lambda_full_nm=full["lambda_nm"], R_full=full["R"], T_full=full["T"], A_rt_full=full["A_rt"], A_ito_full=full["A_ito"],
             note="final3 (P825/h525/pad12%), FDTDX 600 fs, ITO eps(lam,Te) = eps_meas + delta-Drude(Te) refitted per Te; "
                  "A = A_ito (flux difference across the ITO film, used by the feedback); A_rt = 1 - R - T; rows = Te_K, cols = lambda_nm")
    with open(OUT / "final3_Te_family_fdtdx.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["Te_K", "lambda_nm", "R", "T", "A_ito", "A_rt_1_minus_R_minus_T"])
        for i, Te in enumerate(fam["Te_K"]):
            for j, l in enumerate(fam["lambda_nm"]):
                w.writerow([f"{Te:.0f}", f"{l:.1f}", f"{fam['R'][i, j]:.6f}", f"{fam['T'][i, j]:.6f}", f"{fam['A_ito'][i, j]:.6f}", f"{fam['A_rt'][i, j]:.6f}"])
    json.dump(checks, open(OUT / "em_checks.json", "w"), indent=1)
    # diagnostic: the same family with a 9-nm boxcar (one 600-fs ripple period) applied to every row
    sm = {k: np.array([boxcar(fam["lambda_nm"], fam[k][i]) for i in range(len(fam["Te_K"]))]) for k in ("R", "T", "A_rt", "A_ito")}
    np.savez(OUT / "diag_final3_Te_family_fdtdx_smooth9nm.npz", lambda_nm=fam["lambda_nm"], Te_K=fam["Te_K"], R=sm["R"], T=sm["T"], A=sm["A_ito"], A_rt=sm["A_rt"], A_ito=sm["A_ito"],
             note="DIAGNOSTIC ONLY: 9-nm boxcar (one DFT-truncation ripple period at 600 fs) applied to the rows of final3_Te_family_fdtdx.npz")
    make_figures(fam, full, checks)
    print(json.dumps({k: v for k, v in checks.items() if k not in ("old_300K", "ito_fit_check")}, indent=1)[:6000])


def boxcar(lam, y, width_nm=9.0):
    """Running mean over +-width/2 (one DFT-truncation ripple period at 600 fs) on the 1-nm part of the grid."""
    out = np.empty_like(y)
    for i, l in enumerate(lam):
        m = np.abs(lam - l) <= width_nm / 2
        out[i] = y[m].mean()
    return out


def _diff_stats(lam, a, b):
    d = a - b; inb = (lam >= BAND[0]) & (lam <= BAND[1]); cut = (lam >= 1250) & (lam <= 1265)
    ds = boxcar(lam, a) - boxcar(lam, b)
    return dict(max_abs_band=float(np.abs(d[inb]).max()), rms_band=float(np.sqrt(np.mean(d[inb] ** 2))),
                max_abs_1250_1265=float(np.abs(d[cut]).max()), max_abs_band_outside_1250_1265=float(np.abs(d[inb & ~cut]).max()),
                smoothed9nm_max_abs_band=float(np.abs(ds[inb]).max()), smoothed9nm_max_abs_outside_1250_1265=float(np.abs(ds[inb & ~cut]).max()),
                smoothed9nm_max_abs_1250_1265=float(np.abs(ds[cut]).max()))


def extra_checks(ref, full):
    """Time-convergence (400 / 600 / 900 fs) and pole-isolation (300 K, 300 fs) runs, when present."""
    out = {}
    base = {}
    for tag, Te in (("Te300", 300.0), ("Te8000", 8000.0)):
        r = load_run(tag)
        if r is not None:
            base[tag] = spectra_of(r, ref)
    for tag, ref_tag, label in (("Te300_400fs", "Te300", "300 K: 400 fs vs 600 fs"), ("Te8000_400fs", "Te8000", "8000 K: 400 fs vs 600 fs"),
                                ("Te8000_900fs", "Te8000", "8000 K: 900 fs vs 600 fs")):
        r = load_run(tag)
        if r is None or ref_tag not in base:
            continue
        s = spectra_of(r, ref); b = base[ref_tag]
        e = dict(label=label, time_fs=r["meta"]["time_s"] * 1e15, decay=decay_of(r), T=_diff_stats(s["lam"], s["T"], b["T"]), R=_diff_stats(s["lam"], s["R"], b["R"]),
                 A_rt=_diff_stats(s["lam"], s["A"], b["A"]))
        if "A_ito" in s and "A_ito" in b:
            e["A_ito"] = _diff_stats(s["lam"], s["A_ito"], b["A_ito"])
        e["spectra"] = dict(lambda_nm=s["lam"].tolist(), T=s["T"].tolist(), R=s["R"].tolist(), A_rt=s["A"].tolist(), **({"A_ito": s["A_ito"].tolist()} if "A_ito" in s else {}))
        out[tag] = e
    r = load_run("Te300_300fs")
    if r is not None:
        s = spectra_of(r, ref)
        old = np.genfromtxt(CAMP / "final3" / "prod" / "spectra_prod.csv", delimiter=",", names=True)
        common = np.isin(np.round(s["lam"], 6), np.round(old["lambda_nm"], 6)); io = np.isin(np.round(old["lambda_nm"], 6), np.round(s["lam"], 6))
        out["Te300_300fs_vs_existing_prod"] = dict(label="300 K, 300 fs, refit ITO poles vs campaign final3/prod (300 fs, campaign poles): isolates the pole refit",
                                                    decay=decay_of(r), **{k: _diff_stats(s["lam"][common], s[kk][common], old[k][io]) for k, kk in (("R", "R"), ("T", "T"), ("A", "A"))},
                                                    spectra=dict(lambda_nm=s["lam"].tolist(), T=s["T"].tolist(), R=s["R"].tolist(), A_rt=s["A"].tolist(), A_ito=s["A_ito"].tolist()))
    return out


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
    ax.axvspan(1250, 1265, color="0.88", zorder=0)
    ax.text(1257.5, 0.985, "glass (±1,0) cut-off 1251 nm:\nslow mode, truncation-limited", fontsize=6.5, ha="center", va="top", color="0.35", transform=ax.get_xaxis_transform())
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("T"); ax.set_xlim(lam[0], lam[-1]); ax.grid(alpha=0.3)
    ax.set_title(r"final3 (P825 / h525 / pad 12 %), FDTDX (600 fs): $T(\lambda; T_e)$, ITO $\varepsilon(\lambda,T_e)$ = measured + Kane delta-Drude", fontsize=9.5)
    ax.legend(fontsize=7.5, ncol=2, title="$T_e$", title_fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig1_T_lambda_Te_final3.png"); plt.close(fig)
    # Fig 2: 2D map, interpolated linearly in Te between the 9 computed rows (the lookup the feedback uses)
    Te_f = np.linspace(Te[0], Te[-1], 400)
    Tmap = np.array([np.interp(Te_f, Te, fam["T"][:, j]) for j in range(len(lam))]).T
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=130)
    im = ax.pcolormesh(lam, Te_f, Tmap, cmap="viridis", shading="auto", vmin=0, vmax=max(0.5, float(fam["T"].max())))
    for t in Te:
        ax.axhline(t, color="w", lw=0.6, alpha=0.7)
    ax.axvline(1250, color="w", lw=0.8, ls=":", alpha=0.8); ax.axvline(1265, color="w", lw=0.8, ls=":", alpha=0.8)
    ax.set_yticks(Te); ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel(r"$T_e$ (K)  (ticks = computed rows)")
    ax.set_title(f"final3, FDTDX (600 fs): $T(\\lambda, T_e)$ — linear interpolation between the {len(Te)} computed $T_e$ rows (white lines);\n"
                 "dotted: 1250-1265 nm glass cut-off region (truncation-limited)", fontsize=8.5)
    plt.colorbar(im, ax=ax, label="T")
    fig.tight_layout(); fig.savefig(OUT / "fig2_T_map_lambda_Te_final3.png"); plt.close(fig)
    # diagnostic: R, T, A_ito, 1-R-T on the full grid for all Te + 300 K comparison
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.2), dpi=120)
    for (k, lab), ax in zip((("R", "R"), ("T", "T"), ("A_ito", "A_ITO (flux difference across the ITO)"), ("A_rt", "1 - R - T")), axs):
        for i, (t, c) in enumerate(zip(Te, cols)):
            ax.plot(full["lambda_nm"], full[k][i], "-", color=c, lw=1.1, label=f"{t:.0f} K")
        if "old_300K" in checks and k in ("R", "T", "A_rt"):
            o = checks["old_300K"]; ax.plot(o["lambda_nm"], o["A" if k == "A_rt" else k], "k:", lw=1.2, label="300 K existing (300 fs)")
        ax.axvspan(*BAND, color="0.9", zorder=0); ax.axvspan(1250, 1265, color="0.8", zorder=0); ax.set_xlabel("λ [nm]"); ax.set_ylabel(lab); ax.grid(alpha=0.3)
    axs[0].legend(fontsize=6.5, ncol=2)
    fig.suptitle("final3 FDTDX Te family, full recorded band (diagnostic; dark band = 1250-1265 nm glass Rayleigh cut-off region)", fontsize=10); fig.tight_layout()
    fig.savefig(OUT / "diag_RTA_full_band.png"); plt.close(fig)
    # diagnostic: time convergence
    ex = checks.get("extras", {})
    if ex:
        fig, axs = plt.subplots(1, len(ex), figsize=(5.2 * len(ex), 4.0), dpi=120, squeeze=False)
        for ax, (tag, e) in zip(axs[0], ex.items()):
            sp = e["spectra"]; lam_e = np.array(sp["lambda_nm"])
            ax.plot(lam_e, sp["T"], "r-", lw=1.2, label=f"T, {tag}"); ax.plot(lam_e, sp["A_rt"], "k-", lw=1.0, label="1-R-T")
            if "A_ito" in sp:
                ax.plot(lam_e, sp["A_ito"], "b-", lw=1.0, label="A_ITO")
            key = "Te300" if tag.startswith("Te300") else "Te8000"
            if key in [f"Te{t:.0f}" for t in Te] and not tag.endswith("existing_prod"):
                i = list(Te).index(300.0 if key == "Te300" else 8000.0)
                ax.plot(full["lambda_nm"], full["T"][i], "r--", lw=1.0, label="T, 600 fs"); ax.plot(full["lambda_nm"], full["A_ito"][i], "b--", lw=1.0, label="A_ITO, 600 fs")
            if tag.endswith("existing_prod") and "old_300K" in checks:
                o = checks["old_300K"]; ax.plot(o["lambda_nm"], o["T"], "r--", lw=1.0, label="T existing prod"); ax.plot(o["lambda_nm"], o["A"], "k--", lw=1.0, label="A existing prod")
            ax.axvspan(*BAND, color="0.92", zorder=0); ax.axvspan(1250, 1265, color="0.82", zorder=0); ax.set_xlim(1220, 1300); ax.set_xlabel("λ [nm]"); ax.grid(alpha=0.3)
            ax.set_title(e["label"], fontsize=8.5); ax.legend(fontsize=6.5)
        fig.tight_layout(); fig.savefig(OUT / "diag_time_convergence.png"); plt.close(fig)


if __name__ == "__main__":
    main()
