"""Thermal (TTM) feedback on the final3 FDTDX T/A(lambda, Te) lookup table, following the TKBIC ZIP:

    C_e(Te) dTe/dt = A[lam, Te(t)] I(t) / d_ITO - g (Te - T_l),   C_L dT_l/dt = g (Te - T_l)

integrated with the ZIP's own `integrate_ttm` (tkbic_ref/ttm/_te_fluence_map.py, vendored unmodified; batched RK4,
dt = 1 fs, window -1..+3 ps), with the ZIP conventions of tkbic_te_feedback_demo.py section 8:
  * pulse: Gaussian intensity envelope, FWHM 150 fs, int I dt = 1, times the fluence F (module SIG overridden as in the demo),
  * C_e = gamma_e Te (Sommerfeld, HeatCapacity("sommerfeld"), gamma_e = 5.233 J m^-3 K^-2), C_L = 2.6e6 J m^-3 K^-1,
  * g = C_e / tau_ep, tau_ep = 450 fs (ZIP experimental anchor); 1000 fs and g = 0 kept as the model band,
  * A(Te), T(Te): linear interpolation in lambda, then linear in Te, clipped to the table range [300, 8000] K
    (demo `_at` / section-8 Afun); A is re-evaluated from Te(t) inside every RK4 stage,
  * peak intensity -> fluence: F = I_peak * tau_eff, tau_eff = 150 fs * sqrt(pi/2) (demo section 6 `transfer`),
  * intensity grid of the T(I) figure: I = logspace(6, 10.2, 60) W/cm^2 (demo section 7),
  * T_eff = int I(t) T[lam, Te(t)] dt / int I(t) dt  (integrate_ttm probe average `avg["T"]`).
Wavelength selection uses the demo's section-6 metrics (logistic fit on E_in = 0..2.4 nJ per (8 um)^2 pixel) with the
ZIP's `_sel` rules; labels are only assigned when the ZIP criteria are actually met.

    <python> te_feedback.py [--family outputs/final3_Te_family_fdtdx.npz] [--tau 450]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tkbic_ref" / "ttm"))
import _te_fluence_map as ttm                                        # noqa: E402  (ZIP module, unmodified)

OUT = HERE / "outputs"
TAU_EFF = 150e-15 * np.sqrt(np.pi / 2)                                # demo section 6 (notebook convention)
A_PIX_CM2 = (8e-4) ** 2                                               # (8 um)^2 pixel
I_GRID = np.logspace(6.0, 10.2, 60)                                   # W/cm^2, demo section 7
E_SCAN = np.linspace(0.0, 2.4, 49)                                    # nJ, demo section 6 lambda_p scan
TAU_PRIMARY_FS = 450.0
MODELS = {"tau450": ("tau", 450e-15), "tau1000": ("tau", 1000e-15), "g0": ("const", 0.0)}


def I_to_F_mJcm2(I_Wcm2):
    """peak intensity [W/cm^2] -> fluence [mJ/cm^2] with the ZIP rule F[J/m^2] = I * 1e4 * tau_eff."""
    return np.asarray(I_Wcm2, float) * 1e4 * TAU_EFF / 10.0


def E_to_I(E_nJ):
    """demo: I_scan = E * 1e-9 / (tau_eff * A_pix), E = 0 -> 1e3 W/cm^2."""
    E = np.asarray(E_nJ, float)
    return np.where(E > 0, E * 1e-9 / (TAU_EFF * A_PIX_CM2), 1e3)


class Lookup:
    """`_at`-style interpolation of the family: linear in lambda, then linear in Te, clipped to [Te0, Te1]."""
    def __init__(self, lam, Te, T, A, R=None):
        self.lam, self.Te, self.T, self.A, self.R = map(np.asarray, (lam, Te, T, A, R if R is not None else np.zeros_like(T)))

    def fns(self, lp):
        A_col = np.array([np.interp(lp, self.lam, self.A[i]) for i in range(len(self.Te))])
        T_col = np.array([np.interp(lp, self.lam, self.T[i]) for i in range(len(self.Te))])
        Afun = lambda Te: np.interp(np.clip(Te, self.Te[0], self.Te[-1]), self.Te, A_col)      # noqa: E731
        Tfun = lambda Te: np.interp(np.clip(Te, self.Te[0], self.Te[-1]), self.Te, T_col)      # noqa: E731
        return Afun, Tfun, A_col, T_col


def run_ttm(look, lp, F_mJ, g_model, Ce):
    Afun, Tfun, _, _ = look.fns(lp)
    r = ttm.integrate_ttm(np.asarray(F_mJ, float), Afun, Ce, g_model, probe_fns={"T": Tfun, "A": Afun}, dt_fs=1.0,
                          Te_table_max=look.Te[-1])
    return dict(Teff=r["avg"]["T"], Aeff_avgI=r["avg"]["A"], A_eff=r["A_eff"], Te_pk=r["Te_pk"], Te_avgI=r["Te_avgI"],
                Te_end=r["Te_end"], Tl_end=r["Tl_end"], above_table=r["above_table"], T_cold=Tfun(300.0), A_cold=Afun(300.0))


# ----------------------------------------------------------------------------- ZIP section-6 metrics (verbatim)
def fit_logistic(E, T):
    def _r(x):
        c, d, E0, w = x
        return c + d / (1 + np.exp(-(E - E0) / max(w, 1e-3))) - T
    dT0 = T[-1] - T[0]
    try:
        f = least_squares(_r, [T[0], dT0, 0.7, 0.3], bounds=([-0.2, -1.2, 0.0, 0.02], [1.2, 1.2, 3.0, 3.0]))
        ss = 1 - np.sum(_r(f.x) ** 2) / max(np.sum((T - T.mean()) ** 2), 1e-12)
        return f.x, float(ss)
    except Exception:
        return None, -1.0


def metrics_of(lp, E_scan, T_r):
    par, r2 = fit_logistic(E_scan, T_r)
    E0f, wf = (par[2], par[3]) if par is not None else (np.nan, np.nan)
    dT_end = float(T_r[-1] - T_r[0])
    i16 = int(np.argmin(np.abs(E_scan - 1.6)))
    fsat = float((T_r[i16] - T_r[0]) / dT_end) if abs(dT_end) > 1e-6 else 0.0
    return dict(lp=float(lp), T0=float(T_r[0]), dT=dT_end, E0=float(E0f), w=float(wf), r2=r2, fsat=fsat)


def select(mets):
    """ZIP `_sel` rules.  Returns {label: metric or None} with the strict criteria; a separate 'fallback' flag marks
    the demo's relaxed fallbacks so that no behaviour label is forced onto final3."""
    def _sel(cond, score):
        cand = [m for m in mets if cond(m)]
        return max(cand, key=score) if cand else None
    strict = {
        "sigmoid-like": _sel(lambda m: m["dT"] > 0.15 and m["fsat"] >= 0.85 and m["r2"] > 0, lambda m: m["dT"] * m["r2"]),
        "ReLU-like": _sel(lambda m: m["dT"] > 0.15 and m["fsat"] <= 0.82 and m["T0"] < 0.4 and m["E0"] > 0.15,
                          lambda m: m["dT"] * m["r2"] * (0.5 - min(m["T0"], 0.5))),
        "saturable": _sel(lambda m: m["dT"] < -0.1, lambda m: -m["dT"]),
    }
    fallback = {
        "ReLU-like": _sel(lambda m: m["dT"] > 0.1, lambda m: m["dT"] * (0.5 - min(m["T0"], 0.5))),
        "sigmoid-like": _sel(lambda m: m["dT"] > 0.1, lambda m: m["dT"] * m["r2"]),
        "saturable": mets[int(np.argmin([m["dT"] for m in mets]))],
    }
    return strict, fallback


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default=str(OUT / "final3_Te_family_fdtdx.npz"))
    ap.add_argument("--out-prefix", default="fig3")
    ap.add_argument("--title", default="final3 (P825 / h525 / pad 12 %), FDTDX family")
    ap.add_argument("--A-key", default="A_ito", help="absorption column of the family used in the TTM source term (A_ito: flux difference across the ITO; A_rt: 1-R-T)")
    a = ap.parse_args()
    d = np.load(a.family)
    A_key = a.A_key if a.A_key in d.files else "A"
    look = Lookup(d["lambda_nm"], d["Te_K"], d["T"], d[A_key], d["R"] if "R" in d.files else None)
    print(f"[feedback] family {a.family}: absorption column '{A_key}' drives the TTM; Te rows {look.Te.tolist()}; lambda {look.lam[0]}-{look.lam[-1]} nm")
    ttm.SIG = 150e-15 / (2 * np.sqrt(2 * np.log(2)))                  # demo section 8: 150 fs instead of the module's 280 fs
    Ce = ttm.HeatCapacity("sommerfeld")
    lam_scan = np.arange(look.lam[0], look.lam[-1] + 1e-9, 1.0)
    F_I = I_to_F_mJcm2(I_GRID)
    F_E = I_to_F_mJcm2(E_to_I(E_SCAN))

    # --- lambda scan on the E_scan grid (selection metrics) and on the I grid (figure), primary model tau_ep = 450 fs
    scan_E, scan_I, mets = {}, {}, []
    for lp in lam_scan:
        rE = run_ttm(look, lp, F_E, MODELS["tau450"], Ce)
        rI = run_ttm(look, lp, F_I, MODELS["tau450"], Ce)
        scan_E[lp], scan_I[lp] = rE, rI
        mets.append(metrics_of(lp, E_SCAN, rE["Teff"]))
    strict, fallback = select(mets)
    chosen = {}
    for lab in ("ReLU-like", "sigmoid-like", "saturable"):
        m = strict[lab]
        if m is not None:
            chosen[lab] = dict(m, criteria_met=True)

    def _have(m):
        return any(abs(v["lp"] - m["lp"]) < 0.5 for v in chosen.values())
    # representative wavelengths beyond the ZIP labels, chosen from the data (labels state what the curve actually does):
    m_up = max(mets, key=lambda m: m["dT"])                                   # strongest increase
    if not _have(m_up):
        chosen[f"max ΔT (no ZIP label met)"] = dict(m_up, criteria_met=False)
    m_dn = min(mets, key=lambda m: m["dT"])                                   # strongest decrease, only if it is a real one
    if m_dn["dT"] < -0.05 and not _have(m_dn):
        chosen["decreasing (ZIP 'saturable' criteria not met)"] = dict(m_dn, criteria_met=False)
    cut = [m for m in mets if 1250.0 <= m["lp"] <= 1265.0]                   # glass (+-1,0) cut-off zone: truncation-limited rows
    if cut:
        m_cut = max(cut, key=lambda m: m["dT"])
        if not _have(m_cut):
            chosen["cut-off zone 1250-1265 nm (truncation-limited), max ΔT there"] = dict(m_cut, criteria_met=False)
    flat = [m for m in mets if m["lp"] > 1265.0 and m["T0"] < 0.02]          # T pinned near zero at every Te
    if flat:
        m_flat = min(flat, key=lambda m: abs(m["dT"]))
        if not _have(m_flat):
            chosen["no transmission response (T ≈ 0 at all Te)"] = dict(m_flat, criteria_met=False)

    # --- model band (tau 1000 fs, g = 0) for the chosen wavelengths on the I grid
    band = {lab: {k: run_ttm(look, m["lp"], F_I, MODELS[k], Ce) for k in ("tau1000", "g0")} for lab, m in chosen.items()}

    # --- explicit check that the feedback uses A[Te(t)]: compare with a fixed cold-A integration
    fixedA = {}
    for lab, m in chosen.items():
        Afun, Tfun, A_col, T_col = look.fns(m["lp"])
        A0 = float(Afun(300.0))
        r = ttm.integrate_ttm(F_I, lambda Te: A0 + 0.0 * Te, Ce, MODELS["tau450"], probe_fns={"T": Tfun}, dt_fs=1.0)
        fixedA[lab] = dict(Te_pk_fixedA=r["Te_pk"], Teff_fixedA=r["avg"]["T"])

    # --- figure 3
    fig, axs = plt.subplots(2, 1, figsize=(7.4, 7.6), dpi=130, sharex=True, gridspec_kw=dict(height_ratios=[1.4, 1]))
    cols = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
    for (lab, m), c in zip(chosen.items(), cols):
        r = scan_I[m["lp"]]
        axs[0].semilogx(I_GRID, r["Teff"], "-", lw=2.0, color=c, label=f"{lab}: λ = {m['lp']:.0f} nm (ΔT = {m['dT']:+.2f})")
        axs[0].semilogx(I_GRID, band[lab]["g0"]["Teff"], ":", lw=1.0, color=c)
        axs[0].semilogx(I_GRID, band[lab]["tau1000"]["Teff"], "--", lw=0.9, color=c)
        axs[1].semilogx(I_GRID, r["Te_pk"], "-", lw=1.8, color=c)
        axs[1].semilogx(I_GRID, band[lab]["g0"]["Te_pk"], ":", lw=1.0, color=c)
        axs[1].semilogx(I_GRID, band[lab]["tau1000"]["Te_pk"], "--", lw=0.9, color=c)
    axs[0].plot([], [], "k-", lw=2, label=r"TTM $\tau_{ep}$ = 450 fs (primary)")
    axs[0].plot([], [], "k--", lw=0.9, label=r"$\tau_{ep}$ = 1000 fs"); axs[0].plot([], [], "k:", lw=1.0, label="g = 0 (no e-ph loss)")
    axs[0].set_ylabel(r"$T_{\rm eff} = \int I\,T[\lambda,T_e(t)]\,dt \,/\, \int I\,dt$"); axs[0].grid(alpha=0.3, which="both"); axs[0].legend(fontsize=7.2)
    axs[0].set_title(f"{a.title}: pulse-averaged transmission vs peak intensity (150 fs)\n"
                     r"$C_e\dot T_e = A[\lambda,T_e(t)]\,I(t)/d_{\rm ITO} - g(T_e - T_l)$, A and T interpolated from the FDTDX table (clipped at 8000 K)", fontsize=8.5)
    axs[1].axhline(look.Te[-1], color="gray", lw=0.7, ls="-."); axs[1].text(I_GRID[0] * 1.2, look.Te[-1] * 0.93, "table upper limit", fontsize=7, va="top")
    axs[1].set_xlabel("peak intensity (W/cm²), 150 fs"); axs[1].set_ylabel(r"peak $T_e$ (K)"); axs[1].grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(OUT / f"{a.out_prefix}_Teff_vs_I_final3.png"); plt.close(fig)

    # --- scan overview (diagnostic): Teff(E) heat map over lambda_p
    M = np.array([scan_E[lp]["Teff"] for lp in lam_scan])
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=120)
    im = ax.imshow(M, origin="lower", aspect="auto", extent=[E_SCAN.min(), E_SCAN.max(), lam_scan.min(), lam_scan.max()], cmap="viridis")
    for (lab, m), c in zip(chosen.items(), cols):
        ax.axhline(m["lp"], color=c, ls="--", lw=1.0); ax.text(E_SCAN.max() * 0.98, m["lp"] + 0.6, lab, color="w", fontsize=7, ha="right")
    ax.set_xlabel("E_in (nJ / (8 µm)² pixel)"); ax.set_ylabel(r"$\lambda_p$ (nm)"); ax.set_title(r"diagnostic: $T_{\rm eff}(E_{\rm in};\lambda_p)$, TTM $\tau_{ep}$ = 450 fs", fontsize=9)
    plt.colorbar(im, ax=ax, label=r"$T_{\rm eff}$"); fig.tight_layout(); fig.savefig(OUT / f"{a.out_prefix}_diag_scan_Teff_E_lambda.png"); plt.close(fig)

    # --- tables
    sel_lams = np.array([m["lp"] for m in chosen.values()]); sel_labels = np.array(list(chosen.keys()))
    np.savez(OUT / "final3_Teff_vs_I_fdtdx.npz",
             lambda_nm=look.lam, Te_K=look.Te, R=look.R, T=look.T, A=look.A, A_key_used=A_key,
             **({"A_rt": d["A_rt"]} if "A_rt" in d.files else {}), **({"A_ito": d["A_ito"]} if "A_ito" in d.files else {}),
             I_grid_Wcm2=I_GRID, F_mJcm2=F_I, tau_eff_s=TAU_EFF, pulse_fwhm_s=150e-15,
             selected_lambda_nm=sel_lams, selected_labels=sel_labels, selected_criteria_met=np.array([m["criteria_met"] for m in chosen.values()]),
             Teff_tau450=np.array([scan_I[m["lp"]]["Teff"] for m in chosen.values()]),
             Teff_tau1000=np.array([band[l]["tau1000"]["Teff"] for l in chosen]), Teff_g0=np.array([band[l]["g0"]["Teff"] for l in chosen]),
             Te_pk_tau450=np.array([scan_I[m["lp"]]["Te_pk"] for m in chosen.values()]),
             Te_pk_tau1000=np.array([band[l]["tau1000"]["Te_pk"] for l in chosen]), Te_pk_g0=np.array([band[l]["g0"]["Te_pk"] for l in chosen]),
             above_table_tau450=np.array([scan_I[m["lp"]]["above_table"] for m in chosen.values()]),
             Teff_fixed_cold_A_tau450=np.array([fixedA[l]["Teff_fixedA"] for l in chosen]), Te_pk_fixed_cold_A_tau450=np.array([fixedA[l]["Te_pk_fixedA"] for l in chosen]),
             scan_lambda_nm=lam_scan, scan_Teff_I_tau450=np.array([scan_I[lp]["Teff"] for lp in lam_scan]), scan_Te_pk_I_tau450=np.array([scan_I[lp]["Te_pk"] for lp in lam_scan]),
             E_scan_nJ=E_SCAN, scan_Teff_E_tau450=M,
             gamma_e_Jm3K2=ttm.GAM_E, C_L_Jm3K=ttm.C_L, d_ITO_m=ttm.T_ITO_M, dt_fs=1.0, window_ps=[-1.0, 3.0])
    with open(OUT / "final3_Teff_vs_I_fdtdx.csv", "w") as f:
        f.write("# T_eff(I) for the selected wavelengths, TTM tau_ep = 450 fs (columns), plus g=0 / tau 1000 fs band\n")
        f.write("I_peak_Wcm2,F_mJcm2," + ",".join(f"Teff_{m['lp']:.0f}nm_tau450,Tepk_{m['lp']:.0f}nm_tau450,Teff_{m['lp']:.0f}nm_g0,Teff_{m['lp']:.0f}nm_tau1000" for m in chosen.values()) + "\n")
        for i in range(len(I_GRID)):
            row = [f"{I_GRID[i]:.4e}", f"{F_I[i]:.4e}"]
            for lab, m in chosen.items():
                row += [f"{scan_I[m['lp']]['Teff'][i]:.5f}", f"{scan_I[m['lp']]['Te_pk'][i]:.1f}", f"{band[lab]['g0']['Teff'][i]:.5f}", f"{band[lab]['tau1000']['Teff'][i]:.5f}"]
            f.write(",".join(row) + "\n")
    summary = dict(selected={lab: {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in m.items()} for lab, m in chosen.items()},
                   strict_labels_found={k: (None if v is None else v["lp"]) for k, v in strict.items()},
                   zip_fallback_would_pick={k: (None if v is None else v["lp"]) for k, v in fallback.items()},
                   scan_extremes=dict(max_dT=m_up, min_dT=m_dn),
                   feedback_uses_A_of_Te={lab: dict(Te_pk_max_feedback=float(scan_I[m["lp"]]["Te_pk"].max()), Te_pk_max_fixed_cold_A=float(fixedA[lab]["Te_pk_fixedA"].max()),
                                                    max_abs_dTeff_vs_fixed_cold_A=float(np.abs(scan_I[m["lp"]]["Teff"] - fixedA[lab]["Teff_fixedA"]).max()),
                                                    A_cold=float(scan_I[m["lp"]]["A_cold"]), A_eff_path_at_Imax=float(scan_I[m["lp"]]["A_eff"][-1]))
                                          for lab, m in chosen.items()},
                   above_table_first_I_Wcm2={lab: (float(I_GRID[np.argmax(scan_I[m["lp"]]["above_table"])]) if scan_I[m["lp"]]["above_table"].any() else None) for lab, m in chosen.items()},
                   absorption_column_used=A_key,
                   conventions=dict(tau_eff_s=TAU_EFF, I_grid="logspace(6, 10.2, 60) W/cm^2", pulse="Gaussian FWHM 150 fs, int I dt = 1 (ZIP integrate_ttm, SIG overridden)",
                                    Ce="Sommerfeld gamma_e Te, gamma_e = %.4f" % ttm.GAM_E, C_L=ttm.C_L, d_ITO_m=ttm.T_ITO_M, g="C_e/tau_ep, tau_ep 450 fs primary; 1000 fs and g=0 band",
                                    interpolation="linear in lambda then linear in Te, clipped to [300, 8000] K (ZIP _at)", Teff="int I T[Te(t)] dt / int I dt (integrate_ttm avg)"))
    json.dump(summary, open(OUT / f"{a.out_prefix}_feedback_summary.json", "w"), indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
