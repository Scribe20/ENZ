"""Temporal coupled-mode theory (TCMT) fitted to and validated against the RCWA lookup.

Model: ONE resonant mode, TWO ports (1 = air side, 2 = glass side), lossless non-resonant background
C (2x2 symmetric unitary), energy conservation + time-reversal constraints imposed EXACTLY
(Fan, Suh, Joannopoulos, JOSA A 20, 569 (2003)):

    S(E) = C + d d^T / (i (E - E0) + (gamma_r + gamma_nr)/2),     C d* = -d,   |d|^2 = gamma_r
    Takagi form: C = U U^T with U unitary, d = i U v with v real  (this solves C d* = -d identically)
    U = e^{i psi/2} [[cos th e^{i a}, sin th e^{i b}], [-sin th e^{-i b}, cos th e^{-i a}]]
    T = |S21|^2, R = |S11|^2 (illumination from port 1), A_TCMT = 1 - T - R = gamma_nr |a|^2 >= 0.
    gamma_1 = |d1|^2 (air), gamma_2 = |d2|^2 (glass); the deck's slide-14 formula is the special case of a
    transparent background with gamma_1 = gamma_2 (A_pk <= 1/2).  Near-unity absorption in this model requires
    a mirror-like background (|C12| -> 0) with gamma_2 -> 0.

Eight real parameters per T_e (one redundant: the right-orthogonal Takagi freedom): E0, gamma_nr, psi, th, a, b, v1, v2.
Fitted to T(lambda) AND R(lambda) simultaneously on the single-order window lambda > n_glass P (below it the
(+-1,0) orders propagate in the glass and a two-port model is not defined) for every tabulated T_e; then re-used
inside the SAME TTM (heating by A_TCMT) to test whether TCMT can replace RCWA in the intensity domain.

    python tcmt_fit.py [--lookup outputs/nonlinear_best/lookup_order7.npz] [--tag order7] [--lam-ops 1302 ...]
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import materials as mat                    # noqa: E402
import ito_nonlinear as nl                 # noqa: E402
import ttm_activation as ttm               # noqa: E402

OUT = HERE / "outputs" / "nonlinear_best"
FIG = OUT / "figures"
HC = 1239.841984                           # eV nm


def tcmt_S(E, p):
    """Constrained single-mode two-port TCMT.  p = [E0, gnr, psi, th, a, b, v1, v2]; returns S11, S21 (port-1 illumination)."""
    E0, gnr, psi, th, a, b, v1, v2 = p
    U = np.exp(0.5j * psi) * np.array([[np.cos(th) * np.exp(1j * a), np.sin(th) * np.exp(1j * b)],
                                        [-np.sin(th) * np.exp(-1j * b), np.cos(th) * np.exp(-1j * a)]])
    C = U @ U.T
    d = 1j * U @ np.array([v1, v2])
    gr = v1 ** 2 + v2 ** 2
    den = 1j * (E - E0) + 0.5 * (gr + gnr)
    S11 = C[0, 0] + d[0] * d[0] / den
    S21 = C[1, 0] + d[1] * d[0] / den
    return S11, S21, d, C, gr


def tcmt_tr(E, p):
    S11, S21, *_ = tcmt_S(E, p)
    return abs(S21) ** 2, abs(S11) ** 2


def unpack(p):
    _, _, d, C, gr = tcmt_S(np.array([1.0]), p)
    return dict(gr=gr, g1=abs(d[0]) ** 2, g2=abs(d[1]) ** 2, C11=C[0, 0], C21=C[1, 0], t_d2=abs(C[1, 0]) ** 2, r_d2=abs(C[0, 0]) ** 2)


def fit_one(E, T, R, E0_guess, g_guess, weights=(1.0, 1.0)):
    def resid(p):
        Tm, Rm = tcmt_tr(E, p)
        return np.concatenate([weights[0] * (Tm - T), weights[1] * (Rm - R)])
    lb = [E.min() - 0.3, 0.0, -2 * np.pi, -np.pi, -2 * np.pi, -2 * np.pi, -3.0, -3.0]
    ub = [E.max() + 0.3, 2.0, 2 * np.pi, np.pi, 2 * np.pi, 2 * np.pi, 3.0, 3.0]
    best = None
    rng = np.random.default_rng(0)
    starts = []
    for th in (0.05, 0.3, 0.8, 1.3):
        for psi in (0.0, np.pi / 2, np.pi, -np.pi / 2):
            for gs in (0.5, 1.0, 2.0):
                for split in (0.9, 0.5):
                    g = g_guess * gs
                    starts.append([E0_guess, 0.5 * g, psi, th, 0.0, 0.0, np.sqrt(split * g), np.sqrt((1 - split) * g)])
    for _ in range(60):
        g = g_guess * 10 ** rng.uniform(-0.5, 0.5); split = rng.uniform(0.02, 0.98)
        starts.append([E0_guess + rng.normal(0, 0.02), g * rng.uniform(0.05, 1.0), rng.uniform(-np.pi, np.pi), rng.uniform(0, np.pi / 2),
                       rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi), np.sqrt(split * g), np.sqrt((1 - split) * g)])
    for p0 in starts:
        try:
            res = least_squares(resid, p0, bounds=(lb, ub), max_nfev=800)
        except Exception:
            continue
        if best is None or res.cost < best.cost:
            best = res
    p = best.x
    Tm, Rm = tcmt_tr(E, p)
    return p, Tm, Rm


def resonance_diagnostics(lam, T, A):
    """Model-free (RCWA) descriptors per T_e: T-min position/value, A-max position/value, A FWHM if resolvable."""
    jT = int(np.argmin(T)); jA = int(np.argmax(A))
    half = 0.5 * (A.max() + A.min())
    above = np.where(A >= half)[0]
    fwhm = (lam[above[-1]] - lam[above[0]]) if len(above) and above[0] > 0 and above[-1] < len(lam) - 1 else np.nan
    return dict(lam_Tmin=float(lam[jT]), Tmin=float(T[jT]), lam_Amax=float(lam[jA]), Amax=float(A[jA]), A_fwhm_nm=float(fwhm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookup", default=str(OUT / "lookup_order7.npz"))
    ap.add_argument("--tag", default="order7")
    ap.add_argument("--lam-ops", type=float, nargs="*", default=None)
    ap.add_argument("--lam-min", type=float, default=None, help="fit window lower edge [nm]; default = n_glass * P + 2 (single-order window)")
    ap.add_argument("--ttm-meta", default=str(OUT / "ttm_meta_base.json"))
    ap.add_argument("--heat", default="base")
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    lk = ttm.Lookup(a.lookup)
    P = float(np.load(a.lookup)["P"])
    lam_min = a.lam_min if a.lam_min else mat.n_glass(1250.0) * P + 2.0
    sel = lk.lam >= lam_min
    lam = lk.lam[sel]; E = HC / lam
    rows, Tfit, Rfit = [], np.zeros((len(lk.Te), len(lam))), np.zeros((len(lk.Te), len(lam)))
    for i, Te in enumerate(lk.Te):
        T, R, A = lk.tab["T"][i, sel], lk.tab["R"][i, sel], lk.tab["A"][i, sel]
        d = resonance_diagnostics(lam, T, A)
        g_guess = HC / d["lam_Amax"] ** 2 * (d["A_fwhm_nm"] if np.isfinite(d["A_fwhm_nm"]) else 60.0)    # eV
        p, Tm, Rm = fit_one(E, T, R, HC / d["lam_Tmin"], max(g_guess, 0.005))
        Tfit[i], Rfit[i] = Tm, Rm
        Am = 1 - Tm - Rm
        E0, gnr = p[0], p[1]
        u = unpack(p); g1, g2 = u["g1"], u["g2"]
        row = dict(Te=float(Te), E0_eV=E0, lam0_nm=HC / E0, g1_meV=1e3 * g1, g2_meV=1e3 * g2, gr_meV=1e3 * (g1 + g2), gnr_meV=1e3 * gnr,
                   Q_loaded=E0 / (g1 + g2 + gnr), Q_rad=E0 / (g1 + g2), Q_nr=E0 / max(gnr, 1e-12), rho_gr_over_gnr=(g1 + g2) / max(gnr, 1e-12),
                   bg_T=u["t_d2"], bg_R=u["r_d2"], params=list(map(float, p)),
                   rmse_T=float(np.sqrt(np.mean((Tm - T) ** 2))), rmse_R=float(np.sqrt(np.mean((Rm - R) ** 2))), rmse_A=float(np.sqrt(np.mean((Am - A) ** 2))),
                   maxerr_T=float(np.max(abs(Tm - T))), maxerr_R=float(np.max(abs(Rm - R))), maxerr_A=float(np.max(abs(Am - A))),
                   lam_Tmin_tcmt=float(lam[int(np.argmin(Tm))]), Tmin_tcmt=float(Tm.min()), Amax_tcmt=float(Am.max()), **d)
        rows.append(row)
        print(f"Te={Te:6.0f}: lam0={row['lam0_nm']:.1f} nm  g_r={row['gr_meV']:.1f} (g1 {row['g1_meV']:.1f} + g2 {row['g2_meV']:.1f}) g_nr={row['gnr_meV']:.1f} meV  "
              f"Q_l={row['Q_loaded']:.1f} bg T/R={row['bg_T']:.2f}/{row['bg_R']:.2f} | rmse T {row['rmse_T']:.4f} R {row['rmse_R']:.4f} A {row['rmse_A']:.4f} max|dT| {row['maxerr_T']:.4f} | "
              f"RCWA T-min {d['Tmin']:.4f}@{d['lam_Tmin']:.0f} A-max {d['Amax']:.3f}@{d['lam_Amax']:.0f} FWHM_A {d['A_fwhm_nm']:.0f}", flush=True)
    # resonance-shift error (shift of the T-min relative to 300 K): RCWA vs TCMT
    for row in rows:
        row["shift_Tmin_rcwa_nm"] = row["lam_Tmin"] - rows[0]["lam_Tmin"]
        row["shift_lam0_tcmt_nm"] = row["lam0_nm"] - rows[0]["lam0_nm"]
        row["shift_Tmin_tcmt_nm"] = row["lam_Tmin_tcmt"] - rows[0]["lam_Tmin_tcmt"]
        row["shift_error_nm"] = row["shift_Tmin_tcmt_nm"] - row["shift_Tmin_rcwa_nm"]
    with open(OUT / f"tcmt_fit_{a.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()})
    np.savez_compressed(OUT / f"tcmt_fit_{a.tag}.npz", Te=lk.Te, lam=lam, T_fit=Tfit, R_fit=Rfit, T_rcwa=lk.tab["T"][:, sel], R_rcwa=lk.tab["R"][:, sel], A_rcwa=lk.tab["A"][:, sel],
                        params=np.array([r["params"] for r in rows]), lam_min=lam_min)

    # ---- intensity-domain validation: TCMT(Te) inside the same TTM at the selected lambda_op
    heat_ok = (OUT / f"heatmap_T_{a.heat}.npz").exists() and Path(a.ttm_meta).exists()
    val = []
    if heat_ok:
        meta = json.load(open(a.ttm_meta))
        z = np.load(OUT / f"heatmap_T_{a.heat}.npz"); zT = np.load(OUT / f"heatmap_Te_peak_{a.heat}.npz")
        I = z["I_peak_Wcm2"]; lam_h = z["lam"]; T_rc = z["value"]; Tep = zT["value"]
        Ttab = np.zeros((len(lk.Te), len(lk.lam))); Rtab = np.zeros_like(Ttab)
        for i in range(len(lk.Te)):
            Tm, Rm = tcmt_tr(HC / lk.lam, rows[i]["params"])
            Ttab[i], Rtab[i] = Tm, Rm
        Atab = 1 - Ttab - Rtab
        lk_t = ttm.Lookup(Te=lk.Te, lam=lk.lam, tab=dict(T=Ttab, R=Rtab, A=Atab, Ftot=Atab), eps=lk.eps)
        th = ttm.Thermo(nl.ITOHot())
        lam_ops = a.lam_ops if a.lam_ops else json.load(open(OUT / "activation_summary.json"))["lam_ops"] if (OUT / "activation_summary.json").exists() else [1302.0]
        fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
        for n, lo in enumerate(lam_ops):
            j = int(np.argmin(abs(lk.lam - lo)))
            if lk.lam[j] < lam_min:
                continue
            o = ttm.integrate(lk_t, th, j, I, meta["G_W_m3_K"], meta["Cl_J_m3_K"], dt=meta["dt_fs"] * 1e-15)
            jh = int(np.argmin(abs(lam_h - lk.lam[j])))
            okm = Tep[jh] <= ttm.TE_MAX_MODEL; okt = Tep[jh] <= ttm.TE_TRUST
            err = o["T"] - T_rc[jh]
            v = dict(lam_op=float(lk.lam[j]), rmse_T_trust=float(np.sqrt(np.mean(err[okt] ** 2))), maxerr_T_trust=float(np.max(abs(err[okt]))),
                     rmse_T_model=float(np.sqrt(np.mean(err[okm] ** 2))), maxerr_T_model=float(np.max(abs(err[okm]))),
                     T_cold_rcwa=float(T_rc[jh, 0]), T_cold_tcmt=float(o["T"][0]), dT_trust_rcwa=float(T_rc[jh, okt][-1] - T_rc[jh, 0]), dT_trust_tcmt=float(o["T"][okt][-1] - o["T"][0]),
                     Te_peak_max_trust_rcwa=float(Tep[jh, okt][-1]), Te_peak_max_trust_tcmt=float(o["Te_peak"][okt][-1]))
            val.append(v)
            c = plt.get_cmap("tab10")(n)
            axs[2].plot(I[okm], T_rc[jh, okm], color=c, lw=2, label=f"RCWA λ_op={lk.lam[j]:.0f}"); axs[2].plot(I[okm], o["T"][okm], color=c, lw=1.2, ls="--", label=f"TCMT (RMSE {v['rmse_T_trust']:.4f})")
        axs[2].set_xscale("log"); axs[2].set_yscale("log"); axs[2].set_xlabel("I_peak [W/cm²]"); axs[2].set_ylabel("⟨T⟩"); axs[2].legend(fontsize=7); axs[2].grid(alpha=0.3)
        axs[2].set_title("(c) intensity domain: TCMT(T_e) inside the same TTM vs RCWA lookup", fontsize=9)
    else:
        fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
    # ---- spectra panel + parameter panel
    cmap = plt.get_cmap("plasma")
    for i, Te in enumerate(lk.Te):
        if Te in (300, 1000, 2000, 3000, 4000, 6000, 8000):
            c = cmap((np.log(Te) - np.log(300)) / (np.log(8000) - np.log(300)))
            axs[0].plot(lam, lk.tab["T"][i, sel], color=c, lw=1.8, label=f"RCWA {Te:.0f} K"); axs[0].plot(lam, Tfit[i], color=c, lw=1, ls="--")
            axs[0].plot(lam, lk.tab["R"][i, sel], color=c, lw=1.0, alpha=0.5); axs[0].plot(lam, Rfit[i], color=c, lw=0.7, ls=":", alpha=0.7)
    axs[0].set_yscale("log"); axs[0].set_ylim(1e-3, 1); axs[0].set_xlabel("λ [nm]"); axs[0].set_ylabel("T (thick) and R (thin); RCWA solid, TCMT dashed/dotted"); axs[0].legend(fontsize=7); axs[0].grid(alpha=0.3)
    axs[0].set_title(f"(a) two-port TCMT fit on the single-order window λ ≥ {lam_min:.0f} nm", fontsize=9)
    Te = np.array([r["Te"] for r in rows])
    axs[1].plot(Te, [r["gr_meV"] for r in rows], "o-", label="γ_r = γ_1 + γ_2 [meV]"); axs[1].plot(Te, [r["gnr_meV"] for r in rows], "s-", label="γ_nr [meV]")
    axs[1].plot(Te, [r["g1_meV"] for r in rows], "^--", ms=4, label="γ_1 (air)"); axs[1].plot(Te, [r["g2_meV"] for r in rows], "v--", ms=4, label="γ_2 (glass)")
    ax1b = axs[1].twinx(); ax1b.plot(Te, [r["lam0_nm"] for r in rows], "k-", lw=1.5, label="λ_0 (TCMT)"); ax1b.plot(Te, [r["lam_Tmin"] for r in rows], "k:", lw=1.5, label="λ(T-min) RCWA")
    ax1b.plot(Te, [nl.ITOHot().enz_crossing(t) for t in Te], color="0.5", lw=1, ls="-.", label="λ_ENZ(T_e) of ITO"); ax1b.set_ylabel("wavelength [nm]"); ax1b.legend(fontsize=7, loc="center right")
    axs[1].axvline(ttm.TE_TRUST, color="k", ls="--", lw=0.8); axs[1].axvline(ttm.TE_MAX_MODEL, color="r", lw=0.8)
    axs[1].set_xlabel("T_e [K]"); axs[1].set_ylabel("rate [meV]"); axs[1].legend(fontsize=7, loc="upper left"); axs[1].grid(alpha=0.3)
    axs[1].set_title("(b) fitted TCMT parameters vs T_e", fontsize=9)
    fig.suptitle(f"fig7 — RCWA vs fitted constrained two-port TCMT ({a.tag}; Lane B ε_ITO(T_e))", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / f"fig7_tcmt_vs_rcwa_{a.tag}.png", dpi=160); plt.close(fig)
    json.dump(dict(lam_min_fit=lam_min, rows=rows, intensity_validation=val), open(OUT / f"tcmt_summary_{a.tag}.json", "w"), indent=1, default=float)
    for v in val:
        print("intensity validation:", json.dumps({k: round(x, 5) for k, x in v.items()}))
    print("[tcmt] done")


if __name__ == "__main__":
    main()
