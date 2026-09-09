"""Figures, activation-curve quantification and CSV/NPZ deliverables for the nonlinear
activation lane of the FROZEN best design (Lane B literature ITO model; see
NONLINEAR_MODEL_AUDIT.md).  Reads the RCWA lookup (build_lookup.py) and the TTM heatmaps
(ttm_activation.py) and writes to outputs/nonlinear_best/{figures, *.csv, *.npz, *.json}.

    python make_figures.py [--tags base G05 G2] [--lam-ops 1302 ...]
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import least_squares
from scipy.interpolate import interp1d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import materials as mat                    # noqa: E402
import ito_nonlinear as nl                 # noqa: E402
import ttm_activation as ttm               # noqa: E402

OUT = HERE / "outputs" / "nonlinear_best"
FIG = OUT / "figures"
TE_SHOW = [300, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000]
DECK_I_PER_NJ = 1e-9 / 1e-6 / (ttm.TAU * np.sqrt(np.pi / (4 * np.log(2))))   # W/cm2 per nJ on a 10 um x 10 um pixel
DECK_E_RANGE_NJ = (0.15, 1.55)                                              # deck input-energy axis (per >=10-um pixel)
HM_KEYS = ("T", "R", "A", "Fz", "Fx", "Fy", "Ftot", "Te_peak", "energy_resid", "absorbed_J_m2")


def load_heat(tag):
    d = {}
    for k in HM_KEYS:
        p = OUT / f"heatmap_{k}_{tag}.npz"
        if not p.exists():
            return None
        z = np.load(p, allow_pickle=True)
        d[k] = z["value"]
        d["lam"], d["I"], d["E"] = z["lam"], z["I_peak_Wcm2"], z["E_cell_J"]
        d["valid_model"], d["valid_trust"] = z["valid_model"], z["valid_trust"]
        d["G"] = float(z["G_W_m3_K"])
    z = np.load(OUT / f"heatmap_deps_{tag}.npz")
    d["d_eps_re"], d["d_eps_im"] = z["d_eps_re"], z["d_eps_im"]
    return d


def add_energy_axis(ax, lam_axis=False):
    """Top axis: incident energy per 825-nm cell (E_cell = I_peak P^2 tau sqrt(pi/(4 ln2)))."""
    ax2 = ax.twiny() if not lam_axis else ax.twinx()
    lo, hi = ax.get_xlim() if not lam_axis else ax.get_ylim()
    if lam_axis:
        ax2.set_yscale("log"); ax2.set_ylim(ttm.e_cell_from_I(lo), ttm.e_cell_from_I(hi)); ax2.set_ylabel("E_cell per 825-nm cell [J]")
    else:
        ax2.set_xscale("log"); ax2.set_xlim(ttm.e_cell_from_I(lo), ttm.e_cell_from_I(hi)); ax2.set_xlabel("incident energy per 825-nm cell, E_cell [J]")
    return ax2


def deck_band(ax, vertical=True):
    lo, hi = DECK_E_RANGE_NJ[0] * DECK_I_PER_NJ, DECK_E_RANGE_NJ[1] * DECK_I_PER_NJ
    if vertical:
        ax.axvspan(lo, hi, color="0.85", alpha=0.5, lw=0, zorder=0)
    else:
        ax.axhspan(lo, hi, color="0.85", alpha=0.5, lw=0, zorder=0)


# ----------------------------------------------------------------------------- fig 1
def fig1_spectra(lk, lam_ze, lam_ops, cold_ref):
    Te_all = lk.Te
    cmap = plt.get_cmap("plasma")
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    for k, ax in zip(("T", "A", "R", "Fz"), axs.flat):
        for Te in TE_SHOW:
            if Te not in Te_all:
                continue
            i = int(np.where(Te_all == Te)[0][0])
            c = cmap((np.log(Te) - np.log(300)) / (np.log(8000) - np.log(300)))
            ax.plot(lk.lam, lk.tab[k][i], color=c, lw=1.6 if Te <= 6000 else 1.2, ls="-" if Te <= 6000 else "--",
                    label=f"{Te:.0f} K" + ("" if Te <= 6000 else " (beyond trusted range)"))
        ax.axvline(lam_ze, color="k", ls=":", lw=1, label=f"λ_ZE = {lam_ze:.1f} nm (cold ε' = 0)" if k == "T" else None)
        if k == "T":
            jmin = int(np.argmin(lk.tab["T"][0]))
            ax.plot(lk.lam[jmin], lk.tab["T"][0][jmin], "kv", ms=8, label=f"cold T-min: {lk.lam[jmin]:.0f} nm, T = {lk.tab['T'][0][jmin]:.4f}")
            ax.plot(cold_ref["lam"], cold_ref["T"], "ko", ms=5, mfc="none", label="cold reference ([11,11], Phase 1)")
            for lo in lam_ops:
                ax.axvline(lo, color="tab:green", ls="--", lw=0.9, alpha=0.8)
            ax.text(lam_ops[0], 0.9, "  selected λ_op", color="tab:green", fontsize=8)
            ax.set_yscale("log"); ax.set_ylim(1e-3, 1.0)
        ax.axvline(mat.n_glass(1250.0) * 825.0, color="0.5", ls="-.", lw=0.8)
        ax.set_ylabel({"T": "T (all orders)", "A": "A = 1 − R − T", "R": "R (all orders)", "Fz": "F_z (longitudinal ITO absorption)"}[k])
        ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=7, ncol=2, loc="lower left")
    axs[1, 0].set_xlabel("wavelength [nm]"); axs[1, 1].set_xlabel("wavelength [nm]")
    axs[0, 1].text(mat.n_glass(1250.0) * 825.0 + 1, 0.05, "Rayleigh (glass) ←", fontsize=7, color="0.4")
    fig.suptitle("fig1 — frozen best design, RCWA [7,7], Lane-B ε_ITO(λ, T_e): T, A, R, F_z vs wavelength for the electron-temperature set", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_T_lambda_Te.png", dpi=170); plt.close(fig)
    # long-format CSV + NPZ
    with open(OUT / "spectra_vs_Te.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Te_K", "lambda_nm", "eps_re", "eps_im", "T", "R", "A", "Fx", "Fy", "Fz", "Ftot", "mean_Ez2_over_Einc2", "within_trusted_Te(<=6000K)"])
        for i, Te in enumerate(Te_all):
            for j, lam in enumerate(lk.lam):
                w.writerow([f"{Te:.0f}", f"{lam:.1f}", f"{lk.eps[i, j].real:.6f}", f"{lk.eps[i, j].imag:.6f}"] +
                           [f"{lk.tab[k][i, j]:.6f}" for k in ("T", "R", "A", "Fx", "Fy", "Fz", "Ftot", "mean_Ez2")] + [int(Te <= ttm.TE_TRUST)])
    np.savez_compressed(OUT / "spectra_vs_Te.npz", Te=Te_all, lam=lk.lam, eps=lk.eps, **lk.tab, order=lk.order)


# ----------------------------------------------------------------------------- fig 2
def fig2_heatmaps(H, lam_ze, lam_ops):
    lam, I, E = H["lam"], H["I"], H["E"]
    inval = ~H["valid_model"]
    def one(ax, val, label, log=False, cmap="viridis", vmin=None, vmax=None):
        v = np.ma.masked_where(inval, val)
        im = ax.pcolormesh(I, lam, v, shading="nearest", cmap=cmap, norm=LogNorm(vmin, vmax) if log else None, vmin=None if log else vmin, vmax=None if log else vmax)
        ax.contour(I, lam, H["Te_peak"], levels=[ttm.TE_TRUST], colors="w", linewidths=1.2, linestyles="--")
        ax.contour(I, lam, H["Te_peak"], levels=[ttm.TE_MAX_MODEL], colors="r", linewidths=1.2)
        ax.set_xscale("log"); ax.axhline(lam_ze, color="w", ls=":", lw=0.8)
        for lo in lam_ops:
            ax.axhline(lo, color="tab:green", ls="--", lw=0.7)
        ax.set_xlabel("I_peak [W/cm²]"); ax.set_ylabel("λ_op [nm]"); ax.set_title(label, fontsize=9)
        plt.colorbar(im, ax=ax, pad=0.01)
        return im
    # main T map
    fig, ax = plt.subplots(figsize=(8.5, 6))
    one(ax, H["T"], "⟨T⟩ (pulse-averaged transmission) — white dashed: T_e,peak = 6000 K (trusted limit); red: 8000 K (model limit; beyond = masked)", log=True, cmap="magma", vmin=1e-3, vmax=1)
    deck_band(ax)
    add_energy_axis(ax)
    ax.text(DECK_E_RANGE_NJ[0] * DECK_I_PER_NJ, lam[1], " deck E_in range\n (0.15–1.55 nJ / 10-µm pixel)", color="0.3", fontsize=7, va="bottom")
    fig.tight_layout(); fig.savefig(FIG / "fig2_T_heatmap.png", dpi=170); plt.close(fig)
    # panel figure
    fig, axs = plt.subplots(2, 4, figsize=(20, 9))
    one(axs[0, 0], H["T"], "⟨T⟩", log=True, cmap="magma", vmin=1e-3, vmax=1)
    one(axs[0, 1], H["R"], "⟨R⟩", cmap="cividis", vmin=0, vmax=1)
    one(axs[0, 2], H["A"], "⟨A⟩ = 1 − R − T", cmap="viridis", vmin=0, vmax=1)
    one(axs[0, 3], H["Fz"], "⟨F_z⟩", cmap="viridis", vmin=0, vmax=1)
    one(axs[1, 0], H["Te_peak"], "peak T_e [K]", log=True, cmap="inferno", vmin=300, vmax=ttm.TE_MAX_MODEL)
    one(axs[1, 1], H["d_eps_re"], "Δε′_ITO(λ_op) at peak T_e", cmap="coolwarm", vmin=-np.nanmax(abs(np.where(inval, 0, H["d_eps_re"]))), vmax=np.nanmax(abs(np.where(inval, 0, H["d_eps_re"]))))
    one(axs[1, 2], H["d_eps_im"], "Δε″_ITO(λ_op) at peak T_e", cmap="coolwarm", vmin=-np.nanmax(abs(np.where(inval, 0, H["d_eps_im"]))), vmax=np.nanmax(abs(np.where(inval, 0, H["d_eps_im"]))))
    dT = H["T"] - H["T"][:, :1]
    m = np.nanmax(abs(np.where(inval, 0, dT)))
    one(axs[1, 3], dT, "⟨T⟩ − T_cold", cmap="RdBu_r", vmin=-m, vmax=m)
    for ax in axs.flat:
        deck_band(ax)
    fig.suptitle(f"fig2 — pulse-averaged response of the frozen best design vs (λ_op, I_peak); 150-fs pulse, TTM with G = {H['G']:.2e} W m⁻³ K⁻¹ (Lane B)", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "fig2_heatmaps_all.png", dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------- selection + quantification
def select_lam_ops(H, lam_ze, extra=()):
    lam, T = H["lam"], H["T"]
    jz = int(np.argmin(abs(lam - lam_ze)))
    jmin = int(np.argmin(T[:, 0]))
    swing = np.zeros(len(lam))
    for j in range(len(lam)):
        ok = np.where(H["valid_trust"][j])[0]
        swing[j] = T[j, ok[-1]] - T[j, 0] if len(ok) else 0.0
    picks = [jz]
    if jmin != jz:
        picks.append(jmin)
    picks += [int(np.argmax(swing)), int(np.argmin(swing))]
    for e in extra:
        picks.append(int(np.argmin(abs(lam - e))))
    seen, out = set(), []
    for j in picks:
        if j not in seen:
            seen.add(j); out.append(j)
    return out, swing


def _fit(fun, p0, x, y, bounds=(-np.inf, np.inf)):
    best = None
    for p in p0:
        try:
            r = least_squares(lambda q: fun(x, *q) - y, p, bounds=bounds, max_nfev=4000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None, np.nan
    return best.x, float(np.sqrt(np.mean((fun(x, *best.x) - y) ** 2)))


def f_lin(x, a): return a * x
def f_lrelu(x, a, b, x0): return a * x + (b - a) * np.maximum(x - x0, 0.0)
def f_softplus(x, c, a, x0, s): return c + a * s * np.log1p(np.exp(np.clip((x - x0) / s, -50, 50)))
def f_sigmoid(x, c, a, x0, s): return c + a / (1.0 + np.exp(np.clip(-(x - x0) / s, -50, 50)))


def quantify(lam_op, I, E, T, R, A, Fz, Te_peak, valid_trust, valid_model, Tsens=None):
    """Activation-curve metrics on the TRUSTED intensity range (T_e,peak <= 6000 K)."""
    ok = np.where(valid_trust)[0]; okm = np.where(valid_model)[0]
    q = dict(lam_op_nm=float(lam_op), n_valid_trust=int(len(ok)), n_valid_model=int(len(okm)))
    if len(ok) < 5:
        q["note"] = "fewer than 5 trusted intensity points"; return q
    it, im = ok[-1], okm[-1]
    logI = np.log10(I)
    q.update(I_max_trust_Wcm2=float(I[it]), E_cell_max_trust_J=float(E[it]), I_max_model_Wcm2=float(I[im]), E_cell_max_model_J=float(E[im]),
             Te_peak_at_I_max_trust=float(Te_peak[it]), T_low=float(T[0]), T_high_trust=float(T[it]), T_high_model=float(T[im]),
             dT_trust=float(T[it] - T[0]), dT_model=float(T[im] - T[0]),
             contrast_trust=float(T[it] / T[0]), contrast_dB_trust=float(10 * np.log10(T[it] / T[0])),
             R_low=float(R[0]), R_high_trust=float(R[it]), A_low=float(A[0]), A_high_trust=float(A[it]), Fz_low=float(Fz[0]), Fz_high_trust=float(Fz[it]))
    dT = T[: it + 1] - T[0]; full = dT[-1]
    # thresholds at 10 / 50 / 90 % of the trusted-range swing (first crossing, interpolated in log I)
    for frac in (0.1, 0.5, 0.9):
        target = frac * full
        idx = np.where(np.sign(full) * (dT - target) >= 0)[0]
        idx = idx[idx > 0]
        if len(idx) and abs(full) > 1e-6:
            k = idx[0]
            x = np.interp(target, [dT[k - 1], dT[k]] if dT[k] >= dT[k - 1] else [dT[k], dT[k - 1]],
                          [logI[k - 1], logI[k]] if dT[k] >= dT[k - 1] else [logI[k], logI[k - 1]])
            q[f"I_{int(frac*100)}_Wcm2"] = float(10 ** x); q[f"E_cell_{int(frac*100)}_J"] = float(ttm.e_cell_from_I(10 ** x))
        else:
            q[f"I_{int(frac*100)}_Wcm2"] = np.nan; q[f"E_cell_{int(frac*100)}_J"] = np.nan
    q["dynamic_range_decades_10_90"] = float(np.log10(q["I_90_Wcm2"] / q["I_10_Wcm2"])) if np.isfinite(q["I_10_Wcm2"]) and np.isfinite(q["I_90_Wcm2"]) else np.nan
    # slopes
    sl = np.gradient(T[: it + 1], logI[: it + 1])
    q["max_abs_dT_dlog10I"] = float(np.max(abs(sl))); q["I_at_max_slope_Wcm2"] = float(I[int(np.argmax(abs(sl)))])
    q["slope_at_I_max_trust_over_max"] = float(abs(sl[-1]) / max(np.max(abs(sl)), 1e-12))
    q["saturating_within_trusted_range"] = bool(abs(sl[-1]) < 0.2 * np.max(abs(sl)) and abs(full) > 1e-3)
    Iout = T * I
    g = np.gradient(Iout[: it + 1], I[: it + 1])
    q["dIout_dIin_min"] = float(g.min()); q["dIout_dIin_max"] = float(g.max()); q["dIout_dIin_cold"] = float(T[0])
    ds = np.diff(T[: it + 1]); tol = 2e-4
    q["monotonic_T_trust"] = "increasing" if np.all(ds > -tol) else ("decreasing" if np.all(ds < tol) else "non-monotonic")
    q["monotonic_Iout_trust"] = bool(np.all(np.diff(Iout[: it + 1]) > 0))
    # activation-function fits on a LINEAR input grid over the trusted range (x = I/I_max_trust, y = I_out/I_out_max)
    x = np.linspace(0.0, 1.0, 201)
    Tx = np.interp(x * I[it], I[: it + 1], T[: it + 1], left=T[0])
    y = Tx * x
    ymax = max(y.max(), 1e-30); yn = y / ymax
    p, e = _fit(f_lin, [[1.0]], x, yn); q["fit_linear_rmse"] = e
    p, e = _fit(f_lrelu, [[T[0] / ymax * I[it] / 1, 1.0, 0.3], [0.5, 1.5, 0.5], [0.0, 1.0, 0.2]], x, yn, bounds=([-10, -10, 0], [10, 10, 1])); q["fit_leakyrelu_rmse"] = e
    q["fit_leakyrelu_params(a,b,x0)"] = [float(v) for v in p] if p is not None else None
    p, e = _fit(f_softplus, [[0, 1, 0.5, 0.1], [0, 2, 0.3, 0.2], [0, 1, 0.7, 0.05]], x, yn, bounds=([-2, 0, -1, 1e-3], [2, 50, 2, 5])); q["fit_softplus_rmse"] = e
    q["fit_softplus_params(c,a,x0,s)"] = [float(v) for v in p] if p is not None else None
    p, e = _fit(f_sigmoid, [[0, 1, 0.5, 0.1], [0, 1, 0.3, 0.2], [0, 1, 0.8, 0.1]], x, yn, bounds=([-2, 0, -1, 1e-3], [2, 50, 2, 5])); q["fit_sigmoid_rmse"] = e
    q["fit_sigmoid_params(c,a,x0,s)"] = [float(v) for v in p] if p is not None else None
    # transmission-vs-linear-input fits (the deck plots T vs E_in): sigmoid in x
    Tn = (Tx - Tx.min()) / max(Tx.max() - Tx.min(), 1e-12)
    p, e = _fit(f_sigmoid, [[0, 1, 0.5, 0.1], [0, 1, 0.2, 0.2], [0, 1, 0.8, 0.1]], x, Tn, bounds=([-2, 0, -1, 1e-3], [2, 50, 2, 5])); q["fit_T_sigmoid_rmse_norm"] = e
    q["fit_T_sigmoid_params(c,a,x0,s)"] = [float(v) for v in p] if p is not None else None
    if Tsens is not None:
        for tag, Ts in Tsens.items():
            q[f"T_high_trust_{tag}"] = float(Ts[it]); q[f"dT_trust_{tag}"] = float(Ts[it] - Ts[0])
    return q


# ----------------------------------------------------------------------------- fig 3 / 4
def fig34_activation(H, Hs, lam_ops_j, Q):
    lam, I, E = H["lam"], H["I"], H["E"]
    cols = plt.get_cmap("tab10")
    fig3, ax3 = plt.subplots(1, 2, figsize=(13, 5))
    fig4, ax4 = plt.subplots(1, len(lam_ops_j), figsize=(4.6 * len(lam_ops_j), 4.4), squeeze=False)
    for n, j in enumerate(lam_ops_j):
        ok = H["valid_trust"][j]; okm = H["valid_model"][j]
        c = cols(n)
        lab = f"λ_op = {lam[j]:.0f} nm"
        ax3[0].plot(I[okm], H["T"][j, okm], color=c, lw=1.0, ls="--")
        ax3[0].plot(I[ok], H["T"][j, ok], color=c, lw=2.2, label=lab)
        if Hs:
            lo = np.min([h["T"][j] for h in Hs.values()], axis=0); hi = np.max([h["T"][j] for h in Hs.values()], axis=0)
            ax3[0].fill_between(I[okm], lo[okm], hi[okm], color=c, alpha=0.15, lw=0)
        ax3[1].plot(I[okm], H["Te_peak"][j, okm], color=c, lw=1.5, label=lab)
        # fig4: I_out(I_in), linear axes over the trusted range
        ax = ax4[0, n]
        it = np.where(ok)[0][-1]
        x = I[: it + 1]; y = H["T"][j, : it + 1] * x
        ax.plot(x, y, color=c, lw=2.2, label="TTM+RCWA (trusted)")
        xm = I[okm]; ax.plot(xm, H["T"][j, okm] * xm, color=c, lw=1, ls="--", label="6000 < T_e,peak ≤ 8000 K")
        ax.plot(x, H["T"][j, 0] * x, color="0.5", lw=1, ls=":", label=f"linear, cold T = {H['T'][j, 0]:.4f}")
        q = Q[n]
        if q.get("fit_leakyrelu_params(a,b,x0)"):
            xx = np.linspace(0, 1, 201); ymax = (H["T"][j, it] * I[it])
            yy = f_lrelu(xx, *q["fit_leakyrelu_params(a,b,x0)"]) * np.max(np.interp(xx * I[it], I[: it + 1], H["T"][j, : it + 1]) * xx) 
            ax.plot(xx * I[it], yy, color="tab:red", lw=0.9, ls="-.", label=f"leaky-ReLU fit, RMSE {q['fit_leakyrelu_rmse']:.3f}")
            yy = f_softplus(xx, *q["fit_softplus_params(c,a,x0,s)"]) * np.max(np.interp(xx * I[it], I[: it + 1], H["T"][j, : it + 1]) * xx)
            ax.plot(xx * I[it], yy, color="tab:purple", lw=0.9, ls="-.", label=f"softplus fit, RMSE {q['fit_softplus_rmse']:.3f}")
            yy = f_sigmoid(xx, *q["fit_sigmoid_params(c,a,x0,s)"]) * np.max(np.interp(xx * I[it], I[: it + 1], H["T"][j, : it + 1]) * xx)
            ax.plot(xx * I[it], yy, color="tab:olive", lw=0.9, ls="-.", label=f"sigmoid fit, RMSE {q['fit_sigmoid_rmse']:.3f}")
        ax.set_xlabel("I_in = I_peak [W/cm²]"); ax.set_ylabel("I_out = ⟨T⟩·I_in [W/cm²]")
        ax.set_title(f"{lab}: ΔT = {q.get('dT_trust', np.nan):+.4f} (T {q.get('T_low', np.nan):.4f} → {q.get('T_high_trust', np.nan):.4f}), {q.get('monotonic_T_trust','')}", fontsize=8)
        ax.legend(fontsize=6.5); ax.grid(alpha=0.3)
        ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    ax3[0].set_xscale("log"); ax3[0].set_yscale("log"); ax3[0].set_xlabel("I_peak [W/cm²]"); ax3[0].set_ylabel("⟨T⟩ (pulse-averaged)")
    ax3[0].set_title("fig3 — ⟨T⟩(I_peak) at selected λ_op (solid: T_e,peak ≤ 6000 K; dashed: ≤ 8000 K; band: G × ½ … × 2)", fontsize=9)
    ax3[0].legend(fontsize=8); ax3[0].grid(alpha=0.3); deck_band(ax3[0]); add_energy_axis(ax3[0])
    ax3[1].set_xscale("log"); ax3[1].set_yscale("log"); ax3[1].axhline(ttm.TE_TRUST, color="k", ls="--", lw=0.8); ax3[1].axhline(ttm.TE_MAX_MODEL, color="r", lw=0.8)
    ax3[1].set_xlabel("I_peak [W/cm²]"); ax3[1].set_ylabel("peak T_e [K]"); ax3[1].grid(alpha=0.3); ax3[1].legend(fontsize=8); deck_band(ax3[1]); add_energy_axis(ax3[1])
    fig3.tight_layout(); fig3.savefig(FIG / "fig3_T_vs_I.png", dpi=170); plt.close(fig3)
    fig4.suptitle("fig4 — input–output characteristic I_out(I_in) on the trusted range (Lane B); fits are shape descriptors, not labels", fontsize=10)
    fig4.tight_layout(); fig4.savefig(FIG / "fig4_Iout_vs_Iin.png", dpi=170); plt.close(fig4)


# ----------------------------------------------------------------------------- fig 5 traces
def fig5_traces(lk, th, j, I_list, G, Cl, dt):
    o = ttm.integrate(lk, th, j, I_list, G, Cl, dt=dt, traces=True)
    tr = o["traces"]; t = tr["t"] * 1e15
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for n, Iv in enumerate(I_list):
        lab = f"I_peak = {Iv:.1e} W/cm² (E_cell = {ttm.e_cell_from_I(Iv):.2e} J)"
        axs[0, 0].plot(t, tr["Te"][:, n], label=lab); axs[0, 0].plot(t, tr["Tl"][:, n], ls="--", color=axs[0, 0].lines[-1].get_color())
        axs[0, 1].plot(t, tr["A"][:, n], label=lab)
        axs[1, 0].plot(t, tr["T"][:, n], label=lab)
        axs[1, 1].plot(t, tr["I"][:, n] / 1e4, label=lab)
    axs[0, 0].axhline(ttm.TE_TRUST, color="k", ls="--", lw=0.8); axs[0, 0].axhline(ttm.TE_MAX_MODEL, color="r", lw=0.8)
    axs[0, 0].set_ylabel("T_e (solid), T_l (dashed) [K]"); axs[0, 0].set_yscale("log")
    axs[0, 1].set_ylabel("A(T_e(t)) — instantaneous total ITO absorption"); axs[1, 0].set_ylabel("T(T_e(t)) — instantaneous transmission"); axs[1, 0].set_yscale("log")
    axs[1, 1].set_ylabel("I(t) [W/cm²]"); axs[1, 1].set_yscale("log"); axs[1, 1].set_ylim(1e4, None)
    for ax in axs.flat:
        ax.set_xlabel("t [fs]"); ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=7)
    fig.suptitle(f"fig5 — TTM traces at λ_op = {lk.lam[j]:.0f} nm; G = {G:.2e} W m⁻³ K⁻¹, C_l = {Cl:.1e} J m⁻³ K⁻¹, dt = {dt*1e15:.2f} fs (Lane B)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig5_TTM_traces.png", dpi=170); plt.close(fig)
    np.savez_compressed(OUT / "TTM_traces.npz", t_s=tr["t"], I_peak_Wcm2=np.array(I_list), Te=tr["Te"], Tl=tr["Tl"], T=tr["T"], A=tr["A"], I_t=tr["I"],
                        lam_op=lk.lam[j], G=G, Cl=Cl, dt=dt, Te_peak=o["Te_peak"], energy_resid=o["energy_resid"])
    with open(OUT / "TTM_traces.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t_fs"] + [f"{k}@{Iv:.2e}" for Iv in I_list for k in ("I_Wcm2", "Te_K", "Tl_K", "A", "T")])
        for i in range(0, len(t), 4):
            w.writerow([f"{t[i]:.2f}"] + [f"{v:.6g}" for n in range(len(I_list)) for v in (tr["I"][i, n] / 1e4, tr["Te"][i, n], tr["Tl"][i, n], tr["A"][i, n], tr["T"][i, n])])
    return o


# ----------------------------------------------------------------------------- fig 6 permittivity
def fig6_eps(lk, H, model, lam_ops_j, lam_ze):
    lam, I = H["lam"], H["I"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    cols = plt.get_cmap("tab10")
    for n, j in enumerate(lam_ops_j):
        okm = H["valid_model"][j]; ok = H["valid_trust"][j]
        axs[0].plot(I[okm], H["d_eps_re"][j, okm], color=cols(n), ls="--", lw=1)
        axs[0].plot(I[ok], H["d_eps_re"][j, ok], color=cols(n), lw=2, label=f"Δε′ at λ_op = {lam[j]:.0f} nm")
        axs[0].plot(I[okm], H["d_eps_im"][j, okm], color=cols(n), ls=":", lw=1.5, label=f"Δε″ at λ_op = {lam[j]:.0f} nm")
    axs[0].set_xscale("log"); axs[0].set_xlabel("I_peak [W/cm²]"); axs[0].set_ylabel("Δε_ITO(λ_op) at peak T_e"); axs[0].legend(fontsize=7); axs[0].grid(alpha=0.3); deck_band(axs[0]); add_energy_axis(axs[0])
    Te = lk.Te
    jz = int(np.argmin(abs(lk.lam - lam_ze)))
    axs[1].plot(Te, lk.eps[:, jz].real, "k-", label=f"ε′ at λ_ZE ({lk.lam[jz]:.0f} nm)"); axs[1].plot(Te, lk.eps[:, jz].imag, "k--", label="ε″ at λ_ZE")
    for n, j in enumerate(lam_ops_j):
        if j != jz:
            axs[1].plot(Te, lk.eps[:, j].real, color=cols(n), label=f"ε′ at {lk.lam[j]:.0f} nm"); axs[1].plot(Te, lk.eps[:, j].imag, color=cols(n), ls="--")
    axs[1].axvline(ttm.TE_TRUST, color="k", ls="--", lw=0.8); axs[1].axvline(ttm.TE_MAX_MODEL, color="r", lw=0.8); axs[1].axhline(0, color="0.6", lw=0.6)
    axs[1].set_xlabel("T_e [K]"); axs[1].set_ylabel("ε_ITO"); axs[1].legend(fontsize=7); axs[1].grid(alpha=0.3)
    Tg = np.linspace(300, 8000, 60); lz = [model.enz_crossing(T) for T in Tg]
    axs[2].plot(Tg, lz, "k-"); axs[2].axvline(ttm.TE_TRUST, color="k", ls="--", lw=0.8); axs[2].axvline(ttm.TE_MAX_MODEL, color="r", lw=0.8)
    axs[2].set_xlabel("T_e [K]"); axs[2].set_ylabel("λ_ENZ(T_e) [nm] (ε′ = 0)"); axs[2].grid(alpha=0.3)
    axs[2].set_title("Kane-band Drude weight, γ const. (Lane B)", fontsize=9)
    fig.suptitle("fig6 — ITO permittivity vs input (Lane B literature model anchored to the supplied cold ITO_nk.csv)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig6_eps_vs_input.png", dpi=170); plt.close(fig)


# ----------------------------------------------------------------------------- composite
def composite(lk, H, lam_ze, lam_ops_j, Q):
    lam, I, E = H["lam"], H["I"], H["E"]
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.4))
    cmap = plt.get_cmap("plasma")
    for Te in TE_SHOW:
        if Te in lk.Te:
            i = int(np.where(lk.Te == Te)[0][0])
            axs[0].plot(lk.lam, lk.tab["T"][i], color=cmap((np.log(Te) - np.log(300)) / (np.log(8000) - np.log(300))), ls="-" if Te <= 6000 else "--", label=f"{Te:.0f} K")
    axs[0].set_yscale("log"); axs[0].set_ylim(1e-3, 1); axs[0].set_xlabel("λ [nm]"); axs[0].set_ylabel("T"); axs[0].legend(fontsize=7, ncol=2); axs[0].grid(alpha=0.3)
    axs[0].axvline(lam_ze, color="k", ls=":", lw=0.8); axs[0].set_title("(a) T(λ; T_e), RCWA [7,7]", fontsize=10)
    inval = ~H["valid_model"]
    im = axs[1].pcolormesh(E, lam, np.ma.masked_where(inval, H["T"]), shading="nearest", cmap="magma", norm=LogNorm(1e-3, 1))
    axs[1].contour(E, lam, H["Te_peak"], levels=[ttm.TE_TRUST], colors="w", linestyles="--"); axs[1].contour(E, lam, H["Te_peak"], levels=[ttm.TE_MAX_MODEL], colors="r")
    axs[1].set_xscale("log"); axs[1].set_xlabel("incident energy per 825-nm cell E_cell [J]"); axs[1].set_ylabel("λ_op [nm]"); plt.colorbar(im, ax=axs[1], label="⟨T⟩")
    axs[1].set_title("(b) ⟨T⟩(λ_op, E_cell); white: T_e,peak = 6000 K, red: 8000 K", fontsize=10)
    for n, j in enumerate(lam_ops_j):
        axs[1].axhline(lam[j], color="tab:green", ls="--", lw=0.7)
        ok = H["valid_trust"][j]; it = np.where(ok)[0][-1]
        x = E[: it + 1]; y = H["T"][j, : it + 1] * x
        axs[2].plot(x / x[-1], y / y[-1], lw=2, label=f"λ_op = {lam[j]:.0f} nm, ΔT = {Q[n].get('dT_trust', np.nan):+.3f}")
    axs[2].plot([0, 1], [0, 1], "k:", lw=0.8, label="linear")
    axs[2].set_xlabel("E_in / E_in,max (trusted range)"); axs[2].set_ylabel("E_out / E_out,max"); axs[2].legend(fontsize=7); axs[2].grid(alpha=0.3)
    axs[2].set_title("(c) normalized activation E_out(E_in) at selected λ_op", fontsize=10)
    fig.suptitle("Slide-17 analogue for the frozen best freeform design (Lane B literature ITO model — provisional, not the measured nonlinear response)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig_composite_slide17_analogue.png", dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=["base", "G05", "G2"])
    ap.add_argument("--lam-ops", type=float, nargs="*", default=[])
    ap.add_argument("--lookup", default=str(OUT / "lookup_order7.npz"))
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    lk = ttm.Lookup(a.lookup)
    model = nl.ITOHot(); th = ttm.Thermo(model)
    lam_ze, _ = mat.ito_zero_crossing()
    cold = json.load(open(OUT / "cold_state_reference.json"))
    cold_ref = dict(lam=cold.get("lam_nm", 1302.0), T=cold.get("T", np.nan))
    H = load_heat(a.tags[0]); assert H is not None, "run ttm_activation.py first"
    Hs = {t: load_heat(t) for t in a.tags[1:]}; Hs = {t: h for t, h in Hs.items() if h is not None}
    lam_ops_j, swing = select_lam_ops(H, lam_ze, extra=a.lam_ops)
    lam_ops = [float(H["lam"][j]) for j in lam_ops_j]
    print("selected lambda_op:", lam_ops)
    fig1_spectra(lk, lam_ze, lam_ops, cold_ref)
    fig2_heatmaps(H, lam_ze, lam_ops)
    Q = []
    for j in lam_ops_j:
        Q.append(quantify(H["lam"][j], H["I"], H["E"], H["T"][j], H["R"][j], H["A"][j], H["Fz"][j], H["Te_peak"][j], H["valid_trust"][j], H["valid_model"][j],
                          Tsens={t: h["T"][j] for t, h in Hs.items()}))
    fig34_activation(H, Hs, lam_ops_j, Q)
    # per-lambda swing table (all lambda_op) for the report
    with open(OUT / "swing_vs_lambda_op.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["lambda_op_nm", "T_cold", "T_at_I_max_trust", "dT_trust", "I_max_trust_Wcm2", "E_cell_max_trust_J", "I_max_model_Wcm2"])
        for j in range(len(H["lam"])):
            ok = np.where(H["valid_trust"][j])[0]; okm = np.where(H["valid_model"][j])[0]
            w.writerow([f"{H['lam'][j]:.0f}", f"{H['T'][j,0]:.6f}", f"{H['T'][j,ok[-1]]:.6f}", f"{swing[j]:+.6f}", f"{H['I'][ok[-1]]:.3e}", f"{H['E'][ok[-1]]:.3e}", f"{H['I'][okm[-1]]:.3e}"])
    # activation_curves.csv
    with open(OUT / "activation_curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lambda_op_nm", "I_peak_Wcm2", "E_cell_J", "E_per_10um_pixel_nJ_equiv", "T_avg", "R_avg", "A_avg", "Fz_avg", "Te_peak_K", "I_out_Wcm2", "valid_trust(Te<=6000K)", "valid_model(Te<=8000K)"] +
                   [f"T_avg_{t}" for t in Hs])
        for j in lam_ops_j:
            for i in range(len(H["I"])):
                w.writerow([f"{H['lam'][j]:.0f}", f"{H['I'][i]:.4e}", f"{H['E'][i]:.4e}", f"{H['I'][i]/DECK_I_PER_NJ:.4e}", f"{H['T'][j,i]:.6f}", f"{H['R'][j,i]:.6f}", f"{H['A'][j,i]:.6f}",
                            f"{H['Fz'][j,i]:.6f}", f"{H['Te_peak'][j,i]:.1f}", f"{H['T'][j,i]*H['I'][i]:.4e}", int(H["valid_trust"][j, i]), int(H["valid_model"][j, i])] +
                           [f"{h['T'][j,i]:.6f}" for h in Hs.values()])
    with open(OUT / "threshold_summary.csv", "w", newline="") as f:
        keys = sorted(set().union(*[q.keys() for q in Q]), key=lambda k: (k != "lam_op_nm", k))
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for q in Q:
            w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in q.items()})
    json.dump(dict(lam_ze=lam_ze, lam_ops=lam_ops, deck_I_per_nJ_Wcm2=DECK_I_PER_NJ, quant=Q, G=H["G"]), open(OUT / "activation_summary.json", "w"), indent=1, default=float)
    # traces at the first lambda_op
    j0 = lam_ops_j[0]
    okt = np.where(H["valid_trust"][j0])[0]; okm = np.where(H["valid_model"][j0])[0]
    I_list = sorted(set([1e8, 1e9, float(H["I"][okt[-1]]), float(H["I"][okm[-1]])]))
    meta = json.load(open(OUT / f"ttm_meta_{a.tags[0]}.json"))
    fig5_traces(lk, th, j0, I_list, meta["G_W_m3_K"], meta["Cl_J_m3_K"], meta["dt_fs"] * 1e-15)
    fig6_eps(lk, H, model, lam_ops_j, lam_ze)
    composite(lk, H, lam_ze, lam_ops_j, Q)
    for q in Q:
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in q.items() if not k.startswith("fit_") or k.endswith("rmse") or k.endswith("norm")}, default=str))
    print("[figures] done")


if __name__ == "__main__":
    main()
