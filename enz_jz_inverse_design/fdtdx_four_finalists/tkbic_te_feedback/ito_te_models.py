"""Electron-temperature-dependent ITO permittivity for the FDTDX final3 runs.

Target (TKBIC ZIP, `tkbic_ref/ttm/_ito_eps_Te.py`, used verbatim):

    eps_target(lam, Te) = eps_meas(lam, 300 K) + d_eps_Drude(lam, Te)
    d_eps_Drude          = -(hwp(Te)^2 - HWP^2) / (E (E + i HGAM)),   E = hc/lam
    hwp(Te)              = HWP sqrt(m*(300)/m*(Te)),  m*(Te)/m0 = 0.35 (1 + 2*0.4191 (1 + (pi^2/4)(kB Te)^2))

The delta term is a Drude pole with NEGATIVE oscillator strength (hwp decreases with Te), which cannot be
handed to FDTDX as a pole of its own.  For every Te the whole target is therefore re-fitted with the same
causal, FDTDX-compatible form the cold final3 model uses (materials_fit.py):

    eps(w) = eps_inf - wp^2/(w^2 + i gd w) + dl w0^2/(w0^2 - w^2 - i gl w)        (exp(-i w t))

with the same multi-start least-squares procedure and bounds, but a fit weight that emphasises the
1230-1280 nm band of this study (weight 1.0 in 1230-1280, 0.5 in 1200-1400, 0.2 in 1150-1450).

    <python> ito_te_models.py            -> outputs/ito_te_models.json, outputs/ito_te_fit_check.png/.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
CAMP = HERE.parent                                   # fdtdx_four_finalists
sys.path.insert(0, str(HERE / "tkbic_ref" / "ttm"))
import _ito_eps_Te as zip_eps                        # noqa: E402  (the ZIP model, unmodified)

OUT = HERE / "outputs"
C0 = 299792458.0
TE_LIST = [300.0, 600.0, 1000.0, 1500.0, 2000.0, 3000.0, 4500.0, 6000.0, 8000.0]   # TKBIC demo grid
BAND = (1230.0, 1280.0)                              # study band [nm]
FIT_LO, FIT_HI = 1150.0, 1450.0                      # fit window, same as materials_fit.py
COLD = json.load(open(CAMP / "materials" / "material_models.json"))["ITO"]


def omega(lam_nm):
    return 2 * np.pi * C0 / (np.asarray(lam_nm, float) * 1e-9)


def eps_drude_lorentz(lam_nm, eps_inf, wp, gd, dl, w0, gl):
    """Same expression as materials_fit.eps_drude_lorentz / fx_sim.eps_model."""
    w = omega(lam_nm)
    return eps_inf - wp ** 2 / (w ** 2 + 1j * gd * w) + dl * w0 ** 2 / (w0 ** 2 - w ** 2 - 1j * gl * w)


def eps_cold_model(lam_nm):
    """The existing cold final3 FDTDX ITO model (materials/material_models.json)."""
    return eps_drude_lorentz(lam_nm, COLD["eps_inf"], COLD["wp_rad_s"], COLD["gamma_d_rad_s"],
                             COLD["delta_eps_L"], COLD["w0_L_rad_s"], COLD["gamma_L_rad_s"])


def eps_target(lam_nm, Te):
    """ZIP target eps_meas(lam) + delta-Drude(lam, Te)."""
    return np.asarray(zip_eps.eps_ito_Te(np.asarray(lam_nm, float), float(Te)), complex)


def fit_weights(lam):
    wt = np.full(lam.shape, 0.2)
    wt[(lam >= 1200) & (lam <= 1400)] = 0.5
    wt[(lam >= BAND[0]) & (lam <= BAND[1])] = 1.0
    return wt


def fit_one(Te, x_seed=None):
    """Multi-start weighted least squares of the Drude+Lorentz form to eps_target(lam, Te)
    (start grid and bounds as in materials_fit.fit_ito; the cold model and the previous-Te solution are
    added as extra starts so that the parameters vary smoothly with Te)."""
    lam = np.arange(FIT_LO, FIT_HI + 0.01, 1.0)
    target = eps_target(lam, Te)
    wt = fit_weights(lam)
    lb = [1.0, 1e14, 1e12, 0.0, 2e14, 1e12]
    ub = [10.0, 1e16, 5e15, 50.0, 1e17, 1e16]

    def resid(p):
        e = eps_drude_lorentz(lam, *p) - target
        return np.concatenate([wt * e.real, wt * e.imag])

    starts = [[COLD["eps_inf"], COLD["wp_rad_s"], COLD["gamma_d_rad_s"], COLD["delta_eps_L"], COLD["w0_L_rad_s"], COLD["gamma_L_rad_s"]]]
    if x_seed is not None:
        starts.append(list(x_seed))
    for w0 in np.geomspace(3e14, 3e16, 12):
        for dl in (0.01, 0.1, 0.5, 2.0, 5.0):
            for gl in (1e13, 1e14, 5e14, 2e15):
                starts.append([3.4, 2.7e15, 1.85e14, dl, w0, gl])
    best = None
    for x0 in starts:
        x0 = np.clip(np.asarray(x0, float), lb, ub)
        try:
            r = least_squares(resid, x0, bounds=(lb, ub), max_nfev=4000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best.x


def fit_all(te_list=TE_LIST):
    models, x_prev = {}, None
    for Te in te_list:
        x = fit_one(Te, x_prev)
        x_prev = x
        models[f"{Te:.0f}"] = dict(Te_K=float(Te), eps_inf=float(x[0]), wp_rad_s=float(x[1]), gamma_d_rad_s=float(x[2]),
                                   delta_eps_L=float(x[3]), w0_L_rad_s=float(x[4]), gamma_L_rad_s=float(x[5]),
                                   hwp_eV=float(zip_eps.hwp_of(Te)))
    return models


def model_eps(m, lam_nm):
    return eps_drude_lorentz(lam_nm, m["eps_inf"], m["wp_rad_s"], m["gamma_d_rad_s"], m["delta_eps_L"], m["w0_L_rad_s"], m["gamma_L_rad_s"])


def check(models):
    """Fit error vs the ZIP target in the study band and the full pulse band; 300 K vs measured and vs cold model."""
    lam_b = np.arange(BAND[0], BAND[1] + 0.01, 0.5)
    lam_w = np.arange(1200.0, 1400.01, 1.0)
    rep = {}
    for k, m in models.items():
        Te = m["Te_K"]
        eb = model_eps(m, lam_b) - eps_target(lam_b, Te)
        ew = model_eps(m, lam_w) - eps_target(lam_w, Te)
        rep[k] = dict(Te_K=Te,
                      band_1230_1280=dict(max_abs=float(np.abs(eb).max()), rms=float(np.sqrt(np.mean(np.abs(eb) ** 2))),
                                          max_re=float(np.abs(eb.real).max()), max_im=float(np.abs(eb.imag).max())),
                      band_1200_1400=dict(max_abs=float(np.abs(ew).max()), rms=float(np.sqrt(np.mean(np.abs(ew) ** 2)))),
                      eps_target_1255=[float(eps_target(1255.0, Te).real), float(eps_target(1255.0, Te).imag)],
                      eps_model_1255=[float(model_eps(m, 1255.0).real), float(model_eps(m, 1255.0).imag)])
    # 300 K limit: hot model == cold data, and vs the existing cold FDTDX model
    m300 = models["300"]
    d_meas = model_eps(m300, lam_b) - np.asarray(zip_eps.eps_meas(lam_b), complex)
    d_cold = model_eps(m300, lam_b) - eps_cold_model(lam_b)
    d_cold_w = model_eps(m300, lam_w) - eps_cold_model(lam_w)
    t_meas = eps_target(lam_b, 300.0) - np.asarray(zip_eps.eps_meas(lam_b), complex)
    rep["check_300K"] = dict(max_abs_target300_minus_measured=float(np.abs(t_meas).max()),
                             max_abs_fit300_minus_measured_1230_1280=float(np.abs(d_meas).max()),
                             max_abs_fit300_minus_coldmodel_1230_1280=float(np.abs(d_cold).max()),
                             max_abs_fit300_minus_coldmodel_1200_1400=float(np.abs(d_cold_w).max()),
                             max_abs_coldmodel_minus_measured_1230_1280=float(np.abs(eps_cold_model(lam_b) - np.asarray(zip_eps.eps_meas(lam_b), complex)).max()))
    return rep


def main():
    OUT.mkdir(exist_ok=True)
    models = fit_all()
    rep = check(models)
    info = dict(target="eps_meas(lam) + delta-Drude(lam, Te) from tkbic_ref/ttm/_ito_eps_Te.py (TKBIC ZIP, unmodified)",
                HWP_eV=float(zip_eps.HWP), HGAM_eV=float(zip_eps.HGAM), hc_eVnm=float(zip_eps.HC_EVNM),
                fit_form="eps_inf - wp^2/(w^2 + i gd w) + dl w0^2/(w0^2 - w^2 - i gl w)  (exp(-i w t); FDTDX DrudePole + LorentzPole)",
                fit_window_nm=[FIT_LO, FIT_HI], fit_weights="1.0 in 1230-1280, 0.5 in 1200-1400, 0.2 in 1150-1450",
                Te_K=TE_LIST, models=models, fit_check=rep)
    json.dump(info, open(OUT / "ito_te_models.json", "w"), indent=1)
    # csv + figure
    lam = np.arange(1200.0, 1400.01, 1.0)
    with open(OUT / "ito_te_fit_check.csv", "w") as f:
        f.write("Te_K,lambda_nm,eps_re_target,eps_im_target,eps_re_fit,eps_im_fit,abs_err\n")
        for k, m in models.items():
            t = eps_target(lam, m["Te_K"]); e = model_eps(m, lam)
            for i, l in enumerate(lam):
                f.write(f"{m['Te_K']:.0f},{l:.1f},{t[i].real:.6f},{t[i].imag:.6f},{e[i].real:.6f},{e[i].imag:.6f},{abs(e[i]-t[i]):.3e}\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    cols = plt.cm.plasma(np.linspace(0.05, 0.9, len(models)))
    for (k, m), c in zip(models.items(), cols):
        t = eps_target(lam, m["Te_K"]); e = model_eps(m, lam)
        axs[0].plot(lam, t.real, "-", color=c, lw=1.2, label=f"{m['Te_K']:.0f} K"); axs[0].plot(lam, e.real, "--", color=c, lw=1.0)
        axs[1].plot(lam, t.imag, "-", color=c, lw=1.2); axs[1].plot(lam, e.imag, "--", color=c, lw=1.0)
        axs[2].plot(lam, np.abs(e - t), "-", color=c, lw=1.0)
    for ax, ttl in zip(axs, ("Re ε_ITO: target (solid) vs FDTDX fit (dashed)", "Im ε_ITO", "|fit − target|")):
        ax.axvspan(*BAND, color="0.9"); ax.set_xlabel("λ [nm]"); ax.set_title(ttl, fontsize=10); ax.grid(alpha=0.3)
    axs[0].axhline(0, color="k", lw=0.5); axs[0].legend(fontsize=7, ncol=3); axs[2].set_yscale("log")
    fig.tight_layout(); fig.savefig(OUT / "ito_te_fit_check.png", dpi=150)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
