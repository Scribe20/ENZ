"""Material models for the FDTDX campaign, fitted to the SAME supplied datasets as the TORCWA Jz campaign
(materials.py of enz_jz_inverse_design): ITO_nk.csv (Drude + Lorentz ADE poles), a-Si:H (single UV Lorentz pole),
soda-lime glass (single UV Lorentz pole).  Saves original vs fitted epsilon and the fit error.

    /opt/venv-fdtdx/bin/python materials_fit.py
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import materials as mat                       # noqa: E402  (campaign loaders: same files, same interpolation)
import config as cfg                          # noqa: E402

OUT = HERE / "materials"
C0 = 299792458.0
FIT_LO, FIT_HI = 1150.0, 1450.0               # nm, fit window (simulation band 1200-1400 nm with margin)
VERIFY = np.arange(1150.0, 1450.01, 1.0)


def omega(lam_nm):
    return 2 * np.pi * C0 / (lam_nm * 1e-9)


def eps_drude_lorentz(lam, eps_inf, wp, gd, dl, w0, gl):
    w = omega(lam)
    return eps_inf - wp ** 2 / (w ** 2 + 1j * gd * w) + dl * w0 ** 2 / (w0 ** 2 - w ** 2 - 1j * gl * w)


def fit_ito():
    """Drude + one Lorentz pole, band-weighted least squares on Re and Im of eps (1200-1400 nm weight 1, margins 0.3).
    The Lorentz pole is free over 2e14-1e17 rad/s; the best solution is a broad mid-IR oscillator that carries the
    frequency dependence of the damping (Drude-only leaves a 0.03-0.04 error in eps'' at the band edges)."""
    lam = np.arange(FIT_LO, FIT_HI + 0.01, 1.0)
    target = np.array([mat.eps_ito(l) for l in lam])
    w = omega(lam)
    wt = np.where((lam >= 1200) & (lam <= 1400), 1.0, 0.3)
    def resid(p):
        e = eps_drude_lorentz(lam, *p) - target
        return np.concatenate([wt * e.real, wt * e.imag])
    best = None
    for w0 in np.geomspace(3e14, 3e16, 12):
        for dl in (0.01, 0.1, 0.5, 2.0, 5.0):
            for gl in (1e13, 1e14, 5e14, 2e15):
                try:
                    r = least_squares(resid, [3.4, 2.7e15, 1.85e14, dl, w0, gl],
                                      bounds=([1.0, 1e14, 1e12, 0.0, 2e14, 1e12], [10.0, 1e16, 5e15, 50.0, 1e17, 1e16]), max_nfev=4000)
                except Exception:
                    continue
                if best is None or r.cost < best.cost:
                    best = r
    p = best.x
    def resid_d(q):
        e = eps_drude_lorentz(lam, q[0], q[1], q[2], 0.0, 1e16, 1e14) - target
        return np.concatenate([wt * e.real, wt * e.imag])
    rd = least_squares(resid_d, [3.4, 2.7e15, 1.85e14], bounds=([1.0, 1e14, 1e12], [10.0, 1e16, 5e15]))
    return p, rd.x


def fit_lorentz_lossless(eps_fn, name, w0_guess=6e15):
    """Single lossless UV Lorentz pole eps = eps_inf + dl w0^2/(w0^2 - w^2) fitted to the real permittivity."""
    lam = np.arange(FIT_LO, FIT_HI + 0.01, 1.0)
    lam = lam[lam <= 1400.0] if name == "aSiH" else lam       # a-Si:H data end at 1400 nm (no extrapolation)
    target = np.array([float(np.real(eps_fn(l))) for l in lam])
    kmax = max(abs(float(np.imag(eps_fn(l)))) for l in lam)
    def resid(p):
        e = p[0] + p[1] * p[2] ** 2 / (p[2] ** 2 - omega(lam) ** 2)
        return e - target
    best = None
    for w0 in (3e15, 5e15, 7e15, 1e16, 1.5e16):
        r = least_squares(resid, [1.0, target.mean() - 1.0, w0], bounds=([1.0, 0.0, 2.5e15], [20.0, 30.0, 1e17]), max_nfev=20000)
        if best is None or r.cost < best.cost:
            best = r
    return best.x, kmax, lam, target


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lam_ze, _ = mat.ito_zero_crossing()
    p, pd = fit_ito()
    eps_inf, wp, gd, dl, w0, gl = p
    lam = VERIFY
    orig = np.array([mat.eps_ito(l) for l in lam])
    fit = eps_drude_lorentz(lam, *p)
    fit_d = eps_drude_lorentz(lam, pd[0], pd[1], pd[2], 0.0, 1e16, 1e14)
    err = fit - orig
    band = (lam >= 1200) & (lam <= 1400)
    # zero crossing of the fitted model
    from scipy.optimize import brentq
    lz_fit = brentq(lambda l: eps_drude_lorentz(l, *p).real, 1250, 1350)
    # a-Si:H and glass
    pa, ka, lam_a, tgt_a = fit_lorentz_lossless(mat.eps_asi, "aSiH")
    pg, kg, lam_g, tgt_g = fit_lorentz_lossless(lambda l: mat.n_glass(l) ** 2, "glass")
    fit_a = pa[0] + pa[1] * pa[2] ** 2 / (pa[2] ** 2 - omega(lam_a) ** 2)
    fit_g = pg[0] + pg[1] * pg[2] ** 2 / (pg[2] ** 2 - omega(lam_g) ** 2)
    info = dict(
        source_files=dict(ITO=str(cfg.ITO_FILE), aSiH=str(cfg.ASI_FILE) if hasattr(cfg, "ASI_FILE") else "see config.py", glass=str(cfg.GLASS_FILE) if hasattr(cfg, "GLASS_FILE") else "see config.py"),
        loader="enz_jz_inverse_design/materials.py (identical interpolation to the TORCWA campaign)",
        lambda_ZE_data_nm=float(lam_ze), lambda_ZE_fit_nm=float(lz_fit),
        fit_window_nm=[FIT_LO, FIT_HI], time_convention="exp(-i omega t): Drude chi = -wp^2/(w^2 + i g w), Lorentz chi = de w0^2/(w0^2 - w^2 - i g w) (FDTDX ADE)",
        ITO=dict(model="eps_inf + Drude(wp, gd) + Lorentz(de, w0, gl)", eps_inf=float(eps_inf), wp_rad_s=float(wp), gamma_d_rad_s=float(gd), delta_eps_L=float(dl), w0_L_rad_s=float(w0), gamma_L_rad_s=float(gl),
                 err_max_abs_1200_1400=float(np.abs(err[band]).max()), err_rms_1200_1400=float(np.sqrt(np.mean(np.abs(err[band]) ** 2))),
                 err_max_abs_1150_1450=float(np.abs(err).max()), err_re_max_1200_1400=float(np.abs(err.real[band]).max()), err_im_max_1200_1400=float(np.abs(err.imag[band]).max()),
                 eps_at_lambda_ZE_data=[float(np.real(mat.eps_ito(lam_ze))), float(np.imag(mat.eps_ito(lam_ze)))],
                 eps_fit_at_lambda_ZE=[float(eps_drude_lorentz(lam_ze, *p).real), float(eps_drude_lorentz(lam_ze, *p).imag)],
                 drude_only_reference=dict(eps_inf=float(pd[0]), wp_rad_s=float(pd[1]), gamma_rad_s=float(pd[2]), err_max_abs_1200_1400=float(np.abs(fit_d - orig)[band].max()))),
        aSiH=dict(model="eps_inf + de w0^2/(w0^2 - w^2) (lossless Lorentz; supplied k = 0 in band, max |Im eps| = %.2e)" % ka, eps_inf=float(pa[0]), delta_eps=float(pa[1]), w0_rad_s=float(pa[2]), gamma_rad_s=0.0,
                  err_max_abs=float(np.abs(fit_a - tgt_a).max()), fit_range_nm=[float(lam_a[0]), float(lam_a[-1])], eps_at_lambda_ZE_data=float(np.real(mat.eps_asi(lam_ze))),
                  eps_fit_at_lambda_ZE=float(pa[0] + pa[1] * pa[2] ** 2 / (pa[2] ** 2 - omega(lam_ze) ** 2))),
        glass=dict(model="eps_inf + de w0^2/(w0^2 - w^2) (lossless Lorentz; supplied k = 0, max |Im eps| = %.2e)" % kg, eps_inf=float(pg[0]), delta_eps=float(pg[1]), w0_rad_s=float(pg[2]), gamma_rad_s=0.0,
                   err_max_abs=float(np.abs(fit_g - tgt_g).max()), n_at_lambda_ZE_data=float(mat.n_glass(lam_ze)), eps_fit_at_lambda_ZE=float(pg[0] + pg[1] * pg[2] ** 2 / (pg[2] ** 2 - omega(lam_ze) ** 2))),
    )
    json.dump(info, open(OUT / "material_models.json", "w"), indent=1)
    with open(OUT / "ITO_eps_original_vs_fit.csv", "w") as f:
        f.write("lambda_nm,eps_re_supplied,eps_im_supplied,eps_re_fit,eps_im_fit,err_re,err_im,abs_err,eps_re_drudeonly,eps_im_drudeonly\n")
        for i, l in enumerate(lam):
            f.write(f"{l:.1f},{orig[i].real:.6f},{orig[i].imag:.6f},{fit[i].real:.6f},{fit[i].imag:.6f},{err[i].real:.2e},{err[i].imag:.2e},{abs(err[i]):.2e},{fit_d[i].real:.6f},{fit_d[i].imag:.6f}\n")
    with open(OUT / "aSiH_glass_eps_original_vs_fit.csv", "w") as f:
        f.write("lambda_nm,eps_aSiH_supplied,eps_aSiH_fit,err_aSiH,eps_glass_supplied,eps_glass_fit,err_glass\n")
        for i, l in enumerate(lam_g):
            ia = int(np.argmin(abs(lam_a - l))) if l <= lam_a[-1] else None
            f.write(f"{l:.1f},{tgt_a[ia] if ia is not None else float('nan'):.6f},{fit_a[ia] if ia is not None else float('nan'):.6f},{(fit_a[ia]-tgt_a[ia]) if ia is not None else float('nan'):.2e},{tgt_g[i]:.6f},{fit_g[i]:.6f},{fit_g[i]-tgt_g[i]:.2e}\n")
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    axs[0].plot(lam, orig.real, "k-", label="ε′ supplied (ITO_nk.csv)"); axs[0].plot(lam, fit.real, "r--", label="ε′ Drude+Lorentz fit")
    axs[0].plot(lam, orig.imag, "k-", alpha=0.5, label="ε″ supplied"); axs[0].plot(lam, fit.imag, "b--", label="ε″ fit"); axs[0].axvline(lam_ze, color="0.5", ls=":")
    axs[0].axvspan(1200, 1400, color="0.9"); axs[0].set_xlabel("λ [nm]"); axs[0].set_ylabel("ε_ITO"); axs[0].legend(fontsize=7); axs[0].grid(alpha=0.3); axs[0].set_title("ITO", fontsize=10)
    axs[1].plot(lam, err.real, "r-", label="Δε′ (fit − supplied)"); axs[1].plot(lam, err.imag, "b-", label="Δε″"); axs[1].plot(lam, (fit_d - orig).real, "r:", label="Δε′ Drude-only"); axs[1].plot(lam, (fit_d - orig).imag, "b:", label="Δε″ Drude-only")
    axs[1].axvspan(1200, 1400, color="0.9"); axs[1].set_xlabel("λ [nm]"); axs[1].set_ylabel("fit error"); axs[1].legend(fontsize=7); axs[1].grid(alpha=0.3); axs[1].set_title("ITO fit error", fontsize=10)
    axs[2].plot(lam_a, tgt_a, "k-", label="ε a-Si:H supplied"); axs[2].plot(lam_a, fit_a, "r--", label="Lorentz fit")
    ax2 = axs[2].twinx(); ax2.plot(lam_g, tgt_g, "k-", alpha=0.5, label="ε glass supplied"); ax2.plot(lam_g, fit_g, "b--", label="Lorentz fit"); ax2.set_ylabel("ε glass")
    axs[2].set_xlabel("λ [nm]"); axs[2].set_ylabel("ε a-Si:H"); axs[2].legend(fontsize=7, loc="upper right"); ax2.legend(fontsize=7, loc="lower left"); axs[2].grid(alpha=0.3); axs[2].set_title("a-Si:H and glass", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "material_fits.png", dpi=150)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
