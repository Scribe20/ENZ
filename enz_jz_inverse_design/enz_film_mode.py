"""Reference: the TM ENZ (long-range / Berreman-type) mode of the BARE
air / ITO(23 nm) / glass film under the NEW materials, as a complex-frequency
pole at fixed real in-plane momentum K = G10(P) = 2 pi / P.

Analytic continuation of eps_ITO to complex omega requires a model: a Drude
form  eps = eps_inf - wp^2 / (w^2 + i gamma w)  is least-squares fitted to the
supplied ITO_nk.csv over 1000-1675 nm (fit residual reported); it is used ONLY
here, never in the RCWA campaign (which uses the tabulated data directly).
Glass: real n_glass(lambda_ZE); a-Si absent.  For P < lambda/n_glass the mode
at K = G10 lies outside the glass light cone -> bound, non-radiative: the
pole's Im(omega) is the ITO material damping of the ENZ mode.

TM dispersion (exp(-j w t); kz_i = sqrt(eps_i k0^2 - K^2), decaying branches
in the claddings):
  (kz1/e1 + kz2/e2)(kz2/e2 + kz3/e3) - (kz1/e1 - kz2/e2)(kz2/e2 - kz3/e3) exp(2 i kz2 d) = 0
"""

import json
import sys

import numpy as np
from scipy.optimize import least_squares

import config
import materials as mat

C = 299.792458          # nm/fs
D = config.D_ITO_NM


def drude_fit(lo=1000.0, hi=1675.0):
    t = mat.ito()
    sel = (t.wl >= lo) & (t.wl <= hi)
    wl = t.wl[sel]; eps = (t.n[sel] + 1j * t.k[sel]) ** 2
    w = 2 * np.pi * C / wl                                 # rad/fs

    def model(p, w):
        einf, wp, g = p
        return einf - wp ** 2 / (w ** 2 + 1j * g * w)

    def resid(p):
        e = model(p, w)
        return np.concatenate([(e.real - eps.real), (e.imag - eps.imag)])
    p0 = [3.9, 2 * np.pi * C / 1302.28 * np.sqrt(3.9), 0.2]
    fit = least_squares(resid, p0, bounds=([1.0, 0.1, 0.0], [10.0, 10.0, 2.0]))
    e = model(fit.x, w)
    return dict(eps_inf=float(fit.x[0]), wp_rad_fs=float(fit.x[1]), gamma_rad_fs=float(fit.x[2]),
                rms_resid=float(np.sqrt(np.mean(np.abs(e - eps) ** 2))), max_abs_resid=float(np.max(np.abs(e - eps))),
                range_nm=[lo, hi], model=model)


def tm_dispersion(w, K, eps_ito_fn, n_glass):
    e1, e2, e3 = 1.0, eps_ito_fn(w), n_glass ** 2
    k0 = w / C
    def kz(e):
        v = np.sqrt(e * k0 ** 2 - K ** 2 + 0j)
        return v if v.imag >= 0 else -v          # decaying into the claddings
    kz1, kz3 = kz(e1), kz(e3)
    kz2 = np.sqrt(e2 * k0 ** 2 - K ** 2 + 0j)
    a, b, c = kz1 / e1, kz2 / e2, kz3 / e3
    return (a + b) * (b + c) - (a - b) * (b - c) * np.exp(2j * kz2 * D)


def solve_pole(K, eps_fn, n_glass, w0, iters=60):
    w = complex(w0)
    for _ in range(iters):
        f = tm_dispersion(w, K, eps_fn, n_glass)
        dw = 1e-6 * abs(w)
        df = (tm_dispersion(w + dw, K, eps_fn, n_glass) - tm_dispersion(w - dw, K, eps_fn, n_glass)) / (2 * dw)
        step = f / df
        w = w - step
        if abs(step) < 1e-12 * abs(w):
            break
    return w, abs(tm_dispersion(w, K, eps_fn, n_glass))


def solve_K(lam, K0, n_glass, iters=80):
    """Complex in-plane momentum K of the TM mode at REAL wavelength lam with the
    TABULATED eps_ITO(lam) (no analytic continuation needed)."""
    w = 2 * np.pi * C / lam
    eps_fn = lambda _w: mat.eps_ito(lam)
    K = complex(K0)
    for _ in range(iters):
        f = tm_dispersion(w, K, eps_fn, n_glass)
        dK = 1e-7 * max(abs(K), 1e-3)
        df = (tm_dispersion(w, K + dK, eps_fn, n_glass) - tm_dispersion(w, K - dK, eps_fn, n_glass)) / (2 * dK)
        step = f / df
        K = K - step
        if abs(step) < 1e-13 * max(abs(K), 1e-9):
            break
    return K, abs(tm_dispersion(w, K, eps_fn, n_glass))


def real_axis_dispersion(n_glass, lams=np.arange(1150.0, 1400.1, 2.0), K0_over_k0=1.75):
    """Track the ENZ-branch TM mode K(lambda) at real lambda (tabulated eps):
    returns Re K/k0, Im K/k0, and the (real) wavelength where Re K = G10(P)."""
    rows = []
    K = None
    for lam in lams:
        k0 = 2 * np.pi / lam
        K0 = K if K is not None else K0_over_k0 * k0 * (1 + 0.05j)
        K, res = solve_K(lam, K0, n_glass)
        rows.append(dict(lam=float(lam), ReK_over_k0=float(K.real / k0), ImK_over_k0=float(K.imag / k0),
                         residual=float(res), eps_ito=[mat.eps_ito(lam).real, mat.eps_ito(lam).imag]))
    return rows


def main(out=config.OUT / "stage0" / "enz_film_mode.json"):
    lam_ze, _ = mat.ito_zero_crossing()
    ng = mat.n_glass(lam_ze)
    fit = drude_fit()
    model = fit.pop("model")
    eps_fn = lambda w: model([fit["eps_inf"], fit["wp_rad_fs"], fit["gamma_rad_fs"]], w)
    w_ze = 2 * np.pi * C / lam_ze
    rows = []
    for P in [550.0, 650.0, 750.0, 825.0, 850.0, 1000.0, 1300.0]:
        K = 2 * np.pi / P
        w, res = solve_pole(K, eps_fn, ng, w_ze * (1 - 0.05j))
        lam = 2 * np.pi * C / w.real
        rows.append(dict(P=P, K_over_k0=float(K / (w_ze / C)), lambda_pole_nm=float(lam), Q=float(w.real / (2 * abs(w.imag))),
                         gamma_rad_fs=float(abs(w.imag)), residual=float(res),
                         bound=bool(K > ng * w.real / C)))
    ez_check = dict(eps_drude_at_ZE=[eps_fn(w_ze).real, eps_fn(w_ze).imag], eps_table_at_ZE=[mat.eps_ito(lam_ze).real, mat.eps_ito(lam_ze).imag])
    disp = real_axis_dispersion(ng)
    # wavelength where the real part of the tracked mode momentum equals G10(P)
    match = []
    L = np.array([d["lam"] for d in disp]); RK = np.array([d["ReK_over_k0"] for d in disp]) * 2 * np.pi / L
    for P in [550.0, 650.0, 750.0, 825.0, 850.0]:
        G = 2 * np.pi / P
        idx = np.where(np.sign(RK[:-1] - G) != np.sign(RK[1:] - G))[0]
        match.append(dict(P=P, lambda_match_nm=[float(np.interp(G, [RK[i], RK[i + 1]], [L[i], L[i + 1]])) for i in idx]))
    rep = dict(lambda_ZE=lam_ze, n_glass=ng,
               complex_omega_drude=dict(drude_fit=fit, check=ez_check, poles=rows,
                                        caveat="Drude continuation of a Drude-Lorentz table (rms resid 0.056): low confidence; kept for transparency"),
               real_axis_tabulated=dict(dispersion=disp, G10_matches=match,
                                        note="complex K(lambda) of the bare-film TM ENZ branch with the tabulated eps (no continuation)"),
               note="bare air/ITO(23)/glass TM mode reference (no a-Si)")
    with open(out, "w") as f:
        json.dump(rep, f, indent=1)
    print("Drude fit:", {k: v for k, v in fit.items()}, "check:", ez_check)
    print("complex-omega poles (Drude, low confidence):", [(r["P"], round(r["lambda_pole_nm"], 1), round(r["Q"], 2)) for r in rows])
    for d in disp[::10]:
        print(f"  lam {d['lam']:.0f}: ReK/k0 {d['ReK_over_k0']:.3f} ImK/k0 {d['ImK_over_k0']:.3f} resid {d['residual']:.1e} eps {d['eps_ito'][0]:+.3f}+{d['eps_ito'][1]:.3f}i")
    print("G10 matches:", match)
    return rep


if __name__ == "__main__":
    main()
