"""Reference: TM ENZ (long-range / Berreman-type) mode of the BARE
air / ITO(23 nm) / glass film under the NEW materials.

Three complementary views (no a-Si):
  (1) complex-frequency pole at fixed real in-plane momentum K = G10(P):
      needs eps_ITO at complex omega -> a Drude form fitted to the supplied
      ITO_nk.csv over 1000-1675 nm (fit residual reported; used ONLY here)
  (2) complex in-plane momentum K(lambda) at REAL lambda with the TABULATED
      eps_ITO (no analytic continuation)
  (3) driven p-polarized absorption of the bare film vs lambda at oblique
      incidence (torcwa, order [1,1]) -> Berreman absorption peak.

TM pole condition of a slab (exp(-j w t); kz_i = sqrt(eps_i k0^2 - K^2),
decaying/outgoing branches in the claddings), normalized Fresnel form:
    D = 1 + r12 r23 exp(2 i kz2 d) = 0,
    r_ij = (kz_i/eps_i - kz_j/eps_j) / (kz_i/eps_i + kz_j/eps_j).
(A first version of this file used '(a+b)(b+c) - (a-b)(b-c) e^{2 i kz2 d}';
the sign was wrong - caught by the source audit against enz_target/
tm_slab_mode.py - and its outputs were discarded.)
"""

import json
import sys

import numpy as np
import torch
from scipy.optimize import least_squares, root

import config
import materials as mat

C = 299.792458          # nm/fs
D_ITO = config.D_ITO_NM


def drude_fit(lo=1000.0, hi=1675.0):
    t = mat.ito()
    sel = (t.wl >= lo) & (t.wl <= hi)
    wl = t.wl[sel]; eps = (t.n[sel] + 1j * t.k[sel]) ** 2
    w = 2 * np.pi * C / wl

    def model(p, w):
        einf, wp, g = p
        return einf - wp ** 2 / (w ** 2 + 1j * g * w)

    def resid(p):
        e = model(p, w)
        return np.concatenate([(e.real - eps.real), (e.imag - eps.imag)])
    fit = least_squares(resid, [3.9, 2 * np.pi * C / 1302.28 * np.sqrt(3.9), 0.2],
                        bounds=([1.0, 0.1, 0.0], [10.0, 10.0, 2.0]))
    e = model(fit.x, w)
    return dict(eps_inf=float(fit.x[0]), wp_rad_fs=float(fit.x[1]), gamma_rad_fs=float(fit.x[2]),
                rms_resid=float(np.sqrt(np.mean(np.abs(e - eps) ** 2))), max_abs_resid=float(np.max(np.abs(e - eps))),
                range_nm=[lo, hi], model=model)


def _kz(e, k0, K):
    v = np.sqrt(e * k0 ** 2 - K ** 2 + 0j)
    return v if (v.imag > 0 or (v.imag == 0 and v.real > 0)) else -v     # decaying / outgoing


def D_fresnel(w, K, eps2, n_glass):
    e1, e3 = 1.0, n_glass ** 2
    k0 = w / C
    kz1, kz3 = _kz(e1, k0, K), _kz(e3, k0, K)
    kz2 = np.sqrt(eps2 * k0 ** 2 - K ** 2 + 0j)
    a, b, c = kz1 / e1, kz2 / eps2, kz3 / e3
    r12, r23 = (a - b) / (a + b), (b - c) / (b + c)
    return 1 + r12 * r23 * np.exp(2j * kz2 * D_ITO)


def solve_pole_omega(K, eps_fn, n_glass, w0):
    f = lambda v: (lambda d: [d.real, d.imag])(D_fresnel(v[0] + 1j * v[1], K, eps_fn(v[0] + 1j * v[1]), n_glass))
    sol = root(f, [w0.real, w0.imag], method="hybr", tol=1e-13)
    w = sol.x[0] + 1j * sol.x[1]
    return w, abs(D_fresnel(w, K, eps_fn(w), n_glass)), bool(sol.success)


def solve_K(lam, K0, n_glass):
    w = 2 * np.pi * C / lam
    e2 = mat.eps_ito(lam)
    f = lambda v: (lambda d: [d.real, d.imag])(D_fresnel(w, v[0] + 1j * v[1], e2, n_glass))
    sol = root(f, [K0.real, K0.imag], method="hybr", tol=1e-13)
    K = sol.x[0] + 1j * sol.x[1]
    return K, abs(D_fresnel(w, K, e2, n_glass)), bool(sol.success)


def real_axis_dispersion(n_glass, lams=np.arange(1150.0, 1400.1, 2.0), K0_over_k0=1.75):
    rows, K = [], None
    for lam in lams:
        k0 = 2 * np.pi / lam
        K0 = K if K is not None else K0_over_k0 * k0 * (1 + 0.05j)
        K, res, ok = solve_K(lam, K0, n_glass)
        rows.append(dict(lam=float(lam), ReK_over_k0=float(K.real / k0), ImK_over_k0=float(K.imag / k0),
                         residual=float(res), converged=ok, eps_ito=[mat.eps_ito(lam).real, mat.eps_ito(lam).imag]))
    return rows


def bare_film_absorption(lams, thetas=(45.0, 60.0, 70.0)):
    """Driven p-polarized absorption of air/ITO/glass (torcwa, order [1,1])."""
    import forward as fwd
    out = {}
    for th in thetas:
        A = []
        for lam in lams:
            with torch.no_grad():
                sim = fwd.build_sim(None, 500.0, 10.0, float(lam), [1, 1], theta_deg=th, pol="p")
                R, T = fwd.rt_all_orders(sim)
            A.append(float(1 - R - T))
        A = np.array(A)
        out[f"theta{th:.0f}"] = dict(A=A.tolist(), lam_peak=float(lams[int(A.argmax())]), A_peak=float(A.max()))
    return out


def main(out=config.OUT / "stage0" / "enz_film_mode.json"):
    lam_ze, _ = mat.ito_zero_crossing()
    ng = mat.n_glass(lam_ze)
    fit = drude_fit(); model = fit.pop("model")
    p = [fit["eps_inf"], fit["wp_rad_fs"], fit["gamma_rad_fs"]]
    eps_fn = lambda w: model(p, w)
    w_ze = 2 * np.pi * C / lam_ze
    # sanity: the validated historical pole convention is reproduced by the '+' form
    poles = []
    for P in [550.0, 650.0, 750.0, 825.0, 850.0, 1000.0]:
        K = 2 * np.pi / P
        best = None
        for lam0 in (1302.0, 1330.0, 1360.0, 1400.0):
            for q in (8.0, 12.0, 20.0):
                w0 = 2 * np.pi * C / lam0 * (1 - 1j / (2 * q))
                w, res, ok = solve_pole_omega(K, eps_fn, ng, w0)
                if ok and res < 1e-9 and w.imag < 0 and 1200 < 2 * np.pi * C / w.real < 1500:
                    cand = dict(lambda_pole_nm=float(2 * np.pi * C / w.real), Q=float(w.real / (2 * abs(w.imag))),
                                gamma_rad_fs=float(abs(w.imag)), residual=float(res))
                    if best is None or abs(cand["lambda_pole_nm"] - lam_ze) < abs(best["lambda_pole_nm"] - lam_ze):
                        best = cand
        poles.append(dict(P=P, K_over_k0=float(K / (w_ze / C)), bound=bool(K > ng * w_ze / C),
                          **(best or dict(lambda_pole_nm=None, Q=None, gamma_rad_fs=None, residual=None))))
    disp = real_axis_dispersion(ng)
    n_conv = sum(d["converged"] for d in disp)
    lams_b = np.arange(1150.0, 1400.1, 2.0)
    berreman = bare_film_absorption(lams_b)
    rep = dict(lambda_ZE=lam_ze, n_glass=ng,
               complex_omega_drude=dict(drude_fit=fit,
                                        check=dict(eps_drude_at_ZE=[eps_fn(w_ze).real, eps_fn(w_ze).imag],
                                                   eps_table_at_ZE=[mat.eps_ito(lam_ze).real, mat.eps_ito(lam_ze).imag]),
                                        poles=poles,
                                        caveat="Drude continuation of a Drude-Lorentz table (rms resid 0.056): moderate confidence"),
               real_axis_tabulated=dict(dispersion=disp, n_converged=int(n_conv), n_total=len(disp),
                                        status=("USED" if n_conv == len(disp) else "NOT CONVERGED - not used in any report"),
                                        note="complex K(lambda) of the bare-film TM branch, tabulated eps, hybr on the normalized D"),
               berreman_driven=dict(lam=lams_b.tolist(), **berreman,
                                    note="bare air/ITO/glass p-pol absorption vs lambda at oblique incidence (torcwa [1,1])"),
               note="bare air/ITO(23)/glass TM mode reference (no a-Si); D = 1 + r12 r23 exp(2 i kz2 d)")
    with open(out, "w") as f:
        json.dump(rep, f, indent=1)
    print("Drude fit:", fit)
    print("complex-omega poles at K=G10(P):", [(r["P"], r["lambda_pole_nm"] and round(r["lambda_pole_nm"], 1), r["Q"] and round(r["Q"], 2)) for r in poles])
    print(f"real-axis K tracker: {n_conv}/{len(disp)} converged -> {rep['real_axis_tabulated']['status']}")
    print("Berreman driven peaks:", {k: (v["lam_peak"], round(v["A_peak"], 3)) for k, v in berreman.items()})
    return rep


if __name__ == "__main__":
    main()
