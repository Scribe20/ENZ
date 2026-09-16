"""Temperature-dependent ITO permittivity and electron thermodynamics for the
nonlinear-activation calculation of the frozen JZ best design.

LANE B (literature-model exploratory lane; see NONLINEAR_MODEL_AUDIT.md):
the SNU deck states that its activation curves come from a two-temperature
model of the ITO free electrons ("electron cloud response changes refractive
index", Alam et al. Science 2016), but neither the deck nor the repository
contains the model equations or parameter values.  This module therefore
implements the standard hot-electron model of Alam et al. (2016) with the
band parameters of Liu et al., APL 105, 181117 (2014), and anchors it to the
SUPPLIED cold ITO data so that eps(lambda, 300 K) == ITO_nk.csv exactly:

    eps(lambda, Te) = eps_csv(lambda) + [eps_D(lambda, Te) - eps_D(lambda, 300 K)]
    eps_D(lambda, Te) = eps_inf - wp^2(Te) / (w^2 + i gamma w)            (Drude)
    wp^2(Te) = (e^2/eps0) * int g(E) f(E; mu(Te), Te) / m*(E) dE          (Drude weight of a
                                                                            Kane band)
    m*(E) = m0* (1 + 2 C E),   E(1 + C E) = hbar^2 k^2 / (2 m0*)             (Kane / Liu 2014)
    g(E) = (1/2pi^2) (2 m0*/hbar^2)^{3/2} sqrt(E(1+CE)) (1+2CE)
    N = int g f dE  (conserved: mu(Te) solved from N),  U(Te) = int E g f dE,
    Ce(Te) = dU/dTe,   eps_inf, gamma: Drude fit of ITO_nk.csv (1150-1450 nm),
    gamma held constant in Te (assumption, as in the repository's Stage 18).

Parameters: m0* = 0.3964 m_e, C = 0.4191 eV^-1 (Liu et al. 2014, confirmed by
web-search snippets; the ITO of Liu/Alam had N = 1.5e27 m^-3).  N of OUR film is
NOT taken from the literature: it is calibrated so that wp(300 K) equals the
Drude fit of the supplied ITO_nk.csv.
"""

import json
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import brentq, least_squares

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import materials as mat          # noqa: E402  (JZ package: supplied cold data)

# ---- constants (SI) -------------------------------------------------------
e = 1.602176634e-19
eps0 = 8.8541878128e-12
me = 9.1093837015e-31
hbar = 1.054571817e-34
kB = 1.380649e-23
c0 = 299792458.0
C_NM_FS = 299.792458

# ---- literature band parameters (Liu et al. APL 2014 / Alam et al. 2016) --
M0_STAR = 0.3964 * me          # band-edge effective mass
C_NP = 0.4191                  # eV^-1, first-order non-parabolicity
T_REF = 300.0

# ---- Drude fit of the SUPPLIED cold data (1150-1450 nm) --------------------
def drude_fit(lo=1150.0, hi=1450.0):
    t = mat.ito()
    sel = (t.wl >= lo) & (t.wl <= hi)
    wl = t.wl[sel]; eps = (t.n[sel] + 1j * t.k[sel]) ** 2
    w = 2 * np.pi * C_NM_FS / wl                                # rad/fs

    def model(p, w):
        return p[0] - p[1] ** 2 / (w ** 2 + 1j * p[2] * w)

    def resid(p):
        d = model(p, w) - eps
        return np.concatenate([d.real, d.imag])
    fit = least_squares(resid, [3.4, 2.7, 0.19], bounds=([1, 0.1, 0], [10, 10, 2]))
    err = np.abs(model(fit.x, w) - eps)
    return dict(eps_inf=float(fit.x[0]), wp_rad_fs=float(fit.x[1]), gamma_rad_fs=float(fit.x[2]),
                gamma_meV=float(fit.x[2] * 1e15 * hbar / e * 1e3), rms_resid=float(np.sqrt(np.mean(err ** 2))),
                max_abs_resid=float(err.max()), range_nm=[lo, hi], n_points=int(sel.sum()))


# ---- Kane band thermodynamics --------------------------------------------
_PREF = (1.0 / (2 * np.pi ** 2)) * (2 * M0_STAR / hbar ** 2) ** 1.5       # m^-3 J^-3/2


def dos(E_eV):
    """Density of states per m^3 per eV for the Kane band."""
    E = np.asarray(E_eV) * e
    gam = E * (1 + C_NP * np.asarray(E_eV))
    return _PREF * np.sqrt(np.maximum(gam, 0.0)) * (1 + 2 * C_NP * np.asarray(E_eV)) * e   # per eV


def _grid(mu_eV, T):
    kT = kB * T / e
    Emax = max(3.0, mu_eV + 40 * kT)
    return np.linspace(0.0, Emax, 6001)


def fermi(E_eV, mu_eV, T):
    x = (np.asarray(E_eV) - mu_eV) / (kB * T / e)
    return 0.5 * (1 - np.tanh(0.5 * np.clip(x, -700, 700)))


def moments(mu_eV, T):
    E = _grid(mu_eV, T)
    g, f = dos(E), fermi(E, mu_eV, T)
    N = trapezoid(g * f, E)
    U = trapezoid(E * e * g * f, E)                       # J/m^3
    inv_m = trapezoid(g * f / (M0_STAR * (1 + 2 * C_NP * E)), E)   # sum over electrons of 1/m*
    return N, U, inv_m


def solve_mu(N, T):
    return brentq(lambda mu: moments(mu, T)[0] - N, -3.0, 5.0, xtol=1e-10)


def wp2_of(N, T):
    """Drude weight: wp^2 = (e^2/eps0) * int g f / m* dE   [rad^2/s^2]."""
    mu = solve_mu(N, T)
    return e ** 2 / eps0 * moments(mu, T)[2], mu


def calibrate_N(wp_rad_fs):
    target = (wp_rad_fs * 1e15) ** 2
    return brentq(lambda N: wp2_of(N, T_REF)[0] - target, 1e25, 1e28, xtol=1e18)


class ITOHot:
    """eps(lambda, Te) with exact reduction to the supplied ITO_nk.csv at 300 K."""

    def __init__(self, fit=None, gamma_scale_per_K=0.0):
        self.fit = fit or drude_fit()
        self.eps_inf, self.wp0, self.gamma = self.fit["eps_inf"], self.fit["wp_rad_fs"], self.fit["gamma_rad_fs"]
        self.N = calibrate_N(self.wp0)
        self.wp2_300, self.mu_300 = wp2_of(self.N, T_REF)
        _, self.U_300, _ = moments(self.mu_300, T_REF)
        self.EF = self.mu_300
        self.gamma_scale_per_K = gamma_scale_per_K

    # thermodynamic tables ------------------------------------------------
    def state(self, Te):
        wp2, mu = wp2_of(self.N, Te)
        _, U, _ = moments(mu, Te)
        return dict(Te=float(Te), mu_eV=float(mu), wp2_ratio=float(wp2 / self.wp2_300),
                    wp_rad_fs=float(np.sqrt(wp2) / 1e15), U_J_m3=float(U), dU_J_m3=float(U - self.U_300),
                    m_ratio=float(self.wp2_300 / wp2))          # <m*>(Te)/<m*>(300)

    def table(self, Te_grid):
        rows = [self.state(T) for T in Te_grid]
        Te = np.array([r["Te"] for r in rows]); U = np.array([r["U_J_m3"] for r in rows])
        Ce = np.gradient(U, Te)
        for r, c in zip(rows, Ce):
            r["Ce_J_m3_K"] = float(c)
        return rows

    # permittivity ----------------------------------------------------------
    def eps_drude(self, lam_nm, Te):
        w = 2 * np.pi * C_NM_FS / lam_nm
        wp2 = self.wp2_300 * self.state(Te)["wp2_ratio"] / 1e30
        g = self.gamma * (1 + self.gamma_scale_per_K * (Te - T_REF))
        return self.eps_inf - wp2 / (w ** 2 + 1j * g * w)

    def eps(self, lam_nm, Te, _cache={}):
        key = round(float(Te), 3)
        if key not in _cache:
            _cache[key] = self.state(Te)["wp2_ratio"]
        w = 2 * np.pi * C_NM_FS / lam_nm
        g = self.gamma * (1 + self.gamma_scale_per_K * (Te - T_REF))
        wp2_T = self.wp0 ** 2 * _cache[key]
        d = -(wp2_T - self.wp0 ** 2) / (w ** 2 + 1j * g * w)
        return mat.eps_ito(lam_nm) + d

    def enz_crossing(self, Te, lo=1150.0, hi=1670.0):
        f = lambda l: self.eps(l, Te).real
        grid = np.arange(lo, hi, 2.0)
        v = np.array([f(l) for l in grid])
        s = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
        if len(s) == 0:
            return float("nan")
        return brentq(f, grid[s[0]], grid[s[0] + 1], xtol=1e-6)


def main(out=HERE / "outputs" / "nonlinear_best"):
    out.mkdir(parents=True, exist_ok=True)
    fit = drude_fit()
    m = ITOHot(fit)
    lam_ze, _ = mat.ito_zero_crossing()
    Te_grid = np.array([300, 400, 500, 600, 800, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 4500, 5000, 6000, 7000, 8000, 10000], float)
    rows = m.table(Te_grid)
    for r in rows:
        r["lambda_ENZ_nm"] = m.enz_crossing(r["Te"])
        for lam in (1250.0, 1302.282, 1350.0):
            ep = m.eps(lam, r["Te"])
            r[f"eps1_{lam:.0f}"] = ep.real; r[f"eps2_{lam:.0f}"] = ep.imag
    # 300-K reduction check over the whole window
    lams = np.arange(1150.0, 1600.0, 1.0)
    dev = max(abs(m.eps(l, 300.0) - mat.eps_ito(l)) for l in lams)
    par = dict(lane="B (literature model: Alam 2016 / Liu 2014 band; N calibrated to the supplied cold Drude fit)",
               drude_fit_of_supplied_ITO_nk_csv=fit, m0_star_over_me=M0_STAR / me, C_nonparabolicity_eV_inv=C_NP,
               N_calibrated_m3=m.N, EF_300K_eV=m.EF, U_300K_J_m3=m.U_300,
               gamma_Te_dependence="none (held constant; assumption)",
               eps_300K_reduction_max_abs_dev=float(dev), lambda_ZE_supplied=lam_ze,
               provenance=dict(m0_C="Liu, Chang, Boyd et al., APL 105, 181117 (2014); used by Alam, De Leon, Boyd, Science 352, 795 (2016)",
                               eps_inf_wp_gamma="least-squares Drude fit of the supplied ITO_nk.csv over 1150-1450 nm (this file)",
                               N="calibrated: wp(300 K) of the Kane-band Drude weight = fitted wp",
                               NOT_from_sources="electron-phonon coupling G, lattice heat capacity, gamma(Te), SNU's own parameter values"))
    with open(out / "nonlinear_parameters.json", "w") as f:
        json.dump(par, f, indent=1)
    import csv
    with open(out / "epsilon_vs_Te.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(json.dumps({k: v for k, v in par.items() if k != "provenance"}, indent=1))
    for r in rows:
        print(f"Te={r['Te']:6.0f} K: mu={r['mu_eV']:.4f} eV  wp2/wp2_300={r['wp2_ratio']:.4f}  <m*>/<m*>300={r['m_ratio']:.4f}  "
              f"lambda_ENZ={r['lambda_ENZ_nm']:.1f} nm  Ce={r['Ce_J_m3_K']:.3e} J/m3K  dU={r['dU_J_m3']:.3e} J/m3  eps(1302)={r['eps1_1302']:+.4f}+{r['eps2_1302']:.4f}i")
    return m, rows


if __name__ == "__main__":
    main()
