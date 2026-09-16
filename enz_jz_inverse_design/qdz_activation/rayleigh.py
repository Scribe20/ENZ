"""Rayleigh (diffraction-threshold) wavelengths with the SUPPLIED dispersive glass index, at any
incidence angle, plus the safety-margin metric used by the selection stage.

A diffraction order (m, n) starts to propagate in a medium of index n_med when
    |k_par + G_mn| = n_med k0,   k_par = k0 sin(theta) (cos(phi), sin(phi)),  G_mn = 2 pi (m, n) / P
i.e. when   (sin(theta) cos(phi) + m lambda/P)^2 + (sin(theta) sin(phi) + n lambda/P)^2 = n_med(lambda)^2.
At normal incidence this is lambda_R = n_med(lambda) P / sqrt(m^2 + n^2), solved self-consistently with
the tabulated n_glass(lambda) (analysis.rayleigh_wavelengths uses one fixed n_glass; the two agree to
< 1 nm over the window and both are reported).  Branch points of r(omega), t(omega) sit at these
wavelengths, so AAA pole extraction excludes them and this package keeps lambda_op away from them.
"""
import numpy as np
from scipy.optimize import brentq

import common as cm                       # noqa: F401
import materials as mat                   # noqa: E402


def _n_med(lam, medium):
    return mat.n_glass(lam) if medium == "glass" else 1.0


def rayleigh_wavelengths(P, lo=1000.0, hi=1400.0, theta_deg=0.0, phi_deg=0.0, max_order=3):
    """All Rayleigh wavelengths in [lo, hi] for glass and air, orders |m|,|n| <= max_order."""
    th, ph = np.radians(theta_deg), np.radians(phi_deg)
    sx, sy = np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph)
    out = []
    grid = np.linspace(lo, hi, 801)
    for m in range(-max_order, max_order + 1):
        for n in range(-max_order, max_order + 1):
            if m == 0 and n == 0:
                continue
            for medium in ("glass", "air"):
                f = lambda l: (sx + m * l / P) ** 2 + (sy + n * l / P) ** 2 - _n_med(l, medium) ** 2
                v = np.array([f(l) for l in grid])
                s = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
                for i in s:
                    lam = brentq(f, grid[i], grid[i + 1], xtol=1e-6)
                    out.append(dict(m=int(m), n=int(n), medium=medium, lam=float(lam)))
    # merge duplicates (symmetric orders coincide at normal incidence)
    kept = []
    for r in sorted(out, key=lambda r: r["lam"]):
        if all(abs(r["lam"] - k["lam"]) > 1e-6 or r["medium"] != k["medium"] for k in kept):
            kept.append(r)
    return kept


def safety_margin(lam_op, rayleigh, fwhm_nm=None):
    """Distance of lambda_op to the nearest Rayleigh wavelength: nm, relative, and in linewidths."""
    if not rayleigh:
        return dict(nearest=None, margin_nm=None, margin_rel=None, margin_linewidths=None)
    r = min(rayleigh, key=lambda x: abs(x["lam"] - lam_op))
    d = float(lam_op - r["lam"])
    return dict(nearest=r, margin_nm=d, margin_rel=d / r["lam"],
                margin_linewidths=(d / fwhm_nm if fwhm_nm else None), above=bool(d > 0))


def single_order_window(P, lo, hi, theta_deg=0.0, phi_deg=0.0):
    """[lambda_min, hi] in which ONLY the (0,0) order propagates in both media (total T == specular T)."""
    ray = rayleigh_wavelengths(P, lo=min(lo, 600.0), hi=hi, theta_deg=theta_deg, phi_deg=phi_deg)
    top = max([r["lam"] for r in ray], default=-np.inf)
    return (max(lo, top), hi), ray
