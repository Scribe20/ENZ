"""Exact post-hoc pole certification of the PARENT resonance (air / a-Si:H / glass).

Reuses the validated extractor of the Jz/Fz campaign unchanged:
    analysis.significant_poles  (AAA rational fit of r(omega) and t(omega), r/t agreement, residue
                                 significance, Rayleigh branch-point exclusion, duplicate merging)
    analysis.rayleigh_wavelengths, analysis.local_refine-style dense rescan.

Because BOTH parent materials are lossless in the supplied data (max |k| = 0 for a-Si:H and 1e-263
for the glass over 1150-1400 nm; R + T = 1 to 1e-6 in every solve), the pole half-width IS the
radiative rate: Q_pole = Q_r.  No loss-scaling continuation (analysis.track_branch / fit_gamma) is
needed for the parent - that machinery is kept for the loaded (real-ITO) certification stage.
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import analysis as an          # noqa: E402
import forward as fwd          # noqa: E402
import materials as mat        # noqa: E402
import parent_fwd as pf        # noqa: E402


def parent_rt(rho, P, h, lams, order):
    """(r, t) specular amplitudes of the parent on a wavelength grid (no ITO)."""
    r, t, R, T = [], [], [], []
    with torch.no_grad():
        for lam in np.atleast_1d(lams):
            sim = fwd.build_sim(rho, P, h, float(lam), order, with_ito=False)
            ri, ti = fwd.specular_rt_amplitudes(sim)
            Ri, Ti = fwd.rt_all_orders(sim)
            r.append(ri); t.append(ti); R.append(float(Ri)); T.append(float(Ti))
    return np.array(r), np.array(t), np.array(R), np.array(T)


def parent_poles(rho, P, h, order, lam_lo=1150.0, lam_hi=1450.0, step=2.0, n_glass=None):
    """All significant radiative poles of the parent in the window, Rayleigh anomalies excluded."""
    ng = float(mat.n_glass(0.5 * (lam_lo + lam_hi))) if n_glass is None else n_glass
    lams = np.arange(lam_lo, lam_hi + 1e-9, step)
    r, t, R, T = parent_rt(rho, P, h, lams, order)
    ray = [x["lam"] for x in an.rayleigh_wavelengths(P, ng, lo=lam_lo, hi=lam_hi)]
    poles = an.significant_poles(lams, r, t, exclude=ray)
    return dict(poles=poles, lams=lams, r=r, t=t, R=R, T=T, rayleigh=ray,
                energy_resid=float(np.max(np.abs(R + T - 1.0))))


def refine_pole(rho, P, h, order, pole, n=41, half_fwhm=6.0, ray=None):
    """Dense local rescan around one pole: sampling-convergence certificate (same protocol as
    analysis.local_refine, applied to the parent r/t)."""
    lam0 = pole["lambda_nm"]; fwhm = lam0 / max(pole["Q"], 1e-6)
    half = max(half_fwhm * fwhm, 4.0)
    lams = np.linspace(lam0 - half, lam0 + half, n)
    r, t, R, T = parent_rt(rho, P, h, lams, order)
    cands = an.significant_poles(lams, r, t, exclude=ray)
    if not cands:
        return dict(converged=False, note="no significant pole in local rescan", span_nm=2 * half, n_points=n)
    q0 = complex(pole["omega_re"], pole["omega_im"])
    q = min(cands, key=lambda p: abs(complex(p["omega_re"], p["omega_im"]) - q0))
    rel = abs(complex(q["omega_re"], q["omega_im"]) - q0) / abs(q0)
    return dict(converged=bool(rel < 0.02), rel_diff=float(rel), lambda_local=q["lambda_nm"],
                Q_local=q["Q"], gamma_local=q["gamma"], span_nm=float(2 * half), n_points=n,
                energy_resid=float(np.max(np.abs(R + T - 1.0))))


def nearest_pole(poles, lam_E):
    return min(poles, key=lambda p: abs(p["lambda_nm"] - lam_E)) if poles else None
