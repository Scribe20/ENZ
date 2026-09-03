"""Channel-agnostic scattering-pole extraction from the zeroth-order S matrix.

Method: sample the complex specular amplitudes r_xx(omega) and t_xx(omega)
(TORCWA S_parameters, power_norm=False -> pure amplitudes, analytic in
omega) on the real axis, fit each with AAA, and accept a pole only if
  * it lies inside the ENZ target window (defined from the bare-film ENZ
    resonance width, NOT from any momentum channel),
  * it is found consistently in BOTH r and t (relative distance < RT_TOL),
  * its residue is non-negligible within the window (>= RES_FRAC of the
    largest in-window residue of that observable),
  * it is stable under 2x coarser resampling (< STAB_TOL).
No QNM overlap, harmonic, or multipole information enters the selection.
Poles outside the window (e.g. the padded class's ~1300 nm Si Mie pole) are
reported for transparency but cannot certify the ENZ resonance.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import AAA

PKG = Path(__file__).resolve().parent.parent / "enz_inverse_design"
sys.path.insert(0, str(PKG))
import config                                   # noqa: E402
import torcwa_forward as fwd                    # noqa: E402
from validate_with_without_ito import build_sim  # noqa: E402

C_NM_FS = 299.792458

LAMBDA_E = 1433.488
Q_ENZ_BARE = 5.80                       # validated bare-film QNM at G10(850)
HWHM_ENZ = LAMBDA_E / (2 * Q_ENZ_BARE)  # 123.6 nm
WINDOW = (LAMBDA_E - HWHM_ENZ, LAMBDA_E + HWHM_ENZ)   # (1309.9, 1557.1) nm
SCAN = np.arange(1250.0, 1701.0, 10.0)
RT_TOL, RES_FRAC, STAB_TOL, DAMP_MIN = 0.02, 0.05, 0.02, 0.002
PEAK_FRAC = 0.05   # |res|/|Im w| (Lorentzian peak contribution) >= 5% of max|f|


def rt_scan(rho_t, with_ito, lams=SCAN, order=None):
    r, t = [], []
    with torch.no_grad():
        for lam in lams:
            sim = build_sim(rho_t, lam, with_ito, order=order)
            rr = sim.S_parameters(orders=[0, 0], direction="forward",
                                  port="reflection", polarization="xx",
                                  ref_order=[0, 0], power_norm=False)
            tt = sim.S_parameters(orders=[0, 0], direction="forward",
                                  port="transmission", polarization="xx",
                                  ref_order=[0, 0], power_norm=False)
            r.append(complex(rr.cpu().numpy().ravel()[0]))
            t.append(complex(tt.cpu().numpy().ravel()[0]))
    return np.array(r), np.array(t)


def _poles(oms, vals):
    fit = AAA(oms, vals)
    out = []
    for q, res in zip(fit.poles(), fit.residues()):
        if q.imag < -DAMP_MIN:
            out.append((complex(q), abs(res)))
    return out


def _in_window(q):
    lam = 2 * np.pi * C_NM_FS / q.real
    return WINDOW[0] <= lam <= WINDOW[1]


def select_pole(lams, r, t):
    """Returns dict with the certified pole (or None), the list of ALL
    certified in-window poles, and the full table.

    Significance: r and t of one structure share rational structure, so
    r/t agreement alone admits Froissart doublets (bare ITO produced a
    spurious agreeing pair).  A physical pole must also contribute
    visibly: |res|/|Im w| >= PEAK_FRAC * max|f| for BOTH r and t."""
    oms = 2 * np.pi * C_NM_FS / lams
    pr, pt = _poles(oms, r), _poles(oms, t)
    scale_r, scale_t = float(np.max(np.abs(r))), float(np.max(np.abs(t)))
    max_r = max([a for q, a in pr if _in_window(q)] + [1e-300])
    max_t = max([a for q, a in pt if _in_window(q)] + [1e-300])
    table, certified_all, best = [], [], None
    for q_r, a_r in pr:
        match = min(pt, key=lambda p: abs(p[0] - q_r)) if pt else None
        agree = (match is not None
                 and abs(match[0] - q_r) / abs(q_r) < RT_TOL)
        lam = 2 * np.pi * C_NM_FS / q_r.real
        peak_r = a_r / abs(q_r.imag) / scale_r
        peak_t = (match[1] / abs(match[0].imag) / scale_t) if agree else np.nan
        row = dict(lambda_nm=lam, Q=abs(q_r.real / (2 * q_r.imag)),
                   res_r_norm=a_r / max_r if _in_window(q_r) else a_r,
                   rt_agree=agree, in_window=_in_window(q_r),
                   res_t_norm=(match[1] / max_t) if agree else np.nan,
                   peak_r=peak_r, peak_t=peak_t,
                   significant=bool(agree and peak_r >= PEAK_FRAC
                                    and peak_t >= PEAK_FRAC))
        table.append(row)
        if (row["in_window"] and agree and row["significant"]
                and row["res_r_norm"] >= RES_FRAC
                and row["res_t_norm"] >= RES_FRAC):
            score = row["res_r_norm"] + row["res_t_norm"]
            certified_all.append(dict(lambda_nm=lam, Q=row["Q"],
                                      peak_r=peak_r, peak_t=peak_t))
            if best is None or score > best[0]:
                best = (score, q_r, match[0], row)
    if best is None:
        return dict(certified=None, certified_all=certified_all, table=table)
    _, q_r, q_t, row = best
    # resampling stability (2x coarser), on r
    pr2 = _poles(oms[::2], r[::2])
    q2 = min(pr2, key=lambda p: abs(p[0] - q_r))[0] if pr2 else None
    stab = abs(q2 - q_r) / abs(q_r) if q2 is not None else np.inf
    q_avg = 0.5 * (q_r + q_t)
    return dict(certified=dict(
        lambda_pole_nm=2 * np.pi * C_NM_FS / q_avg.real,
        Q_pole=abs(q_avg.real / (2 * q_avg.imag)),
        rt_rel_diff=abs(q_r - q_t) / abs(q_r),
        stability_rel=stab, stable=stab < STAB_TOL,
        res_r_norm=row["res_r_norm"], res_t_norm=row["res_t_norm"],
        peak_r=row["peak_r"], peak_t=row["peak_t"]),
        certified_all=certified_all, table=table)


def certify(rho_t, with_ito=True, order=None):
    r, t = rt_scan(rho_t, with_ito, order=order)
    out = select_pole(SCAN, r, t)
    out["r"], out["t"] = r, t
    return out
