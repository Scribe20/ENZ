"""Channel-agnostic r/t pole reconnaissance and loss-scaling branch tracking
with FIELD-OVERLAP continuity (generalizes enz_highq_enz_campaign/
stage_a_decompose: same loss-scaling philosophy, same AAA machinery, but
per-call P/h/angle and explicit branch identity checks).

Pole acceptance (as pole_rt): found in BOTH r and t within RT_TOL, damped,
significant Lorentzian peak contribution (PEAK_FRAC) in both.
Branch tracking: at each loss level s the candidate poles are ranked by
complex-frequency distance to the previous level AND by the normalized
overlap of the driven ITO Ez map at the pole wavelength with the previous
level's map; a jump is flagged when the chosen pole's overlap < OVL_MIN or
when the distance-nearest and overlap-best candidates disagree.
High-Q certification: after a pole is found, a dense local rescan
(+-6 FWHM, 41 points) is fitted separately; the pole is 'sampling-converged'
when the local and global fits agree within STAB_TOL.
"""

import numpy as np
import torch
from scipy.interpolate import AAA

import common as cm
import forward_multi as fm

C = cm.C_NM_FS
RT_TOL, DAMP_MIN, PEAK_FRAC, STAB_TOL, OVL_MIN = 0.02, 5e-4, 0.03, 0.02, 0.6
SCAN_DENSE = np.arange(1280.0, 1620.1, 4.0)       # reconnaissance (86 pts)
SCAN_COARSE = np.arange(1300.0, 1600.1, 6.0)      # tracking (51 pts)
WINDOW = (1309.9, 1557.1)                          # ENZ band (bare-film HWHM)


def rt_scan(rho, P, h, s, lams, theta=0.0, phi=0.0, order=cm.ORDER,
            with_ito=True):
    r, t = [], []
    with torch.no_grad():
        for lam in lams:
            sim = fm.build_sim(rho, P, h, lam=lam, theta_deg=theta, phi_deg=phi,
                               order=order, with_ito=with_ito, ito_loss_scale=s)
            rr = sim.S_parameters(orders=[0, 0], direction="forward", port="reflection",
                                  polarization="xx", ref_order=[0, 0], power_norm=False)
            tt = sim.S_parameters(orders=[0, 0], direction="forward", port="transmission",
                                  polarization="xx", ref_order=[0, 0], power_norm=False)
            r.append(complex(rr.ravel()[0])); t.append(complex(tt.ravel()[0]))
    return np.array(r), np.array(t)


def _aaa_poles(lams, vals):
    oms = 2 * np.pi * C / np.asarray(lams)
    fit = AAA(oms, vals)
    return [(complex(q), abs(res)) for q, res in zip(fit.poles(), fit.residues())
            if q.imag < -DAMP_MIN]


def significant_poles(lams, r, t):
    """All r/t-agreeing, significant poles (any wavelength) with metrics."""
    pr, pt = _aaa_poles(lams, r), _aaa_poles(lams, t)
    sr, st = float(np.max(np.abs(r))), float(np.max(np.abs(t)))
    out = []
    for q, a in pr:
        if not pt:
            continue
        m = min(pt, key=lambda p: abs(p[0] - q))
        if abs(m[0] - q) / abs(q) >= RT_TOL:
            continue
        peak_r, peak_t = a / abs(q.imag) / sr, m[1] / abs(m[0].imag) / st
        if peak_r < PEAK_FRAC or peak_t < PEAK_FRAC:
            continue
        qa = 0.5 * (q + m[0])
        lam = 2 * np.pi * C / qa.real
        out.append(dict(omega=qa, lambda_nm=lam, Q=abs(qa.real / (2 * qa.imag)),
                        gamma=abs(qa.imag), res_r=a, res_t=m[1], peak_r=peak_r,
                        peak_t=peak_t, rt_rel_diff=abs(m[0] - q) / abs(q),
                        in_window=WINDOW[0] <= lam <= WINDOW[1]))
    return sorted(out, key=lambda p: p["lambda_nm"])


def local_refine(rho, P, h, s, pole, theta=0.0, phi=0.0, order=cm.ORDER,
                 n=41, half_fwhm=6.0):
    """Dense local rescan around a pole: sampling-convergence certificate."""
    lam0 = pole["lambda_nm"]; fwhm = lam0 / max(pole["Q"], 1e-6)
    half = max(half_fwhm * fwhm, 6.0)
    lams = np.linspace(max(lam0 - half, 1200.0), min(lam0 + half, 1750.0), n)
    r, t = rt_scan(rho, P, h, s, lams, theta, phi, order)
    cands = significant_poles(lams, r, t)
    if not cands:
        return dict(converged=False, note="no significant pole in local rescan")
    q = min(cands, key=lambda p: abs(p["omega"] - pole["omega"]))
    rel = abs(q["omega"] - pole["omega"]) / abs(pole["omega"])
    return dict(converged=bool(rel < STAB_TOL), rel_diff=rel,
                lambda_local=q["lambda_nm"], Q_local=q["Q"], span_nm=2 * half,
                n_points=n)


def ez_map(rho, P, h, lam, s, theta=0.0, phi=0.0, order=cm.ORDER, n=48):
    with torch.no_grad():
        sim = fm.build_sim(rho, P, h, lam=lam, theta_deg=theta, phi_deg=phi,
                           order=order, ito_loss_scale=s)
        x, y = fm.cell_axes(P, n)
        E, _ = sim.field_xy(1, x, y, cm.D_ITO / 2)
    return E[2].numpy()


def overlap(a, b):
    return float(abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def track_branch(rho, P, h, omega0, s_levels=cm.S_LEVELS, theta=0.0, phi=0.0,
                 order=cm.ORDER, lams=SCAN_COARSE, log=print, refine_high_q=True,
                 tag=""):
    """Follow one pole branch through the loss levels.  omega0: complex
    starting pole at s = s_levels[0] (from a reconnaissance scan)."""
    rows, prev, prev_map = [], None, None
    for s in s_levels:
        r, t = rt_scan(rho, P, h, s, lams, theta, phi, order)
        cands = significant_poles(lams, r, t)
        ref = prev["omega"] if prev is not None else omega0
        if not cands:
            log(f"  [{tag}] s={s:.2f}: no significant pole"); continue
        by_dist = sorted(cands, key=lambda p: abs(p["omega"] - ref))
        near = [p for p in by_dist if abs(p["omega"] - ref) / abs(ref) < 0.06] or by_dist[:1]
        chosen, ovl, ovl_alt = near[0], None, {}
        if prev_map is not None and len(near) > 1:
            for p in near[:3]:
                p["_map"] = ez_map(rho, P, h, p["lambda_nm"], s, theta, phi, order)
                ovl_alt[f"{p['lambda_nm']:.1f}"] = overlap(prev_map, p["_map"])
            best = max(near[:3], key=lambda p: overlap(prev_map, p["_map"]))
            chosen = best
        cmap = chosen.get("_map") if "_map" in chosen else ez_map(rho, P, h, chosen["lambda_nm"], s, theta, phi, order)
        ovl = overlap(prev_map, cmap) if prev_map is not None else 1.0
        jump = bool(ovl < OVL_MIN or (len(near) > 1 and near[0] is not chosen))
        row = dict(s=s, lambda_pole=chosen["lambda_nm"], Q=chosen["Q"], gamma=chosen["gamma"],
                   omega_re=chosen["omega"].real, omega_im=chosen["omega"].imag,
                   res_r=chosen["res_r"], res_t=chosen["res_t"], peak_r=chosen["peak_r"],
                   peak_t=chosen["peak_t"], rt_rel_diff=chosen["rt_rel_diff"],
                   overlap_prev=ovl, alternatives=ovl_alt, jump_flag=jump,
                   n_candidates_near=len(near))
        if refine_high_q and chosen["Q"] > 25:
            row["local_refine"] = local_refine(rho, P, h, s, chosen, theta, phi, order)
        rows.append(row)
        log(f"  [{tag}] s={s:.2f}: pole {chosen['lambda_nm']:8.2f} nm Q={chosen['Q']:8.2f} "
            f"gamma={chosen['gamma']:.5f} ovl={ovl:.3f}{' JUMP?' if jump else ''}"
            + (f" refine:{row['local_refine'].get('converged')}" if 'local_refine' in row else ""))
        prev, prev_map = chosen, cmap
    return rows


def fit_gamma(rows):
    """gamma(s) = gamma_rad + s*gamma_nr on the non-jumping rows."""
    use = [r for r in rows if not r.get("jump_flag")] or rows
    if len(use) < 3:
        return dict(error="insufficient tracked levels", n=len(use))
    S = np.array([r["s"] for r in use]); G = np.array([r["gamma"] for r in use])
    slope, icpt = np.polyfit(S, G, 1)
    resid = float(np.max(np.abs(G - (slope * S + icpt))) / G.max())
    w0 = use[0]["omega_re"]
    g_rad, g_nr = float(icpt), float(slope)
    lossless = next((r for r in rows if r["s"] == 0.0 and not r.get("jump_flag")), None)
    return dict(gamma_rad=g_rad, gamma_nr=g_nr,
                Q_rad=(w0 / (2 * g_rad) if g_rad > 0 else float("inf")),
                Q_nr=w0 / (2 * g_nr) if g_nr > 0 else float("inf"),
                Q_loaded=rows[0]["Q"], gamma_ratio=g_rad / g_nr if g_nr > 0 else float("inf"),
                linearity_resid=resid, n_levels=len(use),
                lossless_pole=(dict(lambda_nm=lossless["lambda_pole"], Q=lossless["Q"]) if lossless else None),
                Q_rad_from_lossless=(lossless["Q"] if lossless else None),
                pole_shift_nm=(rows[0]["lambda_pole"] - lossless["lambda_pole"]) if lossless else None)
