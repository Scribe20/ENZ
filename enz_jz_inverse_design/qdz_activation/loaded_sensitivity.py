"""STAGE B - real-ITO small-signal TRANSMISSION SENSITIVITY of a frozen geometry (certification/selection,
NOT a differentiable objective).

For a frozen hard-binary design the loaded stack  air / a-Si:H(h, rho) / ITO 23 nm / glass  is solved
TWICE at every wavelength with the validated Jz/Fz machinery (forward.evaluate with the eps_ito override,
exactly as nonlinear_activation_best/build_lookup.py does), once with the physical cold permittivity
eps0(lambda) and once with the perturbed eps1(lambda) of ito_perturbation.py.  Same geometry, same
wavelength, same order, same polarization, same incidence, same z-quadrature, same normalization
(|E_inc| = 1, P_inc = 0.5 P^2), same diffraction-order bookkeeping (R, T = all propagating orders,
both output polarizations; the (0,0) specular power is reported separately from the per-order table).

Per wavelength:
    T0, T1, dT = T1 - T0,  R0, R1, dR,  A0, A1, dA  (A = 1 - R - T)
    F_x, F_y, F_z, F_tot (cold and perturbed), eta_z = F_z / F_tot, <|E_z/E_inc|^2>_ITO
    identity residual F_tot - A  (the only lossy layer is the ITO, so this is the Fourier-truncation error)
    S_T = dT / |eps1 - eps0|      normalized sensitivity (meaningful because, for the native model, the
                                  complex DIRECTION of eps1 - eps0 is fixed by the supplied Drude fit;
                                  see ito_perturbation.py)
    T_00 (specular) cold / perturbed, r, t complex specular amplitudes (cold / perturbed) for poles
The operating wavelength lambda_op is selected POST HOC from the scan (largest positive dT inside the
diffraction-safe window; the largest negative dT and the largest |S_T| are reported next to it), refined
with a fine local rescan, and never assumed to be lambda_E or lambda_r.  The dominant response channel at
lambda_op (T-opening vs absorption -> reflection etc.) is classified from the signs of dT, dR, dA so that
the "absorption -> reflection with no transmission opening" failure mode stays visible.

Heating: the cold-state TOTAL ITO absorption F_tot (per unit area, = A for the supplied lossless a-Si:H
and glass) is the heating metric; eta_z = F_z/F_tot is the separate longitudinal-mechanism diagnostic.
Neither is optimized here.

Numerics: (i) order convergence of dT, T0, T1 at the selected wavelengths across the listed orders with
explicit tolerances -> flag `numerically_unresolved`; (ii) linearity / direction checks with a smaller
and a larger perturbation and with the real-only / imaginary-only projections; (iii) loaded poles (AAA on
the cold and on the perturbed specular amplitudes, Rayleigh anomalies excluded) -> pole shift and width
change under the perturbation; (iv) Rayleigh safety margin of lambda_op.

Angle hook: every routine takes theta_deg / phi_deg / pol and passes them to forward.build_sim, so an
angle ensemble is a loop over theta outside this module (rayleigh.py handles oblique thresholds).
"""
import time

import numpy as np
import torch

import common as cm                       # noqa: F401
import config                             # noqa: E402
import forward as fwd                     # noqa: E402
import materials as mat                   # noqa: E402
import analysis as an                     # noqa: E402  (exact AAA poles)
import parent_fwd as pf                   # noqa: E402
import rayleigh as ry                     # noqa: E402
import ito_perturbation as ip             # noqa: E402

GEO = config.GEO_DTYPE
KEYS_ARR = ("T0", "T1", "dT", "R0", "R1", "dR", "A0", "A1", "dA", "Fx0", "Fy0", "Fz0", "Ftot0", "eta_z0",
            "mean_Ez2_0", "Fz1", "Ftot1", "eta_z1", "mean_Ez2_1", "T00_0", "T00_1", "dT00", "resid0", "resid1",
            "S_T", "abs_deps", "n_orders0")


def _sim_kwargs(theta_deg, phi_deg, pol):
    return dict(theta_deg=float(theta_deg), phi_deg=float(phi_deg), pol=pol)


def evaluate_pair(rho_t, P, h, lam, order, pert, *, n_z=config.Z_SAMPLES_ITO, theta_deg=0.0, phi_deg=0.0, pol="labx"):
    """One wavelength, both material states, identical everything else.  Returns a flat dict."""
    e0, e1 = pert.eps0(lam), pert.eps1(lam)
    kw = _sim_kwargs(theta_deg, phi_deg, pol)
    with torch.no_grad():
        d0 = fwd.evaluate(rho_t, P, h, float(lam), list(order), n_z=n_z, eps_ito=e0, **kw)
        d1 = fwd.evaluate(rho_t, P, h, float(lam), list(order), n_z=n_z, eps_ito=e1, **kw)
        R0, T0, tab0 = fwd.rt_all_orders(d0["sim"], per_order=True)
        R1, T1, tab1 = fwd.rt_all_orders(d1["sim"], per_order=True)
        r0, t0 = fwd.specular_rt_amplitudes(d0["sim"]); r1, t1 = fwd.specular_rt_amplitudes(d1["sim"])
    f0, f1 = fwd.to_floats(d0), fwd.to_floats(d1)
    T00_0 = next((x["T"] for x in tab0 if x["m"] == 0 and x["n"] == 0), 0.0)
    T00_1 = next((x["T"] for x in tab1 if x["m"] == 0 and x["n"] == 0), 0.0)
    dT = f1["T"] - f0["T"]
    de = abs(e1 - e0)
    row = dict(lam=float(lam), eps0=e0, eps1=e1, delta_eps=e1 - e0, abs_deps=de,
               T0=f0["T"], T1=f1["T"], dT=dT, R0=f0["R"], R1=f1["R"], dR=f1["R"] - f0["R"],
               A0=f0["A"], A1=f1["A"], dA=f1["A"] - f0["A"],
               Fx0=f0["Fx"], Fy0=f0["Fy"], Fz0=f0["Fz"], Ftot0=f0["Ftot"], eta_z0=f0["Fz"] / max(f0["Ftot"], 1e-30),
               mean_Ez2_0=f0["mean_Ez2"], Fz1=f1["Fz"], Ftot1=f1["Ftot"], eta_z1=f1["Fz"] / max(f1["Ftot"], 1e-30),
               mean_Ez2_1=f1["mean_Ez2"], T00_0=T00_0, T00_1=T00_1, dT00=T00_1 - T00_0,
               resid0=f0["Ftot"] - f0["A"], resid1=f1["Ftot"] - f1["A"],
               S_T=(dT / de if de > 0 else 0.0), n_orders0=len(tab0), r0=r0, t0=t0, r1=r1, t1=t1,
               order=list(order), n_z=int(n_z), **kw)
    return row


def scan(rho_t, P, h, lams, order, pert, log=None, tag="", **kw):
    rows, t0 = [], time.time()
    for i, lam in enumerate(np.atleast_1d(lams)):
        rows.append(evaluate_pair(rho_t, P, h, float(lam), order, pert, **kw))
        if log and (i % 15 == 0 or i == len(lams) - 1):
            r = rows[-1]
            log(f"  [B {tag}] {lam:7.1f} nm: T0={r['T0']:.4f} T1={r['T1']:.4f} dT={r['dT']:+.4f} A0={r['A0']:.4f} "
                f"Fz0={r['Fz0']:.4f} eta_z={r['eta_z0']:.2f} ({time.time()-t0:.0f} s)")
    return rows


def arrays(rows):
    out = {k: np.array([r[k] for r in rows], float) for k in KEYS_ARR}
    out["lam"] = np.array([r["lam"] for r in rows], float)
    for k in ("r0", "t0", "r1", "t1", "eps0", "eps1", "delta_eps"):
        out[k] = np.array([r[k] for r in rows], complex)
    return out


def classify_channel(r, rel=0.2, floor=1e-5):
    """Which output channel absorbs the change?  Signs of (dT, dR, dA); a component smaller than
    rel * max(|dT|, |dR|, |dA|) (or below an absolute floor) counts as unchanged.  The classification is
    relative so that it stays meaningful for small-signal perturbations."""
    small = max(rel * max(abs(r["dT"]), abs(r["dR"]), abs(r["dA"])), floor)
    s = lambda v: ("+" if v > small else ("-" if v < -small else "0"))
    pat = s(r["dT"]) + s(r["dR"]) + s(r["dA"])
    names = {"+-0": "R->T (reflection to transmission)", "+0-": "A->T (absorption to transmission, useful)",
             "+--": "R,A->T (useful)", "0+-": "A->R (absorption to reflection; NO transmission opening)",
             "-+0": "T->R", "-0+": "T->A", "0-+": "R->A", "000": "negligible", "++-": "A->T,R",
             "--+": "T,R->A", "+-+": "R->T,A", "-++": "T->R,A", "-+-": "T,A->R", "0+0": "R only", "00+": "A only",
             "00-": "A only", "0-0": "R only", "+00": "T only", "-00": "T only", "+++": "inconsistent", "---": "inconsistent",
             "0++": "A,R up", "0--": "A,R down", "-0-": "T,A down", "+0+": "T,A up"}
    return dict(pattern=pat, label=names.get(pat, pat), threshold=small, rel=rel)


def loaded_poles(lams, r, t, exclude):
    try:
        return an.significant_poles(np.asarray(lams), np.asarray(r), np.asarray(t), exclude=exclude)
    except Exception as e:                      # AAA can fail on degenerate data; never block the campaign
        return [dict(error=f"{type(e).__name__}: {e}")]


def nearest_pole(poles, lam):
    ok = [p for p in poles if "lambda_nm" in p]
    return min(ok, key=lambda p: abs(p["lambda_nm"] - lam)) if ok else None


def select_lambda_op(A, safe_lo, safe_hi):
    """Post-hoc operating points inside the diffraction-safe window."""
    m = (A["lam"] >= safe_lo) & (A["lam"] <= safe_hi)
    if not m.any():
        return dict(error="no wavelength inside the safe window")
    idx = np.where(m)[0]
    i_pos = idx[int(np.argmax(A["dT"][idx]))]
    i_neg = idx[int(np.argmin(A["dT"][idx]))]
    i_abs = idx[int(np.argmax(np.abs(A["S_T"][idx])))]
    i_T00 = idx[int(np.argmax(A["dT00"][idx]))]
    return dict(i_pos=int(i_pos), i_neg=int(i_neg), i_absS=int(i_abs), i_pos_specular=int(i_T00),
                lam_pos=float(A["lam"][i_pos]), lam_neg=float(A["lam"][i_neg]), lam_absS=float(A["lam"][i_abs]),
                dT_max_pos=float(A["dT"][i_pos]), dT_max_neg=float(A["dT"][i_neg]),
                S_T_max_abs=float(A["S_T"][i_abs]), window=[float(safe_lo), float(safe_hi)], n_in_window=int(m.sum()))


def point_summary(r, rayleigh, loaded_p0, loaded_p1, extra=None):
    pole0 = nearest_pole(loaded_p0, r["lam"]); pole1 = nearest_pole(loaded_p1, r["lam"])
    d = dict(lam_nm=r["lam"], T0=r["T0"], T1=r["T1"], dT=r["dT"], R0=r["R0"], R1=r["R1"], dR=r["dR"], A0=r["A0"], A1=r["A1"],
             dA=r["dA"], S_T=r["S_T"], abs_delta_eps=r["abs_deps"], eps0=r["eps0"], eps1=r["eps1"],
             Ftot0=r["Ftot0"], Fz0=r["Fz0"], Fx0=r["Fx0"], Fy0=r["Fy0"], eta_z0=r["eta_z0"], mean_Ez2_0=r["mean_Ez2_0"],
             Ftot1=r["Ftot1"], Fz1=r["Fz1"], eta_z1=r["eta_z1"], identity_resid0=r["resid0"],
             T00_0=r["T00_0"], T00_1=r["T00_1"], total_minus_specular_T0=r["T0"] - r["T00_0"], n_propagating_orders=int(r["n_orders0"]),
             channel=classify_channel(r), contrast_ratio=(r["T1"] / r["T0"] if r["T0"] > 0 else None),
             rayleigh_margin=ry.safety_margin(r["lam"], rayleigh, fwhm_nm=(pole0["fwhm_nm"] if pole0 else None)),
             loaded_pole_cold=pole0, loaded_pole_hot=pole1,
             pole_shift_nm=((pole1["lambda_nm"] - pole0["lambda_nm"]) if (pole0 and pole1) else None),
             pole_Q_cold=(pole0["Q"] if pole0 else None), pole_Q_hot=(pole1["Q"] if pole1 else None))
    if extra:
        d.update(extra)
    return d


def coupling_estimate(cand, pert, lam_E, loaded_poles_cold, parent_info=None):
    """FIRST-ORDER estimate of the ITO-induced non-radiative Q of the parent mode from PARENT quantities
    only (no loaded solve):  the energy decay rate through ITO absorption is
        Gamma_nr = <P_abs>/U = omega eps'' INT_ITO |E|^2 dV / INT eps |E|^2 dV
    and with the longitudinal field in the ITO taken from the continuous normal displacement,
    |E_z|^2 = |D_z|^2 / |eps_ITO|^2  (tangential E neglected -> a LOWER bound on the loss), and the
    stored energy taken as the design-layer Parseval energy W_layer,
        Q_nr,est = |eps_ITO|^2 / (eps''_ITO * eta_Dz_P),      gamma_nr / gamma_r = Q_r / Q_nr,est,
    evaluated with the cold eps_ITO at lambda_E.  It assumes the parent field is not changed by the
    loading (weak-perturbation limit), which fails precisely when the ratio is >> 1 - but then the
    qualitative conclusion "over-damped by the ITO" is robust.  Checked against the exact loaded pole Q
    (AAA on the loaded r, t) whenever one is found: Q_loaded,est = 1 / (1/Q_r + 1/Q_nr,est)."""
    c = dict(cand.get("cert") or {})
    src = "certification"
    if (not c.get("Q_r") or c.get("eta_Dz_P_lamE") is None) and parent_info:
        c.update({k: v for k, v in parent_info.items() if v is not None}); src = "Stage-A values computed here (not a certification)"
    if not c.get("Q_r") or c.get("eta_Dz_P_lamE") is None:
        return dict(available=False, reason="no Q_r / eta_Dz_P for this candidate")
    e0 = pert.eps0(lam_E)
    Q_nr = abs(e0) ** 2 / (e0.imag * c["eta_Dz_P_lamE"])
    Q_r = c["Q_r"]
    Q_ld = 1.0 / (1.0 / Q_r + 1.0 / Q_nr)
    near = nearest_pole(loaded_poles_cold, c.get("lambda_r") or lam_E)
    return dict(available=True, source=src, eps_ITO_lamE=e0, eta_Dz_P_lamE=c["eta_Dz_P_lamE"], Q_r=Q_r, Q_nr_est=Q_nr,
                gamma_nr_over_gamma_r=Q_r / Q_nr, Q_loaded_est=Q_ld,
                far_field_visibility_est=Q_ld / Q_r,
                Q_loaded_exact_nearest=(near["Q"] if near else None), lambda_loaded_exact_nearest=(near["lambda_nm"] if near else None),
                eta_Dz_P_for_critical_coupling=abs(e0) ** 2 / (e0.imag * Q_r),
                note="first-order parent-only estimate; tangential E in the ITO neglected (lower bound on loss); compare with the exact loaded pole")


def order_check(rho_t, P, h, lams, orders, pert, tol_abs=0.02, tol_rel=0.2, log=None, tag="", **kw):
    """dT, T0, T1 at each wavelength for every order; convergence flags between consecutive orders."""
    rows = []
    for lam in lams:
        per = []
        for od in orders:
            t0 = time.time()
            r = evaluate_pair(rho_t, P, h, float(lam), od, pert, **kw)
            per.append(dict(order=list(od), T0=r["T0"], T1=r["T1"], dT=r["dT"], R0=r["R0"], A0=r["A0"], Fz0=r["Fz0"],
                            Ftot0=r["Ftot0"], S_T=r["S_T"], wall_s=time.time() - t0))
        d_hi = per[-1]["dT"]; d_lo = per[-2]["dT"] if len(per) > 1 else d_hi
        change = abs(d_hi - d_lo)
        unresolved = bool(change > max(tol_abs, tol_rel * abs(d_hi)))
        rows.append(dict(lam=float(lam), by_order=per, dT_change_last_two=change, dT_rel_change=(change / abs(d_hi) if abs(d_hi) > 0 else None),
                         T0_change_last_two=abs(per[-1]["T0"] - per[-2]["T0"]) if len(per) > 1 else 0.0,
                         numerically_unresolved=unresolved, tol_abs=tol_abs, tol_rel=tol_rel))
        if log:
            log(f"  [conv {tag}] {lam:7.1f} nm: " + " ".join(f"{p['order'][0]}:{p['dT']:+.4f}" for p in per) +
                f"  -> {'UNRESOLVED' if unresolved else 'ok'} (|d|={change:.4f})")
    return rows


def linearity_and_direction(rho_t, P, h, lams, order, pert, Te_list=(600.0, 1000.0, 2000.0), log=None, tag="", **kw):
    """S_T vs perturbation size (linear-response check) and the real / imaginary decomposition."""
    out = []
    for lam in lams:
        rec = dict(lam=float(lam), by_Te=[], decomposition={})
        for Te in Te_list:
            p = ip.ITOPerturbation("hot_electron_Te", Te=Te)
            r = evaluate_pair(rho_t, P, h, float(lam), order, p, **kw)
            rec["by_Te"].append(dict(Te=float(Te), abs_deps=r["abs_deps"], dT=r["dT"], S_T=r["S_T"], dR=r["dR"], dA=r["dA"]))
        S = [b["S_T"] for b in rec["by_Te"]]
        rec["S_T_spread_rel"] = float((max(S) - min(S)) / max(abs(S[0]), 1e-12)) if S else None
        rec["linear_response"] = bool(rec["S_T_spread_rel"] is not None and rec["S_T_spread_rel"] < 0.25)
        for kind in ("real_only", "imag_only"):
            p = ip.ITOPerturbation(kind, Te=pert.Te if pert.kind != "custom" else 1000.0)
            r = evaluate_pair(rho_t, P, h, float(lam), order, p, **kw)
            rec["decomposition"][kind] = dict(dT=r["dT"], dR=r["dR"], dA=r["dA"], delta_eps=r["delta_eps"])
        full = next((b for b in rec["by_Te"] if abs(b["Te"] - pert.Te) < 1e-6), None)
        if full and abs(full["dT"]) > 0:
            rec["fraction_dT_from_real_part"] = float(rec["decomposition"]["real_only"]["dT"] / full["dT"])
            rec["additivity_resid"] = float(rec["decomposition"]["real_only"]["dT"] + rec["decomposition"]["imag_only"]["dT"] - full["dT"])
        out.append(rec)
        if log:
            log(f"  [lin {tag}] {lam:7.1f} nm: S_T(Te)=" + ", ".join(f"{b['Te']:.0f}K:{b['S_T']:+.3f}" for b in rec["by_Te"]) +
                f"  Re-frac={rec.get('fraction_dT_from_real_part')}")
    return out


def stage_b(cand, rho, pert, *, order=(9, 9), lam_lo=None, lam_hi=None, step=2.0, m_R=10.0, fine_half=4.0, fine_step=0.5,
            orders_check=((5, 5), (7, 7), (9, 9)), n_z=config.Z_SAMPLES_ITO, theta_deg=0.0, phi_deg=0.0, pol="labx",
            do_linearity=True, log=print, parent_lambda_r=None, parent_info=None):
    """parent_info (optional): dict(Q_r, lambda_r, eta_Dz_P_lamE, parent_poles=[{lam, Q, fwhm}]) from Stage A,
    used for the refinement windows / coupling estimate when the candidate has no certification."""
    P, h, tag = cand["P"], cand["h"], cand["tag"]
    lam_E = pf.lambda_E(); t0 = time.time()
    rho_t = None if rho is None else torch.as_tensor(rho, dtype=GEO)
    kw = dict(n_z=n_z, theta_deg=theta_deg, phi_deg=phi_deg, pol=pol)
    lam_hi = float(mat.asi().hi) if lam_hi is None else float(lam_hi)
    (safe_lo, safe_hi), ray = ry.single_order_window(P, 900.0, lam_hi, theta_deg=theta_deg, phi_deg=phi_deg)
    safe_lo = safe_lo + m_R if safe_lo > 900.0 + 1e-9 else float(lam_lo or 1160.0)
    lam_lo = float(lam_lo) if lam_lo is not None else max(1160.0, safe_lo - 20.0)
    lams = np.arange(lam_lo, lam_hi + 1e-9, step)
    rows = scan(rho_t, P, h, lams, list(order), pert, log=log, tag=tag, **kw)
    # ---- dense refinement windows: around every PARENT pole in the window (the loaded line is expected
    # within a few nm of it and a 2-nm grid does not resolve Q ~ 500) and around the coarse loaded T0
    # minimum; merged into one non-uniform grid (AAA and the argmax selection work on any grid) --------
    refine = []
    A0 = arrays(rows)
    ppoles = (cand.get("cert") or {}).get("parent_poles") or (parent_info or {}).get("parent_poles") or []
    for p in ppoles:
        if safe_lo - 10 <= p["lam"] <= safe_hi + 10 and p["Q"] > 30:
            fw = max(p["fwhm"], 0.5)
            refine.append(dict(center=p["lam"], half=3.0 * fw, step=max(fw / 5.0, 0.25), reason=f"parent pole Q={p['Q']:.0f}"))
    m0 = (A0["lam"] >= safe_lo) & (A0["lam"] <= safe_hi)
    if m0.any():
        lmin = float(A0["lam"][m0][int(np.argmin(A0["T0"][m0]))])
        refine.append(dict(center=lmin, half=1.5 * step, step=step / 4.0, reason="coarse loaded T0 minimum"))
    have = set(np.round(lams, 6))
    for rw in refine:
        lf = np.arange(rw["center"] - rw["half"], rw["center"] + rw["half"] + 1e-9, rw["step"])
        lf = np.array([l for l in lf if lam_lo <= l <= lam_hi and round(float(l), 6) not in have])[:60]
        rw["n"] = int(len(lf))
        if len(lf):
            rows += scan(rho_t, P, h, lf, list(order), pert, **kw)
            have |= set(np.round(lf, 6))
    rows.sort(key=lambda r: r["lam"])
    lams = np.array([r["lam"] for r in rows])
    A = arrays(rows)
    ray_l = [r["lam"] for r in ray if lam_lo - 5 <= r["lam"] <= lam_hi + 5]
    poles0 = loaded_poles(A["lam"], A["r0"], A["t0"], ray_l)
    poles1 = loaded_poles(A["lam"], A["r1"], A["t1"], ray_l)
    sel = select_lambda_op(A, safe_lo, safe_hi)
    # ---- fine local rescan around the coarse positive optimum ---------------------------------
    fine = None
    if "i_pos" in sel:
        lf = np.arange(max(sel["lam_pos"] - fine_half, safe_lo), min(sel["lam_pos"] + fine_half, safe_hi) + 1e-9, fine_step)
        rf = scan(rho_t, P, h, lf, list(order), pert, **kw)
        Af = arrays(rf)
        j = int(np.argmax(Af["dT"]))
        fine = dict(lam=[float(x) for x in Af["lam"]], dT=[float(x) for x in Af["dT"]], T0=[float(x) for x in Af["T0"]],
                    T1=[float(x) for x in Af["T1"]], lam_pos_fine=float(Af["lam"][j]), dT_max_pos_fine=float(Af["dT"][j]),
                    row=rf[j])
    r_pos = fine["row"] if fine else (rows[sel["i_pos"]] if "i_pos" in sel else None)
    iE = int(np.argmin(np.abs(A["lam"] - lam_E)))
    pts = {}
    if r_pos is not None:
        pts["lambda_op_pos"] = point_summary(r_pos, ray, poles0, poles1, extra=dict(selection="max positive dT in the safe window (fine rescan)"))
        pts["lambda_op_neg"] = point_summary(rows[sel["i_neg"]], ray, poles0, poles1, extra=dict(selection="max negative dT in the safe window"))
        pts["lambda_op_absS"] = point_summary(rows[sel["i_absS"]], ray, poles0, poles1, extra=dict(selection="max |S_T| in the safe window"))
    pts["lambda_E"] = point_summary(rows[iE], ray, poles0, poles1, extra=dict(selection="ENZ crossing of the supplied ITO (reference, not selected)"))
    if parent_lambda_r and lam_lo <= parent_lambda_r <= lam_hi:
        rr = evaluate_pair(rho_t, P, h, float(parent_lambda_r), list(order), pert, **kw)
        pts["parent_lambda_r"] = point_summary(rr, ray, poles0, poles1, extra=dict(selection="certified PARENT pole wavelength (reference)"))
    # ---- global scan descriptors -------------------------------------------------------------
    m = (A["lam"] >= safe_lo) & (A["lam"] <= safe_hi)
    glob = dict(T0_max_in_window=float(A["T0"][m].max()), T0_min_in_window=float(A["T0"][m].min()),
                lam_T0_min=float(A["lam"][m][int(np.argmin(A["T0"][m]))]), A0_max_in_window=float(A["A0"][m].max()),
                lam_A0_max=float(A["lam"][m][int(np.argmax(A["A0"][m]))]), Ftot0_max=float(A["Ftot0"][m].max()),
                Fz0_max=float(A["Fz0"][m].max()), lam_Fz0_max=float(A["lam"][m][int(np.argmax(A["Fz0"][m]))]),
                dR_max_pos=float(A["dR"][m].max()), lam_dR_max=float(A["lam"][m][int(np.argmax(A["dR"][m]))]),
                dA_min=float(A["dA"][m].min()), max_identity_resid=float(np.abs(A["resid0"][m]).max()),
                max_total_minus_specular=float(np.abs(A["T0"][m] - A["T00_0"][m]).max()),
                A_to_R_dominant=bool(abs(A["dR"][m]).max() > 2 * abs(A["dT"][m]).max()))
    # ---- convergence + linearity at the selected points ---------------------------------------
    chk_l = sorted({round(pts[k]["lam_nm"], 3) for k in pts if k in ("lambda_op_pos", "lambda_op_neg", "lambda_E")})
    conv = order_check(rho_t, P, h, chk_l, [list(o) for o in orders_check], pert, log=log, tag=tag, **kw) if orders_check else []
    lin = linearity_and_direction(rho_t, P, h, [pts["lambda_op_pos"]["lam_nm"]] if "lambda_op_pos" in pts else [], list(order), pert,
                                  log=log, tag=tag, **kw) if (do_linearity and pert.kind != "custom") else []
    unres = {c["lam"]: c["numerically_unresolved"] for c in conv}
    for k in ("lambda_op_pos", "lambda_op_neg", "lambda_E"):
        if k in pts:
            pts[k]["numerically_unresolved"] = unres.get(round(pts[k]["lam_nm"], 3))
    cc = coupling_estimate(cand, pert, lam_E, poles0, parent_info=parent_info)
    res = dict(tag=tag, stage="B", P=P, h=h, lam_E=lam_E, order=list(order), n_z=int(n_z), incidence=dict(theta_deg=theta_deg, phi_deg=phi_deg, pol=pol),
               critical_coupling_estimate=cc,
               grid=dict(lam_lo=lam_lo, lam_hi=lam_hi, step=step, n=len(lams), fine_half_nm=fine_half, fine_step_nm=fine_step,
                         refinement_windows=refine),
               safe_window=[safe_lo, safe_hi], m_R_nm=m_R, rayleigh=ray,
               perturbation=pert.describe(lams=[pts[k]["lam_nm"] for k in pts]),
               selection=sel, fine_rescan=(None if fine is None else {k: v for k, v in fine.items() if k != "row"}),
               points=pts, global_scan=glob, loaded_poles_cold=poles0, loaded_poles_hot=poles1,
               order_convergence=conv, linearity_direction=lin, wall_s=time.time() - t0)
    if log:
        p = pts.get("lambda_op_pos", {})
        log(f"[B {tag}] lam_op+={p.get('lam_nm')} dT={p.get('dT')} T0={p.get('T0')} T1={p.get('T1')} A0={p.get('A0')} Ftot0={p.get('Ftot0')} "
            f"eta_z={p.get('eta_z0')} S_T={p.get('S_T')} chan={p.get('channel', {}).get('label')} | dT-={sel.get('dT_max_neg')}@{sel.get('lam_neg')} "
            f"| A->R dominant={glob['A_to_R_dominant']} ({res['wall_s']:.0f} s)")
    return res, A
