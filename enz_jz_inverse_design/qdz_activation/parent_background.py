"""STAGE A - is the no-ITO parent a USEFUL photonic state?  T_bg, resonance contrast, visibility.

The existing certification (qdz_parent/certify_parent.py) characterises a parent by the exact pole
(lambda_r, Q_r) and by eta_Dz / eta_Dz_P at lambda_E and lambda_r.  It does not say whether the
resonance is visible in the far field on a transmissive background, which is what a transmission-mode
activation needs.  This stage adds, for a FROZEN hard-binary parent (no ITO):

  T_bg     background total transmission of the parent over a SYSTEMATIC off-resonance set  L_bg
  C_res    resonance contrast  |T_bg - T(lambda_r)|  (pole centre)  and  the extremum form
           C_res_ext = max_{|lambda - lambda_r| <= k_loc FWHM} |T_bg - T(lambda)|  (a Fano feature's
           far-field extremum is generally offset from the pole centre), with the local modulation
           depth  T_max - T_min  over the same local window.

The off-resonance set is NOT arbitrary.  It is built from
  (1) the ENZ operating window   W_op = [lambda_E - D_op, lambda_E + D_op]           (D_op sweepable)
  (2) the single-order window    lambda > lambda_R(glass, 1 0)(P) + m_R                (Rayleigh margin)
  (3) the supplied-material window lambda <= 1400 nm (end of the a-Si:H file)
  (4) minus  +-min(k_excl FWHM, hw_max)  around EVERY significant parent pole (not only the one
      nearest lambda_E), from the certified pole list (or an exact-pole pass computed here when no
      certification exists).  The cap hw_max (default D_op / 2) exists because a broad low-Q pole
      (Q ~ 20, FWHM ~ 60 nm) IS the background of the window: excluding 3 FWHM of it would empty the
      set - when that happens the background is reported as undefined ("resonance as broad as the
      window"), which is itself a result, and the cap activity is recorded per pole.
The surviving wavelengths, all parameters (k_excl, hw_max, m_R, D_op) and the full parent R(lambda),
T(lambda) arrays are stored, so the selection stage can re-derive T_bg for other (k_excl, D_op) choices
without re-solving (threshold / definition sweeps).  T is the TOTAL transmitted power over all
propagating orders (forward.rt_all_orders); in the single-order window it coincides with the specular
(0,0) power, verified numerically at three wavelengths per candidate with the per-order table.

Spectra: the certification stores R(lambda), T(lambda) at the certification order ([9,9],
1160-1400 nm, 2 nm) in parent_spectrum.npz; that file is used when present (its order is recorded);
otherwise the spectrum is computed here (one pass, which also yields the exact poles).  The local window
around lambda_r is always re-solved densely at the spectrum order, because a 2-nm grid does not resolve
a Q ~ 500 line (FWHM 2.6 nm).  The parent metrics eta_Dz / eta_Dz_P / U_mid at lambda_E and lambda_r are
recomputed here at the same order for every candidate: for certified ones this is a REGRESSION CHECK
against certification.json (recorded as `regression_vs_certification`), for uncertified ones it is the
only value available and is labelled as such.
"""
import time
from pathlib import Path

import numpy as np
import torch

import common as cm                       # noqa: F401
import config                             # noqa: E402
import forward as fwd                     # noqa: E402
import materials as mat                   # noqa: E402
import analysis as an                     # noqa: E402
import parent_fwd as pf                   # noqa: E402  (qdz_parent: lambda_E, parent_metrics)
import poles_parent as pp                 # noqa: E402  (qdz_parent: exact parent poles / parent_rt)
import rayleigh as ry                     # noqa: E402

GEO = config.GEO_DTYPE
LAM_ASI_MAX = float(mat.asi().hi)          # 1400 nm: end of the supplied a-Si:H file


def per_order_table(rho, P, h, lam, order, with_ito=False, **kw):
    with torch.no_grad():
        sim = fwd.build_sim(rho, P, h, float(lam), order, with_ito=with_ito, **kw)
        R, T, tab = fwd.rt_all_orders(sim, per_order=True)
    t00 = next((x for x in tab if x["m"] == 0 and x["n"] == 0), dict(R=0.0, T=0.0))
    return dict(lam=float(lam), R_tot=float(R), T_tot=float(T), T_00=t00["T"], R_00=t00["R"],
                n_propagating_orders=len(tab), orders=tab,
                total_minus_specular_T=float(T) - t00["T"])


def background_set(lams, T, R, poles, lam_E, lam_R, *, D_op=50.0, m_R=10.0, k_excl=3.0, hw_max=None,
                   lam_lo=1160.0, lam_hi=LAM_ASI_MAX, n_min=5):
    """The systematic off-resonance set and T_bg statistics (pure numpy: re-usable post hoc)."""
    hw_max = 0.5 * D_op if hw_max is None else hw_max
    w_lo, w_hi = max(lam_E - D_op, lam_R + m_R, lam_lo), min(lam_E + D_op, lam_hi)
    keep = (lams >= w_lo) & (lams <= w_hi)
    excl = []
    for p in poles:
        hw_raw = k_excl * max(p["fwhm"], 1e-6)
        hw = min(hw_raw, hw_max)
        excl.append(dict(lam=p["lam"], Q=p["Q"], fwhm=p["fwhm"], excluded=[p["lam"] - hw, p["lam"] + hw],
                         cap_active=bool(hw_raw > hw_max)))
        keep &= np.abs(lams - p["lam"]) > hw
    ok = int(keep.sum()) >= n_min
    Tb, Rb = T[keep], R[keep]
    return dict(n_points=int(keep.sum()), defined=ok, window=[float(w_lo), float(w_hi)],
                lambdas_nm=[float(x) for x in lams[keep]],
                T_bg_median=(float(np.median(Tb)) if ok else None), T_bg_mean=(float(np.mean(Tb)) if ok else None),
                T_bg_min=(float(np.min(Tb)) if ok else None), T_bg_max=(float(np.max(Tb)) if ok else None),
                T_bg_std=(float(np.std(Tb)) if ok else None), R_bg_median=(float(np.median(Rb)) if ok else None),
                parameters=dict(D_op_nm=D_op, m_R_nm=m_R, k_excl=k_excl, hw_max_nm=hw_max, n_min=n_min),
                exclusions=excl,
                note=(None if ok else "background undefined: after excluding the parent poles fewer than n_min "
                                      "samples remain (resonances as broad as the operating window)"))


def stage_a(cand, rho, *, order=(9, 9), D_op=50.0, m_R=10.0, k_excl=3.0, hw_max=None, k_loc=3.0, n_loc=25,
            lam_lo=1160.0, lam_hi=LAM_ASI_MAX, step=2.0, log=print):
    P, h, tag = cand["P"], cand["h"], cand["tag"]
    lam_E = pf.lambda_E()
    t0 = time.time()
    rho_t = None if rho is None else torch.as_tensor(rho, dtype=GEO)
    ng = float(mat.n_glass(1300.0))
    ray = ry.rayleigh_wavelengths(P, lo=900.0, hi=lam_hi)
    lam_R = max([r["lam"] for r in ray if r["lam"] < lam_hi], default=0.0)      # highest anomaly below the window end
    # ---- spectrum + poles: reuse the certification spectrum / pole list if present -------------
    spec_src, spec_order, lams, R, T = None, list(order), None, None, None
    if cand.get("cert_path"):
        f = Path(cand["cert_path"]).parent / "parent_spectrum.npz"
        if f.exists():
            z = np.load(f)
            lams, R, T = z["lam_nm"], z["R"], z["T"]
            spec_order = [int(x) for x in z["order"]]
            spec_src = str(f.relative_to(cm.PKG))
    poles = list(cand.get("cert", {}).get("parent_poles") or [])
    src_poles = "certification.json (order %s)" % (cand.get("cert", {}).get("cert_order"),)
    if lams is None or (not poles and cand.get("cert") == {}):
        if rho_t is None:
            lams = np.arange(lam_lo, lam_hi + 1e-9, step); R = np.zeros_like(lams); T = np.zeros_like(lams)
            for i, l in enumerate(lams):
                with torch.no_grad():
                    sim = fwd.build_sim(None, P, h, float(l), list(order), with_ito=False)
                    Ri, Ti = fwd.rt_all_orders(sim); R[i], T[i] = float(Ri), float(Ti)
            poles, src_poles, spec_src = [], "no design layer: no poles", f"computed here at order {list(order)}"
        else:
            pk = pp.parent_poles(rho_t, P, h, list(order), lam_lo, lam_hi, step, n_glass=ng)
            if lams is None:
                lams, R, T = pk["lams"], pk["R"], pk["T"]; spec_src = f"computed here at order {list(order)}"
            if not poles:
                poles = [dict(lam=p["lambda_nm"], Q=p["Q"], fwhm=p["fwhm_nm"]) for p in pk["poles"]]
                src_poles = f"exact AAA poles computed here at order {list(order)} ({lam_lo}-{lam_hi} nm, {step} nm); NOT a certification"
    lams, R, T = np.asarray(lams, float), np.asarray(R, float), np.asarray(T, float)
    energy_resid = float(np.max(np.abs(R + T - 1.0)))
    near = min(poles, key=lambda p: abs(p["lam"] - lam_E)) if poles else None
    bg = background_set(lams, T, R, poles, lam_E, lam_R, D_op=D_op, m_R=m_R, k_excl=k_excl, hw_max=hw_max, lam_lo=lam_lo, lam_hi=lam_hi)
    # ---- resonance contrast: exact solve at lambda_r + dense local window ---------------------
    res = dict(pole=near, T_at_lambda_r=None, C_res_center=None, C_res_ext=None, local=None)
    if near and rho_t is not None and near["lam"] > lam_R + 1e-6:
        hw = k_loc * max(near["fwhm"], 0.5)
        loc = np.linspace(max(near["lam"] - hw, lam_lo), min(near["lam"] + hw, lam_hi), n_loc)
        loc = np.unique(np.concatenate([loc, [near["lam"]]]))
        _, _, Rl, Tl = pp.parent_rt(rho_t, P, h, loc, spec_order)
        i_r = int(np.argmin(np.abs(loc - near["lam"])))
        Tb = bg["T_bg_median"]
        res.update(T_at_lambda_r=float(Tl[i_r]), R_at_lambda_r=float(Rl[i_r]),
                   C_res_center=(abs(Tb - float(Tl[i_r])) if Tb is not None else None),
                   C_res_ext=(float(np.max(np.abs(Tb - Tl))) if Tb is not None else None),
                   lambda_ext=(float(loc[int(np.argmax(np.abs(Tb - Tl)))]) if Tb is not None else None),
                   local=dict(lam=[float(x) for x in loc], T=[float(x) for x in Tl], R=[float(x) for x in Rl],
                              T_min=float(Tl.min()), T_max=float(Tl.max()), depth=float(Tl.max() - Tl.min()),
                              lam_T_min=float(loc[int(np.argmin(Tl))]), lam_T_max=float(loc[int(np.argmax(Tl))]),
                              half_window_nm=float(hw), order=spec_order))
    elif near and near["lam"] <= lam_R + 1e-6:
        res["note"] = "nearest pole lies below the glass Rayleigh wavelength (multi-order region): contrast not defined here"
    # ---- parent metrics recomputed here (regression check for certified, only value otherwise) --
    pm, reg = {}, {}
    if rho_t is not None:
        for label, lam in (("lambda_E", lam_E), ("lambda_r", near["lam"] if near else None)):
            if lam is None:
                continue
            f = pf.to_floats(pf.parent_metrics(rho_t, P, h, float(lam), spec_order, n_z=9))
            pm[label] = dict(lambda_nm=float(lam), order=spec_order, **{k: f[k] for k in ("eta_Dz", "eta_Dz_P", "U_mid", "R", "T", "W_Si", "S_Dz_ratio_glass_over_Si")})
        c = cand.get("cert") or {}
        if c.get("eta_Dz_lamE") is not None and "lambda_E" in pm:
            reg = dict(eta_Dz_cert=c["eta_Dz_lamE"], eta_Dz_here=pm["lambda_E"]["eta_Dz"],
                       rel_diff=abs(pm["lambda_E"]["eta_Dz"] - c["eta_Dz_lamE"]) / max(abs(c["eta_Dz_lamE"]), 1e-30),
                       T_cert=c.get("T_parent_lamE"), T_here=pm["lambda_E"]["T"],
                       note="certification metrics at lambda_E recomputed with the unchanged qdz_parent code at the certification order")
    # ---- total vs specular check at three wavelengths ----------------------------------------
    chk = []
    for lam in (lam_R + m_R, lam_E, min(lam_hi, 1398.0)):
        if lam_lo <= lam <= lam_hi:
            chk.append(per_order_table(rho_t, P, h, lam, list(order), with_ito=False))
    iE = int(np.argmin(np.abs(lams - lam_E)))
    out = dict(tag=tag, stage="A", P=P, h=h, lam_E=lam_E, order_spectrum=spec_order, order_here=list(order),
               spectrum_source=spec_src, pole_source=src_poles, energy_resid_max=energy_resid,
               parameters=dict(D_op_nm=D_op, m_R_nm=m_R, k_excl=k_excl, hw_max_nm=bg["parameters"]["hw_max_nm"], k_loc=k_loc,
                               n_loc=n_loc, lam_lo=lam_lo, lam_hi=lam_hi),
               rayleigh=ray, lambda_R_top=lam_R,
               rayleigh_fixed_index_check=[r["lam"] for r in an.rayleigh_wavelengths(P, ng, lo=900.0, hi=lam_hi)],
               poles=poles, background=bg, resonance=res, parent_metrics_here=pm, regression_vs_certification=reg,
               T_parent_lamE=float(T[iE]), R_parent_lamE=float(R[iE]), lam_grid_nearest_E=float(lams[iE]),
               spectrum=dict(lam=[float(x) for x in lams], R=[float(x) for x in R], T=[float(x) for x in T]),
               specular_vs_total=chk, wall_s=time.time() - t0)
    log(f"[A {tag}] T_bg={bg['T_bg_median']} (n={bg['n_points']} in [{bg['window'][0]:.0f},{bg['window'][1]:.0f}]) "
        f"T(lam_r)={res.get('T_at_lambda_r')} C_res={res.get('C_res_center')} C_ext={res.get('C_res_ext')} "
        f"T(lam_E)={float(T[iE]):.3f} eta_reg={reg.get('rel_diff')} ({out['wall_s']:.0f} s)")
    return out
