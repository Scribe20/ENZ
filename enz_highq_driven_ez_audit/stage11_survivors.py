"""Stages 11-14 for the surviving candidates only:
 11  nonlinear-relevance proxies (linear only: no validated nonlinear ITO
     model exists in the repository);
 12  fair normalization across periods (fixed intensity / fixed power per
     cell / fixed pulse energy per logical pixel = per cell);
 13  fabrication + locality (+-1 px, h +-10 nm, P +-2 %, padding +1 px,
     2x2 supercell neighbour test);
 14  numerical convergence ladder [7,7] -> [9,9] -> [11,11] for driven
     quantities and for the target-near pole (with dense local rescan).
"""
import numpy as np
import torch
from scipy import ndimage

import common as cm
import forward_multi as fm
import poles as pl

torch.set_num_threads(4)
pareto = cm.jload(cm.OUT / "stage4_pareto.json")
br = cm.jload(cm.OUT / "stage2_branches.json")
ang = cm.jload(cm.OUT / "stage5_angular_driven.json")
d1 = cm.jload(cm.OUT / "stage1_driven_lambdaE.json")
names = []
for k in ("strongest_driven_Ez", "highest_Q_Ez_rich_parent", "best_accessible_highQ", "knee"):
    v = pareto.get(k)
    if v:
        names.append(v.split("|")[0])
for n in ("P925 robust", "P800 robust", "padded QNM"):
    names.append(n)
SURV = []
for n in names:
    if n not in SURV and n in cm.CANDIDATES:
        SURV.append(n)
SURV = SURV[:4]
print("[stage11] survivors:", SURV, flush=True)
res = {}


def near_pole(rho, P, h, order=cm.ORDER, lam_ref=cm.LAMBDA_E, prefer=None):
    r, t = pl.rt_scan(rho, P, h, 1.0, pl.SCAN_COARSE, order=order)
    sig = pl.significant_poles(pl.SCAN_COARSE, r, t)
    if not sig:
        return None, []
    ref = prefer if prefer is not None else lam_ref
    p = min(sig, key=lambda q: abs(q["lambda_nm"] - ref))
    if p["Q"] > 25:
        p["local_refine"] = pl.local_refine(rho, P, h, 1.0, p, order=order)
    return p, sig


def q_rad_3level(rho, P, h, p):
    rows = pl.track_branch(rho, P, h, p["omega"], s_levels=(1.0, 0.25, 0.0), refine_high_q=False, log=lambda *a: None)
    return pl.fit_gamma(rows)


for name in SURV:
    rho, P, h = cm.load_candidate(name)
    B = rho.numpy() > 0.5
    brs = br.get(name, [])
    # the branch of interest: narrow near-target for P800, nearest otherwise
    prefer = 1437.6 if name == "P800 robust" else None
    rec = dict(P=P, h=h)
    # ---- 11/12 proxies and normalization -----------------------------------
    a = ang[name]
    le30 = [x for x in a if x["theta"] <= 30]
    rec["proxies"] = dict(
        A_lambdaE=d1[name]["A_rt"], F_Ez=d1[name]["F_Ez"], peak_Ez2=d1[name]["peak_Ez2"],
        Ez2_p95=d1[name]["Ez2_p95"], Ez2_p99=d1[name]["Ez2_p99"],
        eta_ENZ_z_driven=d1[name]["energy_lambda_E"]["eta_ENZ_z"],
        robust_angular_F_Ez_mean_le30=float(np.mean([x["F_Ez"] for x in le30])),
        robust_angular_F_Ez_min_le30=float(np.min([x["F_Ez"] for x in le30])),
        robust_angular_A_mean_le30=float(np.mean([x["A"] for x in le30])),
        robust_angular_A_min_le30=float(np.min([x["A"] for x in le30])),
        note="linear proxies only; no validated nonlinear ITO model in the repository; no authoritative pulse spectrum -> no pulse weighting")
    A0 = d1[name]["A_rt"]
    rec["normalization"] = dict(
        fixed_intensity_abs_energy_density_per_fluence=A0 / cm.D_ITO,           # U/V per (J/m^2): A/d  [1/nm]
        fixed_power_per_cell_abs_energy_density=A0 / (P ** 2 * cm.D_ITO),       # relative to the cell
        fixed_energy_per_logical_pixel_abs_energy_density=A0 / (P ** 2 * cm.D_ITO),
        cell_area_nm2=P ** 2, V_ITO_nm3=P ** 2 * cm.D_ITO,
        note="A: same for all periods up to A itself (intensity fixed); B/C: divide by cell area - a larger cell "
             "intercepts more energy but spreads it over more ITO volume")
    # ---- 13 fabrication / locality ------------------------------------------
    p0, _ = near_pole(rho, P, h, prefer=prefer)
    base = dict(lambda_pole=p0["lambda_nm"] if p0 else None, Q_loaded=p0["Q"] if p0 else None,
                F_Ez=d1[name]["F_Ez"], A=A0)
    variants = {}
    st = np.ones((3, 3), bool)
    geoms = {"erode_1px": (torch.as_tensor(ndimage.binary_erosion(B, st).astype(float), dtype=cm.GEO), P, h),
             "dilate_1px": (torch.as_tensor(ndimage.binary_dilation(B, st).astype(float), dtype=cm.GEO), P, h),
             "h_-10nm": (rho, P, h - 10), "h_+10nm": (rho, P, h + 10),
             "P_-2pct": (rho, P * 0.98, h), "P_+2pct": (rho, P * 1.02, h)}
    pad_px = int(round(cm.CANDIDATES[name][3] / (P / 128)))
    if pad_px > 0:
        ring = np.zeros((128, 128), bool); ring[pad_px + 1:127 - pad_px, pad_px + 1:127 - pad_px] = True
        geoms["pad_+1px"] = (torch.as_tensor((B & ring).astype(float), dtype=cm.GEO), P, h)
    for tag, (r_, P_, h_) in geoms.items():
        m = cm.driven_metrics(r_, P_, h_, n_xy=64)
        p, _ = near_pole(r_, P_, h_, prefer=p0["lambda_nm"] if p0 else prefer)
        v = dict(F_Ez=m["F_Ez"], A=m["A_rt"], eta_z=m["eta_z"], lambda_pole=p["lambda_nm"] if p else None,
                 Q_loaded=p["Q"] if p else None)
        if tag in ("erode_1px", "dilate_1px") and p:
            v["Q_rad_3level"] = q_rad_3level(r_, P_, h_, p)
        variants[tag] = v
        print(f"  {name} {tag:11s}: F_Ez={m['F_Ez']:.3f} A={m['A_rt']:.3f} pole={v['lambda_pole']} Q={v['Q_loaded']}"
              + (f" Q_rad~{v['Q_rad_3level'].get('Q_rad')}" if 'Q_rad_3level' in v else ""), flush=True)
    lab, ncomp = ndimage.label(B)
    dx = P / 128
    def min_scale(mask):
        for rr in range(1, 12):
            s_ = np.ones((2 * rr + 1, 2 * rr + 1), bool)
            if ndimage.binary_opening(mask, s_).sum() < 0.98 * mask.sum():
                return (2 * rr - 1) * dx
        return 23 * dx
    edge = B[0, :].any() or B[-1, :].any() or B[:, 0].any() or B[:, -1].any()
    rec["fabrication"] = dict(base=base, variants=variants)
    rec["locality"] = dict(n_components=int(ncomp), boundary_contact=bool(edge),
                           realized_padding_nm=cm.CANDIDATES[name][3], min_feature_nm=min_scale(B),
                           min_gap_nm=min_scale(~B), fill=float(B.mean()))
    # 2x2 supercell neighbour test (order [11,11] on the 2P cell ~ [5,5] per cell)
    try:
        big = np.block([[B, B], [B, B]]).astype(float)
        Bp = ndimage.binary_erosion(B, np.ones((5, 5), bool))
        big_p = np.block([[B, B], [B, Bp]]).astype(float)
        m_same = cm.driven_metrics(torch.as_tensor(big, dtype=cm.GEO), 2 * P, h, order=[11, 11], n_xy=64)
        m_pert = cm.driven_metrics(torch.as_tensor(big_p, dtype=cm.GEO), 2 * P, h, order=[11, 11], n_xy=64)
        m_single = cm.driven_metrics(rho, P, h, order=[5, 5], n_xy=64)
        rec["supercell"] = dict(single_cell_order55=dict(A=m_single["A_rt"], F_Ez=m_single["F_Ez"]),
                                supercell_identical=dict(A=m_same["A_rt"], F_Ez=m_same["F_Ez"]),
                                supercell_one_neighbour_eroded_2px=dict(A=m_pert["A_rt"], F_Ez=m_pert["F_Ez"]),
                                note="2x2 supercell at order [11,11] (= [5,5] per cell); identical-cell case must reproduce the single cell at [5,5]")
        print(f"  {name} supercell: single[5,5] A={m_single['A_rt']:.4f} F={m_single['F_Ez']:.3f} | 2x2 identical A={m_same['A_rt']:.4f} "
              f"F={m_same['F_Ez']:.3f} | one neighbour eroded A={m_pert['A_rt']:.4f} F={m_pert['F_Ez']:.3f}", flush=True)
    except Exception as e:      # noqa: BLE001
        rec["supercell"] = dict(error=str(e))
    # ---- 14 convergence ladder ----------------------------------------------
    conv = {}
    for od in ([7, 7], [9, 9], [11, 11]):
        m = cm.driven_metrics(rho, P, h, order=od, n_xy=64)
        p, _ = near_pole(rho, P, h, order=od, prefer=p0["lambda_nm"] if p0 else prefer)
        conv[str(od)] = dict(F_Ez=m["F_Ez"], A=m["A_rt"], eta_z=m["eta_z"],
                             lambda_pole=p["lambda_nm"] if p else None, Q_loaded=p["Q"] if p else None,
                             local_refine=(p.get("local_refine") if p else None))
        print(f"  {name} order {od}: F_Ez={m['F_Ez']:.3f} A={m['A_rt']:.4f} pole={conv[str(od)]['lambda_pole']} Q={conv[str(od)]['Q_loaded']}", flush=True)
    rec["convergence"] = conv
    res[name] = rec
    cm.jdump(res, cm.OUT / "stage11_survivors.json")
print("[stage11] done")
