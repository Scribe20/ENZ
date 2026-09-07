"""Stage 6: spectral-alignment test - small adaptive P/h sweeps for P925 and
P800 with the hard topology fixed in normalized cell coordinates.  At each
point: target-near pole (s = 1 scan), Q_loaded, F_Ez(lambda_E),
A_ITO(lambda_E); 3-level Q_rad estimate (s = 1, 0.25, 0) for the best two
points per candidate."""
import numpy as np
import torch

import common as cm
import poles as pl

torch.set_num_threads(4)
CANDS = {"P925 robust": (925.0, 240.0), "P800 robust": (800.0, 200.0)}
DP, DH = 0.04, 20.0
out = {}


def point(rho, P, h, tag):
    r, t = pl.rt_scan(rho, P, h, 1.0, pl.SCAN_COARSE)
    sig = pl.significant_poles(pl.SCAN_COARSE, r, t)
    near = min(sig, key=lambda p: abs(p["lambda_nm"] - cm.LAMBDA_E)) if sig else None
    m = cm.driven_metrics(rho, P, h, n_xy=64)
    rec = dict(P=P, h=h, F_Ez=m["F_Ez"], A=m["A_rt"], eta_z=m["eta_z"],
               near_pole=(dict(lambda_nm=near["lambda_nm"], Q=near["Q"], omega=[near["omega"].real, near["omega"].imag]) if near else None),
               all_poles=[dict(lambda_nm=p["lambda_nm"], Q=p["Q"]) for p in sig])
    print(f"  [{tag}] P={P:.1f} h={h:.1f}: F_Ez={m['F_Ez']:.3f} A={m['A_rt']:.3f} near-pole="
          f"{(near or {}).get('lambda_nm', float('nan')):.1f} nm Q={(near or {}).get('Q', float('nan')):.1f}", flush=True)
    return rec


for name, (P0, h0) in CANDS.items():
    rho, _, _ = cm.load_candidate(name)
    pts = []
    for fp in (-DP, 0.0, DP):
        for dh in (-DH, 0.0, DH):
            pts.append(point(rho, P0 * (1 + fp), h0 + dh, name))
    best = max(pts, key=lambda r: r["F_Ez"])
    for fp in (-DP / 2, DP / 2):
        for dh in (-DH / 2, DH / 2):
            pts.append(point(rho, best["P"] * (1 + fp), best["h"] + dh, name + " refine"))
    best2 = sorted(pts, key=lambda r: -r["F_Ez"])[:2]
    aligned = min([p for p in pts if p["near_pole"]], key=lambda r: abs(r["near_pole"]["lambda_nm"] - cm.LAMBDA_E))
    for rec in best2 + ([aligned] if aligned not in best2 else []):
        if not rec["near_pole"]:
            continue
        rows = pl.track_branch(rho, rec["P"], rec["h"], complex(*rec["near_pole"]["omega"]),
                               s_levels=(1.0, 0.25, 0.0), refine_high_q=False,
                               tag=f"{name} P{rec['P']:.0f} h{rec['h']:.0f}")
        rec["Q_rad_3level"] = pl.fit_gamma(rows)
        rec["track_rows"] = rows
    out[name] = dict(base=dict(P=P0, h=h0), points=pts,
                     best_F_Ez=dict(P=best2[0]["P"], h=best2[0]["h"], F_Ez=best2[0]["F_Ez"]),
                     best_aligned=dict(P=aligned["P"], h=aligned["h"], lambda_pole=aligned["near_pole"]["lambda_nm"]))
cm.jdump(out, cm.OUT / "stage6_alignment.json")
print("[stage6] done")
