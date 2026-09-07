"""Stage 5: staged angular tests.
 5a  inexpensive driven quantities (A_ITO, F_Ez, eta_z) for ALL candidates at
     theta in {0,10,20,30,40}, phi in {0,45,90} (lab-x polarization);
 5b  sparse pole tracking (s = 1 only: lambda_pole, Q_loaded of the
     target-near branch) at 7 angles for the three focus candidates.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as cm
import poles as pl

torch.set_num_threads(4)
THETAS, PHIS = [0.0, 10.0, 20.0, 30.0, 40.0], [0.0, 45.0, 90.0]
FOCUS = ["P925 robust", "P800 robust", "padded QNM"]
SPARSE = [(0, 0), (20, 0), (20, 45), (20, 90), (40, 0), (40, 45), (40, 90)]
br = cm.jload(cm.OUT / "stage2_branches.json")

ang = {}
for name in cm.CANDIDATES:
    rho, P, h = cm.load_candidate(name)
    ang[name] = []
    for ph in PHIS:
        for th in THETAS:
            if th == 0 and ph != 0:
                continue
            m = cm.driven_metrics(rho, P, h, theta=th, phi=ph, n_xy=64)
            ang[name].append(dict(theta=th, phi=ph, A=m["A_rt"], F_Ez=m["F_Ez"],
                                  eta_z=m["eta_z"], peak_Ez2=m["peak_Ez2"], orders=m["orders"]))
    print(f"{name:18s} " + " ".join(f"({a['theta']:.0f},{a['phi']:.0f}):A{a['A']:.3f}/F{a['F_Ez']:.2f}/z{a['eta_z']:.2f}"
                                    for a in ang[name]), flush=True)
cm.jdump(ang, cm.OUT / "stage5_angular_driven.json")

fig, axs = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
for j, ph in enumerate(PHIS):
    for name, rows in ang.items():
        rr = [r for r in rows if r["phi"] == ph or r["theta"] == 0]
        rr = sorted(rr, key=lambda r: r["theta"])
        for i, key in enumerate(("A", "F_Ez", "eta_z")):
            axs[i, j].plot([r["theta"] for r in rr], [r[key] for r in rr], marker="o", ms=3,
                           label=name, lw=2 if name in FOCUS else 1)
    axs[0, j].set_title(f"phi = {ph:.0f} deg (lab-x pol)")
    axs[2, j].set_xlabel("theta (deg)")
for i, key in enumerate(("A_ITO(lambda_E)", "F_Ez(lambda_E)", "eta_z driven")):
    axs[i, 0].set_ylabel(key)
    for j in range(3):
        axs[i, j].grid(alpha=.3)
axs[0, 0].legend(fontsize=7)
fig.savefig(cm.FIG / "stage5_angular_driven.png", dpi=150, bbox_inches="tight")

# 5b sparse pole tracking (s = 1): follow the target-near branch by field overlap
sparse = {}
for name in FOCUS:
    rho, P, h = cm.load_candidate(name)
    brs = br.get(name, [])
    if not brs:
        continue
    # target-near branch = the one with the smallest |lambda_pole - lambda_E| at s=1
    b0 = min(brs, key=lambda b: abs(b["rows"][0]["lambda_pole"] - cm.LAMBDA_E))
    ref_omega = complex(b0["rows"][0]["omega_re"], b0["rows"][0]["omega_im"])
    ref_map = pl.ez_map(rho, P, h, b0["rows"][0]["lambda_pole"], 1.0)
    sparse[name] = dict(branch=b0["branch"], points=[])
    for th, ph in SPARSE:
        r, t = pl.rt_scan(rho, P, h, 1.0, pl.SCAN_COARSE, th, ph)
        sig = pl.significant_poles(pl.SCAN_COARSE, r, t)
        best = None
        for p in sorted(sig, key=lambda p: abs(p["omega"] - ref_omega))[:3]:
            ov = pl.overlap(ref_map, pl.ez_map(rho, P, h, p["lambda_nm"], 1.0, th, ph))
            p["overlap_with_normal"] = ov
            if best is None or ov > best["overlap_with_normal"]:
                best = p
        rec = dict(theta=th, phi=ph,
                   all_poles=[dict(lambda_nm=p["lambda_nm"], Q=p["Q"], peak_r=p["peak_r"]) for p in sig],
                   tracked=(dict(lambda_nm=best["lambda_nm"], Q=best["Q"],
                                 overlap=best["overlap_with_normal"]) if best else None))
        sparse[name]["points"].append(rec)
        print(f"  {name} ({th},{ph}): tracked {rec['tracked']} | all: "
              + ", ".join(f"{p['lambda_nm']:.0f}/Q{p['Q']:.1f}" for p in sig), flush=True)
cm.jdump(sparse, cm.OUT / "stage5_sparse_poles.json")
print("[stage5] done")
