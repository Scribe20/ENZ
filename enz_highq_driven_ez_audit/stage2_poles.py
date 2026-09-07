"""Stage 2: dense r/t reconnaissance + loss-scaling branch tracking with
field-overlap continuity for the four primary candidates (padded QNM,
padded F_ENZ, P800 robust, P925 robust) and the P925/h220 control (s=1 only).

Outputs: stage2_recon.json (all significant poles at s=1), stage2_branches.json
(per branch: rows over s, gamma fit, Q_rad/Q_nr/ratio), fig5 trajectories.
"""
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as cm
import poles as pl

torch.set_num_threads(4)
LOG = open(cm.OUT / "stage2.log", "a")


def log(s):
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


HIST = cm.jload(cm.ROOT / "enz_highq_enz_campaign/outputs/stage_a_decomposition.json")
recon, branches = {}, {}
for name in cm.PRIMARY + ["P925/h220 robust"]:
    rho, P, h = cm.load_candidate(name)
    log(f"\n=== {name} (P={P:.0f}, h={h:.0f}) reconnaissance s=1, {len(pl.SCAN_DENSE)} pts ===")
    r, t = pl.rt_scan(rho, P, h, 1.0, pl.SCAN_DENSE)
    sig = pl.significant_poles(pl.SCAN_DENSE, r, t)
    np.savez(cm.OUT / f"stage2_recon_{name.replace(' ', '_').replace('/', '_')}.npz",
             lam=pl.SCAN_DENSE, r=r, t=t)
    for p in sig:
        p["local_refine"] = pl.local_refine(rho, P, h, 1.0, p) if p["Q"] > 25 else None
        log(f"  pole {p['lambda_nm']:8.2f} nm  Q={p['Q']:8.2f}  peak(r,t)=({p['peak_r']:.3f},{p['peak_t']:.3f}) "
            f"rt_diff={p['rt_rel_diff']:.1e} in_window={p['in_window']}"
            + (f" refine_conv={p['local_refine']['converged']}" if p["local_refine"] else ""))
    recon[name] = sig
    if name not in cm.PRIMARY:
        continue
    # track every significant pole within the ENZ band +-60 nm
    branches[name] = []
    for p in sig:
        if not (pl.WINDOW[0] - 60 <= p["lambda_nm"] <= pl.WINDOW[1] + 60):
            continue
        bid = f"{name}|{p['lambda_nm']:.0f}nm"
        log(f"--- tracking branch {bid} (Q_loaded={p['Q']:.2f}) ---")
        rows = pl.track_branch(rho, P, h, p["omega"], log=log, tag=bid)
        fit = pl.fit_gamma(rows)
        log(f"  fit: {({k: (round(v, 5) if isinstance(v, float) else v) for k, v in fit.items()})}")
        branches[name].append(dict(branch=bid, start=p, rows=rows, fit=fit))
    # consistency with the historical Stage-A decomposition
    key = {"padded QNM": "padded QNM winner", "padded F_ENZ": "padded F_ENZ winner"}.get(name)
    if key:
        hst = HIST[key]
        log(f"  historical Stage-A: Q_loaded={hst['Q_loaded']:.2f} Q_rad={hst['Q_rad']:.1f} "
            f"Q_nr={hst['Q_nr']:.2f} ratio={hst['gamma_ratio']:.4f} (rows: "
            + ", ".join(f"s={r['s']}:{r['lambda_pole']:.1f}nm/Q{r['Q']:.1f}" for r in hst["rows"]) + ")")
cm.jdump(recon, cm.OUT / "stage2_recon.json")
cm.jdump(branches, cm.OUT / "stage2_branches.json")

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
for name, brs in branches.items():
    for b in brs:
        lam = [r["lambda_pole"] for r in b["rows"]]; Q = [r["Q"] for r in b["rows"]]
        ss = [r["s"] for r in b["rows"]]
        axs[0].plot(lam, Q, marker="o", label=b["branch"])
        for l, q, s in zip(lam, Q, ss):
            axs[0].annotate(f"{s:g}", (l, q), fontsize=6, textcoords="offset points", xytext=(2, 2))
        axs[1].plot([r["s"] for r in b["rows"]], [r["gamma"] for r in b["rows"]], marker="o", label=b["branch"])
axs[0].set_yscale("log"); axs[0].axvline(cm.LAMBDA_E, ls="--", c="k", lw=.8)
axs[0].set_xlabel("pole wavelength (nm)"); axs[0].set_ylabel("Q (loss-scaled)")
axs[0].set_title("Fig. 5 - pole trajectories under ITO-loss scaling (labels: s)")
axs[0].legend(fontsize=6)
axs[1].set_xlabel("loss scale s"); axs[1].set_ylabel("gamma = |Im omega| (rad/fs)")
axs[1].set_title("gamma(s) = gamma_rad + s gamma_nr"); axs[1].legend(fontsize=6); axs[1].grid(alpha=.3)
fig.savefig(cm.FIG / "fig5_pole_trajectories.png", dpi=150, bbox_inches="tight")
log("[stage2] done")
