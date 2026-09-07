"""Stage 7: loss / coupling sweep as a physics experiment.  For the focus
branches, at every ITO-loss scale s evaluate the DRIVEN F_Ez, A_ITO and
stored energy U at the tracked pole lambda(s) and at lambda_E, and plot
them against gamma_rad / (s gamma_nr)."""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as cm

torch.set_num_threads(4)
br = cm.jload(cm.OUT / "stage2_branches.json")
FOCUS = {"P925 robust": None, "P800 robust": 1437.6, "padded QNM": None, "padded F_ENZ": None}
res = {}
for name, lam_sel in FOCUS.items():
    brs = br.get(name, [])
    if not brs:
        continue
    if lam_sel is None:
        b = min(brs, key=lambda b: abs(b["rows"][0]["lambda_pole"] - cm.LAMBDA_E))
    else:
        b = min(brs, key=lambda b: abs(b["rows"][0]["lambda_pole"] - lam_sel))
    rho, P, h = cm.load_candidate(name)
    fit = b["fit"]
    rows = []
    for r in b["rows"]:
        s = r["s"]
        mp = cm.driven_metrics(rho, P, h, lam=r["lambda_pole"], s=s, n_xy=64)
        mE = cm.driven_metrics(rho, P, h, lam=cm.LAMBDA_E, s=s, n_xy=64)
        ep = cm.energy_participation(rho, P, h, r["lambda_pole"], s)
        ratio = (fit["gamma_rad"] / (s * fit["gamma_nr"])) if (s > 0 and "gamma_nr" in fit) else float("inf")
        rows.append(dict(s=s, gamma_ratio=ratio, lambda_pole=r["lambda_pole"], Q=r["Q"],
                         F_Ez_pole=mp["F_Ez"], A_pole=mp["A_rt"], U_pole=ep["U"],
                         eta_ENZ_z_pole=ep["eta_ENZ_z"], F_Ez_E=mE["F_Ez"], A_E=mE["A_rt"]))
        print(f"  {b['branch']:28s} s={s:.2f} ratio={ratio:8.3f} F_Ez(pole)={mp['F_Ez']:.3f} A(pole)={mp['A_rt']:.3f} "
              f"U={ep['U']:.3e} F_Ez(E)={mE['F_Ez']:.3f} A(E)={mE['A_rt']:.3f}", flush=True)
    res[b["branch"]] = rows
cm.jdump(res, cm.OUT / "stage7_loss_sweep.json")
fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
for bid, rows in res.items():
    x = [r["gamma_ratio"] if np.isfinite(r["gamma_ratio"]) else 1e3 for r in rows]
    axs[0].plot(x, [r["F_Ez_pole"] for r in rows], marker="o", label=bid)
    axs[1].plot(x, [r["A_pole"] for r in rows], marker="o", label=bid)
    axs[2].plot(x, [r["U_pole"] for r in rows], marker="o", label=bid)
for ax, yl in zip(axs, ("F_Ez at pole", "A_ITO at pole", "stored energy U at pole (a.u.)")):
    ax.set_xscale("log"); ax.set_xlabel("gamma_rad / (s gamma_nr)  (s -> 0: right)"); ax.set_ylabel(yl); ax.grid(alpha=.3)
axs[2].set_yscale("log"); axs[0].legend(fontsize=7)
fig.suptitle("Fig. 4b - driven response vs radiative/non-radiative balance (ITO-loss scaling)")
fig.savefig(cm.FIG / "stage7_loss_sweep.png", dpi=150, bbox_inches="tight")
print("[stage7] done")
