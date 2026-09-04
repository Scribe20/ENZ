"""Stage 2: outer screen of (P, h, p_pad) with shortened multi-seed
topology runs (screen order, screen angular set).  Restartable: runs whose
result.json exists are skipped.

Run:  python stage2_screen.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import optimizer as opt
import robust_config as rc

OUT = rc.OUT / "stage2"
LOGF = OUT / "stage2.log"


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOGF, "a") as f:
        f.write(s + "\n")


def run_tag(P, h, pad, seed):
    return f"P{P:.0f}_h{h:.0f}_pad{pad:.3f}_s{seed}"


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    pre = json.load(open(rc.OUT / "preflight.json"))
    beta = pre["beta_calibration"]["beta"]
    n_runs = len(rc.P_SCREEN) * len(rc.H_SCREEN) * len(rc.PAD_SCREEN) \
        * len(rc.SEEDS_SCREEN)
    t_scr = pre["timing_s_per_angle"][str(rc.ORDER_SCREEN)] * len(rc.ANGLES_SCREEN)
    n_iter = int(np.clip(rc.BUDGET_H["stage2"] * 3600 / (n_runs * t_scr),
                         *rc.N_ITER_SCREEN_RANGE))
    log(f"[stage2] beta={beta:.2f} n_iter={n_iter} order={rc.ORDER_SCREEN} "
        f"angles={rc.ANGLES_SCREEN}")
    rows = []
    for P in rc.P_SCREEN:
        for h in rc.H_SCREEN:
            for pad in rc.PAD_SCREEN:
                for seed in rc.SEEDS_SCREEN:
                    tag = run_tag(P, h, pad, seed)
                    d = OUT / "runs" / tag
                    if (d / "result.json").exists():
                        res = json.load(open(d / "result.json"))
                        log(f"[stage2] {tag}: cached J_hard={res['J_hard']:.5f}")
                    else:
                        res = opt.optimize(P, h, pad, seed, n_iter,
                                           rc.ORDER_SCREEN, rc.ANGLES_SCREEN,
                                           beta, d, log=log, tag=tag)
                    rows.append(dict(P=P, h=h, pad=pad, seed=seed,
                                     J_hard=res["J_hard"], J_soft=res["J_soft"],
                                     A_hard_min=res["A_hard_min"],
                                     A_hard=res["A_hard"],
                                     fill=res["fill_fraction_active"],
                                     s_flip=res["s_flip_final"],
                                     wall_s=res["wall_s"], tag=tag))
                    pd.DataFrame(rows).to_csv(OUT / "stage2_results.csv",
                                              index=False)
    df = pd.DataFrame(rows)
    # best over seeds per (P,h,pad)
    best = df.sort_values("J_hard", ascending=False).groupby(
        ["P", "h", "pad"], as_index=False).first()
    best = best.sort_values("J_hard", ascending=False)
    best.to_csv(OUT / "stage2_best_per_combo.csv", index=False)
    top = best.head(rc.N_TOP_STAGE2)
    summary = dict(beta=beta, n_iter=n_iter,
                   top=[dict(P=float(r.P), h=float(r.h), pad=float(r.pad),
                             seed=int(r.seed), J_hard=float(r.J_hard),
                             tag=r.tag) for r in top.itertuples()],
                   seed_spread_median=float(df.groupby(["P", "h", "pad"])
                                            ["J_hard"].agg(lambda x: x.max() - x.min()).median()))
    json.dump(summary, open(OUT / "stage2_summary.json", "w"), indent=1)
    log("[stage2] top combos: " + json.dumps(summary["top"]))
    landscape_plots(df, best)


def landscape_plots(df, best):
    pads = sorted(df["pad"].unique())
    fig, axs = plt.subplots(1, len(pads), figsize=(6 * len(pads), 4.5),
                            squeeze=False)
    vmin, vmax = best["J_hard"].min(), best["J_hard"].max()
    for ax, pad in zip(axs[0], pads):
        sub = best[best["pad"] == pad].pivot(index="h", columns="P",
                                             values="J_hard")
        im = ax.imshow(sub.values, origin="lower", aspect="auto",
                       vmin=vmin, vmax=vmax, cmap="viridis",
                       extent=[sub.columns.min() - 50, sub.columns.max() + 50,
                               sub.index.min() - 10, sub.index.max() + 10])
        for (hh, PP), v in np.ndenumerate(sub.values):
            ax.text(sub.columns[PP], sub.index[hh], f"{v:.3f}", ha="center",
                    va="center", color="w", fontsize=8)
        ax.set_title(f"Stage 2: best-of-seeds J_robust (hard), pad = {pad:.2f} P")
        ax.set_xlabel("period P (nm)"); ax.set_ylabel("a-Si height h (nm)")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(rc.OUT / "figures" / "stage2_landscape_J.png", dpi=150,
                bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for pad in pads:
        s = df[df["pad"] == pad]
        ax.scatter(s["P"] + 8 * (s["seed"] % 7) + 200 * (s["h"] - 140) / 60,
                   s["J_hard"], label=f"pad {pad:.2f}", s=18)
    ax.set_xlabel("P (nm) (offset by h and seed)"); ax.set_ylabel("J_hard")
    ax.legend(); ax.grid(alpha=.3)
    ax.set_title("Stage 2: all runs (seed scatter)")
    fig.savefig(rc.OUT / "figures" / "stage2_seed_scatter.png", dpi=150,
                bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
