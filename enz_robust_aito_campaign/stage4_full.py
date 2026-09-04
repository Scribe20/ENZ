"""Stage 4: full topology optimization of the finalists at the full order
and full angular set:
  * each finalist warm-started from its Stage-3 raw variable
    (projection ramp restarted at beta_proj = 4),
  * plus ONE from-scratch run (fresh seed, random init, beta_proj ramp
    from 1) at the best finalist's (P, h, p_pad),
then dense angular evaluation (phi = 0, 90, 45 planes; lab-x, p, s) of every
hard-binary candidate.  Restartable.

Run:  python stage4_full.py
"""

import json

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import forward_multi as fm
import optimizer as opt
import robust_config as rc

OUT = rc.OUT / "stage4"
LOGF = OUT / "stage4.log"
SCRATCH_SEED = 2024


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOGF, "a") as f:
        f.write(s + "\n")


def dense_eval(rho, P, h, order=rc.ORDER_FULL):
    out = {}
    with torch.no_grad():
        for pl, angs in rc.ANGLES_EVAL_PLANES.items():
            for pol in (("labx", "p", "s") if pl != "phi45" else ("labx",)):
                key = f"{pl}_{pol}"
                out[key] = []
                for th, ph in angs:
                    sim = fm.build_sim(rho, P, h, theta_deg=th, phi_deg=ph,
                                       order=order, pol=pol)
                    A, R, T = fm.a_ito(sim)
                    out[key].append(dict(theta=th, phi=ph, A=float(A),
                                         R=float(R), T=float(T)))
    return out


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    pre = json.load(open(rc.OUT / "preflight.json"))
    s3 = json.load(open(rc.OUT / "stage3" / "stage3_summary.json"))
    beta = s3["beta"]
    n_runs = len(s3["finalists"]) + len(rc.SCRATCH_SEEDS)
    t_full = pre["timing_s_per_angle"][str(rc.ORDER_FULL)] * len(rc.ANGLES_FULL)
    n_iter = int(np.clip(rc.BUDGET_H["stage4"] * 3600 / (n_runs * t_full),
                         60, rc.N_ITER_FULL))
    log(f"[stage4] beta={beta:.2f} n_iter={n_iter} order={rc.ORDER_FULL} "
        f"angles={rc.ANGLES_FULL}")
    cands = []
    for i, f in enumerate(s3["finalists"]):
        tag = f"finalist{i}_{f['tag']}_warm"
        d = OUT / "runs" / tag
        if (d / "result.json").exists():
            res = json.load(open(d / "result.json"))
        else:
            rho0 = np.load(rc.OUT / "stage3" / "runs" / f["tag"]
                           / "rho_raw_final.npy")
            res = opt.optimize(f["P"], f["h"], f["pad"], 333, n_iter,
                               rc.ORDER_FULL, rc.ANGLES_FULL, beta, d,
                               rho_init=rho0, beta_proj_start=4.0, log=log,
                               tag=tag)
        cands.append((tag, res))
    f0 = s3["finalists"][0]
    for sd in rc.SCRATCH_SEEDS:
        tag = f"scratch_{f0['tag']}_seed{sd}"
        d = OUT / "runs" / tag
        if (d / "result.json").exists():
            res = json.load(open(d / "result.json"))
        else:
            res = opt.optimize(f0["P"], f0["h"], f0["pad"], sd, n_iter,
                               rc.ORDER_FULL, rc.ANGLES_FULL, beta, d, log=log,
                               tag=tag)
        cands.append((tag, res))

    rows, dense = [], {}
    for tag, res in cands:
        rho = torch.as_tensor(np.load(OUT / "runs" / tag / "rho_hard_binary.npy"),
                              dtype=fm.GEO_DTYPE)
        dense[tag] = dense_eval(rho, res["P"], res["h"])
        allA = [c["A"] for k in dense[tag] if k.endswith("labx")
                for c in dense[tag][k] if c["theta"] <= 30]
        rows.append(dict(tag=tag, P=res["P"], h=res["h"], pad=res["pad_frac"],
                         J_hard=res["J_hard"], A_hard_min=res["A_hard_min"],
                         A_normal=res["A_hard"][0], A_hard=res["A_hard"],
                         dense_labx_min_le30=min(allA),
                         dense_labx_mean_le30=float(np.mean(allA)),
                         s_flip=res["s_flip_final"],
                         fill=res["fill_fraction_active"],
                         warm=res["warm_start"]))
        log(f"[stage4] {tag}: J_hard={res['J_hard']:.5f} A_hard={[round(a,4) for a in res['A_hard']]} "
            f"dense(<=30deg,labx) min={min(allA):.4f} mean={np.mean(allA):.4f} "
            f"S_flip={res['s_flip_final']:.3f}")
    df = pd.DataFrame(rows).sort_values("J_hard", ascending=False)
    df.to_csv(OUT / "stage4_results.csv", index=False)
    json.dump(dense, open(OUT / "stage4_dense_angular.json", "w"), indent=1)
    winner = df.iloc[0]["tag"]
    json.dump(dict(beta=beta, n_iter=n_iter, winner=winner,
                   ranking=df["tag"].tolist()),
              open(OUT / "stage4_summary.json", "w"), indent=1)
    log(f"[stage4] winner = {winner}")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, pl in zip(axs, ("phi0", "phi90", "phi45")):
        for tag in dense:
            c = dense[tag][f"{pl}_labx"]
            ax.plot([x["theta"] for x in c], [x["A"] for x in c], marker="o",
                    ms=3, label=tag)
        ax.axvspan(0, 30, color="#eee", zorder=0)
        ax.set_title(f"A_ITO vs theta, plane {pl}, lab-x pol, order {rc.ORDER_FULL}")
        ax.set_xlabel("theta (deg)"); ax.grid(alpha=.3)
    axs[0].set_ylabel("A_ITO(lambda_E)"); axs[0].legend(fontsize=7)
    fig.savefig(rc.OUT / "figures" / "stage4_dense_angular.png", dpi=150,
                bbox_inches="tight"); plt.close(fig)
    fig, axs = plt.subplots(1, len(cands), figsize=(4 * len(cands), 4))
    for ax, (tag, res) in zip(np.atleast_1d(axs), cands):
        rho = np.load(OUT / "runs" / tag / "rho_hard_binary.npy")
        ax.imshow(rho.T, origin="lower", cmap="gray_r",
                  extent=[0, res["P"], 0, res["P"]])
        ax.set_title(f"{tag}\nJ={res['J_hard']:.4f} S_flip={res['s_flip_final']:.2f}",
                     fontsize=8)
        ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
    fig.savefig(rc.OUT / "figures" / "stage4_geometries.png", dpi=150,
                bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
