"""Stage 3: adaptive local refinement of (P, h, p_pad) around the Stage-2
top combos.  Each neighbour is a shortened warm-started run (Stage-2 raw
variable carried over on the normalized grid; mask re-applied; projection
ramp restarted at beta_proj = 8).  If a neighbour beats its centre, one
further step in that direction (same step) is tried (adaptive round 2)
while the budget allows.  Restartable.

Run:  python stage3_refine.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import optimizer as opt
import robust_config as rc

OUT = rc.OUT / "stage3"
LOGF = OUT / "stage3.log"
STEP = dict(P=50.0, h=20.0, pad=0.03)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOGF, "a") as f:
        f.write(s + "\n")


def clip(P, h, pad):
    return (float(np.clip(P, *rc.P_BOUNDS)), float(np.clip(h, *rc.H_BOUNDS)),
            float(np.clip(pad, *rc.PAD_BOUNDS)))


def tag_of(P, h, pad):
    return f"P{P:.0f}_h{h:.0f}_pad{pad:.3f}"


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    pre = json.load(open(rc.OUT / "preflight.json"))
    s2 = json.load(open(rc.OUT / "stage2" / "stage2_summary.json"))
    beta = s2["beta"]
    t_iter = pre["sizing"]["s_per_iter_screen"]
    tops = s2["top"][:rc.N_TOP_STAGE2]
    n_runs = len(tops) * 7
    n_iter = int(np.clip(rc.BUDGET_H["stage3"] * 3600 / (n_runs * t_iter),
                         *rc.N_ITER_REFINE_RANGE))
    log(f"[stage3] beta={beta:.2f} n_iter={n_iter} centres={tops}")
    rows = []

    def run(P, h, pad, rho_init, seed, origin):
        P, h, pad = clip(P, h, pad)
        tag = tag_of(P, h, pad)
        d = OUT / "runs" / tag
        if (d / "result.json").exists():
            res = json.load(open(d / "result.json"))
        else:
            res = opt.optimize(P, h, pad, seed, n_iter, rc.ORDER_SCREEN,
                               rc.ANGLES_SCREEN, beta, d, rho_init=rho_init,
                               beta_proj_start=8.0, log=log, tag=tag)
        rows.append(dict(P=P, h=h, pad=pad, origin=origin, J_hard=res["J_hard"],
                         A_hard_min=res["A_hard_min"], A_hard=res["A_hard"],
                         fill=res["fill_fraction_active"],
                         s_flip=res["s_flip_final"], tag=tag))
        pd.DataFrame(rows).to_csv(OUT / "stage3_results.csv", index=False)
        return res["J_hard"], tag

    for c in tops:
        src = rc.OUT / "stage2" / "runs" / c["tag"]
        rho0 = np.load(src / "rho_raw_final.npy")
        centre = (c["P"], c["h"], c["pad"])
        J_c, _ = run(*centre, rho0, c["seed"], f"centre<-{c['tag']}")
        best_dir, best_J = None, J_c
        for key, i in (("P", 0), ("h", 1), ("pad", 2)):
            for sgn in (+1, -1):
                v = list(centre); v[i] += sgn * STEP[key]
                if tuple(clip(*v)) == tuple(clip(*centre)):
                    continue
                J, _ = run(*v, rho0, c["seed"], f"nb({key}{sgn:+d})<-{c['tag']}")
                if J > best_J + 0.002:
                    best_J, best_dir = J, (i, sgn)
        # adaptive round 2: one more step along the best improving direction
        if best_dir is not None:
            i, sgn = best_dir
            v = list(centre); v[i] += 2 * sgn * STEP[["P", "h", "pad"][i]]
            if tuple(clip(*v)) != tuple(clip(*centre)):
                run(*v, rho0, c["seed"], f"step2<-{c['tag']}")

    df = pd.DataFrame(rows).sort_values("J_hard", ascending=False)
    df.to_csv(OUT / "stage3_results.csv", index=False)
    finalists = []
    for r in df.itertuples():
        if all(abs(r.P - f["P"]) + abs(r.h - f["h"]) > 1e-6 or
               abs(r.pad - f["pad"]) > 1e-6 for f in finalists):
            finalists.append(dict(P=float(r.P), h=float(r.h), pad=float(r.pad),
                                  J_hard=float(r.J_hard), tag=r.tag))
        if len(finalists) >= rc.N_FINALISTS:
            break
    json.dump(dict(beta=beta, n_iter=n_iter, finalists=finalists),
              open(OUT / "stage3_summary.json", "w"), indent=1)
    log("[stage3] finalists: " + json.dumps(finalists))


if __name__ == "__main__":
    main()
