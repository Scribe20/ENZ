"""Build the Stage-3 (long, full-order) job list from the Stage-2 results.

Rule (predeclared):
  * finalists = the top-N_FINALISTS Stage-2 runs by hard-binary F_z that are
    geometrically distinct: (P, h) differ OR the normalized-cell correlation of
    the binary designs is < 0.9; at least one finalist must be of from-scratch
    lineage (Stage-1 seed lineage, i.e. not a CTRL_ warm start from a
    historical design) - if the top runs are all CTRL_, the best non-CTRL run
    is added.
  * every finalist: warm start from its Stage-2 rho_raw_final at order [7,7],
    150 iterations, beta_start = 4 (Example6 ramp restarted from a mildly
    binarized state), final evaluation at [9,9]
  * one pure from-scratch [7,7] 150-iteration run (beta ramp from 1) at the
    best finalist's (P, h, pad), seed 8080, as a control of path dependence
"""

import argparse
import json
from pathlib import Path

import numpy as np

import config

OUT2 = config.OUT / "stage2"
OUT3 = config.OUT / "stage3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-finalists", type=int, default=3)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--corr-max", type=float, default=0.9)
    a = ap.parse_args()
    rows = json.load(open(OUT2 / "results.json"))
    rows.sort(key=lambda r: -r["Fz_hard"])
    fin, geos = [], []
    for r in rows:
        g = np.load(OUT2 / "runs" / r["tag"] / "rho_hard_binary.npy").ravel()
        distinct = True
        for f, gf in zip(fin, geos):
            if (f["P"], f["h"]) == (r["P"], r["h"]):
                c = float(np.corrcoef(g, gf)[0, 1]) if g.std() > 0 and gf.std() > 0 else 1.0
                if c >= a.corr_max:
                    distinct = False
                    break
        if distinct:
            fin.append(r); geos.append(g)
        if len(fin) >= a.n_finalists:
            break
    if all(f["tag"].startswith("CTRL_") for f in fin):
        best_scratch = next(r for r in rows if not r["tag"].startswith("CTRL_"))
        fin.append(best_scratch)
    jobs = []
    for i, r in enumerate(fin):
        jobs.append(dict(tag=f"final{i}_{r['tag']}", P=r["P"], h=r["h"], pad=r["pad_frac"], seed=r["seed"], n_iter=a.iters,
                         order=list(config.ORDER_FULL), warm=str(OUT2 / "runs" / r["tag"] / "rho_raw_final.npy"),
                         beta_start=4.0, eval_order=[9, 9], n_threads=1, save_every=10))
    b = fin[0]
    jobs.append(dict(tag=f"scratch_s8080_P{b['P']:.0f}_h{b['h']:.0f}_pad{b['pad_frac']:.2f}", P=b["P"], h=b["h"], pad=b["pad_frac"], seed=8080,
                     n_iter=a.iters, order=list(config.ORDER_FULL), warm=None, beta_start=1.0, eval_order=[9, 9], n_threads=1, save_every=10))
    OUT3.mkdir(parents=True, exist_ok=True)
    with open(OUT3 / "jobs.json", "w") as f:
        json.dump(jobs, f, indent=1)
    with open(OUT3 / "finalists.json", "w") as f:
        json.dump(fin, f, indent=1)
    for r in fin:
        print(f"  finalist {r['tag']:40s} Fz_hard={r['Fz_hard']:.4f} P={r['P']:.0f} h={r['h']:.0f} pad={r['pad_frac']:.2f} warm={r.get('warm_from')}")
    print(f"{len(jobs)} jobs -> {OUT3/'jobs.json'}")


if __name__ == "__main__":
    main()
