"""Stage 1: low-cost outer screening of (P, h, p_pad) x seeds at order [5,5].

Every job is an independent from-scratch F_z topology optimization
(optimizer.optimize) on the normalized 128x128 cell; 4 single-thread worker
processes run concurrently (this session: 4-core CPU).  Jobs whose
result.json exists are skipped, so the driver is restartable.

    python run_stage1.py [--iters 80] [--workers 4] [--seeds 333 1001 7777]
                         [--P ...] [--h ...] [--pad ...] [--collect-only]
"""

import argparse
import csv
import itertools
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import config

OUT = config.OUT / "stage1"


def job_tag(P, h, pad, seed):
    return f"P{P:.0f}_h{h:.0f}_pad{pad:.2f}_s{seed}"


def worker(job):
    P, h, pad, seed, n_iter, order = job
    import torch
    torch.set_num_threads(1)
    import materials as mat
    import optimizer as opt
    tag = job_tag(P, h, pad, seed)
    d = OUT / "runs" / tag
    if (d / "result.json").exists():
        return tag, json.load(open(d / "result.json"))["Fz_hard"], "cached"
    d.mkdir(parents=True, exist_ok=True)
    lam, _ = mat.ito_zero_crossing()
    with open(d / "log.txt", "w") as logf:
        res = opt.optimize(P, h, pad, seed, n_iter, order, d, lam, save_every=20,
                           log=lambda *a: print(*a, file=logf, flush=True), tag=tag, n_threads=1)
    return tag, res["Fz_hard"], f"{res['wall_s']:.0f}s"


FIELDS = ["tag", "stage", "P", "h", "pad_frac", "realized_pad_nm", "seed", "n_iter", "order", "warm_start",
          "Fz_soft", "Fz_hard", "Fx_hard", "Fy_hard", "Ftot_hard", "A_hard", "R_hard", "T_hard",
          "eta_z_hard", "identity_resid_hard", "mean_Ez2_hard", "max_Ez2_hard",
          "Fz_best_history", "fill", "fill_active", "s_flip_lr", "s_flip_ud", "s_inv",
          "binarization_final", "wall_s"]


def row_from_result(res, stage):
    hd = res["hard"]
    return dict(tag=res["tag"], stage=stage, P=res["P"], h=res["h"], pad_frac=res["pad_frac"],
                realized_pad_nm=res["mask"]["realized_pad_nm"], seed=res["seed"], n_iter=res["n_iter"],
                order=res["order"][0], warm_start=res["warm_start"],
                Fz_soft=res["Fz_soft"], Fz_hard=hd["Fz"], Fx_hard=hd["Fx"], Fy_hard=hd["Fy"], Ftot_hard=hd["Ftot"],
                A_hard=hd["A"], R_hard=hd["R"], T_hard=hd["T"], eta_z_hard=hd["Fz"] / max(hd["Ftot"], 1e-30),
                identity_resid_hard=hd["Ftot"] - hd["A"], mean_Ez2_hard=hd["mean_Ez2"], max_Ez2_hard=hd.get("max_Ez2", float("nan")),
                Fz_best_history=max(res["history"]["Fz"]), fill=res["fill_fraction"], fill_active=res["fill_fraction_active"],
                s_flip_lr=res["s_flip_final"]["lr"], s_flip_ud=res["s_flip_final"]["ud"], s_inv=res["s_flip_final"]["inv"],
                binarization_final=res["history"]["binarization"][-1], wall_s=res["wall_s"])


def collect(out=OUT, stage="stage1"):
    rows = []
    for f in sorted((out / "runs").glob("*/result.json")):
        rows.append(row_from_result(json.load(open(f)), stage))
    rows.sort(key=lambda r: -r["Fz_hard"])
    with open(out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(out / "results.json", "w") as fh:
        json.dump(rows, fh, indent=1)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="*", default=config.SEEDS_SCREEN)
    ap.add_argument("--P", type=float, nargs="*", default=config.P_SCREEN)
    ap.add_argument("--h", type=float, nargs="*", default=config.H_SCREEN)
    ap.add_argument("--pad", type=float, nargs="*", default=config.PAD_SCREEN)
    ap.add_argument("--order", type=int, nargs=2, default=config.ORDER_SCREEN)
    ap.add_argument("--collect-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.collect_only:
        rows = collect()
        print(f"collected {len(rows)} runs -> {OUT/'results.csv'}")
        for r in rows[:15]:
            print(f"  {r['tag']:28s} Fz_hard={r['Fz_hard']:.4f} Fz_soft={r['Fz_soft']:.4f} A={r['A_hard']:.3f} eta_z={r['eta_z_hard']:.3f} fill={r['fill']:.3f}")
        return
    jobs = [(P, h, pad, s, a.iters, list(a.order))
            for s in a.seeds for h in a.h for P in a.P for pad in a.pad]   # seed-major: full grid coverage early
    print(f"[stage1] {len(jobs)} jobs, {a.iters} iters, order {a.order}, {a.workers} workers", flush=True)
    t0 = time.time()
    done = 0
    with open(OUT / "progress.log", "a") as prog:
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(worker, j): j for j in jobs}
            for fut in as_completed(futs):
                done += 1
                try:
                    tag, fz, note = fut.result()
                    msg = f"[{done}/{len(jobs)} {time.time()-t0:.0f}s] {tag}: Fz_hard={fz:.4f} ({note})"
                except Exception as e:  # keep going; record the failure
                    msg = f"[{done}/{len(jobs)}] FAILED {futs[fut][:4]}: {e!r}"
                print(msg, flush=True)
                print(msg, file=prog, flush=True)
    rows = collect()
    print(f"[stage1] done in {time.time()-t0:.0f}s; {len(rows)} results")
    for r in rows[:15]:
        print(f"  {r['tag']:28s} Fz_hard={r['Fz_hard']:.4f} Fz_soft={r['Fz_soft']:.4f} A={r['A_hard']:.3f} eta_z={r['eta_z_hard']:.3f} fill={r['fill']:.3f}")


if __name__ == "__main__":
    main()
