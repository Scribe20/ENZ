"""Generic restartable multi-process job runner for Stages 2/3 (and re-runs).

A job file (JSON list) holds dicts with keys
    tag, P, h, pad, seed, n_iter, order (2-list), warm (path to a raw rho .npy
    or null), beta_start (projection sharpness at the first iteration),
    eval_order (2-list or null), n_threads (int)
Results go to <out>/runs/<tag>/ (optimizer.optimize outputs) and are
collected into <out>/results.csv|json (same columns as Stage 1).

    python run_jobs.py --jobs outputs/stage2/jobs.json --out outputs/stage2 --workers 4
"""

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import config
from run_stage1 import row_from_result, FIELDS


def worker(args):
    job, out = args
    import torch
    torch.set_num_threads(int(job.get("n_threads", 1)))
    import materials as mat
    import optimizer as opt
    d = Path(out) / "runs" / job["tag"]
    if (d / "result.json").exists():
        return job["tag"], json.load(open(d / "result.json"))["Fz_hard"], "cached"
    d.mkdir(parents=True, exist_ok=True)
    lam, _ = mat.ito_zero_crossing()
    rho_init = np.load(job["warm"]) if job.get("warm") else None
    with open(d / "log.txt", "w") as logf:
        res = opt.optimize(job["P"], job["h"], job["pad"], job["seed"], job["n_iter"], job["order"], d, lam,
                           rho_init=rho_init, beta_proj_start=job.get("beta_start", 1.0),
                           eval_order_final=job.get("eval_order"), save_every=job.get("save_every", 20),
                           log=lambda *a: print(*a, file=logf, flush=True), tag=job["tag"],
                           n_threads=int(job.get("n_threads", 1)))
    res["warm_from"] = job.get("warm")
    with open(d / "result.json", "w") as f:
        json.dump(res, f, indent=1)
    return job["tag"], res["Fz_hard"], f"{res['wall_s']:.0f}s"


def collect(out, stage):
    out = Path(out)
    rows = []
    for f in sorted((out / "runs").glob("*/result.json")):
        r = json.load(open(f))
        row = row_from_result(r, stage)
        row["warm_from"] = r.get("warm_from")
        rows.append(row)
    rows.sort(key=lambda r: -r["Fz_hard"])
    import csv
    with open(out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS + ["warm_from"])
        w.writeheader(); w.writerows(rows)
    with open(out / "results.json", "w") as fh:
        json.dump(rows, fh, indent=1)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stage", default=None)
    ap.add_argument("--collect-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stage = a.stage or out.name
    if a.collect_only:
        rows = collect(out, stage)
        for r in rows[:20]:
            print(f"  {r['tag']:40s} Fz_hard={r['Fz_hard']:.4f} Fz_soft={r['Fz_soft']:.4f} A={r['A_hard']:.3f} eta_z={r['eta_z_hard']:.3f} fill={r['fill']:.3f} warm={bool(r['warm_from'])}")
        return
    jobs = json.load(open(a.jobs))
    print(f"[{stage}] {len(jobs)} jobs, {a.workers} workers", flush=True)
    t0 = time.time(); done = 0
    with open(out / "progress.log", "a") as prog:
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(worker, (j, str(out))): j for j in jobs}
            for fut in as_completed(futs):
                done += 1
                try:
                    tag, fz, note = fut.result()
                    msg = f"[{done}/{len(jobs)} {time.time()-t0:.0f}s] {tag}: Fz_hard={fz:.4f} ({note})"
                except Exception as e:
                    msg = f"[{done}/{len(jobs)}] FAILED {futs[fut]['tag']}: {e!r}"
                print(msg, flush=True); print(msg, file=prog, flush=True)
    rows = collect(out, stage)
    print(f"[{stage}] done in {time.time()-t0:.0f}s; {len(rows)} results")
    for r in rows[:20]:
        print(f"  {r['tag']:40s} Fz_hard={r['Fz_hard']:.4f} Fz_soft={r['Fz_soft']:.4f} A={r['A_hard']:.3f} eta_z={r['eta_z_hard']:.3f} fill={r['fill']:.3f} warm={bool(r['warm_from'])}")


if __name__ == "__main__":
    main()
