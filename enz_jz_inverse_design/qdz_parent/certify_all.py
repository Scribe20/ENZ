"""Certify every hard-binary candidate produced by the frontier/pilot runs (exact, post hoc)."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import config, forward as fwd            # noqa: E402
import certify_parent as cp              # noqa: E402

GEO = config.GEO_DTYPE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=["frontier", "pilot"])
    ap.add_argument("--orders", type=int, nargs="*", default=[5, 5, 7, 7, 9, 9])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--only-acquired", action="store_true",
                    help="skip runs whose fitted curvature never became positive (no resonance at lambda_E)")
    ap.add_argument("--worker", type=int, default=0); ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args()
    fwd.set_threads(a.threads)
    orders = [a.orders[i:i + 2] for i in range(0, len(a.orders), 2)]
    jobs = []
    for d in a.dirs:
        for p in sorted((HERE / d).glob("*/result.json")):
            r = json.load(open(p))
            if r.get("campaign") != "qdz_parent":
                continue
            acq = max(r["history"]["curvature_u"]) > 0.1
            if a.only_acquired and not acq:
                print(f"  skip {r['tag']}: never acquired a resonance (max u = {max(r['history']['curvature_u']):.3f})")
                continue
            jobs.append((p.parent, r, acq))
    mine = [j for i, j in enumerate(jobs) if i % a.nworkers == a.worker]
    print(f"[certify worker {a.worker}] {len(mine)} of {len(jobs)} candidates", flush=True)
    for d, r, acq in mine:
        rho = torch.as_tensor(np.load(d / "rho_hard_binary.npy"), dtype=GEO)
        out = HERE / "outputs" / "certified" / r["tag"]
        t = time.time()
        try:
            c = cp.certify(rho, r["P"], r["h"], r["tag"], out, orders=orders, n_z=9)
            c["run_dir"] = str(d.relative_to(HERE)); c["Q_target"] = r["Q_target"]
            c["acquired_resonance_during_run"] = bool(acq)
            c["proxy_at_end_of_run"] = dict(Q_proxy=r["hard"].get("Q_proxy"), lambda_r_proxy=r["hard"].get("lambda_r_proxy"),
                                            curvature_u=r["hard"].get("curvature_u"))
            json.dump(c, open(out / "certification.json", "w"), indent=1, default=float)
        except Exception as e:
            print(f"  FAILED {r['tag']}: {type(e).__name__}: {e}", flush=True)
    print(f"[certify worker {a.worker}] done", flush=True)


if __name__ == "__main__":
    main()
