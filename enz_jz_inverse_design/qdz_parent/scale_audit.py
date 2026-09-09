"""WEIGHT SCALE AUDIT - are lambda_Q and lambda_lambda active enough?

The failure mode to exclude is eta_Dz improving while the resonance constraint is abandoned (the
fitted curvature u = c/c_target staying <= 0, i.e. no peak at lambda_E).  Short runs over a grid of
lambda_Q with the loss-term magnitudes recorded every iteration decide the production weights.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import forward as fwd                     # noqa: E402
import optimize_parent as op              # noqa: E402

OUT = HERE / "outputs" / "phaseA"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--h", type=float, default=525.0)
    ap.add_argument("--q-target", type=float, default=100.0)
    ap.add_argument("--lam-Q", type=float, nargs="*", default=[1.0, 3.0, 10.0, 30.0])
    ap.add_argument("--warm", default="final3")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    fwd.set_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    rho0 = np.load(HERE.parent / "outputs" / "stage3" / "runs" / op.WARM[a.warm] / "rho_hard_binary.npy") if a.warm else None
    rows = {}
    for lq in a.lam_Q:
        tag = f"scale_lamQ{lq:g}"
        r = op.run(825.0, a.h, a.q_target, 333, a.iters, [5, 5], HERE / "outputs" / "phaseA" / tag,
                   rho_init=rho0, lam_Q=lq, lam_lam=lq, n_z=5, tag=tag, n_threads=a.threads, save_every=1000,
                   log=lambda m: print(m, flush=True))
        H = r["history"]
        u = np.array(H["curvature_u"]); eta = np.array(H["eta_Dz"])
        rows[f"lambda_Q={lq:g}"] = dict(
            lambda_Q=lq, u_start=float(u[0]), u_end=float(u[-1]), u_max=float(u.max()),
            reached_resonance=bool((u > 0.1).any()), first_iter_u_positive=(int(np.argmax(u > 0.1)) if (u > 0.1).any() else None),
            eta_start=float(eta[0]), eta_end=float(eta[-1]), eta_gain=float(eta[-1] / eta[0]),
            L_obj_end=float(H["L_obj"][-1]), L_Q_end=float(H["L_Q"][-1]), L_lam_end=float(H["L_lam"][-1]),
            L_Q_over_L_obj_end=float(abs(lq * H["L_Q"][-1]) / max(abs(H["L_obj"][-1]), 1e-9)),
            Q_proxy_end=float(H["Q_proxy"][-1]), eta_hard=r["hard"]["eta_Dz"], u_hard=r["hard"]["curvature_u"],
            wall_s=r["wall_s"])
        print(f"  lambda_Q={lq:g}: u {u[0]:+.3f} -> {u[-1]:+.3f} (max {u.max():+.3f}), eta {eta[0]:.4f} -> {eta[-1]:.4f}, "
              f"resonance reached: {rows[f'lambda_Q={lq:g}']['reached_resonance']}", flush=True)
    json.dump(rows, open(OUT / "weight_scale_audit.json", "w"), indent=1, default=float)
    print(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
