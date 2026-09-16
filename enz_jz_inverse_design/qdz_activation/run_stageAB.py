"""Campaign driver: Stage A (parent background / contrast) + Stage B (real-ITO sensitivity) for every
frozen candidate of candidates.registry(), or a selection of tags.  One JSON per stage per candidate plus
a compact NPZ of the Stage-B scan, under outputs/candidates/<tag>/ (or --out).

    python run_stageAB.py                              # all candidates, order [9,9], Te = 1000 K
    python run_stageAB.py --tags fr_Q200_fresh4242_h700 deck_cylinder_P588_h970 --order 7 7
    python run_stageAB.py --worker 0 --nworkers 2 --threads 2   (two processes on the 4-core host)

Every output carries the provenance block (git commit, material SHA256, grid, order, d_ITO, ...), the
geometry SHA256, the perturbation description (native parameters + resulting complex epsilon values) and
all thresholds / parameters used.
"""
import argparse, time
from pathlib import Path

import numpy as np

import common as cm                       # noqa: F401
import config                             # noqa: E402
import forward as fwd                     # noqa: E402
import candidates as cd                   # noqa: E402
import parent_background as pb            # noqa: E402
import loaded_sensitivity as ls           # noqa: E402
import ito_perturbation as ip             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--kinds", nargs="*", default=None, help="qdz_certified qdz_uncertified jzfz_baseline reference")
    ap.add_argument("--order", type=int, nargs=2, default=[9, 9])
    ap.add_argument("--order-A", type=int, nargs=2, default=[9, 9])
    ap.add_argument("--orders-check", type=int, nargs="*", default=[5, 5, 7, 7, 9, 9, 11, 11])
    ap.add_argument("--lam-lo", type=float, default=None)
    ap.add_argument("--lam-hi", type=float, default=None)
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--kind", default="hot_electron_Te", choices=ip.KINDS)
    ap.add_argument("--Te", type=float, default=1000.0)
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--custom", default=None, help="re,im of a constant delta_eps (kind=custom; NON-PHYSICAL, diagnostics only)")
    ap.add_argument("--m-R", type=float, default=10.0, help="Rayleigh safety margin [nm] for the operating window")
    ap.add_argument("--D-op", type=float, default=50.0, help="half-width [nm] of the ENZ operating window for T_bg")
    ap.add_argument("--k-excl", type=float, default=3.0, help="linewidths excluded around each parent pole for T_bg")
    ap.add_argument("--n-z", type=int, default=config.Z_SAMPLES_ITO)
    ap.add_argument("--theta", type=float, default=0.0); ap.add_argument("--phi", type=float, default=0.0)
    ap.add_argument("--pol", default="labx")
    ap.add_argument("--skip-A", action="store_true"); ap.add_argument("--skip-B", action="store_true")
    ap.add_argument("--no-linearity", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--worker", type=int, default=0); ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    fwd.set_threads(a.threads)
    out_root = Path(a.out) if a.out else cm.OUT / "candidates"
    pert = ip.from_args(a.kind, a.Te, a.delta, a.custom)
    reg = cd.registry()
    if a.tags:
        reg = [c for c in reg if c["tag"] in a.tags]
    if a.kinds:
        reg = [c for c in reg if c["kind"] in a.kinds]
    mine = [c for i, c in enumerate(reg) if i % a.nworkers == a.worker]
    logf = open(cm.HERE / "logs" / f"run_w{a.worker}.log", "a") if (cm.HERE / "logs").exists() or (cm.HERE / "logs").mkdir(parents=True, exist_ok=True) is None else None
    def log(m):
        print(m, flush=True)
        if logf:
            logf.write(m + "\n"); logf.flush()
    log(f"[worker {a.worker}] {len(mine)} of {len(reg)} candidates; order {a.order}; perturbation {a.kind} Te={a.Te}")
    prov = cm.provenance(order=a.order, order_A=a.order_A, orders_check=a.orders_check, step=a.step, n_z=a.n_z,
                         incidence=dict(theta_deg=a.theta, phi_deg=a.phi, pol=a.pol), m_R_nm=a.m_R, D_op_nm=a.D_op, k_excl=a.k_excl)
    for c in mine:
        t0 = time.time()
        rho = cd.get_rho(c)
        out = out_root / c["tag"]; out.mkdir(parents=True, exist_ok=True)
        meta = dict(candidate={k: v for k, v in c.items()}, rho_sha256=(cm.sha256_array(rho) if rho is not None else None),
                    fill_fraction=(float(np.mean(rho)) if rho is not None else None), provenance=prov)
        try:
            pinfo = None
            if not a.skip_A:
                ra = pb.stage_a(c, rho, order=tuple(a.order_A), D_op=a.D_op, m_R=a.m_R, k_excl=a.k_excl, log=log)
                ra["meta"] = meta
                cm.jdump(ra, out / "stageA.json")
                near = (ra.get("resonance") or {}).get("pole") or {}
                pinfo = dict(Q_r=near.get("Q"), lambda_r=near.get("lam"),
                             eta_Dz_P_lamE=(ra.get("parent_metrics_here", {}).get("lambda_E") or {}).get("eta_Dz_P"),
                             parent_poles=ra.get("poles"))
            elif (out / "stageA.json").exists():
                ra = cm.jload(out / "stageA.json"); near = (ra.get("resonance") or {}).get("pole") or {}
                pinfo = dict(Q_r=near.get("Q"), lambda_r=near.get("lam"),
                             eta_Dz_P_lamE=(ra.get("parent_metrics_here", {}).get("lambda_E") or {}).get("eta_Dz_P"), parent_poles=ra.get("poles"))
            if not a.skip_B:
                lam_r = (c.get("cert") or {}).get("lambda_r") or (pinfo or {}).get("lambda_r")
                rb, A = ls.stage_b(c, rho, pert, order=tuple(a.order), lam_lo=a.lam_lo, lam_hi=a.lam_hi, step=a.step, m_R=a.m_R,
                                   orders_check=[tuple(a.orders_check[i:i + 2]) for i in range(0, len(a.orders_check), 2)],
                                   n_z=a.n_z, theta_deg=a.theta, phi_deg=a.phi, pol=a.pol, do_linearity=not a.no_linearity,
                                   log=log, parent_lambda_r=lam_r, parent_info=pinfo)
                rb["meta"] = meta
                cm.jdump(rb, out / "stageB.json")
                np.savez_compressed(out / "stageB_scan.npz", **{k: v for k, v in A.items()}, order=np.array(a.order),
                                    P=c["P"], h=c["h"], tag=c["tag"], Te=a.Te, kind=a.kind)
        except Exception as e:
            import traceback
            log(f"  FAILED {c['tag']}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        log(f"[worker {a.worker}] {c['tag']} done in {time.time()-t0:.0f} s")
    log(f"[worker {a.worker}] all done")


if __name__ == "__main__":
    main()
