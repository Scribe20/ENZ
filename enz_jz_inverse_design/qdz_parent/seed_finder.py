"""SEED SELECTION for the frontier.

The pilot established (outputs/phaseA/weight_scale_audit.json and the pilot histories) that a
topology whose parent spectrum is FLAT at lambda_E cannot be driven to grow a resonance there by the
Q constraint at any weight: the fitted curvature u = c/c_target stays ~0, the loss gradient is finite
but the sensitivity of u to rho vanishes, and the optimizer simply maximizes eta_Dz off-resonance.
The parent resonance wavelength is instead controlled at leading order by the layer height, so each
frontier run is started from a (topology, h) pair whose exact parent pole already lies within a few
linewidths of lambda_E and whose Q is within reach of the target.

For a given topology this scans h, extracts the exact parent poles (analysis.significant_poles) and
scores each height by
      score = (detuning / linewidth)^2 + [log(Q_pole / Q_target)]^2
i.e. the same two constraints the optimizer will then enforce on the topology at fixed h.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import config, forward as fwd, materials as mat       # noqa: E402
import parent_fwd as pf, poles_parent as pp           # noqa: E402
import optimize_parent as op                          # noqa: E402

GEO = config.GEO_DTYPE


def topology(name, P=825.0):
    if name in op.WARM:
        return torch.as_tensor(np.load(HERE.parent / "outputs" / "stage3" / "runs" / op.WARM[name] / "rho_hard_binary.npy"), dtype=GEO)
    if name.startswith("fresh"):
        import audit_qproxy as a1
        return a1.rand_ff(config.NX, P, int(name[5:]))
    if name.startswith("disc"):
        import audit_qproxy as a1
        return a1.disc(config.NX, P, float(name[4:]))
    if name.endswith(".npy"):
        return torch.as_tensor(np.load(name), dtype=GEO)
    raise ValueError(name)


def scan(rho, P, heights, order, lam_E, n_glass, log=print, tag=""):
    rows = []
    for h in heights:
        pk = pp.parent_poles(rho, P, float(h), order, 1180.0, 1398.0, 4.0, n_glass=n_glass)
        for q in pk["poles"]:
            rows.append(dict(h=float(h), lambda_r=q["lambda_nm"], Q=q["Q"], fwhm=q["fwhm_nm"],
                             detune_nm=q["lambda_nm"] - lam_E, detune_lw=(q["lambda_nm"] - lam_E) / max(q["fwhm_nm"], 1e-9)))
        log(f"    {tag} h={h:5.0f}: " + ", ".join(f"{q['lambda_nm']:.0f}/Q{q['Q']:.0f}" for q in pk["poles"]))
    return rows


def score(r, Q_target):
    return (r["detune_lw"]) ** 2 + (np.log(max(r["Q"], 1e-6) / Q_target)) ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topologies", nargs="*", default=["final3", "final1", "final0", "final2", "fresh8080", "fresh4242"])
    ap.add_argument("--heights", type=float, nargs="*", default=list(np.arange(320.0, 761.0, 20.0)))
    ap.add_argument("--q-targets", type=float, nargs="*", default=[20, 50, 100, 200])
    ap.add_argument("--order", type=int, nargs=2, default=[5, 5])
    ap.add_argument("--P", type=float, default=825.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    fwd.set_threads(a.threads)
    lam_E = pf.lambda_E(); ng = float(mat.n_glass(1300.0))
    all_rows, t0 = {}, time.time()
    for name in a.topologies:
        rho = topology(name, a.P)
        all_rows[name] = scan(rho, a.P, a.heights, a.order, lam_E, ng, tag=name)
        print(f"  [{name}] {len(all_rows[name])} poles ({time.time()-t0:.0f} s)", flush=True)
    seeds = {}
    for Qt in a.q_targets:
        cand = [dict(topology=k, **r, score=score(r, Qt)) for k, rows in all_rows.items() for r in rows]
        cand.sort(key=lambda r: r["score"])
        seeds[f"Q{Qt:.0f}"] = cand[:8]
        print(f"\n  Q_target = {Qt:.0f}: best seeds")
        for c in cand[:6]:
            print(f"    {c['topology']:<10} h={c['h']:5.0f}  lam_r={c['lambda_r']:7.1f} ({c['detune_nm']:+6.1f} nm = "
                  f"{c['detune_lw']:+5.2f} linewidths)  Q={c['Q']:7.1f}  score={c['score']:.3f}")
    out = Path(a.out) if a.out else HERE / "outputs" / "phaseA" / "seeds.json"
    json.dump(dict(lam_E=lam_E, order=a.order, P=a.P, heights=a.heights, poles=all_rows, seeds=seeds,
                   scoring="(detuning/linewidth)^2 + [log(Q_pole/Q_target)]^2", wall_s=time.time() - t0),
              open(out, "w"), indent=1, default=float)
    print(f"\n[seeds] -> {out} ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
