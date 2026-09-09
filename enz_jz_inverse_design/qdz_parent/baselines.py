"""BASELINES - evaluate (never re-optimize) existing structures under the new PARENT metrics.

  * the four Jz/Fz finalists  final3 / final1 / final0 / final2  at their own certified P, h, pad
  * one conventional dielectric meta-atom already present in the repository: the SNU deck's
    fabricated a-Si:H cylinder (d = 405.7 nm, h = 970.5 nm, P = 588 nm; SOURCE_AUDIT.md section 8),
    reproduced with forward.cuboid-style rasterisation on the same 128 x 128 grid, plus the same
    cylinder placed in the campaign's own 825-nm cell for a like-for-like comparison.

For each: parent Q_r, lambda_r, eta_Dz, U_mid (+ everything certify_parent.py produces).
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import config, forward as fwd                       # noqa: E402
import certify_parent as cp                          # noqa: E402

GEO = config.GEO_DTYPE
FINALISTS = {
    "final3": dict(tag="final3_div_F1_P825_h500_pad0.12_s333_h525", P=825.0, h=525.0, pad=0.12),
    "final1": dict(tag="final1_F0_P825_h600_pad0.12_s1001_h575", P=825.0, h=575.0, pad=0.12),
    "final0": dict(tag="final0_F2_P850_h600_pad0.12_s333_P825", P=825.0, h=600.0, pad=0.12),
    "final2": dict(tag="final2_F0_P825_h600_pad0.12_s1001_pad0.08", P=825.0, h=600.0, pad=0.08),
}


def disc(nx, P, d):
    x = (np.arange(nx) + 0.5) / nx * P
    X, Y = np.meshgrid(x, x, indexing="ij")
    return torch.as_tensor((((X - P / 2) ** 2 + (Y - P / 2) ** 2) < (d / 2) ** 2).astype(float), dtype=GEO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--orders", type=int, nargs="*", default=[5, 5, 7, 7, 9, 9])
    a = ap.parse_args()
    fwd.set_threads(a.threads)
    orders = [a.orders[i:i + 2] for i in range(0, len(a.orders), 2)]
    OUT = HERE / "baselines"
    jobs = {}
    for name, d in FINALISTS.items():
        p = HERE.parent / "outputs" / "stage3" / "runs" / d["tag"] / "rho_hard_binary.npy"
        jobs[name] = (torch.as_tensor(np.load(p), dtype=GEO), d["P"], d["h"], str(p))
    jobs["deck_cylinder_P588_h970"] = (disc(config.NX, 588.0, 405.7), 588.0, 970.5,
                                       "SNU deck fabricated design (SOURCE_AUDIT.md section 8): d=405.7 nm cylinder")
    jobs["cylinder_P825_h525_d405"] = (disc(config.NX, 825.0, 405.7), 825.0, 525.0,
                                       "the same cylinder in the campaign's own 825-nm cell at h = 525 nm")
    summary = {}
    for name, (rho, P, h, prov) in jobs.items():
        if a.which and name not in a.which:
            continue
        print(f"=== baseline {name}: P={P} h={h}", flush=True)
        r = cp.certify(rho, P, h, name, OUT / name, orders=orders, n_z=9)
        r["provenance"] = prov
        json.dump(r, open(OUT / name / "certification.json", "w"), indent=1, default=float)
        m = r["metrics"].get("lambda_E", {}).get("by_order", [{}])[-1]
        summary[name] = dict(P=P, h=h, lambda_r=r.get("lambda_r"), Q_r=r.get("Q_r"),
                             detuning_nm=r.get("detuning_nm"), detuning_in_linewidths=r.get("detuning_in_linewidths"),
                             eta_Dz_at_lambda_E=m.get("eta_Dz"), eta_Dz_P_at_lambda_E=m.get("eta_Dz_P"),
                             U_mid_at_lambda_E=m.get("U_mid"), R_at_lambda_E=m.get("R"),
                             eta_Dz_at_lambda_r=r["metrics"].get("lambda_r", {}).get("by_order", [{}])[-1].get("eta_Dz"),
                             U_mid_at_lambda_r=r["metrics"].get("lambda_r", {}).get("by_order", [{}])[-1].get("U_mid"),
                             n_parent_poles=len(r.get("parent_poles", [])), sha256=r["sha256"], provenance=prov)
        p = OUT / "baselines_summary.json"
        old = json.load(open(p)) if p.exists() else {}
        old.update(summary); json.dump(old, open(p, "w"), indent=1, default=float)
    for k, v in summary.items():
        print(f"  {k:28s} lam_r={v['lambda_r']} Q_r={v['Q_r']} eta_Dz(lam_E)={v['eta_Dz_at_lambda_E']} U_mid={v['U_mid_at_lambda_E']}")


if __name__ == "__main__":
    main()
