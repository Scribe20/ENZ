"""Cheap height scan of the PARENT resonance for reference geometries: where does a parent mode sit
relative to lambda_E, and with what radiative Q?  One RCWA solve per (geometry, h, lambda).
Informs the height prior of the campaign (the Jz/Fz prior was h = 525-600 nm, chosen for absorption,
not for a parent resonance at lambda_E)."""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import config, forward as fwd, materials as mat        # noqa: E402
import parent_fwd as pf, poles_parent as pp, audit_qproxy as a1   # noqa: E402

GEO = config.GEO_DTYPE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heights", type=float, nargs="*", default=list(np.arange(300.0, 901.0, 25.0)))
    ap.add_argument("--order", type=int, nargs=2, default=[5, 5])
    ap.add_argument("--P", type=float, default=825.0)
    ap.add_argument("--which", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    fwd.set_threads(a.threads)
    lam_E = pf.lambda_E(); ng = float(mat.n_glass(1300.0))
    lib = a1.library(a.P)
    for name, tag in (("final3", "final3_div_F1_P825_h500_pad0.12_s333_h525"),
                      ("final1", "final1_F0_P825_h600_pad0.12_s1001_h575"),
                      ("final0", "final0_F2_P850_h600_pad0.12_s333_P825"),
                      ("final2", "final2_F0_P825_h600_pad0.12_s1001_pad0.08")):
        p = HERE.parent / "outputs" / "stage3" / "runs" / tag / "rho_hard_binary.npy"
        if p.exists():
            lib[f"jzfz_{name}"] = torch.as_tensor(np.load(p), dtype=GEO)
    sel = a.which or ["jzfz_final3", "jzfz_final1", "jzfz_final0", "jzfz_final2", "disc_d540", "square_w500", "rand_s11", "rand_s202"]
    out, t0 = {}, time.time()
    for name in sel:
        rho = lib[name]; rows = []
        for h in a.heights:
            pk = pp.parent_poles(rho, a.P, float(h), a.order, 1180.0, 1398.0, 4.0, n_glass=ng)
            near = pp.nearest_pole(pk["poles"], lam_E)
            f = pf.to_floats(pf.parent_metrics(rho, a.P, float(h), lam_E, a.order, n_z=5))
            rows.append(dict(h=float(h), n_poles=len(pk["poles"]),
                             poles=[dict(lam=p["lambda_nm"], Q=p["Q"]) for p in pk["poles"]],
                             lam_r=(near["lambda_nm"] if near else None), Q_r=(near["Q"] if near else None),
                             detune_nm=(near["lambda_nm"] - lam_E if near else None),
                             eta_Dz=f["eta_Dz"], U_mid=f["U_mid"], W_Si=f["W_Si"], R=f["R"]))
            print(f"  {name} h={h:5.0f}: {len(pk['poles'])} poles "
                  f"{['%.0f/Q%.0f' % (p['lambda_nm'], p['Q']) for p in pk['poles']]} "
                  f"nearest {('%.1f nm Q=%.0f (%+.1f nm)' % (near['lambda_nm'], near['Q'], near['lambda_nm']-lam_E)) if near else '-'} "
                  f"eta={f['eta_Dz']:.4f} ({time.time()-t0:.0f} s)", flush=True)
        out[name] = rows
        json.dump(out, open(HERE / "outputs" / "phaseA" / "height_scan.json", "w"), indent=1, default=float)
    print(f"[hscan] {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
