"""Cheap, clearly separated first look at angular robustness (NOT an ensemble optimization): dT(theta) and
S_T(theta) of one frozen candidate at its selected wavelengths for a few incidence angles, using the angle
hook of loaded_sensitivity.evaluate_pair (theta_deg, phi_deg passed to forward.build_sim; lab-x polarization
is mapped to torcwa's p/s basis by forward.lab_x_amplitude_ps).  Rayleigh thresholds move with theta and are
reported per angle so a near-threshold angle is not mistaken for a physical effect.

    python angle_check.py --tag <tag> [--thetas 0 2 4 6 8] [--phi 0] [--order 9 9]
Output: outputs/angle_check/<tag>.json  (mean and spread of dT over the angle set at every wavelength)
"""
import argparse

import numpy as np
import torch

import common as cm                       # noqa: F401
import config                             # noqa: E402
import forward as fwd                     # noqa: E402
import candidates as cd                   # noqa: E402
import loaded_sensitivity as ls           # noqa: E402
import ito_perturbation as ip             # noqa: E402
import rayleigh as ry                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--thetas", type=float, nargs="*", default=[0.0, 2.0, 4.0, 6.0, 8.0])
    ap.add_argument("--phi", type=float, default=0.0); ap.add_argument("--order", type=int, nargs=2, default=[9, 9])
    ap.add_argument("--lams", type=float, nargs="*", default=None); ap.add_argument("--Te", type=float, default=1000.0)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(); fwd.set_threads(a.threads)
    c = cd.by_tag(a.tag); rho = torch.as_tensor(cd.get_rho(c), dtype=config.GEO_DTYPE)
    B = cm.jload(cm.OUT / "candidates" / a.tag / "stageB.json")
    lams = a.lams or sorted({round(B["points"][k]["lam_nm"], 3) for k in ("lambda_op_pos", "lambda_E") if k in B["points"]})
    pert = ip.ITOPerturbation("hot_electron_Te", Te=a.Te)
    rows = []
    for th in a.thetas:
        ray = ry.rayleigh_wavelengths(c["P"], lo=600.0, hi=1400.0, theta_deg=th, phi_deg=a.phi)
        for lam in lams:
            r = ls.evaluate_pair(rho, c["P"], c["h"], lam, a.order, pert, theta_deg=th, phi_deg=a.phi)
            rows.append(dict(theta=th, lam=lam, T0=r["T0"], T1=r["T1"], dT=r["dT"], S_T=r["S_T"], R0=r["R0"], A0=r["A0"], Fz0=r["Fz0"], Ftot0=r["Ftot0"],
                             n_orders=r["n_orders0"], rayleigh=ry.safety_margin(lam, ray)))
            rm = rows[-1]["rayleigh"]
            print(f"  theta={th:4.1f} lam={lam:8.2f}: T0={r['T0']:.4f} dT={r['dT']:+.5f} S_T={r['S_T']:+.4f} A0={r['A0']:.4f} n_orders={r['n_orders0']} "
                  f"nearest Rayleigh {(rm['nearest'] or {}).get('lam', float('nan')):.1f} nm ({(rm['margin_nm'] if rm['margin_nm'] is not None else float('nan')):+.1f})", flush=True)
    summ = {}
    for lam in lams:
        d = np.array([r["dT"] for r in rows if r["lam"] == lam]); s = np.array([r["S_T"] for r in rows if r["lam"] == lam])
        summ[f"{lam:.3f}"] = dict(dT_mean=float(d.mean()), dT_std=float(d.std()), dT_min=float(d.min()), dT_max=float(d.max()), S_T_mean=float(s.mean()), S_T_std=float(s.std()),
                                  sign_stable=bool(np.all(np.sign(d) == np.sign(d[0]))))
    cm.jdump(dict(tag=a.tag, thetas=a.thetas, phi=a.phi, order=a.order, Te=a.Te, rows=rows, summary=summ, provenance=cm.provenance(),
                  note="normal-incidence physics established first; this is a cheap angle sweep through the existing hook, not an NA-weighted ensemble"),
             cm.OUT / "angle_check" / f"{a.tag}.json")
    print(summ)


if __name__ == "__main__":
    main()
