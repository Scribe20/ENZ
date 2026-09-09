"""HYBRID CERTIFICATION - insert the real measured 23-nm dispersive ITO into a frozen parent design.

This is CERTIFICATION ONLY: F_z is never re-introduced as an optimization objective.  Everything is
computed with the unchanged Jz/Fz machinery (forward.evaluate with with_ito=True, the measured
ITO_nk.csv dispersion, analysis.spectrum / significant_poles / track_branch / fit_gamma).

Per candidate:  R(lambda), T(lambda), A(lambda), F_x, F_y, F_z, F_tot, eta_z_abs, <|Ez|^2> in the ITO,
the loaded poles, the parent (unloaded) poles for the same geometry, and the loss-scaling continuation
gamma(s) = gamma_rad + s gamma_nr that separates radiative from non-radiative damping.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import config, forward as fwd, materials as mat, analysis as an     # noqa: E402
import parent_fwd as pf, poles_parent as pp                          # noqa: E402

GEO = config.GEO_DTYPE


def loaded_spectrum(rho, P, h, lams, order, n_z=config.Z_SAMPLES_ITO):
    rows = []
    with torch.no_grad():
        for lam in lams:
            d = fwd.evaluate(rho, P, h, float(lam), order, n_z=n_z)
            f = fwd.to_floats(d)
            r, t = fwd.specular_rt_amplitudes(d["sim"])
            f.update(lam=float(lam), r=complex(r), t=complex(t))
            rows.append(f)
    return rows


def certify_loaded(rho, P, h, tag, out_dir, order=config.ORDER_FULL, lam_lo=1180.0, lam_hi=1398.0,
                   step=2.0, n_z=config.Z_SAMPLES_ITO, log=print, track=True):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    lam_E = pf.lambda_E(); t0 = time.time()
    ng = float(mat.n_glass(1300.0))
    lams = np.arange(lam_lo, lam_hi + 1e-9, step)
    rows = loaded_spectrum(rho, P, h, lams, order, n_z)
    arr = {k: np.array([r[k] for r in rows]) for k in ("R", "T", "A", "Fx", "Fy", "Fz", "Ftot", "mean_Ez2")}
    r_c = np.array([r["r"] for r in rows]); t_c = np.array([r["t"] for r in rows])
    np.savez_compressed(out / "loaded_spectrum.npz", lam_nm=lams, r=r_c, t=t_c, order=order, **arr)
    ray = [x["lam"] for x in an.rayleigh_wavelengths(P, ng, lo=lam_lo, hi=lam_hi)]
    loaded_poles = an.significant_poles(lams, r_c, t_c, exclude=ray)
    par = pp.parent_poles(rho, P, h, order, lam_lo, lam_hi, step, n_glass=ng)
    iE = int(np.argmin(np.abs(lams - lam_E)))
    res = dict(tag=tag, P=float(P), h=float(h), lam_E=lam_E, order=list(order), n_z=int(n_z),
               at_lambda_E={k: float(arr[k][iE]) for k in arr},
               eta_z_abs=float(arr["Fz"][iE] / max(arr["Ftot"][iE], 1e-30)),
               identity_resid=float(arr["Ftot"][iE] - arr["A"][iE]),
               A_max=float(arr["A"].max()), lambda_A_max=float(lams[int(np.argmax(arr["A"]))]),
               Fz_max=float(arr["Fz"].max()), lambda_Fz_max=float(lams[int(np.argmax(arr["Fz"]))]),
               loaded_poles=loaded_poles, parent_poles=par["poles"], rayleigh_nm=ray,
               parent_energy_resid=par["energy_resid"], spectrum_file="loaded_spectrum.npz")
    if track and loaded_poles:
        p0 = min(loaded_poles, key=lambda p: abs(p["lambda_nm"] - lam_E))
        try:
            tr = an.track_branch(rho, P, h, p0["omega_re"], lams, order=order, tag=tag, log=lambda *a_, **k_: None)
            res["loss_scaling_rows"] = tr
            res["gamma_fit"] = an.fit_gamma(tr)
        except Exception as e:                       # never block the campaign on the continuation
            res["loss_scaling_error"] = f"{type(e).__name__}: {e}"
    res["wall_s"] = time.time() - t0
    json.dump(res, open(out / "loaded_certification.json", "w"), indent=1, default=float)
    a_ = res["at_lambda_E"]
    log(f"[loaded {tag}] at lambda_E: R={a_['R']:.4f} T={a_['T']:.4f} A={a_['A']:.4f} Fz={a_['Fz']:.4f} "
        f"Ftot={a_['Ftot']:.4f} eta_z={res['eta_z_abs']:.3f} <|Ez|^2>={a_['mean_Ez2']:.2f} | "
        f"A_max={res['A_max']:.4f}@{res['lambda_A_max']:.0f} nm | {len(loaded_poles)} loaded / {len(par['poles'])} parent poles ({res['wall_s']:.0f} s)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", required=True); ap.add_argument("--h", type=float, required=True)
    ap.add_argument("--P", type=float, default=825.0); ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=None); ap.add_argument("--order", type=int, nargs=2, default=config.ORDER_FULL)
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--no-track", action="store_true")
    a = ap.parse_args()
    fwd.set_threads(a.threads)
    rho = torch.as_tensor(np.load(a.rho), dtype=GEO)
    out = Path(a.out) if a.out else HERE / "hybrid_certification" / a.tag
    certify_loaded(rho, a.P, a.h, a.tag, out, order=a.order, track=not a.no_track)


if __name__ == "__main__":
    main()
