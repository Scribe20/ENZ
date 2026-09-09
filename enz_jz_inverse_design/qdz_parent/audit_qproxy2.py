"""CHECK 4 (second pass) - PROBE-SPAN SELECTION for the differentiable Q proxy.

The first calibration (outputs/phaseA/qproxy_calibration.json) showed that the 1/W quadratic fit is
accurate only when the probe window is NARROW compared with the linewidth: the closed form is exact
for an isolated Lorentzian at any span, so the error is entirely the non-Lorentzian background and
neighbouring resonances, which dominate once the probes reach the line wings.  This study selects the
span, comparing two candidate spectral quantities:
    W_Si    = rho-weighted electric energy in the a-Si:H (the eta_Dz denominator, real-space)
    W_layer = Parseval full-cell energy of the design layer (no real-space synthesis, no Gibbs error)
over span in {0.125 ... 2} x kappa, centre offsets in {0, 0.25, 0.5} x kappa, and a 2x mismatch
between Q_target and the true Q (which the optimizer will always have).
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import config, forward as fwd, materials as mat        # noqa: E402
import parent_fwd as pf, qproxy as qp, poles_parent as pp, audit_qproxy as a1   # noqa: E402

OUT = HERE / "outputs" / "phaseA"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, nargs=2, default=[5, 5])
    ap.add_argument("--P", type=float, default=825.0)
    ap.add_argument("--n-z", type=int, default=5)
    ap.add_argument("--max-poles", type=int, default=16)
    a = ap.parse_args()
    fwd.set_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    prev = json.load(open(OUT / "qproxy_calibration.json"))
    seen, poles = set(), []
    for r in prev["rows"]:
        k = (r["structure"], r["h"], round(r["lam_exact"], 2))
        if k not in seen:
            seen.add(k); poles.append(dict(structure=r["structure"], h=r["h"], lam=r["lam_exact"], Q=r["Q_exact"]))
    poles = sorted(poles, key=lambda p: p["Q"])[:a.max_poles]
    lib = a1.library(a.P)
    rows, t0 = [], time.time()
    for ip, pl in enumerate(poles):
        rho = lib[pl["structure"]]; h = pl["h"]; Q_ex = pl["Q"]; lam_r = pl["lam"]
        w_r = qp.omega_of(lam_r); k_ex = w_r / Q_ex
        cfgs = [(sp, off, 1.0) for sp in (0.125, 0.25, 0.5, 1.0, 2.0) for off in (0.0, 0.25, 0.5)]
        cfgs += [(sp, 0.25, m) for sp in (0.25, 0.5) for m in (0.5, 2.0)]
        for span, off, mism in cfgs:
            w_c = w_r + off * k_ex
            Q_t = Q_ex * mism                                    # probe kappa = w_c / Q_t
            lams, om, kt = qp.probe_wavelengths(qp.lam_of(w_c), Q_t, n=5, span=span)
            if lams.max() > 1399.0 or lams.min() < 250.0:
                continue
            d = [pf.parent_metrics(rho, a.P, h, float(l), a.order, n_z=a.n_z) for l in lams]
            for key in ("W_Si", "W_layer"):
                W = torch.stack([x[key] for x in d])
                fit = qp.lorentz_fit(om, W, w_c)
                rows.append(dict(structure=pl["structure"], h=h, Q_exact=Q_ex, lam_exact=lam_r, quantity=key,
                                 span=span, offset=off, mismatch=mism, Q_proxy=float(fit["Q"]),
                                 lambda_r_proxy=float(fit["lambda_r"]), rel_err_Q=float(abs(float(fit["Q"]) - Q_ex) / Q_ex),
                                 lam_err_nm=float(abs(float(fit["lambda_r"]) - lam_r)), fit_rel_resid=float(fit["rel_resid"]),
                                 well_posed=bool(fit["well_posed"])))
        print(f"  [{ip+1}/{len(poles)}] {pl['structure']} h={h:.0f} lam={lam_r:.1f} Q={Q_ex:.1f} ({time.time()-t0:.0f} s)", flush=True)
    def sel(**kw):
        return [r for r in rows if all(r[k] == v for k, v in kw.items())]
    def stat(sub):
        if not sub:
            return None
        e = np.array([r["rel_err_Q"] for r in sub]); l = np.array([r["lam_err_nm"] for r in sub])
        lo = np.log(np.clip([r["Q_proxy"] for r in sub], 1e-6, None)); le = np.log([r["Q_exact"] for r in sub])
        return dict(n=len(sub), median_rel_err_Q=float(np.median(e)), p90=float(np.percentile(e, 90)),
                    max=float(e.max()), median_lam_err_nm=float(np.median(l)),
                    log_corr=(float(np.corrcoef(le, lo)[0, 1]) if len(sub) > 2 else None),
                    frac_within_20pct=float(np.mean(e < 0.20)), frac_within_35pct=float(np.mean(e < 0.35)))
    table = {}
    for key in ("W_Si", "W_layer"):
        for span in (0.125, 0.25, 0.5, 1.0, 2.0):
            table[f"{key}|span={span}|centred"] = stat(sel(quantity=key, span=span, offset=0.0, mismatch=1.0))
            table[f"{key}|span={span}|offset=0.25k"] = stat(sel(quantity=key, span=span, offset=0.25, mismatch=1.0))
        for m in (0.5, 2.0):
            for span in (0.25, 0.5):
                table[f"{key}|span={span}|Qtarget x{m}"] = stat(sel(quantity=key, span=span, offset=0.25, mismatch=m))
    json.dump(dict(order=a.order, n_rows=len(rows), table=table, rows=rows, wall_s=time.time() - t0),
              open(OUT / "qproxy_span_selection.json", "w"), indent=1, default=float)
    print("\n" + "=" * 108)
    print(f"{'configuration':<42} {'n':>4} {'median|dQ|/Q':>13} {'p90':>8} {'<20%':>6} {'<35%':>6} {'logcorr':>8} {'|dlam| nm':>10}")
    for k, v in table.items():
        if v:
            print(f"{k:<42} {v['n']:>4} {v['median_rel_err_Q']:>13.3f} {v['p90']:>8.3f} {v['frac_within_20pct']:>6.2f} "
                  f"{v['frac_within_35pct']:>6.2f} {str(round(v['log_corr'],3)) if v['log_corr'] is not None else '-':>8} {v['median_lam_err_nm']:>10.3f}")


if __name__ == "__main__":
    main()
