"""CHECK 4 - calibration of the differentiable Q proxy against the exact post-hoc pole Q.

Protocol.  A library of parent structures (analytic discs/squares/rings of several sizes and heights,
filtered-random freeforms, and the four Jz/Fz finalists) is scanned for exact radiative poles with
analysis.significant_poles.  For every pole found in the window, the 5-point Lorentzian proxy is
evaluated with the probe window centred on that pole and a span set by an ASSUMED Q_target, and the
proxy Q is compared with the exact pole Q.  Two stress axes are covered, because during optimization
neither is exact:
    - Q_target / Q_exact  in {1/4, 1/2, 1, 2, 4}   (probe span mismatched to the true linewidth)
    - the centre offset   in {0, 0.25, 0.5} * kappa_exact  (resonance not exactly at the probe centre)
Outputs the median/robust relative error per regime, i.e. the VALID RANGE of the proxy.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import config, forward as fwd, materials as mat            # noqa: E402
from optimizer import gaussian_kernel_fft, filter_rho, project_rho  # noqa: E402
import parent_fwd as pf, qproxy as qp, poles_parent as pp  # noqa: E402

OUT = HERE / "outputs" / "phaseA"
GEO = config.GEO_DTYPE
NX = config.NX


def disc(nx, P, d):
    x = (np.arange(nx) + 0.5) / nx * P
    X, Y = np.meshgrid(x, x, indexing="ij")
    return torch.as_tensor(((X - P / 2) ** 2 + (Y - P / 2) ** 2 < (d / 2) ** 2).astype(float), dtype=GEO)


def ring(nx, P, d_out, d_in):
    x = (np.arange(nx) + 0.5) / nx * P
    X, Y = np.meshgrid(x, x, indexing="ij")
    r2 = (X - P / 2) ** 2 + (Y - P / 2) ** 2
    return torch.as_tensor(((r2 < (d_out / 2) ** 2) & (r2 > (d_in / 2) ** 2)).astype(float), dtype=GEO)


def square(nx, P, w):
    x = (np.arange(nx) + 0.5) / nx * P
    X, Y = np.meshgrid(x, x, indexing="ij")
    return torch.as_tensor(((np.abs(X - P / 2) < w / 2) & (np.abs(Y - P / 2) < w / 2)).astype(float), dtype=GEO)


def rand_ff(nx, P, seed, pad=0.12, radius=60.0, beta=8.0):
    M_np, _ = fwd.build_pad_mask(nx, P, pad)
    M = torch.as_tensor(M_np, dtype=GEO)
    g = gaussian_kernel_fft(nx, nx, P / nx, P / nx, radius)
    torch.manual_seed(seed)
    return (project_rho(filter_rho(torch.rand((nx, nx), dtype=GEO) * M, g), beta) * M > 0.5).to(GEO) * M


def library(P):
    lib = {}
    for d in (380.0, 460.0, 540.0, 620.0):
        lib[f"disc_d{d:.0f}"] = disc(NX, P, d)
    for w in (400.0, 500.0, 600.0):
        lib[f"square_w{w:.0f}"] = square(NX, P, w)
    lib["ring_620_300"] = ring(NX, P, 620.0, 300.0)
    lib["ring_700_420"] = ring(NX, P, 700.0, 420.0)
    for s in (11, 202, 3003):
        lib[f"rand_s{s}"] = rand_ff(NX, P, s)
    for name, tag in (("final3", "final3_div_F1_P825_h500_pad0.12_s333_h525"),
                      ("final0", "final0_F2_P850_h600_pad0.12_s333_P825")):
        p = HERE.parent / "outputs" / "stage3" / "runs" / tag / "rho_hard_binary.npy"
        if p.exists():
            lib[f"jzfz_{name}"] = torch.as_tensor(np.load(p), dtype=GEO)
    return lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, nargs=2, default=[5, 5])
    ap.add_argument("--P", type=float, default=825.0)
    ap.add_argument("--heights", type=float, nargs="*", default=[525.0, 600.0])
    ap.add_argument("--n-z", type=int, default=5)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fwd.set_threads(4)
    lam_E = pf.lambda_E()
    lib = library(a.P)
    rows, t0 = [], time.time()
    ng = float(mat.n_glass(1300.0))
    ray = [x["lam"] for x in pp.an.rayleigh_wavelengths(a.P, ng, lo=1160.0, hi=1400.0)]
    stress_budget = 6                      # full 5x3 stress grid on this many poles; ideal setting on all
    n_stress = 0
    for name, rho in lib.items():
        for h in a.heights:
            pk = pp.parent_poles(rho, a.P, h, a.order, 1160.0, 1400.0, 3.0, n_glass=ng)
            for pole in pk["poles"]:
                if not (1200.0 <= pole["lambda_nm"] <= 1390.0) or pole["Q"] < 3 or pole["Q"] > 3000:
                    continue
                lam_r, Q_ex = pole["lambda_nm"], pole["Q"]
                w_r = qp.omega_of(lam_r); k_ex = w_r / Q_ex
                # probe wavelengths must stay inside the supplied a-Si:H range [246, 1400] nm
                if qp.lam_of(w_r - 4.0 * k_ex) > 1399.0:
                    continue
                stress = n_stress < stress_budget
                n_stress += 1 if stress else 0
                grid = [(r_, o_) for r_ in (0.25, 0.5, 1.0, 2.0, 4.0) for o_ in (0.0, 0.25, 0.5)] if stress else [(1.0, 0.0), (0.5, 0.25), (2.0, 0.25)]
                for ratio, off in grid:
                    if True:
                        Q_t = Q_ex / ratio                        # probe span = w_center/Q_t
                        w_c = w_r + off * k_ex
                        lams, omegas, kt = qp.probe_wavelengths(qp.lam_of(w_c), Q_t)
                        W = torch.stack([pf.parent_metrics(rho, a.P, h, float(l), a.order, n_z=a.n_z)["W_Si"] for l in lams])
                        fit = qp.lorentz_fit(omegas, W, w_c)
                        rows.append(dict(structure=name, h=h, lam_exact=lam_r, Q_exact=Q_ex, span_ratio=ratio,
                                         centre_offset_kappa=off, Q_proxy=float(fit["Q"]), lambda_r_proxy=float(fit["lambda_r"]),
                                         kappa_proxy=float(fit["kappa"]), kappa_exact=float(k_ex),
                                         rel_err_Q=float(abs(float(fit["Q"]) - Q_ex) / Q_ex),
                                         lam_err_nm=float(abs(float(fit["lambda_r"]) - lam_r)),
                                         fit_rel_resid=float(fit["rel_resid"]), well_posed=bool(fit["well_posed"]),
                                         peak_r=pole["peak_r"], energy_resid=pk["energy_resid"]))
            print(f"  {name} h={h:.0f}: {len(pk['poles'])} poles "
                  f"{['%.0f nm/Q%.0f' % (p['lambda_nm'], p['Q']) for p in pk['poles'] if 1200 <= p['lambda_nm'] <= 1400]} "
                  f"({time.time()-t0:.0f} s)", flush=True)
    R = rows
    def summarise(sel, label):
        e = np.array([r["rel_err_Q"] for r in R if sel(r)])
        l = np.array([r["lam_err_nm"] for r in R if sel(r)])
        if not len(e):
            return dict(label=label, n=0)
        return dict(label=label, n=int(len(e)), median_rel_err_Q=float(np.median(e)), p90_rel_err_Q=float(np.percentile(e, 90)),
                    max_rel_err_Q=float(e.max()), median_lam_err_nm=float(np.median(l)), p90_lam_err_nm=float(np.percentile(l, 90)))
    summ = [summarise(lambda r: True, "all"),
            summarise(lambda r: r["span_ratio"] == 1.0 and r["centre_offset_kappa"] == 0.0, "matched span, centred (ideal)"),
            summarise(lambda r: 0.5 <= r["span_ratio"] <= 2.0 and r["centre_offset_kappa"] <= 0.25, "span within 2x, offset <= kappa/4 (working regime)"),
            summarise(lambda r: r["span_ratio"] >= 4.0, "probe span 4x too narrow"),
            summarise(lambda r: r["span_ratio"] <= 0.25, "probe span 4x too wide"),
            summarise(lambda r: r["centre_offset_kappa"] >= 0.5, "centre offset kappa/2")]
    for s in [summarise(lambda r, q=q: q[0] <= r["Q_exact"] < q[1], f"Q_exact in [{q[0]},{q[1]})") for q in ((3, 30), (30, 80), (80, 150), (150, 400), (400, 3000))]:
        summ.append(s)
    corr = np.corrcoef(np.log([r["Q_exact"] for r in R]), np.log([max(r["Q_proxy"], 1e-6) for r in R]))[0, 1] if len(R) > 2 else float("nan")
    ok = [r for r in R if 0.5 <= r["span_ratio"] <= 2.0 and r["centre_offset_kappa"] <= 0.25]
    corr_ok = np.corrcoef(np.log([r["Q_exact"] for r in ok]), np.log([max(r["Q_proxy"], 1e-6) for r in ok]))[0, 1] if len(ok) > 2 else float("nan")
    out = dict(order=a.order, P=a.P, heights=a.heights, n_z=a.n_z, lam_E=lam_E, n_points=len(R),
               log_correlation_all=float(corr), log_correlation_working_regime=float(corr_ok),
               summary=summ, rows=R, wall_s=time.time() - t0,
               valid_range="probe span within a factor 2 of the true linewidth and resonance within kappa/4 of the "
                           "probe centre; the spectral-alignment loss L_lambda keeps the optimizer inside that regime")
    json.dump(out, open(OUT / "qproxy_calibration.json", "w"), indent=1, default=float)
    print("\n" + "=" * 100)
    for s in summ:
        if s.get("n"):
            print(f"  {s['label']:<45} n={s['n']:4d}  median|dQ|/Q={s['median_rel_err_Q']:.3f}  p90={s['p90_rel_err_Q']:.3f}  "
                  f"median|dlam|={s['median_lam_err_nm']:.3f} nm")
    print(f"  log-log correlation Q_proxy vs Q_exact: all {corr:.4f}, working regime {corr_ok:.4f}")
    print(f"[qproxy] {len(R)} points in {time.time()-t0:.0f} s -> {OUT/'qproxy_calibration.json'}")


if __name__ == "__main__":
    main()
