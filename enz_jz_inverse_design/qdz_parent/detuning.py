"""STRONG-COUPLING CERTIFICATION - controlled height detuning of one frozen topology.

For each height in a sweep, BOTH systems are solved with the identical in-plane topology:
    A. the PARENT   air / a-Si:H(h) / glass                      -> bare photonic branch (lossless: Q = Q_r)
    B. the LOADED   air / a-Si:H(h) / 23-nm measured ITO / glass  -> hybrid branches
Exact poles are extracted for both with analysis.significant_poles (Rayleigh anomalies excluded), and
the loaded branches are additionally tracked through the near-field observables (F_z, <|Ez|^2>, A).

The two-mode non-Hermitian fit
        H = [[w_Si(h) - i k_Si/2, g], [g, w_ENZ - i k_ENZ/2]]
is performed ONLY if the data support it (at least two loaded branches present over a contiguous part
of the sweep on both sides of the bare crossing).  Otherwise the negative result is reported and no
coupling constant is invented.  Fit residuals and parameter uncertainties are always saved.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import config, forward as fwd, materials as mat, analysis as an     # noqa: E402
import parent_fwd as pf, poles_parent as pp, hybrid_certify as hc   # noqa: E402

GEO = config.GEO_DTYPE
C_NM_FS = 299.792458


def w_of(lam):
    return 2 * np.pi * C_NM_FS / np.asarray(lam, float)


def two_mode_branches(h, p, w_ENZ_ref=None):
    """Eigenvalues of the 2x2 non-Hermitian Hamiltonian for a linear bare-mode dispersion in h."""
    w0, dwdh, k_Si, w_E_, k_E_, g = p
    wS = w0 + dwdh * (np.asarray(h, float) - np.mean(h))
    A = wS - 0.5j * k_Si
    B = w_E_ - 0.5j * k_E_
    m = 0.5 * (A + B); d = 0.5 * (A - B)
    r = np.sqrt(d ** 2 + g ** 2)
    return m - r, m + r


def fit_two_mode(h_arr, br_lo, br_hi, kap_lo, kap_hi):
    """Least squares of both complex branches; returns params, residual and jacobian-based sigmas."""
    obs = np.concatenate([br_lo - 0.5j * kap_lo, br_hi - 0.5j * kap_hi])
    def resid(p):
        lo, hi = two_mode_branches(h_arr, p)
        e = np.concatenate([lo - (br_lo - 0.5j * kap_lo), hi - (br_hi - 0.5j * kap_hi)])
        return np.concatenate([e.real, e.imag])
    w_mid = float(np.mean(np.concatenate([br_lo, br_hi])))
    p0 = [w_mid, (br_hi[-1] - br_hi[0]) / (h_arr[-1] - h_arr[0]) if len(h_arr) > 1 else 0.0,
          float(np.mean(kap_hi)), w_mid, float(np.mean(kap_lo)), 0.02 * w_mid]
    best = None
    for gs in (0.002, 0.01, 0.05, 0.2):
        try:
            r = least_squares(resid, [p0[0], p0[1], p0[2], p0[3], p0[4], gs * w_mid], max_nfev=20000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None
    p = best.x
    J = best.jac; dof = max(len(best.fun) - len(p), 1)
    s2 = 2 * best.cost / dof
    try:
        cov = s2 * np.linalg.inv(J.T @ J); sig = np.sqrt(np.abs(np.diag(cov)))
    except Exception:
        sig = np.full(len(p), np.nan)
    g = abs(p[5]); k_Si = abs(p[2]); k_E = abs(p[4])
    return dict(omega0=p[0], domega_dh=p[1], kappa_Si=k_Si, omega_ENZ=p[3], kappa_ENZ=k_E, g=g,
                sigma=[float(x) for x in sig], rms_resid=float(np.sqrt(np.mean(best.fun ** 2))),
                max_resid=float(np.max(np.abs(best.fun))), n_points=int(len(h_arr)), dof=int(dof),
                C=float(4 * g ** 2 / max(k_Si * k_E, 1e-30)),
                splitting_2g=float(2 * g), mean_damping=float(0.5 * (k_Si + k_E)),
                g_over_mean_damping=float(g / max(0.5 * (k_Si + k_E), 1e-30)),
                strong_coupling_inequality_4g_gt_kSi_kENZ=bool(4 * g ** 2 > k_Si * k_E),
                note="omega in rad/fs; kappa are FULL widths (gamma_total = kappa); C = 4 g^2 / (kappa_Si kappa_ENZ)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--P", type=float, default=825.0)
    ap.add_argument("--heights", type=float, nargs="*", default=None)
    ap.add_argument("--h-center", type=float, default=None); ap.add_argument("--h-span", type=float, default=120.0)
    ap.add_argument("--n-h", type=int, default=13)
    ap.add_argument("--order", type=int, nargs=2, default=[5, 5])
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    fwd.set_threads(a.threads)
    rho = torch.as_tensor(np.load(a.rho), dtype=GEO)
    lam_E = pf.lambda_E(); ng = float(mat.n_glass(1300.0)); w_E = w_of(lam_E)
    heights = np.array(a.heights) if a.heights else np.linspace(a.h_center - a.h_span, a.h_center + a.h_span, a.n_h)
    out = Path(a.out) if a.out else HERE / "detuning" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    lams = np.arange(1180.0, 1398.1, a.step)
    ray = [x["lam"] for x in an.rayleigh_wavelengths(a.P, ng, lo=1180.0, hi=1398.0)]
    rows, t0 = [], time.time()
    for h in heights:
        pk = pp.parent_poles(rho, a.P, float(h), a.order, 1180.0, 1398.0, a.step, n_glass=ng)
        lo = hc.loaded_spectrum(rho, a.P, float(h), lams, a.order, n_z=config.Z_SAMPLES_ITO)
        r_c = np.array([r["r"] for r in lo]); t_c = np.array([r["t"] for r in lo])
        lp = an.significant_poles(lams, r_c, t_c, exclude=ray)
        iE = int(np.argmin(np.abs(lams - lam_E)))
        rows.append(dict(h=float(h),
                         parent_poles=[dict(lam=p["lambda_nm"], Q=p["Q"], w=p["omega_re"], gam=p["gamma"]) for p in pk["poles"]],
                         loaded_poles=[dict(lam=p["lambda_nm"], Q=p["Q"], w=p["omega_re"], gam=p["gamma"]) for p in lp],
                         Fz_at_lamE=float(lo[iE]["Fz"]), A_at_lamE=float(lo[iE]["A"]),
                         mean_Ez2_at_lamE=float(lo[iE]["mean_Ez2"]),
                         A_max=float(max(r["A"] for r in lo)), lam_A_max=float(lams[int(np.argmax([r["A"] for r in lo]))]),
                         Fz_max=float(max(r["Fz"] for r in lo)), lam_Fz_max=float(lams[int(np.argmax([r["Fz"] for r in lo]))])))
        np.savez_compressed(out / f"loaded_spectrum_h{h:.0f}.npz", lam_nm=lams,
                            **{k: np.array([r[k] for r in lo]) for k in ("R", "T", "A", "Fz", "Ftot", "mean_Ez2")})
        print(f"  h={h:6.1f}: parent {['%.0f/Q%.0f' % (p['lam'], p['Q']) for p in rows[-1]['parent_poles']]} | "
              f"loaded {['%.0f/Q%.0f' % (p['lam'], p['Q']) for p in rows[-1]['loaded_poles']]} | "
              f"A_max={rows[-1]['A_max']:.3f}@{rows[-1]['lam_A_max']:.0f} ({time.time()-t0:.0f} s)", flush=True)
    # ---- branch assembly: nearest loaded poles below / above lambda_E per height
    hs, blo, bhi, klo, khi, hs_ok = [], [], [], [], [], []
    for r in rows:
        below = [p for p in r["loaded_poles"] if p["w"] > w_E]        # higher omega = shorter lambda
        above = [p for p in r["loaded_poles"] if p["w"] <= w_E]
        if below and above:
            b = min(below, key=lambda p: p["w"] - w_E); a_ = max(above, key=lambda p: p["w"])
            hs_ok.append(r["h"]); bhi.append(b["w"]); blo.append(a_["w"])
            khi.append(2 * b["gam"]); klo.append(2 * a_["gam"])
        hs.append(r["h"])
    res = dict(tag=a.tag, P=a.P, heights=[float(x) for x in heights], lam_E=lam_E, order=list(a.order),
               rows=rows, n_heights_with_two_loaded_branches=len(hs_ok), wall_s=time.time() - t0)
    if len(hs_ok) >= 5:
        fit = fit_two_mode(np.array(hs_ok), np.array(blo), np.array(bhi), np.array(klo), np.array(khi))
        res["two_mode_fit"] = fit
        res["two_mode_fit_supported"] = True
        res["branch_data"] = dict(h=hs_ok, omega_lower=blo, omega_upper=bhi, kappa_lower=klo, kappa_upper=khi)
    else:
        res["two_mode_fit"] = None
        res["two_mode_fit_supported"] = False
        res["two_mode_fit_reason"] = (f"only {len(hs_ok)} of {len(rows)} heights show loaded poles on BOTH sides of "
                                      f"omega_E; a stable two-mode fit is not supported by the data and no coupling "
                                      f"constant is reported")
    json.dump(res, open(out / "detuning.json", "w"), indent=1, default=float)
    print(f"[detuning {a.tag}] {len(rows)} heights, two-sided branches at {len(hs_ok)}; "
          f"two-mode fit {'DONE' if res['two_mode_fit_supported'] else 'NOT SUPPORTED'} ({time.time()-t0:.0f} s)")
    if res.get("two_mode_fit"):
        f = res["two_mode_fit"]
        print(f"   g={f['g']:.5f} rad/fs  kappa_Si={f['kappa_Si']:.5f}  kappa_ENZ={f['kappa_ENZ']:.5f}  "
              f"C={f['C']:.3f}  2g/(mean damping)={2*f['g_over_mean_damping']:.3f}  rms resid={f['rms_resid']:.2e}")


if __name__ == "__main__":
    main()
