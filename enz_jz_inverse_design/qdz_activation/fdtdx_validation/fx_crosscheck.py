"""Shift-aware TORCWA <-> FDTDX cross-check of one candidate (post hoc, from the saved spectra):
  1. best spectral shift  d  minimizing rms[T_fdtdx(lam) - T_torcwa(lam + d)] over the single-order window, and the
     residual rms / max after the shift (resonance-position discrepancy vs line-shape discrepancy);
  2. material-fit isolation: TORCWA re-run with the FDTDX ADE material models (fx_sim.eps_model) instead of the
     tabulated data over a window around the line -> how much of d the material fits explain;
  3. field / loss comparison at SHIFT-EQUIVALENT wavelengths: TORCWA at lam + d vs FDTDX at lam for the recorded
     field wavelengths (same detuning from the respective resonance), plus the fixed-wavelength values.
Writes <tag>/spectra/crosscheck_<tag>.json.
    python fx_crosscheck.py --design <tag>   (system python with torch; the FDTDX venv is not needed)
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as cm                       # noqa: E402
import config                             # noqa: E402
import forward as fwd                     # noqa: E402
import candidates as cd                   # noqa: E402
import materials as mat                   # noqa: E402
sys.path.insert(0, str(HERE.parent.parent / "fdtdx_four_finalists"))
import importlib.util                     # noqa: E402
import torch                              # noqa: E402

# fx_sim imports jax; load only its material-model function without importing the module
_MODELS = json.load(open(HERE.parent.parent / "fdtdx_four_finalists" / "materials" / "material_models.json"))
C0 = 299792458.0


def eps_model(name, lam_nm):
    m = _MODELS[name]; w = 2 * np.pi * C0 / (lam_nm * 1e-9)
    if name == "ITO":
        return m["eps_inf"] - m["wp_rad_s"] ** 2 / (w ** 2 + 1j * m["gamma_d_rad_s"] * w) + m["delta_eps_L"] * m["w0_L_rad_s"] ** 2 / (m["w0_L_rad_s"] ** 2 - w ** 2 - 1j * m["gamma_L_rad_s"] * w)
    return m["eps_inf"] + m["delta_eps"] * m["w0_rad_s"] ** 2 / (m["w0_rad_s"] ** 2 - w ** 2 - 1j * max(m["gamma_rad_s"], 1e9) * w)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--design", required=True); ap.add_argument("--order", type=int, nargs=2, default=[9, 9])
    ap.add_argument("--threads", type=int, default=2); a = ap.parse_args(); fwd.set_threads(a.threads)
    tag = a.design; c = cd.by_tag(tag); rho = cd.get_rho(c); rho_t = None if rho is None else torch.as_tensor(rho, dtype=config.GEO_DTYPE)
    z = np.load(HERE / tag / "spectra" / f"spectra_{tag}.npz"); lam_f, T_f, R_f, A_f = z["lam_nm"], z["T"], z["R"], z["A"]
    lam_t, T_t, R_t, A_t = z["torcwa_lam"], z["torcwa_T"], z["torcwa_R"], z["torcwa_A"]
    chk = json.load(open(HERE / tag / "spectra" / f"checks_{tag}.json"))
    lo = max(1252.0, lam_t.min() + 15.0); hi = min(1385.0, lam_t.max() - 15.0)
    m = (lam_f >= lo) & (lam_f <= hi)
    shifts = np.arange(-30.0, 30.01, 0.25)
    rms = np.array([np.sqrt(np.mean((T_f[m] - np.interp(lam_f[m] + d, lam_t, T_t)) ** 2)) for d in shifts])
    d_best = float(shifts[int(np.argmin(rms))]); rms0 = float(rms[np.argmin(np.abs(shifts))]); rms_best = float(rms.min())
    at_bound = bool(abs(abs(d_best) - 30.0) < 1e-9)
    Ts = np.interp(lam_f[m] + d_best, lam_t, T_t); Rs = np.interp(lam_f[m] + d_best, lam_t, R_t); As = np.interp(lam_f[m] + d_best, lam_t, A_t)
    res = dict(tag=tag, window_nm=[lo, hi], shift_scan_nm=[-30.0, 30.0, 0.25], best_shift_nm=d_best,
               meaning="TORCWA(lam + d) matches FDTDX(lam): d > 0 means the FDTDX line is BLUE-shifted relative to TORCWA",
               shift_at_scan_bound=at_bound, rms_dT_no_shift=rms0, rms_dT_after_shift=rms_best, max_dT_after_shift=float(np.abs(T_f[m] - Ts).max()),
               max_dR_after_shift=float(np.abs(R_f[m] - Rs).max()), max_dA_after_shift=float(np.abs(A_f[m] - As).max()),
               lam_T_min=dict(fdtdx=float(lam_f[m][np.argmin(T_f[m])]), torcwa=float(lam_t[np.argmin(T_t)])),
               T_min=dict(fdtdx=float(T_f[m].min()), torcwa=float(T_t.min())), A_max=dict(fdtdx=float(A_f[m].max()), torcwa=float(A_t.max())))
    # ---- material-fit isolation: TORCWA with the FDTDX ADE models around the line ------------------------
    lam_c = res["lam_T_min"]["torcwa"]; grid = np.arange(lam_c - 20.0, lam_c + 20.01, 1.0)
    Tm = []
    with torch.no_grad():
        for l in grid:
            sim = fwd.build_sim(rho_t, c["P"], c["h"], float(l), list(a.order), eps_asi=complex(eps_model("aSiH", l)).real,
                                eps_ito=complex(eps_model("ITO", l)), n_glass=float(np.sqrt(complex(eps_model("glass", l)).real)))
            _, T = fwd.rt_all_orders(sim); Tm.append(float(T))
    Tm = np.array(Tm); Tt = np.interp(grid, lam_t, T_t)
    res["material_fit_isolation"] = dict(grid_nm=[float(grid[0]), float(grid[-1]), 1.0], lam_T_min_torcwa_tabulated=float(grid[np.argmin(Tt)]),
                                         lam_T_min_torcwa_fdtdx_fits=float(grid[np.argmin(Tm)]), shift_from_material_fits_nm=float(grid[np.argmin(Tm)] - grid[np.argmin(Tt)]),
                                         max_dT_tabulated_vs_fits=float(np.abs(Tm - Tt).max()),
                                         eps_at_line=dict(ITO_tab=[complex(mat.eps_ito(lam_c)).real, complex(mat.eps_ito(lam_c)).imag], ITO_fit=[complex(eps_model("ITO", lam_c)).real, complex(eps_model("ITO", lam_c)).imag],
                                                          aSiH_tab=float(np.real(mat.eps_asi(lam_c))), aSiH_fit=float(complex(eps_model("aSiH", lam_c)).real),
                                                          n_glass_tab=float(mat.n_glass(lam_c)), n_glass_fit=float(np.sqrt(complex(eps_model("glass", lam_c)).real))),
                                         note="remaining shift = discretization (64 x 64 in-plane sub-pixel raster, 12.9-nm a-Si:H z-cells, 4.6-nm ITO cells, CFL 0.95) and DFT/finite-time effects")
    # ---- field / loss comparison at shift-equivalent wavelengths --------------------------------------------
    fe = chk["ito_field_enhancement"]; comp = {}
    with torch.no_grad():
        for k, v in fe.items():
            l = float(k); ls_ = float(np.clip(l + d_best, 1160.0, 1399.5))      # stay inside the supplied a-Si:H data
            d0 = fwd.to_floats(fwd.evaluate(rho_t, c["P"], c["h"], l, list(a.order)))
            d1 = fwd.to_floats(fwd.evaluate(rho_t, c["P"], c["h"], ls_, list(a.order)))
            comp[k] = dict(shift_equivalent_lambda_clipped=bool(abs(ls_ - (l + d_best)) > 1e-9), fdtdx=dict(mean_Ez2=v["mean_Ez2_ito_fdtdx"], max_Ez2=v["max_Ez2_ito_fdtdx"], Fz=v["Fz_fdtdx"], Ftot=v["Ftot_fdtdx"], A_flux=v["A_fdtdx_flux"]),
                           torcwa_same_lambda=dict(mean_Ez2=d0["mean_Ez2"], Fz=d0["Fz"], Ftot=d0["Ftot"], A=d0["A"], T=d0["T"]),
                           torcwa_shift_equivalent=dict(lam=ls_, mean_Ez2=d1["mean_Ez2"], Fz=d1["Fz"], Ftot=d1["Ftot"], A=d1["A"], T=d1["T"]),
                           ratio_mean_Ez2_fdtdx_over_torcwa_shifted=v["mean_Ez2_ito_fdtdx"] / max(d1["mean_Ez2"], 1e-12),
                           ratio_Fz_fdtdx_over_torcwa_shifted=v["Fz_fdtdx"] / max(d1["Fz"], 1e-12))
    res["fields_vs_torcwa"] = comp; res["provenance"] = cm.provenance(order=a.order)
    cm.jdump(res, HERE / tag / "spectra" / f"crosscheck_{tag}.json")
    print(json.dumps({k: v for k, v in res.items() if k not in ("fields_vs_torcwa", "provenance")}, indent=1))
    for k, v in comp.items():
        print(k, "FDTDX", {kk: round(x, 3) for kk, x in v["fdtdx"].items()}, "| TORCWA shifted", {kk: (round(x, 3) if isinstance(x, float) else x) for kk, x in v["torcwa_shift_equivalent"].items()})


if __name__ == "__main__":
    main()
