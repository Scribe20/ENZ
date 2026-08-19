"""Branch-wide analysis of the bare air/ITO/glass ENZ mode for metasurface
target selection (real-K / complex-omega QNM formulation).

What this adds (and what it deliberately does not redo)
-------------------------------------------------------
The existing package already provides, and this module REUSES unchanged:
  * ITOMaterial (CSV loading, passivity, material crossing),
  * D_fresnel / D_vassant (audited TM pole equations),
  * kz_branch (explicit sheet selection),
  * ModeField (analytic field reconstruction + BC residuals + localization,
    generalized to complex k0 for QNM evaluation),
  * the Drude fit used by run_all.py's complex-omega cross-check.
run_all.py's complex_omega_branch() computed a QNM branch but persisted only
its stationary-point wavelength; here the FULL branch is solved on an
absolute-K grid, field metrics are evaluated at every point, and everything
is persisted (outputs/enz_qnm_branch.npz, outputs/enz_branch_metrics.csv).

Conventions
-----------
* exp(-i*omega*t); passive medium Im(eps) > 0; QNM poles in the lower
  half-plane, omega_tilde = omega_r - i*gamma with gamma > 0;
  Q = omega_r / (2*gamma).
* K is the REAL in-plane propagation constant of the bare film (isotropic
  in-plane: only |K| matters; (K,0) chosen w.l.o.g.).  It is NOT the Bloch
  wavevector of a metasurface; for a normally incident periodic structure
  the accessible harmonics are K_mn = |G_mn|.
* Cladding decay lengths: fields go like exp(+i*kz1*(z-d)) in air and
  exp(-i*kz3*z) in glass, so the amplitude 1/e length is 1/Im(kz) when
  Im(kz) > 0; Im(kz) <= 0 marks a leaky (radiating/growing) channel and the
  length is reported as NaN.
* |Ez|^2 integrals are INTENSITY measures, not electromagnetic energy
  (lossy dispersive medium): column name Ez_intensity_localization_fraction.
* All QNM quantities depend on the Drude analytical continuation of the
  tabulated permittivity; both eps(omega_tilde) (continued, used for the
  eigenfield) and eps(Re omega_tilde) (measured-axis CSV value) are stored.

Run:  python analyze_enz_branch_for_metasurface.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, root

import config
from ito_material import ITOMaterial
from tm_slab_mode import D_fresnel, ModeField

C_NM_FS = 299.792458
EPS1 = config.EPS_AIR
EPS3 = config.N_GLASS ** 2

OUT_DIR = config._HERE / "outputs"

# absolute-K grid (nm^-1): spans Lambda_10 from ~2510 nm down to ~180 nm,
# covering the leaky (glass-cone) segment, the Karimi first-shell region,
# the previous 770nm/(3,0) target, and the overdamped high-K continuation.
K_GRID = np.arange(0.0025, 0.03501, 0.000125)

# validity thresholds
RES_OK = 1e-9
BC_OK = 1e-8
CONTINUITY_NM = 8.0     # flag |d lambda| jumps larger than this between pts


# ---------------------------------------------------------------------------
def drude_fit_report(ito):
    """Fit + the validation demanded in task section 23."""
    w = 2 * np.pi * C_NM_FS / ito.wl
    eps_dat = ito.eps(ito.wl)

    def drude(p, om):
        return p[0] - p[1] ** 2 / (om ** 2 + 1j * p[2] * om)

    p = least_squares(lambda q: np.concatenate(
        [(drude(q, w) - eps_dat).real, (drude(q, w) - eps_dat).imag]),
        [4.0, 2.7, 0.2]).x
    err = drude(p, w) - eps_dat
    err_max = float(np.max(np.abs(err)))
    err_rms = float(np.sqrt(np.mean(np.abs(err) ** 2)))
    # material crossing reproduced by the fit: Re eps = 0 at
    # omega^2 = wp^2/eps_inf - gamma^2
    w_ze = np.sqrt(p[1] ** 2 / p[0] - p[2] ** 2)
    lam_ze_fit = 2 * np.pi * C_NM_FS / w_ze
    lam_ze_csv = ito.zero_crossing_nm()
    print(f"[drude] eps_inf = {p[0]:.5f}, wp = {p[1]:.6f} rad/fs, "
          f"gamma = {p[2]:.6f} rad/fs")
    print(f"[drude] fit error over 1200-1700 nm: max {err_max:.2e}, "
          f"rms {err_rms:.2e}")
    print(f"[drude] material crossing: fit {lam_ze_fit:.2f} nm vs CSV "
          f"{lam_ze_csv:.2f} nm (diff {abs(lam_ze_fit-lam_ze_csv)*1e3:.0f} pm)")
    print("[drude] NOTE: every complex-omega quantity below depends on this "
          "analytical continuation.")
    return p, drude, dict(err_max=err_max, err_rms=err_rms,
                          lam_ze_fit=lam_ze_fit, lam_ze_csv=lam_ze_csv)


# ---------------------------------------------------------------------------
def solve_qnm(K, w_seed, drude, p, d_nm):
    """Complex-omega pole at fixed real K; returns (omega, |D|) or None."""
    def F(x):
        Dv = D_fresnel(K, (x[0] + 1j * x[1]) / C_NM_FS,
                       drude(p, x[0] + 1j * x[1]), d_nm, EPS1, EPS3)
        return [Dv.real, Dv.imag]

    sol = root(F, [w_seed.real, w_seed.imag], method="hybr",
               options={"xtol": 1e-13, "maxfev": 300})
    w = sol.x[0] + 1j * sol.x[1]
    res = abs(D_fresnel(K, w / C_NM_FS, drude(p, w), d_nm, EPS1, EPS3))
    if res > RES_OK or w.imag >= 0:      # QNM must decay in time
        return None, res
    return w, res


def point_metrics(K, w, drude, p, ito, d_nm):
    """All branch-point diagnostics from the QNM eigenfield."""
    eps_cont = complex(drude(p, w))               # continued (used for field)
    lam_r = 2 * np.pi * C_NM_FS / w.real
    in_csv_range = ito.wl[0] <= lam_r <= ito.wl[-1]
    eps_meas = complex(ito.eps(lam_r)) if in_csv_range else complex(np.nan, np.nan)

    mode = ModeField(K, w / C_NM_FS, eps_cont, d_nm, EPS1, EPS3)
    kz1, kz2, kz3 = mode.kz
    bc = mode.interface_residuals()
    bc_max = max(bc.values())

    z = np.linspace(0.0, d_nm, 401)
    Ez = mode.Ez(z)
    Ex = mode.Ex(z)
    aEz, aEx = np.abs(Ez), np.abs(Ex)

    # Metric A: longitudinal character
    R_rms = float(np.sqrt(np.trapezoid(aEz ** 2, z)
                          / max(np.trapezoid(aEx ** 2, z), 1e-300)))
    R_max = float(aEz.max() / max(aEx.max(), 1e-300))
    # Metric B: flatness (guarded)
    cv = float(aEz.std() / max(aEz.mean(), 1e-300))
    mn = aEz.min()
    flat_ratio = float(aEz.max() / mn) if mn > 1e-12 * aEz.max() else np.inf
    # Metric D: decay lengths (amplitude 1/e; NaN when channel leaks)
    L_air = 1.0 / kz1.imag if kz1.imag > 0 else np.nan
    L_glass = 1.0 / kz3.imag if kz3.imag > 0 else np.nan
    # Metric C: localization (only meaningful when both claddings decay)
    loc = mode.ez_localization(n_pts=2001) if (kz1.imag > 0 and kz3.imag > 0) \
        else np.nan

    k0r = w.real / C_NM_FS
    bound_air = K > k0r          # real-frequency light-line relation
    bound_glass = K > config.N_GLASS * k0r
    if kz1.imag > 0 and kz3.imag > 0:
        char = "bound"
    elif kz1.imag <= 0 and kz3.imag <= 0:
        char = "leaky_both"
    elif kz3.imag <= 0:
        char = "leaky_glass"
    else:
        char = "leaky_air"

    return dict(
        K_per_nm=K, K_over_k0=K / k0r, lambda_qnm_nm=lam_r,
        omega_real=w.real, omega_imag=w.imag,
        Q=abs(w.real / (2 * w.imag)),
        linewidth_THz=2 * abs(w.imag) / (2 * np.pi) * 1e3,
        eps_ito_cont_real=eps_cont.real, eps_ito_cont_imag=eps_cont.imag,
        eps_ito_meas_real=eps_meas.real, eps_ito_meas_imag=eps_meas.imag,
        Ez_to_Ex_rms=R_rms, Ez_to_Ex_max=R_max,
        Ez_flatness_cv=cv, Ez_flatness_max_min=flat_ratio,
        Ez_intensity_localization_fraction=loc,
        air_decay_length_nm=L_air, glass_decay_length_nm=L_glass,
        max_boundary_condition_residual=bc_max,
        outside_air_lightline=bound_air, outside_glass_lightline=bound_glass,
        bound_or_leaky=char,
        Lambda_10_nm=2 * np.pi / K,
        Lambda_11_nm=2 * np.pi * np.sqrt(2) / K,
        Lambda_20_nm=4 * np.pi / K,
    )


# ---------------------------------------------------------------------------
def trace_branch(ito, p, drude, d_nm=config.D_ITO_NM, k_grid=K_GRID):
    """Continuation over the absolute-K grid, seeded in the Karimi region."""
    i0 = int(np.argmin(np.abs(k_grid - 2 * np.pi / 850.0)))
    w_seed0 = 2 * np.pi * C_NM_FS / 1460.0 * (1 - 0.09j)

    results = {}
    for direction in (+1, -1):
        w_prev = w_seed0
        idx = range(i0, len(k_grid)) if direction > 0 else range(i0 - 1, -1, -1)
        for i in idx:
            w, res = solve_qnm(k_grid[i], w_prev, drude, p, d_nm)
            if w is None or abs(w - w_prev) > 0.25 * abs(w_prev):
                print(f"[branch] stopped ({'up' if direction>0 else 'down'}) "
                      f"at K = {k_grid[i]:.5f} nm^-1 "
                      f"(|D| = {res:.1e})")
                break
            results[i] = (w, res)
            w_prev = w

    rows = []
    prev_lam = None
    for i in sorted(results):
        w, res = results[i]
        m = point_metrics(k_grid[i], w, drude, p, ito, d_nm)
        m["pole_residual"] = res
        m["physical_branch"] = (res < RES_OK and
                                m["max_boundary_condition_residual"] < BC_OK)
        jump = (prev_lam is not None
                and abs(m["lambda_qnm_nm"] - prev_lam) > CONTINUITY_NM)
        m["continuity_ok"] = not jump
        if jump:
            print(f"[branch] continuity flag at K = {k_grid[i]:.5f}: "
                  f"d lambda = {m['lambda_qnm_nm']-prev_lam:+.1f} nm")
        prev_lam = m["lambda_qnm_nm"]
        rows.append(m)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(exist_ok=True)
    ito = ITOMaterial()
    p, drude, fit_rep = drude_fit_report(ito)

    df = trace_branch(ito, p, drude)
    n_bound = int((df.bound_or_leaky == "bound").sum())
    print(f"[branch] {len(df)} QNM points; {n_bound} bound, "
          f"{len(df)-n_bound} leaky; max pole residual "
          f"{df.pole_residual.max():.1e}; max BC residual "
          f"{df.max_boundary_condition_residual.max():.1e}")

    csv_path = OUT_DIR / "enz_branch_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")

    np.savez(
        OUT_DIR / "enz_qnm_branch.npz",
        K=df.K_per_nm.to_numpy(),
        K_over_k0=df.K_over_k0.to_numpy(),
        omega_complex=(df.omega_real + 1j * df.omega_imag).to_numpy(),
        lambda_real_nm=df.lambda_qnm_nm.to_numpy(),
        Q=df.Q.to_numpy(),
        eps_ito_continued=(df.eps_ito_cont_real
                           + 1j * df.eps_ito_cont_imag).to_numpy(),
        eps_ito_measured_axis=(df.eps_ito_meas_real
                               + 1j * df.eps_ito_meas_imag).to_numpy(),
        pole_residual=df.pole_residual.to_numpy(),
        bound=(df.bound_or_leaky == "bound").to_numpy(),
        drude_eps_inf_wp_gamma=np.array(p),
        drude_fit_err_max=fit_rep["err_max"],
        drude_fit_err_rms=fit_rep["err_rms"],
        material_crossing_csv_nm=fit_rep["lam_ze_csv"],
        material_crossing_fit_nm=fit_rep["lam_ze_fit"],
        ito_thickness_nm=config.D_ITO_NM,
        glass_index=config.N_GLASS,
        time_convention="exp(-i*omega*t); omega_tilde = omega_r - i*gamma",
        note=("real-K/complex-omega QNM branch of bare air/ITO/glass; "
              "fields and metrics from eps(omega_tilde) (Drude continuation); "
              "|Ez|^2 fractions are intensity, not energy"),
    )
    print(f"[saved] {OUT_DIR/'enz_qnm_branch.npz'}")
    return df, p, drude, ito


if __name__ == "__main__":
    main()
