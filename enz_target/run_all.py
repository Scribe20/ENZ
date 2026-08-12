"""End-to-end driver: material analysis -> pole search -> branch continuation
-> target selection -> validation -> figures -> saved outputs.

Run:  python run_all.py
"""

import numpy as np
from scipy.optimize import least_squares, root

import config
from ito_material import ITOMaterial
from tm_slab_mode import ModeField, D_fresnel
from solve_enz_dispersion import (initial_scan, classify_root, continue_branch,
                                  solve_pole, k0_of, EPS1, EPS3)
import validate_mode as vm
import visualize_enz_mode as viz

C_NM_FS = 299.792458          # speed of light in nm/fs
C_SI = 299792458.0            # m/s


# ---------------------------------------------------------------------------
def drude_fit(ito):
    """Least-squares Drude fit of the tabulated data (used ONLY for the
    complex-omega cross-check, never for the saved real-omega target mode)."""
    w = 2 * np.pi * C_NM_FS / ito.wl              # rad/fs
    eps_dat = ito.eps(ito.wl)

    def drude(p, om):
        return p[0] - p[1] ** 2 / (om ** 2 + 1j * p[2] * om)

    def resid(p):
        d = drude(p, w) - eps_dat
        return np.concatenate([d.real, d.imag])

    p = least_squares(resid, [4.0, 2.7, 0.2]).x
    err = np.max(np.abs(drude(p, w) - eps_dat))
    return p, drude, err


def complex_omega_branch(ito, p, drude, u_min=1.02, u_max=8.0, du=0.02,
                         lam_ref=1450.0):
    """Complex-omega pole at fixed real K (continuation in K).  Returns the
    branch and its stationary (zero-group-velocity) point."""
    k0r = 2 * np.pi / lam_ref

    def D_om(om, K):
        return D_fresnel(K, om / C_NM_FS, drude(p, om), config.D_ITO_NM,
                         EPS1, EPS3, +1, +1)

    us = np.arange(u_min, u_max + du / 2, du)
    w_prev = 2 * np.pi * C_NM_FS / 1460.0 * (1 - 0.12j)
    u_ok, lam_re, w_im, qf = [], [], [], []
    for uK in us:
        K = uK * k0r
        sol = root(lambda x: [D_om(x[0] + 1j * x[1], K).real,
                              D_om(x[0] + 1j * x[1], K).imag],
                   [w_prev.real, w_prev.imag], method="hybr",
                   options={"xtol": 1e-14})
        if not sol.success:
            continue
        wc = sol.x[0] + 1j * sol.x[1]
        if abs(wc - w_prev) > 0.3 * abs(w_prev):
            continue
        w_prev = wc
        u_ok.append(uK)
        lam_re.append(2 * np.pi * C_NM_FS / wc.real)
        w_im.append(wc.imag)
        qf.append(abs(wc.real / (2 * wc.imag)))
    u_ok, lam_re = np.array(u_ok), np.array(lam_re)
    # stationary point of the branch (flat-dispersion center), restricted to
    # the bound region outside the glass light cone
    inner = (u_ok > config.N_GLASS) & (u_ok < u_max - 0.2)
    dl = np.gradient(lam_re, u_ok)
    i_st = np.where(inner)[0][np.argmin(np.abs(dl[inner]))]
    return {"u_K": u_ok, "lam_re": lam_re, "w_im": np.array(w_im),
            "Q": np.array(qf), "u_stat": float(u_ok[i_st]),
            "lam_stat": float(lam_re[i_st])}


# ---------------------------------------------------------------------------
def main(run_thickness_sweep=True):
    print("=" * 72)
    print("STEP 1: ITO material analysis")
    print("=" * 72)
    ito = ITOMaterial()
    print("CSV columns:", ito.columns)
    print(f"using: {ito.re_col} / {ito.im_col} "
          f"({'recommended-physical' if ito.used_recommended else 'as-labeled'})")
    print("as-labeled (possibly swapped) columns present in file:",
          ito.labeled_cols_present)
    print("passivity:", ito.passivity_report())
    lam_ze = ito.zero_crossing_nm()
    eps_at_ze = ito.eps(lam_ze)
    print(f"material ENZ zero crossing (interpolated): {lam_ze:.3f} nm, "
          f"eps = {eps_at_ze:.5f}")

    p_drude, drude, fit_err = drude_fit(ito)
    print(f"Drude fit (for complex-omega cross-check only): eps_inf={p_drude[0]:.4f}, "
          f"wp={p_drude[1]:.5f} rad/fs, gamma={p_drude[2]:.5f} rad/fs, "
          f"max fit error {fit_err:.1e}")

    print()
    print("=" * 72)
    print("STEP 2: initial multi-seed pole scan (lambda = 1450 nm)")
    print("=" * 72)
    wl_scan = 1450.0
    eps2 = complex(ito.eps(wl_scan))
    found = {}
    for s1, s3, tag in ((+1, +1, "proper air / proper glass"),
                        (-1, +1, "improper air / proper glass")):
        roots = initial_scan(wl_scan, eps2, config.D_ITO_NM, s1, s3)
        found[(s1, s3)] = roots
        for u, res in roots:
            c = classify_root(wl_scan, u, s1, s3)
            print(f"  [{tag}] K/k0 = {u.real:+.4f}{u.imag:+.4f}j  |D| = {res:.1e}")
            print(f"      outside air cone: {c['outside_air_cone']}, "
                  f"outside glass cone: {c['outside_glass_cone']}, "
                  f"air decay: {c['decays_air']}, glass decay: {c['decays_glass']}")

    # identify seeds: confined branch on proper/proper, Berreman on improper-air
    u_enz_seed = min((u for u, _ in found[(+1, +1)] if abs(u) < 15),
                     key=lambda u: abs(u))
    u_ber_seed = min((u for u, _ in found[(-1, +1)] if abs(u) < 5),
                     key=lambda u: abs(u))

    print()
    print("=" * 72)
    print("STEP 3: branch continuation (real omega, complex K)")
    print("=" * 72)
    wl_e, us_e, res_e = continue_branch(ito, wl_scan, u_enz_seed,
                                        config.D_ITO_NM, sheet1=+1, sheet3=+1)
    print(f"confined-ENZ branch: {wl_e[0]:.0f}-{wl_e[-1]:.0f} nm "
          f"({len(wl_e)} pts), max |D| = {res_e.max():.1e}")
    wl_b, us_b, res_b = continue_branch(ito, wl_scan, u_ber_seed,
                                        config.D_ITO_NM, sheet1=-1, sheet3=+1)
    print(f"Berreman leaky branch: {wl_b[0]:.0f}-{wl_b[-1]:.0f} nm "
          f"({len(wl_b)} pts), max |D| = {res_b.max():.1e}")

    # per-wavelength diagnostics on the ENZ branch
    loc = np.full(len(wl_e), np.nan)
    bound = np.zeros(len(wl_e), bool)
    for i, (w, u) in enumerate(zip(wl_e, us_e)):
        k0 = k0_of(w)
        f = ModeField(u * k0, k0, complex(ito.eps(w)), config.D_ITO_NM, EPS1, EPS3)
        kz1, _, kz3 = f.kz
        bound[i] = (kz1.imag > 0) and (kz3.imag > 0)
        if bound[i]:
            l = f.ez_localization(n_pts=4001)
            loc[i] = l if l is not None else np.nan
    wl_bound_min = wl_e[bound].min()
    print(f"bound (both claddings decaying) for lambda >= {wl_bound_min:.0f} nm "
          f"(material crossing: {lam_ze:.1f} nm); "
          "below: virtual/antibound continuation (fields grow into claddings)")

    print()
    print("=" * 72)
    print("STEP 4: complex-omega cross-check (Drude fit, real K)")
    print("=" * 72)
    cw = complex_omega_branch(ito, p_drude, drude)
    print(f"flat ENZ-mode branch: Re(lambda_pole) spans "
          f"{cw['lam_re'].min():.0f}-{cw['lam_re'].max():.0f} nm for "
          f"K/k0 in [{cw['u_K'][0]:.2f}, {cw['u_K'][-1]:.2f}], Q ~ "
          f"{np.median(cw['Q']):.1f}")
    print(f"stationary (zero-group-velocity) point: K/k0 = {cw['u_stat']:.2f}, "
          f"lambda = {cw['lam_stat']:.1f} nm  -> modal ENZ central wavelength")
    print("(Karimi et al. report ~1460 nm for their sample with crossing at "
          "1410 nm - same formulation, their permittivity dataset)")

    print()
    print("=" * 72)
    print("STEP 5: target-point selection")
    print("=" * 72)
    i_t = int(np.nanargmax(loc))
    wl_E, u_E = float(wl_e[i_t]), complex(us_e[i_t])
    print("criterion: maximum Ez-localization fraction in the ITO film on the")
    print("bound (proper-sheet) branch - the most strongly confined point.")
    print(f"TARGET: lambda_E = {wl_E:.1f} nm, K/k0 = {u_E:.4f}, "
          f"localization = {loc[i_t]:.4f}")

    print()
    print("=" * 72)
    print("STEP 6: validation")
    print("=" * 72)
    rep = vm.validate_target(wl_E, u_E, ito)
    print()
    vm.driven_consistency(ito)
    print()
    vm.berreman_driven_check(ito, wl_b, us_b, theta_deg=50.0)

    print()
    print("=" * 72)
    print("STEP 7: field reconstruction, normalization, saving")
    print("=" * 72)
    k0 = k0_of(wl_E)
    eps2_E = complex(ito.eps(wl_E))
    mode = ModeField(u_E * k0, k0, eps2_E, config.D_ITO_NM, EPS1, EPS3)
    La, Lg = mode.decay_lengths_nm()

    # normalization: integral over ITO of |Ez|^2 dz = 1  (per unit area; z in nm)
    z_ito = np.linspace(0.0, config.D_ITO_NM, config.N_Z_ITO)
    I_raw = np.trapezoid(np.abs(mode.Ez(z_ito)) ** 2, z_ito)
    s = 1.0 / np.sqrt(I_raw)
    print(f"normalization scale s = {s:.6e} "
          f"(so that integral_ITO |Ez|^2 dz = 1, z in nm)")

    z_air = config.D_ITO_NM + np.linspace(0, config.Z_PAD_FACTOR * La,
                                          config.N_Z_CLAD + 1)[1:]
    z_glass = -np.linspace(0, config.Z_PAD_FACTOR * Lg, config.N_Z_CLAD + 1)[1:][::-1]
    z_all = np.concatenate([z_glass, z_ito, z_air])
    Hy = mode.Hy(z_all) * s
    Ex = mode.Ex(z_all) * s
    Ez = mode.Ez(z_all) * s

    lam_par = 2 * np.pi / abs(u_E.real) / k0
    K_E = u_E * k0
    np.savez(
        config.OUT_NPZ,
        # target point
        wavelength_nm=wl_E,
        omega_rad_per_s=2 * np.pi * C_SI / (wl_E * 1e-9),
        k0_per_nm=k0,
        K_real_per_nm=K_E.real,
        K_imag_per_nm=K_E.imag,
        K_note=("K with Re K > 0 chosen; D(K)=D(-K), and the +x-amplitude-"
                "decaying representative of the pair is -K (overdamped/backward "
                "in-plane character). Field ansatz: F(z) exp(+i K x) exp(-i w t)."),
        # z-grid and fields (complex)
        z_nm=z_all,
        Ez=Ez, Ex=Ex, Hy=Hy,
        ito_z_range_nm=np.array([0.0, config.D_ITO_NM]),
        # layer kz values (branch: decaying into both claddings)
        kz_air_per_nm=complex(mode.kz[0]),
        kz_ito_per_nm=complex(mode.kz[1]),
        kz_glass_per_nm=complex(mode.kz[2]),
        # material / geometry
        eps_ito_real=eps2_E.real, eps_ito_imag=eps2_E.imag,
        ito_thickness_nm=config.D_ITO_NM,
        glass_index=config.N_GLASS,
        glass_index_is_paper_value=config.N_GLASS_IS_PAPER_VALUE,
        eps_air=EPS1,
        # conventions and normalization
        time_convention="exp(-i*omega*t)",
        normalization="integral_ITO_abs_Ez_squared_dz_equals_1_z_in_nm",
        field_units=("Ez, Ex in nm^-1/2 (after normalization); Hy in the same "
                     "arbitrary mode amplitude with E = (kz/(omega*eps0*eps))*Hy "
                     "relations; only ratios/overlaps are meaningful"),
        # diagnostics
        localization_fraction=float(loc[i_t]),
        pole_residual=float(abs(D_fresnel(K_E, k0, eps2_E, config.D_ITO_NM,
                                          EPS1, EPS3))),
        air_decay_length_nm=La,
        glass_decay_length_nm=Lg,
        inplane_modal_wavelength_nm=lam_par,
        material_zero_crossing_nm=lam_ze,
        modal_central_wavelength_complex_omega_nm=cw["lam_stat"],
        drude_fit_eps_inf_wp_gamma=(np.array([p_drude[0], p_drude[1], p_drude[2]])),
        # full branches for later re-evaluation at any wavelength
        branch_wavelength_nm=wl_e,
        branch_K_over_k0=us_e,
        branch_residual=res_e,
        branch_localization=loc,
        branch_bound=bound,
        berreman_wavelength_nm=wl_b,
        berreman_K_over_k0=us_b,
    )
    print(f"saved: {config.OUT_NPZ}")

    import pandas as pd
    pd.DataFrame({
        "wavelength_nm": wl_e,
        "K_re_over_k0": us_e.real, "K_im_over_k0": us_e.imag,
        "pole_residual": res_e, "bound_both_claddings": bound,
        "Ez_localization_fraction": loc,
    }).to_csv(config.OUT_BRANCH_CSV, index=False)
    print(f"saved: {config.OUT_BRANCH_CSV}")

    print()
    print("=" * 72)
    print("STEP 8: figures")
    print("=" * 72)
    viz.fig1_epsilon(ito, lam_ze)
    viz.fig2_dispersion({"wl": wl_e, "u": us_e}, {"wl": wl_b, "u": us_b},
                        cw, lam_ze, (wl_E, u_E))
    viz.fig3_field_1d(mode, wl_E, s)
    viz.fig4_field_xz(mode, wl_E, s)

    bcmax = max(rep["bc_residuals"].values())
    rows = [
        ("material ENZ wavelength (Re eps = 0)", f"{lam_ze:.1f} nm"),
        ("modal central wavelength (complex-omega)", f"{cw['lam_stat']:.0f} nm"),
        ("target modal wavelength (max confinement)", f"{wl_E:.0f} nm"),
        ("ITO thickness", f"{config.D_ITO_NM:.0f} nm (Karimi et al.)"),
        ("glass index (input parameter, not paper value)", f"{config.N_GLASS}"),
        ("eps_ITO at target", f"{eps2_E.real:+.3f} + {eps2_E.imag:.3f}i"),
        ("Re(K)/k0, Im(K)/k0", f"{u_E.real:+.3f}, {u_E.imag:+.3f}"),
        ("in-plane modal wavelength 2pi/ReK", f"{lam_par:.0f} nm"),
        ("air light line", f"Re K = {u_E.real:.2f} k0 > 1 k0 (outside)"),
        ("glass light line", f"Re K = {u_E.real:.2f} k0 > {config.N_GLASS} k0 (outside)"),
        ("in-plane amplitude decay 1/|Im K|",
         f"{1/abs(u_E.imag)/k0:.1f} nm (overdamped: |ImK|>|ReK|)"),
        ("air decay length 1/Im kz1", f"{La:.1f} nm"),
        ("glass decay length 1/Im kz3", f"{Lg:.1f} nm"),
        ("ITO Ez localization fraction", f"{loc[i_t]:.3f}"),
        ("pole residual |D|", f"{rep['abs_D_fresnel']:.1e}"),
        ("max boundary-condition residual", f"{bcmax:.1e}"),
    ]
    viz.fig5_diagnostics(rows)

    if run_thickness_sweep:
        print()
        print("=" * 72)
        print("STEP 9 (optional): ITO thickness sweep")
        print("=" * 72)
        sweep = []
        for d in (10, 15, 20, 23, 25, 30, 40):
            seed = u_enz_seed * config.D_ITO_NM / d      # quasi-static K ~ 1/d
            try:
                wls, uss, _ = continue_branch(ito, wl_scan, seed, d)
            except RuntimeError:
                print(f"  d = {d} nm: seed did not converge, skipped")
                continue
            locs = np.full(len(wls), np.nan)
            for i, (w, u) in enumerate(zip(wls, uss)):
                k0i = k0_of(w)
                f = ModeField(u * k0i, k0i, complex(ito.eps(w)), d, EPS1, EPS3)
                kz1, _, kz3 = f.kz
                if kz1.imag > 0 and kz3.imag > 0:
                    l = f.ez_localization(n_pts=2001)
                    locs[i] = l if l is not None else np.nan
            if np.all(np.isnan(locs)):
                print(f"  d = {d} nm: no bound point found in window")
                continue
            i_m = int(np.nanargmax(locs))
            sweep.append((d, wls[i_m], uss[i_m], locs[i_m]))
            print(f"  d = {d:2d} nm: lambda_E = {wls[i_m]:6.0f} nm, "
                  f"K/k0 = {uss[i_m].real:+7.3f}{uss[i_m].imag:+8.3f}j, "
                  f"max localization = {locs[i_m]:.3f}")

    print()
    print("done.")
    return wl_E, u_E


if __name__ == "__main__":
    main()
