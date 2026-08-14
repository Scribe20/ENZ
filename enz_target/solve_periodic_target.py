"""Periodic-design-consistent ENZ target: real K = m*G, complex-omega pole.

Motivation (review finding on the first Phase-2 run): a periodic RCWA
metasurface at normal incidence supplies REAL reciprocal-lattice momenta
K = m*G.  The Phase-1 target (complex K at real omega, lambda = 1527 nm)
is NOT a pole once Im K is dropped: |D(Re K, 1527 nm)| ~ 1.86, and
|D(3G, 1527 nm)| ~ 1.86.  The self-consistent periodic target is instead
the complex-omega pole at fixed real K:

    D(K = m*G, omega_tilde) = 0 .

For p = 770 nm, m = 3 this gives Re(lambda_pole) ~ 1470.8 nm, Q ~ 5.5.
This script solves that pole (Drude fit of the supplied CSV, as in the
Phase-1 complex-omega cross-check), reconstructs and validates the mode
profile at the exact complex frequency, and saves
target_enz_mode_periodic.npz in the same key layout used by Phase 2.

Run:  python solve_periodic_target.py [--period 770 --order 3]
"""

import argparse

import numpy as np
from scipy.optimize import least_squares, root

import config
from ito_material import ITOMaterial
from tm_slab_mode import D_fresnel, ModeField

C_NM_FS = 299.792458
C_SI = 299792458.0
EPS1 = config.EPS_AIR
EPS3 = config.N_GLASS ** 2


def drude_fit(ito):
    w = 2 * np.pi * C_NM_FS / ito.wl
    eps_dat = ito.eps(ito.wl)

    def drude(p, om):
        return p[0] - p[1] ** 2 / (om ** 2 + 1j * p[2] * om)

    p = least_squares(lambda q: np.concatenate(
        [(drude(q, w) - eps_dat).real, (drude(q, w) - eps_dat).imag]),
        [4.0, 2.7, 0.2]).x
    err = float(np.max(np.abs(drude(p, w) - eps_dat)))
    return p, drude, err


def main(period_nm=770.0, m_order=3, d_nm=config.D_ITO_NM,
         out_path=None, seed_lambda_nm=1470.0, seed_im_ratio=-0.09):
    ito = ITOMaterial()
    p, drude, fit_err = drude_fit(ito)
    print(f"[drude] eps_inf={p[0]:.5f}, wp={p[1]:.6f} rad/fs, "
          f"gamma={p[2]:.6f} rad/fs, max fit err {fit_err:.1e}")

    K = m_order * 2 * np.pi / period_nm
    print(f"[target] K = {m_order}*G(p={period_nm:.0f} nm) = {K:.6f} nm^-1")

    # honesty diagnostic: the old real-lambda/real-K combination is NOT a pole
    for lam_chk in (1527.0,):
        k0c = 2 * np.pi / lam_chk
        Dv = abs(D_fresnel(K, k0c, complex(ito.eps(lam_chk)), d_nm, EPS1, EPS3))
        print(f"[diagnostic] |D(K=3G, lambda={lam_chk:.0f} nm real)| = "
              f"{Dv:.3f}  (NOT a pole - motivates this recalculation)")

    # complex-omega pole at fixed real K
    def D_om(om):
        return D_fresnel(K, om / C_NM_FS, drude(p, om), d_nm, EPS1, EPS3)

    w_seed = 2 * np.pi * C_NM_FS / seed_lambda_nm * (1 + 1j * seed_im_ratio)
    sol = root(lambda x: [D_om(x[0] + 1j * x[1]).real,
                          D_om(x[0] + 1j * x[1]).imag],
               [w_seed.real, w_seed.imag], method="hybr",
               options={"xtol": 1e-13})
    w_pole = sol.x[0] + 1j * sol.x[1]
    resid = abs(D_om(w_pole))
    if resid > 1e-9:
        raise RuntimeError(f"complex-omega pole search failed (|D|={resid:.1e})")
    lam_pole = 2 * np.pi * C_NM_FS / w_pole.real
    Q = abs(w_pole.real / (2 * w_pole.imag))
    eps_ito_pole = complex(drude(p, w_pole))
    eps_ito_realw = complex(ito.eps(lam_pole))
    print(f"[pole] omega = {w_pole.real:.6f}{w_pole.imag:+.6f}i rad/fs, "
          f"|D| = {resid:.2e}")
    print(f"[pole] Re lambda = {lam_pole:.2f} nm, Q = {Q:.3f}")
    print(f"[pole] eps_ITO(Drude at complex omega) = {eps_ito_pole:.5f}; "
          f"eps_ITO(CSV at real lambda) = {eps_ito_realw:.5f}")

    # mode profile at the exact pole (true source-free QNM of the bare slab)
    k0_pole = w_pole / C_NM_FS
    mode = ModeField(K, k0_pole, eps_ito_pole, d_nm, EPS1, EPS3)
    bc = mode.interface_residuals()
    print("[validate] boundary residuals:",
          {k: f"{v:.1e}" for k, v in bc.items()})
    assert max(bc.values()) < 1e-8, "boundary conditions violated"
    kz1, kz2, kz3 = mode.kz
    print(f"[validate] kz_air = {kz1:.5f} nm^-1 (Im>0 decay: {kz1.imag > 0}), "
          f"kz_glass = {kz3:.5f} nm^-1 (Im>0 decay: {kz3.imag > 0})")

    # sample the profile: glass tail | ITO | air tail (Phase-1 z convention:
    # z = 0 at ITO/glass, z = d at the air-side interface)
    La = 1.0 / kz1.imag
    Lg = 1.0 / kz3.imag
    z_ito = np.linspace(0.0, d_nm, config.N_Z_ITO)
    z_air = d_nm + np.linspace(0, 3 * La, 301)[1:]
    z_glass = -np.linspace(0, 3 * Lg, 301)[1:][::-1]
    z_all = np.concatenate([z_glass, z_ito, z_air])
    Ez = mode.Ez(z_all)
    Ex = mode.Ex(z_all)
    Hy = mode.Hy(z_all)

    # normalization: integral over ITO of |Ez|^2 dz = 1  (z in nm)
    I = np.trapezoid(np.abs(mode.Ez(z_ito)) ** 2, z_ito)
    s = 1.0 / np.sqrt(I)
    Ez, Ex, Hy = Ez * s, Ex * s, Hy * s
    print(f"[normalize] integral_ITO |Ez|^2 dz = 1 (residual "
          f"{abs(np.trapezoid(np.abs(Ez[(z_all>=0)&(z_all<=d_nm)])**2, z_all[(z_all>=0)&(z_all<=d_nm)])-1):.1e})")

    out_path = out_path or (config._HERE / "target_enz_mode_periodic.npz")
    np.savez(
        out_path,
        # design point for the periodic optimization: drive at Re(lambda_pole)
        wavelength_nm=lam_pole,
        omega_rad_per_s=2 * np.pi * C_SI / (lam_pole * 1e-9),
        k0_per_nm=2 * np.pi / lam_pole,
        # the target momentum is REAL and exactly on the reciprocal lattice
        K_real_per_nm=K,
        K_imag_per_nm=0.0,
        K_note=(f"real K = {m_order}*2*pi/{period_nm}nm (reciprocal-lattice "
                "harmonic); mode is the complex-OMEGA pole at this K "
                "(real-K/complex-omega formulation, self-consistent with a "
                "periodic RCWA cell). Profile evaluated at the exact complex "
                "pole frequency (QNM)."),
        # complex pole data
        omega_pole_rad_per_fs_real=w_pole.real,
        omega_pole_rad_per_fs_imag=w_pole.imag,
        pole_Q=Q,
        pole_residual=resid,
        drude_fit_eps_inf_wp_gamma=np.array([p[0], p[1], p[2]]),
        # profile (Phase-1 z convention)
        z_nm=z_all,
        Ez=Ez, Ex=Ex, Hy=Hy,
        ito_z_range_nm=np.array([0.0, d_nm]),
        kz_air_per_nm=complex(kz1),
        kz_ito_per_nm=complex(kz2),
        kz_glass_per_nm=complex(kz3),
        # material / geometry
        eps_ito_real=eps_ito_realw.real, eps_ito_imag=eps_ito_realw.imag,
        eps_ito_pole_real=eps_ito_pole.real,
        eps_ito_pole_imag=eps_ito_pole.imag,
        ito_thickness_nm=d_nm,
        glass_index=config.N_GLASS,
        glass_index_is_paper_value=config.N_GLASS_IS_PAPER_VALUE,
        eps_air=EPS1,
        period_nm=period_nm, harmonic_order=m_order,
        time_convention="exp(-i*omega*t)",
        normalization="integral_ITO_abs_Ez_squared_dz_equals_1_z_in_nm",
        field_units=("Ez, Ex in nm^-1/2 after normalization; eps_ito_real/"
                     "imag = CSV value at Re(lambda_pole) for driven "
                     "simulations; eps_ito_pole_* = Drude value at the "
                     "complex pole (used for the profile)"),
    )
    print(f"[saved] {out_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", type=float, default=770.0)
    ap.add_argument("--order", type=int, default=3)
    args = ap.parse_args()
    main(period_nm=args.period, m_order=args.order)
