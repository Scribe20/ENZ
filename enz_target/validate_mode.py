"""Numerical validation of the identified ENZ eigenmode.

Checks performed (all printed, nothing hidden):
 1. Equivalence of the two independently derived dispersion forms
    (Fresnel-pole form vs. Vassant Eq. (1) tan form) at the found roots.
 2. Pole residual |D| at the target root.
 3. Maxwell boundary-condition residuals (Hy, Ex, Dz) at both interfaces.
 4. Square-root branch verification: explicit Im(kz) signs and cladding decay.
 5. Light-cone relations for air and glass.
 6. Resolution check: localization integral vs. grid density.
 7. Driven-response consistency: the near-field spectrum Im r_p(K real, lambda)
    peaks near the complex-omega pole branch (independent of the root finder).
"""

import numpy as np

import config
from tm_slab_mode import D_fresnel, D_vassant, r123_tm, ModeField, kz_branch
from solve_enz_dispersion import k0_of, EPS1, EPS3


def validate_target(wl_nm, u, ito, verbose=True):
    """Run all checks for one root u = K/k0 at wavelength wl_nm. Returns dict."""
    k0 = k0_of(wl_nm)
    eps2 = complex(ito.eps(wl_nm))
    K = u * k0
    rep = {}

    # 1. dispersion-form equivalence
    rep["abs_D_fresnel"] = abs(D_fresnel(K, k0, eps2, config.D_ITO_NM, EPS1, EPS3))
    rep["abs_D_vassant"] = abs(D_vassant(K, k0, eps2, config.D_ITO_NM, EPS1, EPS3))

    # perturbed point: both forms must be nonzero and finite (no spurious zeros)
    Kp = (u + 0.3 + 0.2j) * k0
    rep["abs_D_fresnel_offroot"] = abs(D_fresnel(Kp, k0, eps2, config.D_ITO_NM, EPS1, EPS3))
    rep["abs_D_vassant_offroot"] = abs(D_vassant(Kp, k0, eps2, config.D_ITO_NM, EPS1, EPS3))

    # 2-4. field, boundary conditions, branch/decay
    f = ModeField(K, k0, eps2, config.D_ITO_NM, EPS1, EPS3)
    rep["bc_residuals"] = f.interface_residuals()
    kz1, kz2, kz3 = f.kz
    rep["kz1_over_k0"] = kz1 / k0
    rep["kz2_over_k0"] = kz2 / k0
    rep["kz3_over_k0"] = kz3 / k0
    rep["air_decays"] = kz1.imag > 0
    rep["glass_decays"] = kz3.imag > 0
    La, Lg = f.decay_lengths_nm()
    rep["L_air_nm"], rep["L_glass_nm"] = La, Lg

    # 5. light cones
    rep["Re_u"] = u.real
    rep["outside_air_cone"] = u.real > 1.0
    rep["outside_glass_cone"] = u.real > config.N_GLASS

    # 6. resolution check on the localization integral
    l_lo = f.ez_localization(n_pts=2001)
    l_hi = f.ez_localization(n_pts=200001)
    rep["localization"] = l_hi
    rep["localization_grid_change"] = (abs(l_hi - l_lo) / l_hi) if l_hi else None

    if verbose:
        print(f"--- validation at lambda = {wl_nm:.2f} nm, K/k0 = {u:.6f} ---")
        print(f"|D| Fresnel form        : {rep['abs_D_fresnel']:.3e}")
        print(f"|D| Vassant tan form    : {rep['abs_D_vassant']:.3e}")
        print(f"|D| off-root (sanity)   : {rep['abs_D_fresnel_offroot']:.3e} / "
              f"{rep['abs_D_vassant_offroot']:.3e}")
        print("boundary-condition residuals:")
        for k, v in rep["bc_residuals"].items():
            print(f"    {k:16s}: {v:.3e}")
        print(f"kz1/k0 = {rep['kz1_over_k0']:.4f}  (Im>0 -> air decay: {rep['air_decays']})")
        print(f"kz3/k0 = {rep['kz3_over_k0']:.4f}  (Im>0 -> glass decay: {rep['glass_decays']})")
        print(f"decay lengths: air {La and f'{La:.1f} nm'}, glass {Lg and f'{Lg:.1f} nm'}")
        print(f"Re K/k0 = {u.real:.3f}: outside air cone: {rep['outside_air_cone']}, "
              f"outside glass cone: {rep['outside_glass_cone']}")
        print(f"Ez localization in ITO  : {rep['localization']:.4f} "
              f"(grid-refinement change {rep['localization_grid_change']:.1e})")
    return rep


def driven_consistency(ito, u_list=(2.0, 3.0, 5.0), wl=np.arange(1350.0, 1701.0, 1.0),
                       verbose=True):
    """Peak of Im r_p(K real; lambda): driven near-field resonance positions."""
    peaks = {}
    for uK in u_list:
        vals = np.array([r123_tm(uK * k0_of(w), k0_of(w), complex(ito.eps(w)),
                                 config.D_ITO_NM, EPS1, EPS3).imag for w in wl])
        peaks[uK] = float(wl[np.argmax(vals)])
    if verbose:
        print("driven near-field resonance, peak of Im r_p(K, lambda):")
        for uK, lp in peaks.items():
            print(f"    K = {uK:.1f} k0 -> {lp:.0f} nm")
    return peaks


def _absorption_ppol(wl_nm, sin_th, ito):
    """p-pol absorptance of air/ITO/glass for plane-wave incidence from air."""
    k0 = k0_of(wl_nm)
    eps2 = complex(ito.eps(wl_nm))
    K = sin_th * k0
    kz1 = complex(kz_branch(EPS1, k0, K))
    kz3 = complex(kz_branch(EPS3, k0, K))
    kz2 = complex(kz_branch(eps2, k0, K))
    r = complex(r123_tm(K, k0, eps2, config.D_ITO_NM, EPS1, EPS3))
    r12 = (eps2 * kz1 - EPS1 * kz2) / (eps2 * kz1 + EPS1 * kz2)
    r23 = (EPS3 * kz2 - eps2 * kz3) / (EPS3 * kz2 + eps2 * kz3)
    ph = np.exp(2j * kz2 * config.D_ITO_NM)
    t123 = (1 + r12) * (1 + r23) * np.exp(1j * kz2 * config.D_ITO_NM) \
        / (1 + r12 * r23 * ph)
    T = (kz3 / EPS3).real / (kz1 / EPS1).real * abs(t123) ** 2
    return 1.0 - abs(r) ** 2 - T


def berreman_driven_check(ito, branch_wl, branch_u, theta_deg=50.0, verbose=True):
    """Spectral Berreman check at fixed oblique angle (how Berreman absorption
    is classically measured): the p-pol absorption peak wavelength should lie
    near the wavelength where the leaky branch satisfies Re K/k0 = sin(theta).
    """
    s = np.sin(np.radians(theta_deg))
    wl = np.arange(1200.0, 1701.0, 1.0)
    A = np.array([_absorption_ppol(w, s, ito) for w in wl])
    wl_pk = wl[np.argmax(A)]
    # wavelength where the leaky branch phase-matches sin(theta)
    j = np.argmin(np.abs(branch_u.real - s))
    wl_match = branch_wl[j]
    if verbose:
        print(f"Berreman driven check, p-pol at {theta_deg:.0f} deg incidence:")
        print(f"    absorption peak at {wl_pk:.0f} nm (A = {A.max():.3f})")
        print(f"    leaky branch phase-match Re K/k0 = sin(theta) at "
              f"{wl_match:.0f} nm (pole Im K/k0 = {branch_u[j].imag:.3f} "
              "-> broad resonance)")
    return wl, A, wl_pk, wl_match
