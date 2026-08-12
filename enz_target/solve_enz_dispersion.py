"""Root finding and branch continuation for the TM pole D(K; omega) = 0.

Strategy (see task spec):
  * real omega (tabulated ITO data), complex K = K' + i*K'' unknown;
  * solve the two real equations Re(D)=0, Im(D)=0 in x = [K'/k0, K''/k0]
    with scipy.optimize.root (hybr);
  * initial wavelength: multi-seed scan + de-duplication + physical filtering;
  * then continuation in wavelength using the previous root as the seed.
"""

import numpy as np
from scipy.optimize import root

import config
from ito_material import ITOMaterial
from tm_slab_mode import D_fresnel, kz_branch

EPS1 = config.EPS_AIR
EPS3 = config.N_GLASS**2


def k0_of(wl_nm):
    return 2.0 * np.pi / wl_nm      # nm^-1


def solve_pole(wl_nm, eps2, d_nm, u_seed, sheet1=+1, sheet3=+1):
    """Solve D(K)=0 at one wavelength from seed u_seed = K/k0 (complex).

    Returns (u_root, |D| residual) or (None, None) if not converged.
    """
    k0 = k0_of(wl_nm)

    def F(x):
        u = x[0] + 1j * x[1]
        Dv = D_fresnel(u * k0, k0, eps2, d_nm, EPS1, EPS3, sheet1, sheet3)
        return [Dv.real, Dv.imag]

    sol = root(F, [u_seed.real, u_seed.imag], method="hybr",
               options={"xtol": config.ROOT_XTOL, "maxfev": 400})
    if not sol.success:
        return None, None
    u = sol.x[0] + 1j * sol.x[1]
    res = abs(D_fresnel(u * k0, k0, eps2, d_nm, EPS1, EPS3, sheet1, sheet3))
    if res > config.RESIDUAL_OK:
        return None, None
    return u, res


def initial_scan(wl_nm, eps2, d_nm, sheet1=+1, sheet3=+1):
    """Multi-seed scan at a single wavelength; de-duplicated root list."""
    roots = []
    for ur in config.SEED_RE:
        for ui in config.SEED_IM:
            u, res = solve_pole(wl_nm, eps2, d_nm, ur + 1j * ui, sheet1, sheet3)
            if u is None:
                continue
            if u.real < 0:          # direction degeneracy K -> -K: keep K' > 0
                u = -u
            if any(abs(u - v) < config.DEDUP_TOL for v, _ in roots):
                continue
            roots.append((u, res))
    return roots


def classify_root(wl_nm, u, sheet1=+1, sheet3=+1):
    """Light-cone location and cladding decay character of a root."""
    k0 = k0_of(wl_nm)
    K = u * k0
    kz1 = complex(kz_branch(EPS1, k0, K, sheet1))
    kz3 = complex(kz_branch(EPS3, k0, K, sheet3))
    return {
        "outside_air_cone": u.real > 1.0,
        "outside_glass_cone": u.real > config.N_GLASS,
        "decays_air": kz1.imag > 0,
        "decays_glass": kz3.imag > 0,
        "kz1": kz1, "kz3": kz3,
    }


def continue_branch(ito: ITOMaterial, wl_start_nm, u_start, d_nm,
                    wl_min=config.LAMBDA_MIN_NM, wl_max=config.LAMBDA_MAX_NM,
                    step=config.LAMBDA_STEP_NM, sheet1=+1, sheet3=+1):
    """Continuation of one branch over wavelength in both directions.

    Stops a direction when the solver fails, the root jumps discontinuously
    (|du| > 0.5), or the wavelength window ends.  Returns sorted arrays.
    """
    out = {}

    def march(direction):
        wl, u_prev = wl_start_nm, u_start
        while True:
            wl_next = wl + direction * step
            if wl_next < wl_min or wl_next > wl_max:
                break
            u, res = solve_pole(wl_next, complex(ito.eps(wl_next)), d_nm,
                                u_prev, sheet1, sheet3)
            if u is None or abs(u - u_prev) > 0.5:
                break
            out[wl_next] = (u, res)
            wl, u_prev = wl_next, u

    u0, res0 = solve_pole(wl_start_nm, complex(ito.eps(wl_start_nm)), d_nm,
                          u_start, sheet1, sheet3)
    if u0 is None:
        raise RuntimeError("branch continuation: seed point did not converge")
    out[wl_start_nm] = (u0, res0)
    march(+1)
    march(-1)

    wls = np.array(sorted(out))
    us = np.array([out[w][0] for w in wls])
    res = np.array([out[w][1] for w in wls])
    return wls, us, res
