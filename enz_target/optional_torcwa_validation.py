"""Optional TORCWA cross-validation of the DRIVEN response of the bare
air / ITO / glass stack (planar, zeroth order only).

What this does and does not do
------------------------------
TORCWA (the supplied version, torcwa-0.1.4.2) is a driven S-matrix solver.
It has NO global source-free eigenmode / pole solver (see the capability
audit in README.md), so it CANNOT be used to find the ENZ pole directly.
What it can validate is the driven physics that follows from the same
material data and conventions: the p-polarized reflectance/transmittance
of the planar stack at oblique incidence, including the Berreman
absorption feature near the ENZ crossing.  This script compares TORCWA's
R/T against the transfer-matrix (Fresnel) result from tm_slab_mode.py.

Requires torch + the supplied torcwa package on sys.path; skips gracefully
otherwise.  Set TORCWA_PATH to the extracted torcwa-main directory.
"""

import os
import sys
import numpy as np

import config
from ito_material import ITOMaterial
from tm_slab_mode import kz_branch, r123_tm
from solve_enz_dispersion import k0_of, EPS1, EPS3

TORCWA_PATH = os.environ.get("TORCWA_PATH", "")


def tmm_RT_ppol(wl_nm, theta_deg, ito):
    """Reference transfer-matrix p-pol R and T (incidence from air)."""
    k0 = k0_of(wl_nm)
    eps2 = complex(ito.eps(wl_nm))
    K = np.sin(np.radians(theta_deg)) * k0
    kz1 = complex(kz_branch(EPS1, k0, K))
    kz2 = complex(kz_branch(eps2, k0, K))
    kz3 = complex(kz_branch(EPS3, k0, K))
    r = complex(r123_tm(K, k0, eps2, config.D_ITO_NM, EPS1, EPS3))
    r12 = (eps2 * kz1 - EPS1 * kz2) / (eps2 * kz1 + EPS1 * kz2)
    r23 = (EPS3 * kz2 - eps2 * kz3) / (EPS3 * kz2 + eps2 * kz3)
    ph = np.exp(2j * kz2 * config.D_ITO_NM)
    t = (1 + r12) * (1 + r23) * np.exp(1j * kz2 * config.D_ITO_NM) / (1 + r12 * r23 * ph)
    T = (kz3 / EPS3).real / (kz1 / EPS1).real * abs(t) ** 2
    return abs(r) ** 2, T


def main(theta_deg=50.0, wavelengths=(1300.0, 1380.0, 1420.0, 1460.0, 1540.0, 1650.0)):
    try:
        import torch
        if TORCWA_PATH:
            sys.path.insert(0, TORCWA_PATH)
        import torcwa
    except ImportError as e:
        print(f"[skip] torch/torcwa not available ({e}); "
              "the independent pole solver does not depend on this check.")
        return None

    ito = ITOMaterial()
    dev = torch.device("cpu")
    dtype = torch.complex128
    L = [300.0, 300.0]                      # lattice period, nm (planar: irrelevant)
    print(f"TORCWA vs transfer-matrix, p-pol, theta = {theta_deg} deg "
          f"(planar stack, order [0,0])")
    print("lambda(nm)   R_torcwa   R_tmm      T_torcwa   T_tmm      "
          "A_torcwa   A_tmm")
    worst = 0.0
    for wl in wavelengths:
        eps2 = complex(ito.eps(wl))
        sim = torcwa.rcwa(freq=1.0 / wl, order=[0, 0], L=L, dtype=dtype, device=dev)
        sim.add_input_layer(eps=EPS1)
        sim.add_output_layer(eps=EPS3)
        sim.set_incident_angle(inc_ang=np.radians(theta_deg), azi_ang=0.0)
        sim.add_layer(thickness=config.D_ITO_NM, eps=eps2)
        sim.solve_global_smatrix()
        # p-pol S-parameters, order (0,0)
        rpp = sim.S_parameters(orders=[0, 0], direction="forward",
                               port="reflection", polarization="pp",
                               ref_order=[0, 0])
        tpp = sim.S_parameters(orders=[0, 0], direction="forward",
                               port="transmission", polarization="pp",
                               ref_order=[0, 0])
        R_t = float(np.abs(rpp.cpu().numpy().ravel()[0]) ** 2)
        T_t = float(np.abs(tpp.cpu().numpy().ravel()[0]) ** 2)
        R_m, T_m = tmm_RT_ppol(wl, theta_deg, ito)
        worst = max(worst, abs(R_t - R_m), abs(T_t - T_m))
        print(f"{wl:9.0f}   {R_t:8.5f}   {R_m:8.5f}   {T_t:8.5f}   {T_m:8.5f}"
              f"   {1-R_t-T_t:8.5f}   {1-R_m-T_m:8.5f}")
    print(f"worst |difference| = {worst:.2e}")
    return worst


if __name__ == "__main__":
    main()
