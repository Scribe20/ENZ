"""TORCWA forward model for the a-Si(freeform) / ITO / glass stack.

Uses the supplied torcwa version verbatim (vendored in third_party/).
Everything returned for the objective stays inside the autograd graph;
the differentiable path is:
    rho -> layer eps (linear mix) -> eps_conv -> P,Q -> Eig.apply ->
    layer S-matrices -> global S + C coefficients -> field_xy -> Ez.
"""

import sys
import numpy as np
import torch

import config

sys.path.insert(0, str(config.TORCWA_DIR))
import torcwa  # noqa: E402  (supplied version)


def grid_axes():
    """Cell-centered x/y axes (nm), same convention as torcwa's rcwa_geo."""
    x = (config.PX_NM / config.NX_DESIGN) * (
        torch.arange(config.NX_DESIGN, dtype=config.GEO_DTYPE,
                     device=config.DEVICE) + 0.5)
    y = (config.PY_NM / config.NY_DESIGN) * (
        torch.arange(config.NY_DESIGN, dtype=config.GEO_DTYPE,
                     device=config.DEVICE) + 0.5)
    return x, y


def build_solved_sim(rho_projected, lam_nm, eps_ito, n_glass):
    """Build and solve the stack for one design (or the bare reference).

    rho_projected: (Nx, Ny) tensor in [0,1], or None for the reference
    (air / ITO / glass: the a-Si layer is present with eps = 1 so that the
    z-coordinates, layer indexing, and phase conventions are IDENTICAL).
    Returns the solved sim with the plane-wave source set.
    """
    order = config.FOURIER_ORDER
    L = [config.PX_NM, config.PY_NM]
    sim = torcwa.rcwa(freq=1.0 / lam_nm, order=order, L=L,
                      dtype=config.SIM_DTYPE, device=config.DEVICE)
    sim.add_input_layer(eps=1.0)                      # air superstrate
    sim.add_output_layer(eps=complex(n_glass) ** 2)   # glass substrate
    sim.set_incident_angle(inc_ang=config.INC_ANGLE_RAD,
                           azi_ang=config.AZI_ANGLE_RAD)

    if rho_projected is None:
        sim.add_layer(thickness=config.ASI_THICKNESS_NM, eps=1.0)
    else:
        eps_layer = rho_projected * (config.EPS_ASI - 1.0) + 1.0
        sim.add_layer(thickness=config.ASI_THICKNESS_NM,
                      eps=eps_layer.to(config.SIM_DTYPE))
    sim.add_layer(thickness=float(config.ITO_THICKNESS_NM), eps=eps_ito)

    sim.solve_global_smatrix()
    amp = [1.0, 0.0] if config.POLARIZATION == "x" else [0.0, 1.0]
    sim.source_planewave(amplitude=amp, direction="forward")
    return sim


ITO_LAYER_NUM = 1   # layer 0 = a-Si design layer, layer 1 = ITO


def ez_in_ito(sim, x_axis, y_axis, z_prop_list):
    """Complex Ez inside the ITO at the given z_prop slices.

    Returns tensor of shape (Nz, Nx, Ny); stays in the autograd graph
    (field_xy for internal layers is pure torch, verified by audit).
    """
    slices = []
    for zp in z_prop_list:
        E, _H = sim.field_xy(ITO_LAYER_NUM, x_axis, y_axis, float(zp))
        slices.append(E[2])
    return torch.stack(slices, dim=0)


def specular_RT(sim):
    """Power reflectance/transmittance diagnostics.

    At lambda = 1527 nm and period 770 nm only the (0,0) order propagates in
    both air and glass, so the specular sums are the totals; A = 1 - R - T.
    """
    inpol = "x" if config.POLARIZATION == "x" else "y"
    R = torch.zeros((), dtype=config.GEO_DTYPE, device=config.DEVICE)
    T = torch.zeros((), dtype=config.GEO_DTYPE, device=config.DEVICE)
    for outpol in ("x", "y"):
        pol = outpol + inpol
        r = sim.S_parameters(orders=[0, 0], direction="forward",
                             port="reflection", polarization=pol,
                             ref_order=[0, 0])
        t = sim.S_parameters(orders=[0, 0], direction="forward",
                             port="transmission", polarization=pol,
                             ref_order=[0, 0])
        R = R + torch.abs(r.ravel()[0]) ** 2
        T = T + torch.abs(t.ravel()[0]) ** 2
    return R, T


def p_inc_cell():
    """Incident power per unit cell in TORCWA's Lorentz-Heaviside units
    (c = eps0 = mu0 = 1): P = 0.5 * n_in * cos(theta) * |E0|^2 * A,
    with |E0| = 1 and air input (n_in = 1)."""
    return 0.5 * np.cos(config.INC_ANGLE_RAD) * config.PX_NM * config.PY_NM


_ITO_SPLINES = None


def eps_ito_of_lambda(lam_nm):
    """Complex ITO permittivity from the Phase-1 CSV (cubic spline).

    Loads the CSV directly (not via the enz_target package) to avoid a
    module-name collision between the two phases' config modules.
    """
    global _ITO_SPLINES
    if _ITO_SPLINES is None:
        import pandas as pd
        from scipy.interpolate import CubicSpline
        df = pd.read_csv(config.ENZ_TARGET_DIR / "data"
                         / "ito_digitized_dense_1nm_physical.csv")
        wl = df["wavelength_nm"].to_numpy(float)
        re_col = [c for c in df.columns if "epsilon_real" in c][0]
        im_col = [c for c in df.columns if "epsilon_imag" in c][0]
        _ITO_SPLINES = (CubicSpline(wl, df[re_col].to_numpy(float)),
                        CubicSpline(wl, df[im_col].to_numpy(float)))
    return complex(float(_ITO_SPLINES[0](lam_nm)),
                   float(_ITO_SPLINES[1](lam_nm)))
