"""Multi-angle, variable-geometry TORCWA forward model for the ROBUST ENZ
ENERGY-TRANSFER campaign.

Differences from enz_inverse_design/torcwa_forward.py (which is left
untouched):
  * period P and a-Si height h are per-call arguments (not config globals);
  * oblique incidence (theta, phi) with the source given in TORCWA's p/s
    notation so that |E_inc| = 1 exactly (in the 'xy' notation the implied
    Ez makes |E_inc| = sqrt(1 + (kx/kz)^2) at oblique incidence);
  * the incident polarization is the LAB-FRAME x polarization projected on
    the transverse plane of the incident wave vector (the plane-wave
    decomposition of an x-polarized beam): a_p = cos(theta) cos(phi),
    a_s = -sin(phi), normalized.  At theta = 0 this is exactly the x-polarized
    source used by every previous campaign; in the phi = 0 plane it is pure
    p (TM), in the phi = 90 deg plane pure s (TE);
  * R_total / T_total sum ALL propagating diffraction orders of the sim
    (TORCWA zeroes evanescent orders) and both output polarizations,
    coherently combining the p and s input components; everything stays in
    the autograd graph.
Time convention exp(-j w t), Lorentz-Heaviside units (TORCWA).
"""

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / "enz_inverse_design"
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))
import config                       # noqa: E402  (enz_inverse_design config)
import torcwa_forward as fwd        # noqa: E402  (material splines only)

sys.path.insert(0, str(config.TORCWA_DIR))
import torcwa                       # noqa: E402

DEVICE = config.DEVICE
SIM_DTYPE = config.SIM_DTYPE
GEO_DTYPE = config.GEO_DTYPE

LAMBDA_E = 1433.488          # bare-film ENZ QNM at K = G10(850 nm), Q = 5.80
D_ITO_NM = 23.0
N_GLASS = 1.4446             # from the Phase-1 target npz (not the paper value)


def eps_ito(lam=LAMBDA_E, loss_scale=1.0):
    e = fwd.eps_ito_of_lambda(lam)
    return complex(e.real, loss_scale * e.imag)


def eps_asi(lam=LAMBDA_E):
    return fwd.eps_asi_of_lambda(lam)


def lab_x_amplitude_ps(theta_deg, phi_deg):
    """[a_p, a_s] of the lab-x polarization in TORCWA's p/s basis.

    TORCWA derives the azimuth of the (0,0) order from its k-vector, which
    is 0 at exact normal incidence irrespective of phi; that case is
    handled explicitly (pure p = x)."""
    th, ph = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    if theta_deg == 0.0:
        return [1.0, 0.0]
    a_p, a_s = np.cos(th) * np.cos(ph), -np.sin(ph)
    n = np.hypot(a_p, a_s)
    return [float(a_p / n), float(a_s / n)]


def build_sim(rho, P, h, lam=LAMBDA_E, theta_deg=0.0, phi_deg=0.0,
              order=None, with_ito=True, ito_loss_scale=1.0, pol="labx",
              eps_asi_val=None, eps_ito_val=None):
    """Solve air / a-Si(h, rho) / [ITO 23 nm] / glass at one angle.

    rho: (nx, ny) tensor in [0,1] on the normalized cell, or None for an
    unpatterned eps = 1 layer (bare reference with identical layer indexing).
    pol: 'labx' (default), 'p', or 's'.
    """
    order = order or config.FOURIER_ORDER
    if eps_asi_val is None:
        eps_asi_val = eps_asi(lam)
    if eps_ito_val is None:
        eps_ito_val = eps_ito(lam, ito_loss_scale)
    sim = torcwa.rcwa(freq=1.0 / lam, order=order, L=[float(P), float(P)],
                      dtype=SIM_DTYPE, device=DEVICE)
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=complex(N_GLASS) ** 2)
    sim.set_incident_angle(inc_ang=np.deg2rad(theta_deg),
                           azi_ang=np.deg2rad(phi_deg))
    if rho is None:
        sim.add_layer(thickness=float(h), eps=1.0)
    else:
        eps_layer = rho * (complex(eps_asi_val) - 1.0) + 1.0
        sim.add_layer(thickness=float(h), eps=eps_layer.to(SIM_DTYPE))
    if with_ito:
        sim.add_layer(thickness=D_ITO_NM, eps=eps_ito_val)
    sim.solve_global_smatrix()
    if pol == "labx":
        amp = lab_x_amplitude_ps(theta_deg, phi_deg)
    elif pol == "p":
        amp = [1.0, 0.0]
    elif pol == "s":
        amp = [0.0, 1.0]
    else:
        raise ValueError(pol)
    sim.source_planewave(amplitude=amp, direction="forward", notation="ps")
    sim._amp_ps = amp
    sim._theta_deg, sim._phi_deg = theta_deg, phi_deg
    return sim


def all_orders(sim):
    ox, oy = torch.meshgrid(sim.order_x, sim.order_y, indexing="ij")
    return torch.stack((ox.reshape(-1), oy.reshape(-1)), dim=1)


def rt_all_orders(sim, per_order=False):
    """Power R_total, T_total summed over ALL propagating orders and both
    output polarizations for the coherent p/s input set on the sim.
    Differentiable.  Optionally returns per-order power tables."""
    orders = all_orders(sim)
    a_p, a_s = sim._amp_ps
    out = {}
    for port in ("reflection", "transmission"):
        S = {pp: sim.S_parameters(orders=orders, direction="forward",
                                  port=port, polarization=pp,
                                  ref_order=[0, 0], power_norm=True)
             for pp in ("pp", "ps", "sp", "ss")}
        E_p = S["pp"] * a_p + S["ps"] * a_s
        E_s = S["sp"] * a_p + S["ss"] * a_s
        pw = torch.abs(E_p) ** 2 + torch.abs(E_s) ** 2
        out[port] = pw
    R, T = torch.sum(out["reflection"]), torch.sum(out["transmission"])
    if per_order:
        tab = [dict(m=int(orders[i, 0]), n=int(orders[i, 1]),
                    R=float(out["reflection"][i]),
                    T=float(out["transmission"][i]))
               for i in range(orders.shape[0])
               if float(out["reflection"][i]) > 0 or
               float(out["transmission"][i]) > 0]
        return R, T, tab
    return R, T


def polarization_split(sim):
    """Specular (0,0) reflected/transmitted power in the co- and cross-
    polarized channels (co = same p/s vector as the input)."""
    a_p, a_s = sim._amp_ps
    res = {}
    for port in ("reflection", "transmission"):
        S = {pp: sim.S_parameters(orders=[0, 0], direction="forward",
                                  port=port, polarization=pp,
                                  ref_order=[0, 0], power_norm=True).ravel()[0]
             for pp in ("pp", "ps", "sp", "ss")}
        E_p = S["pp"] * a_p + S["ps"] * a_s
        E_s = S["sp"] * a_p + S["ss"] * a_s
        co = E_p * a_p + E_s * a_s
        cross = -E_p * a_s + E_s * a_p
        res[port] = dict(co=float(torch.abs(co) ** 2),
                         cross=float(torch.abs(cross) ** 2))
    return res


def a_ito(sim):
    R, T = rt_all_orders(sim)
    return 1.0 - R - T, R, T


# ---------------------------------------------------------------------------
# field-based checks (audit only, no_grad)
# ---------------------------------------------------------------------------
def cell_axes(P, n):
    x = (float(P) / n) * (torch.arange(n, dtype=GEO_DTYPE, device=DEVICE) + 0.5)
    return x, x.clone()


def a_ito_volume(sim, P, lam, theta_deg, n_xy=96, n_z=7, loss_scale=1.0):
    """Absorbed fraction from the volume integral (omega/2) Im(eps) |E|^2
    over the ITO layer (layer index 1), divided by the incident power
    P_inc = 0.5 cos(theta) P^2 |E_inc|^2 with |E_inc| = 1 (p/s source)."""
    e_ito = eps_ito(lam, loss_scale)
    x, y = cell_axes(P, n_xy)
    zs = (np.arange(n_z) + 0.5) * D_ITO_NM / n_z
    dV = (float(P) / n_xy) ** 2 * (D_ITO_NM / n_z)
    E2, Iz = 0.0, 0.0
    with torch.no_grad():
        for zp in zs:
            E, _ = sim.field_xy(1, x, y, float(zp))
            E2 += float(sum(torch.sum(torch.abs(c) ** 2) for c in E)) * dV
            Iz += float(torch.sum(torch.abs(E[2]) ** 2)) * dV
    omega = 2 * np.pi / lam
    p_inc = 0.5 * np.cos(np.deg2rad(theta_deg)) * float(P) ** 2
    A_vol = 0.5 * omega * e_ito.imag * E2 / p_inc
    F_Ez = Iz / (float(P) ** 2 * D_ITO_NM)          # <|Ez/E_inc|^2>_ITO
    return dict(A_vol=A_vol, F_Ez=F_Ez, eta_z=Iz / E2 if E2 > 0 else 0.0)


def propagating_orders(P, lam, theta_deg, phi_deg, medium="glass"):
    """Analytic list of propagating (m,n) in air or glass (independent of
    TORCWA) for the diffraction-order audit."""
    n_med = N_GLASS if medium == "glass" else 1.0
    k0 = 2 * np.pi / lam
    kx0 = k0 * np.sin(np.deg2rad(theta_deg)) * np.cos(np.deg2rad(phi_deg))
    ky0 = k0 * np.sin(np.deg2rad(theta_deg)) * np.sin(np.deg2rad(phi_deg))
    G = 2 * np.pi / P
    out = []
    for m in range(-4, 5):
        for n in range(-4, 5):
            kt2 = (kx0 + m * G) ** 2 + (ky0 + n * G) ** 2
            if kt2 < (n_med * k0) ** 2:
                out.append((m, n))
    return out
