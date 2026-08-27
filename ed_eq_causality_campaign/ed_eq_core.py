"""
ed_eq_core.py — shared machinery for the ED-EQ causality campaign
=================================================================

* Dispersive material (material_model.py; NO clamping).
* TORCWA solve helpers on the campaign's geometry/illumination convention
  (input = silica half-space, forward = +z, E ∥ x, normal incidence).
* DIFFERENTIABLE EXACT current multipoles in torch (hierarchy option A):
  the Alaee-2018 exact kernels j0, j1/x, j2/x², j3/x³ are implemented in
  closed form (sin/cos), with series switchover at small argument, so the
  optimization objective uses the exact finite-size moments — no
  long-wavelength surrogate gap. Validated against the corrected MENP
  port (menp_port.exact_me) at machine precision on identical fields
  (method_validation.py) and against TORCWA channel amplitudes
  (channel_validation.py).
* Dimensionless normalized scores (documented normalization):
      S_px  = C_px  / A_cell,   C_px  = k^4 |px|^2 / (6 pi eps0^2)
      S_Qxz = C_Qxz / A_cell,   C_Qxz = k^6 * 2|Qe_xz|^2 / (720 pi eps0^2)
  i.e. each score is that component's formal vacuum radiation cross
  section (same constants as validated MENP, per-component) divided by
  the unit-cell area -> a dimensionless per-cell scattering efficiency.
  Both scores share one normalization, so log-geometric-mean balancing is
  dimensionally sound. E_inc = 1 in TORCWA units = MENP's |E0| = 1.
* PRIMARY DISCOVERY FoM (contract §7; Q, linewidth, T/R, phase all
  excluded by construction):
      F_ED_EQ = 0.5 * [ log(S_px + 1e-12) + log(S_Qxz + 1e-12) ]
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import torcwa                              # noqa: E402
from material_model import Material, PRIMARY, K_LOSS_SCENARIO  # noqa: E402,F401

EPS0 = 8.854187817e-12
C0 = 299792458.0
SUBSTRATE_EPS = 1.46 ** 2
N_SUB = 1.46
GEO_DTYPE = torch.float32
SIM_DTYPE = torch.complex64
DEVICE = torch.device('cpu')
EPS_F = 1e-12

_PRIMARY_MAT = Material(PRIMARY)


def si_eps(lam_nm, material=None, lossless=False, k_override=None):
    m = material or _PRIMARY_MAT
    return complex(m.eps(lam_nm, lossless=lossless, k_override=k_override))


# ---------------------------------------------------------------------------
# TORCWA solve
# ---------------------------------------------------------------------------

def build_sim(rho_tilda, P, h, lam_nm, order, eps_si=None, substrate_eps=SUBSTRATE_EPS):
    if eps_si is None:
        eps_si = si_eps(lam_nm)
    eps_si_t = torch.tensor(eps_si, dtype=SIM_DTYPE, device=DEVICE)
    sim = torcwa.rcwa(freq=1.0 / lam_nm, order=list(order), L=[float(P), float(P)],
                      dtype=SIM_DTYPE, device=DEVICE)
    sim.add_input_layer(eps=substrate_eps)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    layer_eps = rho_tilda * eps_si_t + (1.0 - rho_tilda) * 1.0
    sim.add_layer(thickness=float(h), eps=layer_eps)
    sim.solve_global_smatrix()
    sim.source_planewave(amplitude=[1.0, 0.0], direction='forward')
    return sim


def fields_3d(sim, P, h, n_xy, nz, keep_H=False):
    """Complex E (and optionally H) on n_xy x n_xy x nz, z in [0,h]
    endpoints included (trapz-ready). Differentiable (torch)."""
    x_ax = torch.linspace(0.0, float(P), n_xy, dtype=GEO_DTYPE, device=DEVICE)
    z_fr = np.linspace(0.0, 1.0, nz)
    E = [[], [], []]
    H = [[], [], []]
    for zf in z_fr:
        Ev, Hv = sim.field_xy(0, x_ax, x_ax, z_prop=float(zf * h))
        for c in range(3):
            E[c].append(Ev[c])
            if keep_H:
                H[c].append(Hv[c])
    E = [torch.stack(comp, dim=-1) for comp in E]           # (nx,ny,nz)
    if keep_H:
        H = [torch.stack(comp, dim=-1) for comp in H]
    return x_ax, torch.tensor(z_fr * h, dtype=GEO_DTYPE), E, (H if keep_H else None)


# ---------------------------------------------------------------------------
# Differentiable exact multipoles (torch)
# ---------------------------------------------------------------------------

def _kernels_torch(x):
    """(j0, j1/x, j2/x^2, j3/x^3) closed-form with series switchover;
    differentiable everywhere (both branches evaluated on safe args)."""
    small = x < 1e-2
    xs = torch.where(small, torch.ones_like(x), x)
    s, c = torch.sin(xs), torch.cos(xs)
    j0 = torch.where(small, 1 - x * x / 6, s / xs)
    k1 = torch.where(small, 1 / 3 - x * x / 30, (s / xs ** 2 - c / xs) / xs)
    j2 = ((3 / xs ** 3 - 1 / xs) * s - 3 / xs ** 2 * c)
    k2 = torch.where(small, 1 / 15 - x * x / 210, j2 / xs ** 2)
    j3 = ((15 / xs ** 4 - 6 / xs ** 2) * s - (15 / xs ** 3 - 1 / xs) * c)
    k3 = torch.where(small, 1 / 105 - x * x / 1890, j3 / xs ** 3)
    return j0, k1, k2, k3


def _trapz3(F, x, y, z):
    F = torch.trapezoid(F, x, dim=0)
    F = torch.trapezoid(F, y, dim=0)
    return torch.trapezoid(F, z, dim=0)


def torch_moments(E, eps_grid3, x_nm, z_nm, lam_nm, origin_nm=None):
    """Exact current multipoles (torch, differentiable).

    E: [Ex,Ey,Ez] complex tensors (nx,ny,nz) in TORCWA units (E_inc=1).
    eps_grid3: complex eps(r) on the same grid (air = 1).
    x_nm: in-plane axis (nm, same for x and y); z_nm: z axis (nm).
    Returns dict of complex128 scalars in SI: px..pz [C m], mx..mz [A m2],
    Qe components [C m2], toroidal T [diagnostic], and k [1/m].

    FLOAT-SAFETY DESIGN: all tensor arithmetic runs in DIMENSIONLESS units
    (coordinates xi = r/lambda, reduced current J' = chi*E, all magnitudes
    O(1e-4..1e2), well inside float32 range); the SI prefactors
    (eps0*lam^3 etc., ~1e-30) are restored on the final scalars after
    casting to complex128. Direct SI arithmetic in float32 underflows
    (|p|^2 ~ 1e-61) and produced inf*0 = NaN scores.
    """
    P = float(x_nm[-1])
    h = float(z_nm[-1])
    if origin_nm is None:
        origin_nm = (P / 2, P / 2, h / 2)
    lam = lam_nm * 1e-9
    k = 2 * math.pi / lam
    omega = 2 * math.pi * C0 / lam
    # dimensionless coordinates xi = r/lambda
    xu = (x_nm - origin_nm[0]) / lam_nm
    yu = (x_nm - origin_nm[1]) / lam_nm
    zu = (z_nm - origin_nm[2]) / lam_nm

    X = xu.reshape(-1, 1, 1)
    Y = yu.reshape(1, -1, 1)
    Z = zu.reshape(1, 1, -1)
    chi = (eps_grid3 - 1.0)
    Jx, Jy, Jz = chi * E[0], chi * E[1], chi * E[2]   # reduced current J'

    rJ = X * Jx + Y * Jy + Z * Jz
    rr = X * X + Y * Y + Z * Z
    r = torch.sqrt(rr)
    kr = 2 * math.pi * r                              # k*r in physical terms
    j0, K1, K2, K3 = _kernels_torch(kr)
    j0 = j0.to(Jx.dtype); K1 = K1.to(Jx.dtype)
    K2 = K2.to(Jx.dtype); K3 = K3.to(Jx.dtype)
    T = lambda F: _trapz3(F, xu, yu, zu).to(torch.complex128)
    tp2 = (2 * math.pi) ** 2

    # SI restoration constants (float64 python complex, exact multiplies)
    c_p = EPS0 * lam ** 3                     # p = c_p * [dimensionless]
    c_m = -1j * omega * EPS0 * lam ** 4 * 1.5 # m
    c_Q = 3 * EPS0 * lam ** 4                 # Qe
    c_T = -1j * omega * EPS0 * lam ** 5 / (10 * C0)

    out = {}
    for tag, Ja, A in [('px', Jx, X), ('py', Jy, Y), ('pz', Jz, Z)]:
        out[tag] = c_p * (T(Ja * j0)
                   + (tp2 / 2) * T((3 * rJ * A - rr * Ja) * K2))
    rxJx = Y * Jz - Z * Jy
    rxJy = Z * Jx - X * Jz
    rxJz = X * Jy - Y * Jx
    out['mx'] = c_m * T(rxJx * K1)
    out['my'] = c_m * T(rxJy * K1)
    out['mz'] = c_m * T(rxJz * K1)

    def Qe(a, b, Ja, Jb, diag):
        d1 = (3 * (b * Ja + a * Jb) - (2 * rJ if diag else 0)) * K1
        d2 = (5 * a * b * rJ - rr * (a * Jb + b * Ja)
              - (rr * rJ if diag else 0)) * K3
        return c_Q * (T(d1) + 2 * tp2 * T(d2))
    out['Qxx'] = Qe(X, X, Jx, Jx, True)
    out['Qyy'] = Qe(Y, Y, Jy, Jy, True)
    out['Qzz'] = Qe(Z, Z, Jz, Jz, True)
    out['Qxy'] = Qe(X, Y, Jx, Jy, False)
    out['Qxz'] = Qe(X, Z, Jx, Jz, False)
    out['Qyz'] = Qe(Y, Z, Jy, Jz, False)
    # magnetic quadrupole, exact kernel j2/(kr)^2, CORRECTED symmetrization
    # (the MENP dQmxz bug is not reproduced here); added by the Stage-A
    # forensic audit - the original Stage-A code omitted Qm entirely.
    c_Qm = -1j * omega * EPS0 * lam ** 5 * 15
    out['Qmxx'] = c_Qm * T(2 * X * rxJx * K2)
    out['Qmyy'] = c_Qm * T(2 * Y * rxJy * K2)
    out['Qmzz'] = c_Qm * T(2 * Z * rxJz * K2)
    out['Qmxy'] = c_Qm * T((X * rxJy + Y * rxJx) * K2)
    out['Qmxz'] = c_Qm * T((X * rxJz + Z * rxJx) * K2)
    out['Qmyz'] = c_Qm * T((Y * rxJz + Z * rxJy) * K2)
    # long-wavelength toroidal diagnostic
    out['Tx'] = c_T * T(rJ * X - 2 * rr * Jx)
    out['Ty'] = c_T * T(rJ * Y - 2 * rr * Jy)
    out['Tz'] = c_T * T(rJ * Z - 2 * rr * Jz)
    out['k'] = k
    return out


def scores_from_moments(mo, A_cell_m2):
    """Dimensionless per-cell radiation-efficiency scores (see header)."""
    k = mo['k']
    cE = k ** 4 / (6 * math.pi * EPS0 ** 2)
    C_px = cE * torch.abs(mo['px']) ** 2
    C_Qxz = (k ** 6 / (720 * math.pi * EPS0 ** 2)) * 2 * torch.abs(mo['Qxz']) ** 2
    return C_px / A_cell_m2, C_Qxz / A_cell_m2


def family_weights(mo):
    """Formal per-family radiation weights (validated MENP constants) for
    monitoring; NOT the periodic radiated power.
    AUDIT NOTE: the Stage-A version returned only (Cp, Cm, CQe); the
    fractions EDEQ_frac=(Cp+CQe)/(Cp+Cm+CQe) and MD_frac=Cm/(Cp+Cm+CQe)
    therefore EXCLUDED the magnetic quadrupole from the denominator (it
    was not computed at all), which is why they summed to 1. The audited
    form includes CQm; use family_weights4 for the complete partition."""
    Cp, Cm, CQe, _ = family_weights4(mo)
    return Cp, Cm, CQe


def family_weights4(mo):
    """Complete exact family partition (Cp, Cm, CQe, CQm) with the same
    MENP constants: CQm = const/120*(k/c)^2*sum|Qm|^2 (off-diag doubled)."""
    k = mo['k']
    cE = k ** 4 / (6 * math.pi * EPS0 ** 2)
    Cp = cE * sum(torch.abs(mo[t]) ** 2 for t in ('px', 'py', 'pz'))
    Cm = cE / C0 ** 2 * sum(torch.abs(mo[t]) ** 2 for t in ('mx', 'my', 'mz'))
    n2Qe = (sum(torch.abs(mo[t]) ** 2 for t in ('Qxx', 'Qyy', 'Qzz'))
            + 2 * sum(torch.abs(mo[t]) ** 2 for t in ('Qxy', 'Qxz', 'Qyz')))
    CQe = cE / 120 * k ** 2 * n2Qe
    n2Qm = (sum(torch.abs(mo[t]) ** 2 for t in ('Qmxx', 'Qmyy', 'Qmzz'))
            + 2 * sum(torch.abs(mo[t]) ** 2 for t in ('Qmxy', 'Qmxz', 'Qmyz')))
    CQm = cE / 120 * (k / C0) ** 2 * n2Qm
    return Cp, Cm, CQe, CQm


def eval_objective(rho_tilda, P, h, lam_nm, order, n_xy=48, nz=7,
                   eps_si_val=None, substrate_eps=SUBSTRATE_EPS):
    """Differentiable F_ED_EQ evaluation. Returns (F, S_px, S_Qxz, mo)."""
    if eps_si_val is None:
        eps_si_val = si_eps(lam_nm)
    sim = build_sim(rho_tilda, P, h, lam_nm, order, eps_si=eps_si_val,
                    substrate_eps=substrate_eps)
    x_ax, z_ax, E, _ = fields_3d(sim, P, h, n_xy, nz)
    # eps on the eval grid from the SAME differentiable density
    nmask = rho_tilda.shape[0]
    idx = (torch.floor(x_ax / P * nmask).long()) % nmask
    rho_s = rho_tilda[idx][:, idx]
    eps3 = (rho_s * (eps_si_val - 1.0) + 1.0)[:, :, None].expand(n_xy, n_xy, nz)
    mo = torch_moments(E, eps3, x_ax, z_ax, lam_nm)
    A = (P * 1e-9) ** 2
    S_px, S_Qxz = scores_from_moments(mo, A)
    F = 0.5 * (torch.log(S_px + EPS_F) + torch.log(S_Qxz + EPS_F))
    return F, S_px, S_Qxz, mo


# ---------------------------------------------------------------------------
# 0th-order channel amplitudes (TORCWA = authority for P_rad)
# ---------------------------------------------------------------------------

def channel_amplitudes(sim):
    """Full 0th-order complex amplitudes (x- and y-pol, transmission up into
    air and reflection down into silica), power-normalized S-parameters."""
    amps = {}
    for pol in ('xx', 'yx'):
        amps['t' + pol] = sim.S_parameters(orders=[0, 0], direction='forward',
                                           port='transmission', polarization=pol,
                                           ref_order=[0, 0])
        amps['r' + pol] = sim.S_parameters(orders=[0, 0], direction='forward',
                                           port='reflection', polarization=pol,
                                           ref_order=[0, 0])
    return amps


def bare_stack_amplitudes(P, h, lam_nm, order):
    """Same stack with an EMPTY (air) patterned layer - the scattering
    background for the induced-current picture."""
    rho0 = torch.zeros((64, 64), dtype=GEO_DTYPE, device=DEVICE)
    sim = build_sim(rho0, P, h, lam_nm, order, eps_si=1.0 + 0j)
    return channel_amplitudes(sim)
