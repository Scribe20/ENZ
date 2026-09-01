"""R-type freeform campaign core (633 nm reflective PB meta-atom).

Conventions:
  - exp(-j w t); stack: glass substrate (input layer) / patterned a-Si
    layer (thickness H) / air. TORCWA 'forward' = glass->air.
  - DEVICE illumination is from the AIR side ('backward' direction):
    the metalens is illuminated from air and reflects back into air.
    (Validated against the paper rectangle in rt_baseline.py.)
  - Jones matrices in the linear basis: r = [[rxx, rxy],[ryx, ryy]],
    first index = output pol, second = input pol.
  - Circular basis via C = (1/sqrt2)[[1, 1],[i, -i]] (sigma+ = (x+iy)/sqrt2):
    R_circ = C^H R C. For diagonal r: cross = (rx - ry)/2, co = (rx+ry)/2.
    PB law validated by explicit rotation in rt_baseline / finals.
  - Materials: a-Si Franta 2013 (ed_eq campaign primary; genuinely
    brackets 633 nm; n = 4.2827 + 0.0687i at 633). Glass: fused silica
    n = 1.457 (Malitson 1965) -> eps = 2.122849.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'ed_eq_causality_campaign'))
sys.path.insert(0, str(ROOT))
import material_model as _mm                                # noqa: E402
import torcwa                                               # noqa: E402

LAM0 = 633.0
N_GLASS = 1.457
EPS_GLASS = N_GLASS ** 2
_ASI = _mm.primary()
EPS_ASI_633 = complex(_ASI.eps(LAM0))
SIM_DTYPE = torch.complex64
DEVICE = torch.device('cpu')

PERIODS = [190.0, 208.0, 226.0, 244.0, 262.0]
HEIGHTS = [110.0, 140.0, 170.0, 200.0, 230.0, 260.0]


def eps_asi(lam=LAM0):
    return complex(_ASI.eps(float(lam)))


def padding(P):
    return max(20.0, 0.10 * P)


def r_design(P):
    return P / 2 - padding(P)


def design_mask(n, P):
    """Hard circular rotation-safe envelope on the n x n density grid."""
    ax = (np.arange(n) + 0.5) / n * P - P / 2
    X, Y = np.meshgrid(ax, ax, indexing='ij')
    return torch.tensor((X ** 2 + Y ** 2 <= r_design(P) ** 2)
                        .astype(np.float32))


def conic_filter_kernel(n, P, radius_nm):
    px = P / n
    r_px = max(1.0, radius_nm / px)
    k = int(math.ceil(r_px))
    ax = torch.arange(-k, k + 1, dtype=torch.float32)
    X, Y = torch.meshgrid(ax, ax, indexing='ij')
    w = torch.clamp(1 - torch.sqrt(X ** 2 + Y ** 2) / r_px, min=0.0)
    return (w / w.sum())[None, None]


def d2_symmetrize(rho):
    """Enforce mirror symmetry in x and y (C2v): guarantees a diagonal
    Jones matrix (principal axes along x/y - the PB element requirement)
    while leaving the multipolar content unconstrained."""
    return 0.25 * (rho + torch.flip(rho, [0]) + torch.flip(rho, [1])
                   + torch.flip(rho, [0, 1]))


def filt_project(rho, kern, beta, eta=0.5, mask=None):
    """D2 symmetrization + conic density filter + tanh projection +
    hard envelope mask."""
    rho = d2_symmetrize(rho)
    if mask is not None:
        rho = rho * mask
    pad = kern.shape[-1] // 2
    f = torch.nn.functional.conv2d(
        rho[None, None], kern, padding=pad)[0, 0]
    be = torch.tensor(float(beta * eta))
    num = torch.tanh(be) + torch.tanh(beta * (f - eta))
    den = torch.tanh(be) + torch.tanh(torch.tensor(float(beta * (1 - eta))))
    out = num / den
    if mask is not None:
        out = out * mask
    return out


def build_sim(rho, P, H, lam=LAM0, order=(7, 7)):
    e = eps_asi(lam)
    sim = torcwa.rcwa(freq=1.0 / float(lam), order=list(order),
                      L=[float(P), float(P)], dtype=SIM_DTYPE, device=DEVICE)
    sim.add_input_layer(eps=EPS_GLASS)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    sim.add_layer(thickness=float(H), eps=rho * (e - 1.0) + 1.0)
    sim.solve_global_smatrix()
    return sim


def jones(sim, direction='backward'):
    """Full 2x2 complex r and t Jones matrices for the given illumination
    direction ('backward' = from air = device port)."""
    r = torch.zeros(2, 2, dtype=torch.complex64)
    t = torch.zeros(2, 2, dtype=torch.complex64)
    r = [[None, None], [None, None]]
    t = [[None, None], [None, None]]
    for j, amp in ((0, [1.0, 0.0]), (1, [0.0, 1.0])):
        sim.source_planewave(amplitude=amp, direction=direction)
        for i, pol in ((0, 'x'), (1, 'y')):
            polstr = pol + ('x' if j == 0 else 'y')
            r[i][j] = sim.S_parameters(orders=[0, 0], direction=direction,
                                       port='reflection', polarization=polstr,
                                       ref_order=[0, 0])
            t[i][j] = sim.S_parameters(orders=[0, 0], direction=direction,
                                       port='transmission',
                                       polarization=polstr, ref_order=[0, 0])
    R = torch.stack([torch.stack([r[0][0].reshape(()), r[0][1].reshape(())]),
                     torch.stack([r[1][0].reshape(()), r[1][1].reshape(())])])
    T = torch.stack([torch.stack([t[0][0].reshape(()), t[0][1].reshape(())]),
                     torch.stack([t[1][0].reshape(()), t[1][1].reshape(())])])
    return R, T


_C = torch.tensor([[1.0 + 0j, 1.0 + 0j], [1j, -1j]]) / math.sqrt(2)


def circular(J):
    """R_circ = C^H J C; returns (co1, cross12, cross21, co2)."""
    Jc = _C.conj().T.to(J.dtype) @ J @ _C.to(J.dtype)
    return Jc


def device_metrics(R, T):
    """All scalar campaign metrics from the Jones pair (device port)."""
    Rc = circular(R)
    Tc = circular(T)
    rx, ry = R[0, 0], R[1, 1]
    dphi = torch.angle(rx) - torch.angle(ry)
    R_tot = (torch.abs(R) ** 2).sum(dim=0)   # per input pol
    T_tot = (torch.abs(T) ** 2).sum(dim=0)
    out = {
        'R_cross': float(torch.abs(Rc[0, 1]) ** 2),
        'R_co': float(torch.abs(Rc[0, 0]) ** 2),
        'R_cross_21': float(torch.abs(Rc[1, 0]) ** 2),
        'R_co_22': float(torch.abs(Rc[1, 1]) ** 2),
        'T_cross': float(torch.abs(Tc[0, 1]) ** 2),
        'T_co': float(torch.abs(Tc[0, 0]) ** 2),
        'abs_rx': float(torch.abs(rx)), 'abs_ry': float(torch.abs(ry)),
        'abs_rxy': float(torch.abs(R[0, 1])),
        'abs_ryx': float(torch.abs(R[1, 0])),
        'dphi_r_deg': float(torch.rad2deg(dphi)),
        'pb_phase_err_deg': float(torch.rad2deg(torch.atan2(
            torch.sin(dphi - math.pi), torch.cos(dphi - math.pi)).abs())),
        'R_total_x': float(R_tot[0]), 'R_total_y': float(R_tot[1]),
        'T_total_x': float(T_tot[0]), 'T_total_y': float(T_tot[1]),
        'A_x': float(1 - R_tot[0] - T_tot[0]),
        'A_y': float(1 - R_tot[1] - T_tot[1]),
    }
    for nm, M in (('rxx', R[0, 0]), ('rxy', R[0, 1]), ('ryx', R[1, 0]),
                  ('ryy', R[1, 1]), ('txx', T[0, 0]), ('tyy', T[1, 1])):
        out[nm + '_re'] = float(M.real)
        out[nm + '_im'] = float(M.imag)
    return out


def objective(R, T, method='B', mode_pens=None):
    """Differentiable loss. Primary: maximize |r_cross|^2 (full Jones,
    circular), penalize transmission and co-pol reflection. Smooth in all
    quantities (no wrapped-angle terms: |r_cross|^2 is itself maximal at
    dphi = pi and |rx| = |ry|)."""
    Rc = circular(R)
    Tc = circular(T)
    r_cross2 = torch.abs(Rc[0, 1]) ** 2 + torch.abs(Rc[1, 0]) ** 2
    r_co2 = torch.abs(Rc[0, 0]) ** 2 + torch.abs(Rc[1, 1]) ** 2
    t_tot = (torch.abs(T) ** 2).sum()
    absorb = 2.0 - (torch.abs(R) ** 2).sum() - t_tot
    L = -0.5 * r_cross2 + 0.15 * t_tot + 0.15 * r_co2 + 0.3 * absorb
    if mode_pens is not None:
        L = L + mode_pens
    return L


def softgate(x, gate, width=0.05):
    """Penalty > 0 when x < gate (smooth)."""
    return torch.nn.functional.softplus((gate - x) / width) * width


def moments_families(rho_t, P, H, lam, order, pol, direction='backward',
                     n_xy=32, nz=5):
    """Exact 4-family fractions + key component purities under the given
    polarization and illumination direction (differentiable)."""
    import ed_eq_core as core
    e = eps_asi(lam)
    sim = torcwa.rcwa(freq=1.0 / float(lam), order=list(order),
                      L=[float(P), float(P)], dtype=SIM_DTYPE, device=DEVICE)
    sim.add_input_layer(eps=EPS_GLASS)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    sim.add_layer(thickness=float(H), eps=rho_t * (e - 1.0) + 1.0)
    sim.solve_global_smatrix()
    amp = [1.0, 0.0] if pol == 'x' else [0.0, 1.0]
    sim.source_planewave(amplitude=amp, direction=direction)
    x_ax, z_ax, E, _ = core.fields_3d(sim, float(P), float(H), n_xy, nz)
    Ex, Ey, Ez = E
    n = rho_t.shape[0]
    idx = (torch.floor(x_ax / P * n).long()) % n
    eps3 = (rho_t[idx][:, idx] * (e - 1.0) + 1.0)[:, :, None] \
        .expand(n_xy, n_xy, nz)
    mo = core.torch_moments((Ex, Ey, Ez), eps3, x_ax, z_ax, float(lam))
    Cp, Cm, CQe, CQm = core.family_weights4(mo)
    tot = Cp + Cm + CQe + CQm
    k = mo['k']
    cE = k ** 4 / (6 * math.pi * core.EPS0 ** 2)
    comp = {}
    comp['px_in_ED'] = cE * torch.abs(mo['px']) ** 2 / (Cp + 1e-300)
    comp['py_in_ED'] = cE * torch.abs(mo['py']) ** 2 / (Cp + 1e-300)
    comp['mx_in_MD'] = cE / core.C0 ** 2 * torch.abs(mo['mx']) ** 2 / (Cm + 1e-300)
    comp['my_in_MD'] = cE / core.C0 ** 2 * torch.abs(mo['my']) ** 2 / (Cm + 1e-300)
    return {'f_ED': Cp / tot, 'f_MD': Cm / tot, 'f_EQ': CQe / tot,
            'f_MQ': CQm / tot, **comp, 'mo': mo}
