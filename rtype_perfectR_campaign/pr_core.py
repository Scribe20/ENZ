"""Perfect-R ceiling campaign core (633 nm ideal reflective PB operator).

Target operator (circular basis, alpha = 0, validated handedness):
    R_ideal = exp(i psi) [[0, 1], [1, 0]],  T_ideal = 0.
Global-phase-invariant coherent fidelity:
    F_ideal = |Tr(U^dag R_circ)|^2 / 4 = |Rc[0,1] + Rc[1,0]|^2 / 4.
F_ideal = 1 only for the unit-amplitude half-wave reflector (checked
numerically in pr_validate.py, including the rotated-operator sign
against the measured -2 alpha law of the previous campaign).

Design-space changes vs previous campaigns:
  - fixed tighter rotation-safe padding: 15 nm (never swept),
    r_design = P/2 - 15
  - three symmetry branches: D2 (x+y mirrors), C2 (180-deg rotation
    only), FULL (none); C2/FULL use the full Jones matrix everywhere
  - multiple internal islands allowed (1-3) during discovery
  - no multipole terms anywhere in the objective (post-hoc only)
  - constraint-continuation on T_tot and R_co with multiplier growth;
    NO independent absorption penalty (F -> 1 penalizes A implicitly).
All conventions (materials, exact p/s oblique Jones, angle mapping)
inherited from the validated wf_core/rt_core stack.
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'rtype_widefov_campaign'))
sys.path.insert(0, str(HERE.parent / 'rtype_freeform_campaign'))
import wf_core as wf                                        # noqa: E402
import rt_core as rc                                        # noqa: E402
import torcwa                                               # noqa: E402

LAM0 = rc.LAM0
PAD = 15.0
DEVICE_P = [226.0, 239.0, 252.0, 264.0, 272.0, 278.0]
CEIL_P = [300.0, 330.0, 360.0, 400.0]
HEIGHTS = [140.0, 170.0, 200.0, 230.0, 260.0, 290.0]


def r_design(P):
    return P / 2 - PAD


def design_mask(n, P):
    ax = (np.arange(n) + 0.5) / n * P - P / 2
    X, Y = np.meshgrid(ax, ax, indexing='ij')
    return torch.tensor((X ** 2 + Y ** 2 <= r_design(P) ** 2)
                        .astype(np.float32))


def symmetrize(rho, branch):
    if branch == 'D2':
        return rc.d2_symmetrize(rho)
    if branch == 'C2':
        return 0.5 * (rho + torch.flip(rho, [0, 1]))
    return rho


def filt_project(rho, kern, beta, mask, branch, eta=0.5):
    rho = symmetrize(rho, branch)
    rho = rho * mask
    pad = kern.shape[-1] // 2
    f = torch.nn.functional.conv2d(rho[None, None], kern,
                                   padding=pad)[0, 0]
    be = torch.tensor(float(beta * eta))
    num = torch.tanh(be) + torch.tanh(beta * (f - eta))
    den = torch.tanh(be) + torch.tanh(torch.tensor(float(beta
                                                         * (1 - eta))))
    return (num / den) * mask


def jones_theta0(rho, P, H, order=(7, 7), eps_override=None):
    e = rc.eps_asi() if eps_override is None else eps_override
    sim = torcwa.rcwa(freq=1.0 / LAM0, order=list(order),
                      L=[float(P), float(P)], dtype=rc.SIM_DTYPE,
                      device=rc.DEVICE)
    sim.add_input_layer(eps=rc.EPS_GLASS)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    sim.add_layer(thickness=float(H), eps=rho * (e - 1.0) + 1.0)
    sim.solve_global_smatrix()
    return rc.jones(sim, 'backward')


def fidelity(Rj, alpha_deg=0.0):
    """Coherent global-phase-invariant fidelity to the rotated ideal
    PB half-wave reflector. Validated sign: Rc[0,1] carries e^{-2ia}."""
    Rc = rc.circular(Rj)
    a = math.radians(alpha_deg)
    tr = (torch.exp(torch.tensor(2j * a)) * Rc[0, 1]
          + torch.exp(torch.tensor(-2j * a)) * Rc[1, 0])
    return torch.abs(tr) ** 2 / 4.0


def port_metrics(Rj, Tj):
    """All perfect-R scalar metrics (differentiable pieces as tensors)."""
    Rc = rc.circular(Rj)
    F = torch.abs(Rc[0, 1] + Rc[1, 0]) ** 2 / 4.0
    # axis-orientation-invariant fidelity = max_alpha F(U_alpha): equals
    # F for D2 (Rc01 = Rc10); for C2/FULL it credits a motif whose
    # principal axes are rotated from x/y, and still requires |Rc01| =
    # |Rc10| = 1 (linear eigenaxes, unit amplitude) to reach 1.
    Faf = (torch.abs(Rc[0, 1]) + torch.abs(Rc[1, 0])) ** 2 / 4.0
    P_T = 0.5 * (torch.abs(Tj) ** 2).sum()
    P_Rco = 0.5 * (torch.abs(Rc[0, 0]) ** 2 + torch.abs(Rc[1, 1]) ** 2)
    R_tot = 0.5 * (torch.abs(Rj) ** 2).sum()
    A = 1.0 - R_tot - P_T
    Rcross = 0.5 * (torch.abs(Rc[0, 1]) ** 2 + torch.abs(Rc[1, 0]) ** 2)
    return {'F': F, 'Faf': Faf, 'T': P_T, 'co': P_Rco, 'Rtot': R_tot,
            'A': A, 'Rcross': Rcross}


def soft_over(x, cap, width=0.02):
    return torch.nn.functional.softplus((x - cap) / width) * width


def pr_loss(m, frac, lamT, lamCo, capT, capCo, branch='D2'):
    """Staged continuation loss (spec sec 19): phase-1 mirror formation
    ramps into full ideal-operator fidelity; augmented penalties on the
    transmission and co-pol caps (multipliers grown externally when
    violated). No absorption term. Non-D2 branches use the
    axis-invariant fidelity."""
    wF = min(1.0, max(0.15, frac / 0.25))
    Fobj = m['F'] if branch == 'D2' else m['Faf']
    L = -(1.0 - wF) * m['Rtot'] - wF * Fobj
    L = L + lamT * soft_over(m['T'], capT) \
        + lamCo * soft_over(m['co'], capCo)
    return L


def scalars(m):
    return {k: float(v) for k, v in m.items()}


def eval_full(rho, P, H, order=(9, 9), eps_override=None):
    """Hard-geometry evaluation at theta = 0: fidelity + channels +
    principal amplitudes/phase (works for non-diagonal Jones too)."""
    with torch.no_grad():
        Rj, Tj = jones_theta0(rho, P, H, order, eps_override)
    m = scalars(port_metrics(Rj, Tj))
    rx, ry = Rj[0, 0], Rj[1, 1]
    dphi = float(torch.rad2deg(torch.angle(rx) - torch.angle(ry)))
    err = abs(((dphi - 180.0 + 180.0) % 360.0) - 180.0)
    m.update({'abs_rx': float(torch.abs(rx)), 'abs_ry': float(torch.abs(ry)),
              'dphi_deg': dphi, 'phase_err_deg': err,
              'offdiag': float(torch.abs(Rj[0, 1]) + torch.abs(Rj[1, 0]))})
    return m
