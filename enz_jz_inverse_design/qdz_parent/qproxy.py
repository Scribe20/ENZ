"""Differentiable radiative-Q proxy for the parent mode, and the exact post-hoc pole certification.

WHY A PROXY.  The trusted pole extractor (analysis.significant_poles) is a scipy AAA rational fit of
r(omega), t(omega) sampled on a wavelength grid: its poles come from an SVD + eigenvalue problem on
NumPy arrays built from many independent RCWA solves.  It is neither differentiable nor cheap, so it
CANNOT be used inside topology optimization.  It is used here only as the exact certification value.

THE PROXY.  The parent (air / a-Si:H / glass) is lossless, so a single resonance makes the driven
internal electric energy W_Si(omega) a Lorentzian on a slowly varying background:

    W(omega) ~ W_pk / [1 + 4 (omega - omega_r)^2 / kappa^2]
    <=>  1/W(omega) = a + b x + c x^2,   x = omega - omega_E      (exactly quadratic in x)

so a WEIGHTED LEAST-SQUARES QUADRATIC FIT OF 1/W over a few probe frequencies gives, in closed form
(hence differentiably, with no eigenvalue solve):

    omega_r = omega_E - b/(2c),   W_pk = 1/(a - b^2/(4c)),   kappa = 2 sqrt((a - b^2/(4c))/c),
    Q_proxy = omega_r / kappa.

Probe frequencies: omega_E + {-1, -1/2, 0, +1/2, +1} * kappa_target with kappa_target = omega_E/Q_target,
i.e. the probe span automatically follows the targeted linewidth.  W_Si (internal modal energy) is used
rather than R or T so that a Fano zero of the far field cannot be mistaken for a high modal Q.

The fit is exact for an isolated Lorentzian; its error against the exact pole Q is measured in
audit_qproxy.py over the structures actually met in this campaign, and the calibration is reported.
"""
import numpy as np
import torch

C_NM_FS = 299.792458


def omega_of(lam_nm):
    return 2 * np.pi * C_NM_FS / np.asarray(lam_nm, dtype=float)      # rad/fs


def lam_of(omega):
    return 2 * np.pi * C_NM_FS / np.asarray(omega, dtype=float)


def probe_wavelengths(lam_E, Q_target, n=5, span=1.0):
    """n probe wavelengths symmetric in FREQUENCY about omega_E, spanning +-span*kappa_target."""
    w_E = omega_of(lam_E)
    k = w_E / float(Q_target)
    offs = np.linspace(-span, span, n) * k
    return lam_of(w_E + offs), w_E + offs, float(k)


def lorentz_fit(omegas, W, w_E, weights=None, eps=1e-30):
    """Closed-form weighted least-squares fit of 1/W to a + b x + c x^2 (x = omega - w_E).
    omegas: (n,) float array; W: (n,) torch tensor (differentiable).  Returns torch scalars."""
    x = torch.as_tensor(np.asarray(omegas, dtype=float) - float(w_E), dtype=W.dtype, device=W.device)
    y = 1.0 / (W + eps)
    A = torch.stack([torch.ones_like(x), x, x ** 2], dim=1)
    if weights is None:
        w = torch.ones_like(x)
    else:
        w = torch.as_tensor(np.asarray(weights, dtype=float), dtype=W.dtype, device=W.device)
    Aw = A * w[:, None]
    coef = torch.linalg.lstsq(Aw, (y * w).unsqueeze(1)).solution.squeeze(1)
    a, b, c = coef[0], coef[1], coef[2]
    c_pos = torch.clamp(c, min=1e-24)                      # a resonance requires an upward parabola in 1/W
    x0 = -b / (2 * c_pos)
    y_min = a - b ** 2 / (4 * c_pos)                       # 1/W at the peak
    y_min = torch.clamp(y_min, min=1e-30)
    half = torch.sqrt(torch.clamp(y_min / c_pos, min=1e-30))
    kappa = 2 * half
    omega_r = float(w_E) + x0
    Q = omega_r / torch.clamp(kappa, min=1e-30)
    resid = torch.sqrt(torch.mean(((A @ coef - y) / torch.clamp(y, min=1e-30)) ** 2))
    return dict(omega_r=omega_r, lambda_r=2 * np.pi * C_NM_FS / omega_r, kappa=kappa, Q=Q,
                W_peak=1.0 / y_min, a=a, b=b, c=c, rel_resid=resid,
                well_posed=bool(float(c) > 0 and float(y_min) > 0))


def shape_loss(omegas, W, w_E, kappa_target):
    """The alternative formulation of the constraint: compare the CENTER-NORMALIZED energy spectrum
    with the target Lorentzian L(omega) = 1/[1 + 4(omega-omega_E)^2/kappa_target^2].  Reported next to
    the fit-based losses; it constrains linewidth and alignment jointly but gives no explicit Q."""
    x = torch.as_tensor(np.asarray(omegas, dtype=float) - float(w_E), dtype=W.dtype, device=W.device)
    ic = int(np.argmin(np.abs(np.asarray(omegas, dtype=float) - float(w_E))))
    Wn = W / torch.clamp(W[ic], min=1e-30)
    L = 1.0 / (1.0 + 4 * x ** 2 / float(kappa_target) ** 2)
    return torch.mean((Wn - L) ** 2), Wn, L


def q_loss(Q, Q_target):
    """L_Q = [log(Q / Q_target)]^2 : symmetric in over/under-shoot, no reward for exceeding target."""
    return (torch.log(torch.clamp(Q, min=1e-6) / float(Q_target))) ** 2


def lam_loss(lambda_r, lam_E, tol_nm):
    return ((lambda_r - float(lam_E)) / float(tol_nm)) ** 2
