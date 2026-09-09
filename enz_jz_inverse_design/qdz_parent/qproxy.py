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

Probe frequencies: omega_E + linspace(-span, span, n) * kappa_target with kappa_target = omega_E/Q_target,
so the probe window automatically follows the targeted linewidth.  W_Si (internal modal energy) is used
rather than R or T so that a Fano zero of the far field cannot be mistaken for a high modal Q.

CALIBRATED SPAN (outputs/phaseA/qproxy_span_selection.json, 11 exact poles with Q = 7 - 605 from discs,
squares, filtered-random freeforms and two Jz/Fz finalists).  The closed form is EXACT for an isolated
Lorentzian at any span, so all error is background / neighbouring resonances, and it grows with span:

    span (x kappa)      0.125   0.25    0.5     1.0     2.0
    median |dQ|/Q       0.040   0.047   0.078   0.210   0.427     (centred)
    log-corr(Q_px,Q_ex) 0.996   0.994   0.990   0.677   0.902
    median |dQ|/Q       0.100   0.114   0.148   0.265   0.494     (centre offset kappa/4)

DEFAULT span = 0.25: median 4.7 % error, p90 14 %, 91 % of cases within 20 %, log-correlation 0.994 when
centred; 11 % median with a quarter-linewidth offset; a factor-2 error in Q_target degrades it to 10-15 %.
VALID RANGE: |lambda_r - lambda_E| <~ kappa/4 and Q_target within ~2x of the true Q - precisely the region
the L_lambda and L_Q constraints hold the optimizer in.  W_layer (the Parseval energy) performs the same
to within 0.01 and is reported alongside.  The exact AAA pole Q remains the certification value.
"""
import numpy as np
import torch

C_NM_FS = 299.792458


def omega_of(lam_nm):
    return 2 * np.pi * C_NM_FS / np.asarray(lam_nm, dtype=float)      # rad/fs


def lam_of(omega):
    return 2 * np.pi * C_NM_FS / np.asarray(omega, dtype=float)


def probe_wavelengths(lam_E, Q_target, n=5, span=0.25):
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
    # A resonance requires an upward parabola in 1/W.  Use a SMOOTH positive part (not clamp) so the
    # gradient never dies when the fit is momentarily ill-posed: an ill-posed fit then yields a very
    # large kappa (Q -> 0) and L_Q pushes the design back towards having a peak.
    scale = torch.mean(torch.abs(y)) / max(float(np.ptp(np.asarray(omegas, dtype=float))) ** 2, 1e-30)
    floor = 1e-6 * torch.clamp(scale, min=1e-30)
    c_pos = 0.5 * (c + torch.sqrt(c ** 2 + floor ** 2)) + 1e-30
    x0 = -b / (2 * c_pos)
    y_raw = a - b ** 2 / (4 * c_pos)                       # 1/W at the peak
    y_min = 0.5 * (y_raw + torch.sqrt(y_raw ** 2 + (1e-6 * torch.mean(torch.abs(y))) ** 2)) + 1e-30
    half = torch.sqrt(y_min / c_pos)
    kappa = 2 * half
    omega_r = float(w_E) + x0
    Q = omega_r / torch.clamp(kappa, min=1e-30)
    resid = torch.sqrt(torch.mean(((A @ coef - y) / torch.clamp(y, min=1e-30)) ** 2))
    return dict(omega_r=omega_r, lambda_r=2 * np.pi * C_NM_FS / omega_r, kappa=kappa, Q=Q,
                W_peak=1.0 / y_min, a=a, b=b, c=c, rel_resid=resid,
                well_posed=bool(float(c.detach()) > 0 and float(y_raw.detach()) > 0))


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
    """L_Q = [log(Q / Q_target)]^2 : symmetric in over/under-shoot, no reward for exceeding target.
    Used for REPORTING and for the certification comparison; the optimizer uses the equivalent but
    globally well-defined curvature form in constraint_losses()."""
    return (torch.log(torch.clamp(Q, min=1e-6) / float(Q_target))) ** 2


def lam_loss(lambda_r, lam_E, tol_nm):
    return ((lambda_r - float(lam_E)) / float(tol_nm)) ** 2


def _softlog(u, u0=0.1):
    """log(u) continued linearly below u0, so it is finite and has a strong upward gradient for u <= 0."""
    return torch.where(u >= u0, torch.log(torch.clamp(u, min=u0)), np.log(u0) + (u - u0) / u0)


def constraint_losses(omegas, W, w_E, Q_target, tol_nm, lam_E, weights=None):
    """Q and wavelength constraints expressed on the quadratic fit COEFFICIENTS of 1/W, which stay
    well defined even when the design momentarily has NO peak at omega_E (fitted curvature c <= 0 -
    the regime in which a Q derived as omega/kappa diverges).

        1/W = a + b x + c x^2,  x = omega - omega_E,  a = 1/W(omega_E) > 0
        For a Lorentzian of linewidth kappa_target the curvature at this energy level would be
            c_target = 4 a / kappa_target^2,      u = c / c_target
        and the peak offset is x0 = -b / (2c).

        L_Q   = [softlog(u)]^2 / 4        (EXACTLY [log(Q/Q_target)]^2 when the peak is centred, since
                                           Q/Q_target = sqrt(u) there; softlog continues it linearly
                                           for u <= 0.1 so a design with no resonance is pushed to make one)
        L_lam = [(-b / (2 c_target)) / (omega tolerance)]^2   (the peak-offset penalty with the target
                                           curvature in the denominator: identical to (x0/tol)^2 when the
                                           Q constraint is met, and bounded when c -> 0)

    Returns (L_Q, L_lam, diagnostics).  tol_nm is converted to a frequency tolerance at lambda_E.
    """
    x = torch.as_tensor(np.asarray(omegas, dtype=float) - float(w_E), dtype=W.dtype, device=W.device)
    y = 1.0 / (W + 1e-30)
    A = torch.stack([torch.ones_like(x), x, x ** 2], dim=1)
    w = torch.ones_like(x) if weights is None else torch.as_tensor(np.asarray(weights, float), dtype=W.dtype)
    coef = torch.linalg.lstsq(A * w[:, None], (y * w).unsqueeze(1)).solution.squeeze(1)
    a, b, c = coef[0], coef[1], coef[2]
    kappa_t = float(w_E) / float(Q_target)
    c_t = 4 * torch.clamp(a, min=1e-30) / kappa_t ** 2
    u = c / c_t
    L_Q = _softlog(u) ** 2 / 4.0
    x0_rob = -b / (2 * c_t)
    tol_w = abs(2 * np.pi * C_NM_FS * (1.0 / float(lam_E) - 1.0 / (float(lam_E) + float(tol_nm))))
    L_lam = (x0_rob / tol_w) ** 2
    resid = torch.sqrt(torch.mean(((A @ coef - y) / torch.clamp(y, min=1e-30)) ** 2))
    diag = dict(a=a, b=b, c=c, u=u, c_target=c_t, x0_robust=x0_rob, tol_omega=tol_w,
                rel_resid=resid, kappa_target=kappa_t)
    return L_Q, L_lam, diag
