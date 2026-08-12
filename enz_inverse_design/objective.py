"""The ENZ-overlap objective (Phase-2 primary figure of merit).

    a_ENZ(rho) = integral_ITO  conj(Ez_target) * Ez_scat(rho)  dV
    F_ENZ(rho) = |a_ENZ|^2 / P_inc_cell

Coherent complex overlap: the conjugate is taken on the target and NO
absolute value is applied before the integration.  Discrete form (uniform
Cartesian sampling, midpoint rule in z):

    a_ENZ = sum_{k,i,j} conj(T[k,i,j]) * Ez_scat[k,i,j] * dx*dy*dz

Everything here is pure torch on complex tensors -> differentiable.
"""

import torch


def overlap_amplitude(target, ez_scat, dV):
    """a = sum(conj(T) * Ez_scat) * dV   (complex scalar tensor)."""
    return torch.sum(torch.conj(target) * ez_scat) * dV


def fom_from_amplitude(a, p_inc):
    """F = |a|^2 / P_inc, implemented as re^2 + im^2 (differentiable)."""
    return (a.real ** 2 + a.imag ** 2) / p_inc


def enz_objective(target_plus, ez_scat, dV, p_inc, target_minus=None,
                  direction="+x"):
    """Primary objective and diagnostics.

    Returns (F_primary, dict of diagnostics incl. a_plus / a_minus).
    direction "+x" (default): F = |a_plus|^2 / P_inc.
    direction "bidir" (explicit config option, not silently substituted):
    F = (|a_plus|^2 + |a_minus|^2) / P_inc.
    """
    a_plus = overlap_amplitude(target_plus, ez_scat, dV)
    diags = {"a_plus": a_plus}
    if target_minus is not None:
        a_minus = overlap_amplitude(target_minus, ez_scat, dV)
        diags["a_minus"] = a_minus

    if direction == "bidir":
        if target_minus is None:
            raise ValueError("bidir objective requires target_minus")
        F = fom_from_amplitude(a_plus, p_inc) + fom_from_amplitude(
            diags["a_minus"], p_inc)
    elif direction in ("+x", "-x"):
        F = fom_from_amplitude(a_plus, p_inc)
    else:
        raise ValueError(f"unknown target direction {direction}")
    return F, diags
