"""Parent-mode forward model and metrics for the Q_r / eta_Dz campaign.

PARENT SYSTEM (no ITO):   air / freeform a-Si:H (h, rho) / soda-lime glass.
Both dielectrics are lossless over 1150-1400 nm in the supplied data (max |k| = 0 for a-Si:H,
1e-263 for the glass), so R + T = 1 to 1e-6 and every pole width is purely RADIATIVE.

PROVENANCE.  Everything physical is reused from the validated Jz/Fz package one directory up:
    forward.build_sim(..., with_ito=False)   air / a-Si:H(h, rho) / glass, x-pol, normal incidence
    forward.rt_all_orders, forward.p_inc_cell, forward.build_pad_mask, forward.cell_axes
    materials.eps_asi / n_glass / ito_zero_crossing  (authoritative lambda_E)
    optimizer.gaussian_kernel_fft / filter_rho / project_rho / preprocess / binarization_metric
    analysis.significant_poles / rayleigh_wavelengths / spec_arrays   (exact post-hoc poles)
Only the *new* quantities below are implemented here.

FIELD EXTRACTION.  The Fourier coefficients of an internal layer are taken with the same linear
algebra as torcwa.rcwa.field_xy (vendored rcwa.py l. 1057-1102), i.e. the routine already used by
forward.ito_fourier_coeffs, but generalised to any layer and additionally returning

    Dz_mn / eps0 = (Ky_norm Hx_mn - Kx_norm Hy_mn)          [torcwa units: eps0 = mu0 = c = 1]

which is exactly the quantity torcwa multiplies by eps_conv^-1 to obtain Ez.  Taking it BEFORE the
inverse convolution avoids the inverse-rule Gibbs error and gives the interface-continuous normal
displacement directly.  Real-space maps are synthesised with an exact zero-padded inverse FFT
(the truncated Fourier series has |m|, |n| <= 7 << nx/2 = 64, so the ifft2 is exact, not aliased).

METRICS (all differentiable w.r.t. rho).

  eta_Dz  =  d_ITO * INT_Acontact rho(x, y) |Dz(x, y, z_int)/eps0|^2 dA
             ------------------------------------------------------------
                    INT_VSi rho(x, y) eps_Si |E(x, y, z)|^2 dV

    z_int  = bottom face of the a-Si:H (the prospective Si/ITO interface), evaluated just inside
             the Si layer; d_ITO = 23 nm is the prospective ITO thickness, so the numerator is the
             |Dz/eps0|^2 integral over the *volume the ITO would occupy* under the contact area.
    rho    = the differentiable design density (the projected density during optimization, the
             hard-binary mask at certification), so only the actual Si-contact region contributes.
    Units  : nm * (field)^2 * nm^2 / ((field)^2 * nm^3) = dimensionless; invariant under any
             rescaling of the field amplitude (numerator and denominator are both quadratic).

  U_mid   =  INT rho eps_Si |E|^2 dV over 0.3 <= z/h <= 0.7, divided by that sub-volume
             (diagnostic only; a spatially averaged central-slab energy density, never a point).

  W_Si    =  the eta_Dz denominator (driven electric energy scale of the parent mode in the Si);
             used as the spectral quantity of the differentiable Q proxy (qproxy.py).
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))
import config                      # noqa: E402
import forward as fwd              # noqa: E402
import materials as mat            # noqa: E402

DEV = config.DEVICE
GEO = config.GEO_DTYPE
SIM = config.SIM_DTYPE
D_ITO_NM = config.D_ITO_NM
ASI_LAYER = 0                      # the freeform a-Si:H layer is the first internal layer
C_NM_FS = 299.792458


def lambda_E():
    """Authoritative ENZ wavelength: the eps' = 0 crossing of the supplied ITO data."""
    lam, _ = mat.ito_zero_crossing()
    return float(lam)


# ---------------------------------------------------------------------------
# layer fields
# ---------------------------------------------------------------------------
def output_fourier(sim, zp=0.0):
    """(Ex_mn, Ey_mn, Ez_mn, Dz_mn) in the homogeneous OUTPUT half-space (glass) at depth zp below
    its top face.  Same algebra as the torcwa.rcwa.field_xy input/output branch (rcwa.py l. 995-1045)
    for layer_num == layer_N with source_direction 'forward'.  Because the substrate is homogeneous,
    eps is a scalar here: Dz = eps_out * Ez carries NO inverse-convolution (Gibbs) error, and the
    normal displacement is continuous across the horizontal Si/glass interface, so this is the
    interface Dz of the parent mode."""
    Kz = torch.sqrt(sim.eps_out * sim.mu_out - sim.Kx_norm_dn ** 2 - sim.Ky_norm_dn ** 2)
    Kz = torch.where(torch.imag(Kz) < 0, torch.conj(Kz), Kz).reshape([-1, 1])
    Kz = torch.vstack((Kz, Kz))
    Exy = torch.matmul(sim.S[0], sim.E_i) * torch.exp(1j * sim.omega * Kz * zp)
    Vo = sim.Vo if hasattr(sim, "Vo") else sim.Vf
    Hxy = torch.matmul(Vo, Exy)
    N = sim.order_N
    Ex, Ey = Exy[:N, 0], Exy[N:, 0]
    Hx, Hy = Hxy[:N, 0], Hxy[N:, 0]
    Dz = torch.matmul(sim.Ky_norm, Hx) - torch.matmul(sim.Kx_norm, Hy)     # = eps_out * Ez (scalar eps)
    Ez = Dz / sim.eps_out
    return Ex, Ey, Ez, Dz


def layer_fourier(sim, zp, layer=ASI_LAYER):
    """(Ex_mn, Ey_mn, Ez_mn, Dz_mn) of the total field in an internal layer at distance zp from its
    input-side (top) boundary.  Same algebra as torcwa.rcwa.field_xy internal branch / the
    validated forward.ito_fourier_coeffs, with Dz_mn = Ky Hx_mn - Kx Hy_mn returned as well."""
    N = sim.order_N
    C = torch.matmul(sim.C[0][layer] if sim.source_direction == "forward" else sim.C[1][layer], sim.E_i)
    kz = sim.kz_norm[layer]
    Ev, Hv = sim.E_eigvec[layer], sim.H_eigvec[layer]
    a_p = torch.exp(1j * sim.omega * kz * zp) * C[:2 * N, 0]
    a_m = torch.exp(1j * sim.omega * kz * (sim.thickness[layer] - zp)) * C[2 * N:, 0]
    Exy = torch.matmul(Ev, a_p) + torch.matmul(Ev, a_m)
    Hxy = torch.matmul(Hv, a_p) - torch.matmul(Hv, a_m)
    Ex, Ey = Exy[:N], Exy[N:]
    Hx, Hy = Hxy[:N], Hxy[N:]
    Dz = torch.matmul(sim.Ky_norm, Hx) - torch.matmul(sim.Kx_norm, Hy)     # = eps_conv @ Ez
    Ez = torch.linalg.solve(sim.eps_conv[layer], Dz)
    return Ex, Ey, Ez, Dz


def to_real(coeff, sim, nx):
    """Exact real-space synthesis of a truncated Fourier series on an nx x nx cell grid via a
    zero-padded inverse FFT (equivalent to torcwa's explicit phase sum, but O(nx^2 log nx))."""
    ox = sim.order[0] if hasattr(sim, "order") else None
    F = torch.zeros((nx, nx), dtype=coeff.dtype, device=coeff.device)
    mx = sim.order_x.to(torch.long) if torch.is_tensor(sim.order_x) else torch.as_tensor(sim.order_x)
    my = sim.order_y.to(torch.long) if torch.is_tensor(sim.order_y) else torch.as_tensor(sim.order_y)
    ix, iy = torch.meshgrid(mx % nx, my % nx, indexing="ij")
    F = F.index_put((ix.reshape(-1), iy.reshape(-1)), coeff.reshape(-1), accumulate=True)
    del ox
    return torch.fft.ifft2(F) * (nx * nx)


def asi_z_slices(h, n_z):
    """Midpoint z_prop samples inside the a-Si:H layer (z_prop measured from the TOP face)."""
    return (np.arange(n_z) + 0.5) * h / n_z


# ---------------------------------------------------------------------------
# parent metrics
# ---------------------------------------------------------------------------
def parent_metrics(rho, P, h, lam, order, *, nx=None, n_z=9, with_grad=False, rho_metric=None,
                   want_maps=False, eps_asi=None, n_glass=None, z_int_eps=1e-9):
    """Solve the parent (air / a-Si:H(rho) / glass) and return the differentiable metrics.

    rho        : density that BUILDS the structure (projected density or hard mask)
    rho_metric : density that WEIGHTS the metric integrals (defaults to rho; the audit uses a
                 separate mask to show that eta_Dz only samples the Si-contact region)
    Returns a dict of torch scalars (eta_Dz, W_Si, S_Dz, U_mid, ...) plus R, T.
    """
    nx = config.NX if nx is None else nx
    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    rw = rho if rho_metric is None else rho_metric
    with ctx:
        eps_a = mat.eps_asi(lam) if eps_asi is None else eps_asi
        ng = mat.n_glass(lam) if n_glass is None else n_glass
        sim = fwd.build_sim(rho, P, h, lam, order, with_ito=False, eps_asi=eps_a, n_glass=ng)
        R, T = fwd.rt_all_orders(sim)
        dA = (P / nx) ** 2
        eps_si = float(np.real(eps_a))

        # ---- interface term: |Dz/eps0|^2 at the prospective Si/ITO interface, weighted by the Si
        # contact density.  Dz is continuous across that horizontal interface and is evaluated on the
        # GLASS side (homogeneous -> no inverse-convolution error); the in-Si value is kept as an audit.
        _, _, _, Dz_mn = output_fourier(sim, 0.0)
        Dz = to_real(Dz_mn, sim, nx)
        Dz2 = Dz.real ** 2 + Dz.imag ** 2
        S_Dz = D_ITO_NM * (rw * Dz2).sum() * dA                       # nm * field^2 * nm^2
        _, _, Ez_si_mn, Dz_si_mn = layer_fourier(sim, float(h) - z_int_eps)
        Dz_si = to_real(Dz_si_mn, sim, nx)
        S_Dz_inSi = D_ITO_NM * (rw * (Dz_si.real ** 2 + Dz_si.imag ** 2)).sum() * dA

        # ---- volume term: rho eps_Si |E|^2 over the Si, and the central-slab diagnostic
        zs = asi_z_slices(float(h), n_z)
        dz = float(h) / n_z
        w_slices, u_slices, ez_slices = [], [], []
        for zp in zs:
            Ex_mn, Ey_mn, Ez_mn, _ = layer_fourier(sim, float(zp))
            E2 = torch.zeros((nx, nx), dtype=torch.float64, device=Dz.device)
            for c in (Ex_mn, Ey_mn, Ez_mn):
                Ec = to_real(c, sim, nx)
                E2 = E2 + Ec.real ** 2 + Ec.imag ** 2
            w_slices.append((rw * E2).sum() * dA * dz * eps_si)
            u_slices.append(E2)
            if want_maps:
                ez_slices.append(to_real(Ez_mn, sim, nx))
        W_Si = torch.stack(w_slices).sum()                            # field^2 * nm^3
        mid = [i for i, z in enumerate(zs) if 0.3 <= z / float(h) <= 0.7]
        V_mid = (rw.sum() * dA) * (len(mid) * dz) if mid else None
        U_mid = (torch.stack([(rw * u_slices[i]).sum() * dA * dz * eps_si for i in mid]).sum() / V_mid) if mid else torch.tensor(float("nan"))
        eta_Dz = S_Dz / W_Si

        # ---- Parseval alternative denominator: the full-cell electric energy of the design layer,
        # W_layer = INT_cell eps(x,y) |E|^2 dV = P^2 sum_mn Re[E*_mn . D_mn]  (exact, no real-space
        # synthesis, no Gibbs error).  Reported alongside as eta_Dz_P = S_Dz / W_layer.
        w_par = []
        for zp in zs:
            Ex_mn, Ey_mn, Ez_mn, Dz_mn_l = layer_fourier(sim, float(zp))
            Dx_mn = torch.matmul(sim.eps_conv[ASI_LAYER], Ex_mn)
            Dy_mn = torch.matmul(sim.eps_conv[ASI_LAYER], Ey_mn)
            w_par.append((torch.conj(Ex_mn) * Dx_mn + torch.conj(Ey_mn) * Dy_mn
                          + torch.conj(Ez_mn) * Dz_mn_l).sum().real)
        W_layer = torch.stack(w_par).sum() * (P ** 2) * dz
        out = dict(eta_Dz=eta_Dz, eta_Dz_P=S_Dz / W_layer, W_layer=W_layer,
                   S_Dz_inSi=S_Dz_inSi, S_Dz_ratio_glass_over_Si=S_Dz / S_Dz_inSi,
                   S_Dz=S_Dz, W_Si=W_Si, U_mid=U_mid, R=R, T=T,
                   mean_Dz2_contact=(rw * Dz2).sum() / rw.sum().clamp(min=1e-30),
                   max_Dz2_contact=(rw * Dz2).max(), mean_Ez2_contact=None, sim=sim,
                   eps_si=eps_si, n_glass=float(ng), lam=float(lam), n_z=int(n_z))
        if want_maps:
            out["Dz_map"] = Dz
            out["Ez_slices"] = torch.stack(ez_slices)
            out["E2_slices"] = torch.stack(u_slices)
            out["z_slices_nm"] = zs
    return out


def to_floats(d):
    return {k: (float(v.detach()) if torch.is_tensor(v) and v.numel() == 1 else v)
            for k, v in d.items() if k not in ("sim", "Dz_map", "Ez_slices", "E2_slices")
            and ((torch.is_tensor(v) and v.numel() == 1) or isinstance(v, (int, float)))}


def spectrum_parent(rho, P, h, lams, order, *, nx=None, n_z=9, rho_metric=None):
    """R, T, W_Si, eta_Dz, U_mid vs wavelength for the parent (no gradient)."""
    rows = []
    for lam in np.atleast_1d(lams):
        d = parent_metrics(rho, P, h, float(lam), order, nx=nx, n_z=n_z, rho_metric=rho_metric)
        f = to_floats(d)
        r, t = fwd.specular_rt_amplitudes(d["sim"])
        f.update(lam=float(lam), r=complex(r), t=complex(t))
        rows.append(f)
    return rows
