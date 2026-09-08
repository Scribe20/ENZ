"""TORCWA forward adapter for the JZ F_z campaign.

Reuses the VALIDATED conventions of the repository (functions copied with
attribution from enz_robust_aito_campaign/forward_multi.py and
enz_highq_driven_ez_audit/common.py, which were validated there by
energy-conservation, volume-loss-identity and finite-difference checks) but
with the NEW materials (materials.py) and per-call P, h, lambda, order.

Conventions (torcwa, vendored upstream, untouched):
  * Lorentz-Heaviside units, c = eps0 = mu0 = 1, exp(-j w t)
  * source_planewave(amplitude=[1,0], notation='ps') at normal incidence =
    x-polarized plane wave with |E_inc| = 1 (validated in the repo preflight)
  * layer 0 = a-Si design layer, layer 1 = ITO (when present);
    field_xy(1, x, y, z_prop) with z_prop measured from the ITO/a-Si side
    boundary (0 <= z_prop <= d_ITO); midpoint quadrature never touches the
    interfaces
  * R_tot/T_tot sum ALL propagating orders and both output polarizations
"""

import sys

import numpy as np
import torch

import config
import materials as mat

sys.path.insert(0, str(config.TORCWA_DIR))
import torcwa                                  # noqa: E402  (vendored upstream)
from torcwa.torch_eig import Eig as _Eig       # noqa: E402

DEVICE, SIM_DTYPE, GEO_DTYPE = config.DEVICE, config.SIM_DTYPE, config.GEO_DTYPE
D_ITO = config.D_ITO_NM

# ---------------------------------------------------------------------------
# Memory-leak patch for long single-process campaigns (verbatim logic of
# enz_robust_aito_campaign/forward_multi.py): the upstream Eig.forward keeps
# its own outputs on ctx, creating an uncollectable reference cycle.  Storing
# detached copies is numerically identical (the backward only reads values).
# The vendored package file is NOT modified.
# ---------------------------------------------------------------------------
_eig_backward_orig = _Eig.backward.__func__ if hasattr(_Eig.backward, "__func__") else _Eig.backward


def _eig_forward_noleak(ctx, x):
    ctx.input_is_complex = torch.is_complex(x)
    eigval, eigvec = torch.linalg.eig(x)
    ctx.eigval = eigval.detach().cpu()
    ctx.eigvec = eigvec.detach().cpu()
    return eigval, eigvec


def _eig_backward_noleak(ctx, grad_eigval, grad_eigvec):
    class _Ctx:
        pass
    c = _Ctx()
    c.eigval, c.eigvec = ctx.eigval, ctx.eigvec
    c.input = torch.zeros((), dtype=torch.complex128 if ctx.input_is_complex else torch.float64)
    return _eig_backward_orig(c, grad_eigval, grad_eigvec)


_Eig.forward = staticmethod(_eig_forward_noleak)
_Eig.backward = staticmethod(_eig_backward_noleak)


def set_threads(n):
    torch.set_num_threads(int(n))


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def cell_axes(P, n):
    """Cell-centred sample coordinates (nm): x_i = (i + 1/2) P / n."""
    x = (float(P) / n) * (torch.arange(n, dtype=GEO_DTYPE, device=DEVICE) + 0.5)
    return x, x.clone()


def ito_z_slices(n_z, d=D_ITO):
    """Midpoint z positions strictly inside the ITO (interfaces excluded)."""
    return (np.arange(n_z) + 0.5) * d / n_z


def build_pad_mask(nx, P, pad_frac):
    """Hard air ring (enz_robust_aito_campaign/optimizer.build_pad_mask):
    pixel centres x_i = (i+0.5) P/nx are active iff pad <= x_i <= P - pad."""
    dx = P / nx
    xc = (np.arange(nx) + 0.5) * dx
    pad = pad_frac * P
    m = (xc >= pad) & (xc <= P - pad)
    M = (m[:, None] & m[None, :]).astype(float)
    ix = np.where(m)[0]
    info = dict(P_nm=float(P), pad_frac=float(pad_frac), requested_pad_nm=float(pad), dx_nm=float(dx),
                active_index_range=[int(ix[0]), int(ix[-1])], active_pixels=int(len(ix)),
                realized_pad_nm=float(xc[ix[0]] - dx / 2),
                realized_pad_frac=float((xc[ix[0]] - dx / 2) / P),
                active_width_nm=float(len(ix) * dx))
    return M, info


def lab_x_amplitude_ps(theta_deg, phi_deg):
    """[a_p, a_s] of the lab-x polarization in TORCWA's p/s basis (validated
    in enz_robust_aito_campaign).  At theta = 0: pure p = x."""
    th, ph = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    if theta_deg == 0.0:
        return [1.0, 0.0]
    a_p, a_s = np.cos(th) * np.cos(ph), -np.sin(ph)
    n = np.hypot(a_p, a_s)
    return [float(a_p / n), float(a_s / n)]


# ---------------------------------------------------------------------------
# forward solve
# ---------------------------------------------------------------------------
def build_sim(rho, P, h, lam, order, *, with_ito=True, ito_loss_scale=1.0,
              eps_asi=None, eps_ito=None, n_glass=None, theta_deg=0.0,
              phi_deg=0.0, pol="labx", d_ito=D_ITO):
    """Solve air / a-Si(h, rho) / [ITO d_ito] / glass for one design.

    rho: (nx, ny) tensor in [0, 1] on the normalized cell, or None for an
    unpatterned eps = 1 layer of height h (bare reference with identical
    layer indexing).  Materials default to the supplied files at lam.
    Returns the solved sim with the plane-wave source set.
    """
    eps_asi = mat.eps_asi(lam) if eps_asi is None else eps_asi
    eps_ito = mat.eps_ito(lam, ito_loss_scale) if eps_ito is None else eps_ito
    n_glass = mat.n_glass(lam) if n_glass is None else n_glass
    sim = torcwa.rcwa(freq=1.0 / float(lam), order=list(order), L=[float(P), float(P)],
                      dtype=SIM_DTYPE, device=DEVICE)
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=complex(n_glass) ** 2)
    sim.set_incident_angle(inc_ang=np.deg2rad(theta_deg), azi_ang=np.deg2rad(phi_deg))
    if rho is None:
        sim.add_layer(thickness=float(h), eps=1.0)
    else:
        eps_layer = rho * (complex(eps_asi) - 1.0) + 1.0
        sim.add_layer(thickness=float(h), eps=eps_layer.to(SIM_DTYPE))
    if with_ito:
        sim.add_layer(thickness=float(d_ito), eps=complex(eps_ito))
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
    sim._P, sim._h, sim._lam, sim._d_ito = float(P), float(h), float(lam), float(d_ito)
    sim._eps_ito, sim._eps_asi, sim._n_glass = complex(eps_ito), complex(eps_asi), float(n_glass)
    sim._with_ito = with_ito
    return sim


def all_orders(sim):
    ox, oy = torch.meshgrid(sim.order_x, sim.order_y, indexing="ij")
    return torch.stack((ox.reshape(-1), oy.reshape(-1)), dim=1)


def rt_all_orders(sim, per_order=False):
    """Power R_tot, T_tot over ALL propagating orders and both output
    polarizations (torcwa zeroes evanescent orders).  Differentiable."""
    orders = all_orders(sim)
    a_p, a_s = sim._amp_ps
    out = {}
    for port in ("reflection", "transmission"):
        S = {pp: sim.S_parameters(orders=orders, direction="forward", port=port,
                                  polarization=pp, ref_order=[0, 0], power_norm=True)
             for pp in ("pp", "ps", "sp", "ss")}
        E_p = S["pp"] * a_p + S["ps"] * a_s
        E_s = S["sp"] * a_p + S["ss"] * a_s
        out[port] = torch.abs(E_p) ** 2 + torch.abs(E_s) ** 2
    R, T = torch.sum(out["reflection"]), torch.sum(out["transmission"])
    if per_order:
        tab = [dict(m=int(orders[i, 0]), n=int(orders[i, 1]),
                    R=float(out["reflection"][i]), T=float(out["transmission"][i]))
               for i in range(orders.shape[0])
               if float(out["reflection"][i]) > 0 or float(out["transmission"][i]) > 0]
        return R, T, tab
    return R, T


def specular_rt_amplitudes(sim):
    """Complex r_xx, t_xx of the (0,0) order (power_norm=False): analytic in
    omega, used for pole extraction (as enz_highq_driven_ez_audit/poles.py)."""
    r = sim.S_parameters(orders=[0, 0], direction="forward", port="reflection",
                         polarization="xx", ref_order=[0, 0], power_norm=False)
    t = sim.S_parameters(orders=[0, 0], direction="forward", port="transmission",
                         polarization="xx", ref_order=[0, 0], power_norm=False)
    return complex(r.ravel()[0]), complex(t.ravel()[0])


def p_inc_cell(P, theta_deg=0.0):
    """Incident power per cell, LH units, |E_inc| = 1, air input:
    P_inc = 0.5 n_in cos(theta) |E0|^2 P^2."""
    return 0.5 * np.cos(np.deg2rad(theta_deg)) * float(P) ** 2


ITO_LAYER = 1


def ito_fields(sim, nx=None, n_z=None, x=None, y=None):
    """REAL-SPACE complex E inside the ITO at n_z midpoint slices via the
    upstream field_xy: tensor (n_z, 3, nx, ny) (differentiable, but ~10x
    slower than the Fourier route; used for maps and cross-checks)."""
    nx = config.NX if nx is None else nx
    n_z = config.Z_SAMPLES_ITO if n_z is None else n_z
    if x is None:
        x, y = cell_axes(sim._P, nx)
    E = []
    for zp in ito_z_slices(n_z, sim._d_ito):
        Ev, _ = sim.field_xy(ITO_LAYER, x, y, float(zp))
        E.append(torch.stack(list(Ev), 0))
    return torch.stack(E, 0)


def ito_fourier_coeffs(sim, zp, layer=ITO_LAYER):
    """Fourier coefficients (Ex_mn, Ey_mn, Ez_mn), each of length order_N, of
    the TOTAL field in internal layer `layer` at distance zp from its
    input-side boundary.  Same linear algebra as the internal-layer branch of
    torcwa.rcwa.field_xy (vendored rcwa.py lines ~1040-1090), before the
    real-space synthesis:  Ez_mn = eps_conv^-1 (Ky Hx_mn - Kx Hy_mn).
    Differentiable."""
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
    eps_inv = torch.linalg.inv(sim.eps_conv[layer])
    Ez = torch.matmul(eps_inv, torch.matmul(sim.Ky_norm, Hx) - torch.matmul(sim.Kx_norm, Hy))
    return Ex, Ey, Ez


def ito_fourier_intensities(sim, n_z=None):
    """Plane-integrated intensities per z slice by Parseval:
    int_cell |E_i|^2 dA = P^2 sum_mn |E_i,mn|^2  (exact for the periodic
    Bloch field; identical to the 128-grid real-space sum when the grid
    resolves 2*order harmonics).  Returns (n_z, 3) tensor of int|E_i|^2 dA."""
    n_z = config.Z_SAMPLES_ITO if n_z is None else n_z
    rows = []
    for zp in ito_z_slices(n_z, sim._d_ito):
        Ex, Ey, Ez = ito_fourier_coeffs(sim, float(zp))
        rows.append(torch.stack([(Ex.real ** 2 + Ex.imag ** 2).sum(),
                                 (Ey.real ** 2 + Ey.imag ** 2).sum(),
                                 (Ez.real ** 2 + Ez.imag ** 2).sum()]))
    return torch.stack(rows) * sim._P ** 2


def loss_components(sim, nx=None, n_z=None, method="fourier", E=None):
    """Component-resolved ITO absorbed-power fractions

        F_i = (w/2) Im[eps_ITO] int_ITO |E_i|^2 dV / P_inc,   i = x, y, z

    plus F_tot = Fx+Fy+Fz and <|Ez/E_inc|^2>_ITO (mean over the ITO volume).
    method 'fourier' (default, exact Parseval in-plane integral, midpoint in
    z) or 'realspace' (upstream field_xy on the nx grid; also returns
    max|Ez|^2).  All tensors stay differentiable (F_z is the objective)."""
    nx = config.NX if nx is None else nx
    n_z = config.Z_SAMPLES_ITO if n_z is None else n_z
    dz = sim._d_ito / n_z
    omega = 2 * np.pi / sim._lam
    V = sim._P ** 2 * sim._d_ito
    out = {}
    if method == "fourier":
        I = ito_fourier_intensities(sim, n_z)            # (n_z, 3): int|E_i|^2 dA
        Ix, Iy, Iz = I[:, 0].sum() * dz, I[:, 1].sum() * dz, I[:, 2].sum() * dz
    else:
        if E is None:
            E = ito_fields(sim, nx, n_z)
        dA = (sim._P / nx) ** 2
        I2 = E.real ** 2 + E.imag ** 2
        Ix, Iy, Iz = I2[:, 0].sum() * dA * dz, I2[:, 1].sum() * dA * dz, I2[:, 2].sum() * dA * dz
        out["max_Ez2"] = I2[:, 2].max()
        out["max_E2"] = I2.sum(1).max()
    pref = 0.5 * omega * sim._eps_ito.imag / p_inc_cell(sim._P, sim._theta_deg)
    out.update(Fx=pref * Ix, Fy=pref * Iy, Fz=pref * Iz, Ftot=pref * (Ix + Iy + Iz),
               mean_Ez2=Iz / V, mean_E2=(Ix + Iy + Iz) / V, pref=pref, method=method)
    return out


def evaluate(rho, P, h, lam, order, *, nx=None, n_z=None, with_grad=False,
             method="fourier", want_maps=False, **kw):
    """Full driven evaluation: F_x, F_y, F_z, F_tot, R, T, A = 1-R-T
    (+ real-space ITO field maps / max|Ez|^2 when want_maps)."""
    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        sim = build_sim(rho, P, h, lam, order, **kw)
        R, T = rt_all_orders(sim)
        d = loss_components(sim, nx, n_z, method=method)
        d.update(R=R, T=T, A=1.0 - R - T)
    if want_maps:
        with torch.no_grad():
            E = ito_fields(sim, nx, n_z)
            I2 = E.real ** 2 + E.imag ** 2
            d["E_ito"] = E
            d["max_Ez2"] = I2[:, 2].max()
            d["max_E2"] = I2.sum(1).max()
    d["sim"] = sim
    return d


def to_floats(d):
    return {k: (float(v.detach()) if torch.is_tensor(v) else v) for k, v in d.items()
            if k not in ("sim", "E_ito") and ((torch.is_tensor(v) and v.numel() == 1) or isinstance(v, (int, float)))}


def propagating_orders(P, lam, n_glass, theta_deg=0.0, phi_deg=0.0, medium="glass"):
    """Analytic list of propagating (m, n) in air or glass (independent of
    TORCWA) for the diffraction-order audit."""
    n_med = n_glass if medium == "glass" else 1.0
    k0 = 2 * np.pi / lam
    kx0 = k0 * np.sin(np.deg2rad(theta_deg)) * np.cos(np.deg2rad(phi_deg))
    ky0 = k0 * np.sin(np.deg2rad(theta_deg)) * np.sin(np.deg2rad(phi_deg))
    G = 2 * np.pi / P
    return [(m, n) for m in range(-4, 5) for n in range(-4, 5)
            if (kx0 + m * G) ** 2 + (ky0 + n * G) ** 2 < (n_med * k0) ** 2]


# ---------------------------------------------------------------------------
# reference geometries (frozen, read-only historical designs + analytic cuboids)
# ---------------------------------------------------------------------------
def cuboid(nx, P, lx, ly):
    xg = (np.arange(nx) + 0.5) / nx * P
    X, Y = np.meshgrid(xg, xg, indexing="ij")
    return ((np.abs(X - P / 2) < lx / 2) & (np.abs(Y - P / 2) < ly / 2)).astype(float)


REFERENCE_FILES = {
    "direct-Ez winner (P850,h140,pad86)":
        ("enz_direct_enz_excitation/outputs/geometries/rho_hard_binary.npy", 850.0, 140.0),
    "padded QNM winner (P850,h140,pad86)":
        ("enz_padding_sideexperiment/outputs/geometries/rho_hard_binary.npy", 850.0, 140.0),
    "robust A finalist0 (P800,h200,pad4%)":
        ("enz_robust_aito_campaign/outputs/stage4/runs/finalist0_P800_h200_pad0.040_warm/rho_hard_binary.npy", 800.0, 200.0),
    "robust A finalist1 (P925,h240,pad4%)":
        ("enz_robust_aito_campaign/outputs/stage4/runs/finalist1_P925_h240_pad0.040_warm/rho_hard_binary.npy", 925.0, 240.0),
}


def load_references(nx=128):
    refs = {"bare ITO (P850,h140)": (None, 850.0, 140.0, "air/ITO(23)/glass, no a-Si"),
            "Karimi EDR cuboid (560x500x140, P850)": (torch.as_tensor(cuboid(nx, 850.0, 560.0, 500.0), dtype=GEO_DTYPE), 850.0, 140.0, "Karimi 2023 EDR"),
            "Karimi MDR cuboid (650x650x210, P810)": (torch.as_tensor(cuboid(nx, 810.0, 650.0, 650.0), dtype=GEO_DTYPE), 810.0, 210.0, "Karimi 2023 MDR")}
    for name, (rel, P, h) in REFERENCE_FILES.items():
        f = config.ROOT / rel
        a = np.load(f)
        assert a.shape == (nx, nx) and set(np.unique(a)) <= {0.0, 1.0}, f
        refs[name] = (torch.as_tensor(a, dtype=GEO_DTYPE), P, h, rel)
    return refs
