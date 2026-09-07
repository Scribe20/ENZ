"""Common conventions for the HIGH-Q + DRIVEN-Ez audit.

Every candidate is evaluated with EXACTLY the same forward model
(enz_robust_aito_campaign/forward_multi.build_sim: air / a-Si(h, rho) /
ITO 23 nm / glass, TORCWA, complex128, exp(-j w t), LH units, p/s-notation
source with |E_inc| = 1, lab-x polarization at normal incidence).

Driven quantities (plane wave, unit |E_inc|):
    A_ITO      = 1 - R_total - T_total (all propagating orders)
    A_vol      = (w/2) Im(eps_ITO) int_ITO |E|^2 dV / P_inc   (cross-check)
    F_Ez       = <|Ez/E_inc|^2>_ITO,   F_Etot = <|E/E_inc|^2>_ITO
    eta_z      = int |Ez|^2 / int |E|^2 over the ITO
    peak / percentiles of |Ez|^2, 50%-volume concentration fractions
Modal / stored-energy quantities (driven-field proxy of the eigenmode,
same construction as enz_highq_enz_campaign/stage_a_decompose):
    U = 1/4 int [Re d(w eps)/dw |E|^2 + |H|^2] dV  over a-Si + ITO + 480-nm
        air/glass near-field slabs with the (0,0) plane-wave harmonic removed
        in the claddings
    eta_ENZ,z  = int_ITO |Ez|^2 dV / U      (units nm^3 / energy; ratios only)
    ITO E-energy fraction = U_E,ITO / U_E
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT / "enz_robust_aito_campaign", ROOT / "enz_absorption_campaign",
          ROOT / "enz_highq_enz_campaign", ROOT / "enz_inverse_design"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import forward_multi as fm          # noqa: E402  (patched Eig, variable P/h)
import references as refs           # noqa: E402

OUT = HERE / "outputs"
FIG = OUT / "figures"
LAMBDA_E = fm.LAMBDA_E
D_ITO = fm.D_ITO_NM
C_NM_FS = 299.792458
ORDER = [7, 7]
S_LEVELS = (1.0, 0.5, 0.25, 0.1, 0.03, 0.0)
GEO = fm.GEO_DTYPE

CANDIDATES = {
    # name: (path relative to ROOT or None for bare, P, h, pad_nm, campaign)
    "bare ITO": (None, 850.0, 140.0, 0.0, "reference (no a-Si)"),
    "EDR cuboid": ("<edr>", 850.0, 140.0, 0.0, "reference 560x500x140 cuboid"),
    "unpadded QNM": ("enz_inverse_design/outputs/geometries/rho_hard_binary.npy",
                     850.0, 140.0, 0.0, "enz_inverse_design (850-nm QNM overlap)"),
    "padded QNM": ("enz_padding_sideexperiment/outputs/geometries/rho_hard_binary.npy",
                   850.0, 140.0, 86.33, "enz_padding_sideexperiment"),
    "padded F_ENZ": ("enz_direct_enz_excitation/outputs/geometries/rho_hard_binary.npy",
                     850.0, 140.0, 86.33, "enz_direct_enz_excitation"),
    "P800 robust": ("enz_robust_aito_campaign/outputs/stage4/runs/finalist0_P800_h200_pad0.040_warm/rho_hard_binary.npy",
                    800.0, 200.0, 31.25, "enz_robust_aito_campaign finalist0"),
    "P925 robust": ("enz_robust_aito_campaign/outputs/stage4/runs/finalist1_P925_h240_pad0.040_warm/rho_hard_binary.npy",
                    925.0, 240.0, 36.1, "enz_robust_aito_campaign finalist1"),
    "P925/h220 robust": ("enz_robust_aito_campaign/outputs/stage4/runs/finalist2_P925_h220_pad0.040_warm/rho_hard_binary.npy",
                         925.0, 220.0, 36.1, "enz_robust_aito_campaign finalist2 (control)"),
}
PRIMARY = ["padded QNM", "padded F_ENZ", "P800 robust", "P925 robust"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_candidate(name):
    src, P, h, pad, camp = CANDIDATES[name]
    if src is None:
        return None, P, h
    if src == "<edr>":
        return torch.as_tensor(refs.edr_cuboid(), dtype=GEO), P, h
    a = np.load(ROOT / src)
    assert a.shape == (128, 128) and set(np.unique(a)) <= {0.0, 1.0}
    return torch.as_tensor(a, dtype=GEO), P, h


def manifest():
    rows = []
    for name, (src, P, h, pad, camp) in CANDIDATES.items():
        rho, _, _ = load_candidate(name)
        rows.append(dict(
            name=name, campaign=camp,
            file=(src if src not in (None, "<edr>") else src or "n/a"),
            sha256=(sha256(ROOT / src) if src not in (None, "<edr>") else
                    ("generated in references.edr_cuboid" if src == "<edr>" else "n/a")),
            P_nm=P, h_nm=h, ITO_nm=D_ITO, padding_nm=pad,
            grid=("128x128" if rho is not None else "n/a"),
            fill=(float(rho.mean()) if rho is not None else 0.0),
            rcwa="TORCWA complex128, exp(-jwt), LH units, order [7,7] unless stated",
            materials="ITO: measured CSV (enz_target/data/ito_digitized_dense_1nm_physical.csv); "
                      "a-Si: POSTECH measured n,k (k=0 in band); glass n=1.4446",
            polarization="lab-x (p at theta=0); p/s-notation source, |E_inc|=1"))
    return rows


# ---------------------------------------------------------------------------
def ito_fields(sim, P, n_xy=96, n_z=7):
    x, y = fm.cell_axes(P, n_xy)
    zs = (np.arange(n_z) + 0.5) * D_ITO / n_z
    E = []
    with torch.no_grad():
        for zp in zs:
            Ev, _ = sim.field_xy(1, x, y, float(zp))
            E.append(torch.stack([c for c in Ev], 0))
    return torch.stack(E, 0)          # (nz, 3, nx, ny)


def driven_metrics(rho, P, h, lam=LAMBDA_E, theta=0.0, phi=0.0, s=1.0,
                   order=ORDER, n_xy=96, n_z=7, want_field=False):
    with torch.no_grad():
        sim = fm.build_sim(rho, P, h, lam=lam, theta_deg=theta, phi_deg=phi,
                           order=order, ito_loss_scale=s)
        R, T, tab = fm.rt_all_orders(sim, per_order=True)
        E = ito_fields(sim, P, n_xy, n_z)
    E2 = (E.abs() ** 2).sum(1)            # (nz,nx,ny)
    Ez2 = E[:, 2].abs() ** 2
    dV = (P / n_xy) ** 2 * (D_ITO / n_z)
    V = P ** 2 * D_ITO
    IE2, IEz2 = float(E2.sum()) * dV, float(Ez2.sum()) * dV
    e_ito = fm.eps_ito(lam, s)
    p_inc = 0.5 * np.cos(np.deg2rad(theta)) * P ** 2
    A_vol = 0.5 * (2 * np.pi / lam) * e_ito.imag * IE2 / p_inc
    ez = Ez2.flatten().numpy()
    srt = np.sort(ez)[::-1]
    cum = np.cumsum(srt) / srt.sum()
    conc_ez = float(np.searchsorted(cum, 0.5) + 1) / ez.size
    e2 = E2.flatten().numpy(); srt2 = np.sort(e2)[::-1]
    conc_abs = float(np.searchsorted(np.cumsum(srt2) / srt2.sum(), 0.5) + 1) / e2.size
    out = dict(lam=lam, theta=theta, phi=phi, s=s, order=list(order),
               A_rt=float(1 - R - T), R=float(R), T=float(T),
               orders=[(o["m"], o["n"]) for o in tab], A_vol=A_vol,
               F_Ez=IEz2 / V, F_Etot=IE2 / V, eta_z=IEz2 / IE2,
               peak_Ez2=float(ez.max()),
               Ez2_p50=float(np.percentile(ez, 50)), Ez2_p90=float(np.percentile(ez, 90)),
               Ez2_p95=float(np.percentile(ez, 95)), Ez2_p99=float(np.percentile(ez, 99)),
               conc50_Ez2=conc_ez, conc50_abs=conc_abs,
               # absorbed energy density per unit incident intensity (fixed
               # intensity convention): U_abs/V_ITO / I_inc = A * cos(theta) / d
               abs_density_fixed_intensity=float((1 - R - T)) * np.cos(np.deg2rad(theta)) / D_ITO,
               # fixed power per cell: divide by cell area as well
               abs_density_fixed_power_per_cell=float((1 - R - T)) / (P ** 2 * D_ITO))
    if want_field:
        out["Ez_mid"] = E[n_z // 2, 2].numpy()
        out["E_mid"] = E[n_z // 2].numpy()
    return out


def d_omega_eps_ito(lam, s):
    w = 2 * np.pi * C_NM_FS / lam
    dw = 1e-3 * w
    def we(wv):
        l = 2 * np.pi * C_NM_FS / wv
        return wv * fm.eps_ito(l, s)
    return ((we(w + dw) - we(w - dw)) / (2 * dw)).real


def energy_participation(rho, P, h, lam, s, theta=0.0, phi=0.0, n=48,
                         order=ORDER):
    """Driven-field proxy of the modal quantities (stage_a_decompose
    construction, generalized to per-call P, h).  Returns U (stored energy
    per unit |E_inc|^2), eta_ENZ_z = int_ITO|Ez|^2/U, eta_z, ITO E-energy
    fraction, and the ITO E-energy fraction of the a-Si+ITO region only."""
    sim = fm.build_sim(rho, P, h, lam=lam, theta_deg=theta, phi_deg=phi,
                       order=order, ito_loss_scale=s)
    xs, ys = fm.cell_axes(P, n)
    dA = (P / n) ** 2
    geo = (ndimage.zoom(rho.numpy(), n / rho.shape[0], order=0)
           if rho is not None else np.zeros((n, n)))
    eps_si = fm.eps_asi(lam).real
    eps_g = fm.N_GLASS ** 2
    dwe = d_omega_eps_ito(lam, s)
    layers = [(0, (np.arange(8) + 0.5) * h / 8, h / 8, geo * eps_si + (1 - geo), False),
              (1, (np.arange(5) + 0.5) * D_ITO / 5, D_ITO / 5, np.full((n, n), dwe), False),
              (-1, -((np.arange(12) + 0.5) * 40.0), 40.0, np.ones((n, n)), True),
              (sim.layer_N, (np.arange(12) + 0.5) * 40.0, 40.0, np.full((n, n), eps_g), True)]
    U_E = U_H = UE_ito = UE_si = Iz = It = 0.0
    with torch.no_grad():
        for lay, zps, dz, w, strip in layers:
            for zp in zps:
                E, H = sim.field_xy(lay, xs, ys, float(zp))
                E = [c.numpy() for c in E]; H = [c.numpy() for c in H]
                if strip:
                    E = [c - c.mean() for c in E]; H = [c - c.mean() for c in H]
                e2 = sum(np.abs(c) ** 2 for c in E); h2 = sum(np.abs(c) ** 2 for c in H)
                U_E += 0.25 * np.sum(w * e2) * dA * dz
                U_H += 0.25 * np.sum(h2) * dA * dz
                if lay == 1:
                    Iz += np.sum(np.abs(E[2]) ** 2) * dA * dz
                    It += np.sum(np.abs(E[0]) ** 2 + np.abs(E[1]) ** 2) * dA * dz
                    UE_ito += 0.25 * np.sum(w * e2) * dA * dz
                if lay == 0:
                    UE_si += 0.25 * np.sum(w * e2) * dA * dz
    U = U_E + U_H
    return dict(U=U, U_E=U_E, U_H=U_H, eta_ENZ_z=Iz / U, eta_z=Iz / (Iz + It),
                ito_E_energy_fraction=UE_ito / U_E,
                ito_over_si_E_energy=UE_ito / max(UE_si, 1e-300))


def jdump(obj, path):
    def d(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, complex):
            return [o.real, o.imag]
        return str(o)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=d)


def jload(path):
    return json.load(open(path))
