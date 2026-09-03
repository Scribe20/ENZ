"""Stage A: radiative / non-radiative decomposition of the saved padded
candidates (pre-analysis; no optimization).

For each candidate (padded QNM winner, padded F_ENZ winner):
  1. loaded pole (physical a-Si/ITO/glass) from channel-agnostic r/t AAA
     -> lambda_pole, Q_loaded;
  2. loss-scaling continuation: Im eps_ITO -> s * Im eps_ITO with
     s in S_LEVELS (dispersive Re eps kept from the measured CSV), pole
     tracked by nearest-continuation; gamma(s) = Im(omega_tilde)(s) fitted
     linearly -> gamma_rad = gamma(s->0) (intercept), gamma_nr = slope.
     Linearity residual = validity check of the "same mode" assumption;
     the s = 0 point is the auxiliary lossless-ITO pole (independent
     Q_rad estimate);
  3. air/glass split of gamma_rad from the r and t residues (single-mode
     CMT: |Res_r|^2 : |Res_t|^2 ~ gamma_air : gamma_glass, port-power
     normalized) - reported as approximate;
  4. normalized modal ENZ participation from the DRIVEN field at the
     resonance of the lossless auxiliary structure (resonant term dominates
     when Q_rad >> 1; stated as a proxy):
        eta_ENZ,z = integral_ITO |Ez|^2 dV / U,
        U = 1/4 integral [Re d(w eps)/dw |E|^2 + |H|^2] dV   (LH units)
     over the cell with a-Si + ITO + near-field air/glass slabs, where in
     air/glass the (0,0) plane-wave harmonic is removed so only the
     evanescent near field counts; plus eta_z and ITO energy participation.
Run:  python stage_a_decompose.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / "enz_inverse_design"
PAD = HERE.parent / "enz_padding_sideexperiment"
EZC = HERE.parent / "enz_direct_enz_excitation"
ABS = HERE.parent / "enz_absorption_campaign"
for p in (PKG, ABS):
    sys.path.insert(0, str(p))

import config                     # noqa: E402
import target_mode                # noqa: E402
import torcwa_forward as fwd      # noqa: E402
import pole_rt                    # noqa: E402

C_NM_FS = 299.792458
LAMBDA_E = 1433.488
S_LEVELS = (1.0, 0.5, 0.25, 0.1, 0.03, 0.0)
SCAN = np.arange(1300.0, 1601.0, 6.0)      # denser than pole_rt.SCAN
OUT = HERE / "outputs"


def setup():
    torch.set_num_threads(config.N_THREADS)
    tgt = target_mode.load_target_npz()
    config.ITO_THICKNESS_NM = float(tgt["ito_thickness_nm"])
    config.N_GLASS = float(tgt["glass_index"])
    config.EPS_ASI = fwd.eps_asi_of_lambda(LAMBDA_E)


def eps_ito_scaled(lam, s):
    e = fwd.eps_ito_of_lambda(lam)
    return complex(e.real, s * e.imag)


def rt_scan_scaled(rho_t, s, lams=SCAN):
    r, t = [], []
    with torch.no_grad():
        for lam in lams:
            sim = fwd.build_solved_sim(rho_t, lam, eps_ito_scaled(lam, s),
                                       config.N_GLASS,
                                       eps_asi=fwd.eps_asi_of_lambda(lam))
            rr = sim.S_parameters(orders=[0, 0], direction="forward",
                                  port="reflection", polarization="xx",
                                  ref_order=[0, 0], power_norm=False)
            tt = sim.S_parameters(orders=[0, 0], direction="forward",
                                  port="transmission", polarization="xx",
                                  ref_order=[0, 0], power_norm=False)
            r.append(complex(rr.cpu().numpy().ravel()[0]))
            t.append(complex(tt.cpu().numpy().ravel()[0]))
    return np.array(r), np.array(t)


def tracked_pole(lams, r, t, prev=None):
    """Certified in-window pole; if several, the one nearest the previous
    loss level (mode tracking)."""
    from scipy.interpolate import AAA
    oms = 2 * np.pi * C_NM_FS / lams
    fr, ft = AAA(oms, r), AAA(oms, t)
    cands = []
    for q, res in zip(fr.poles(), fr.residues()):
        if q.imag >= -1e-4 or not pole_rt._in_window(q):
            continue
        m = min(zip(ft.poles(), ft.residues()), key=lambda p: abs(p[0] - q))
        if abs(m[0] - q) / abs(q) < pole_rt.RT_TOL:
            cands.append((0.5 * (q + m[0]), abs(res), abs(m[1])))
    if not cands:
        return None
    if prev is not None:
        q, ar, at = min(cands, key=lambda c: abs(c[0] - prev))
    else:
        q, ar, at = max(cands, key=lambda c: c[1] + c[2])
    return dict(omega=q, res_r=ar, res_t=at)


def d_omega_eps_ito(lam, s):
    """Re d(omega*eps)/d omega for the ITO at loss scale s (finite diff on
    the measured real-axis dispersion)."""
    w = 2 * np.pi * C_NM_FS / lam
    dw = 1e-3 * w
    def we(wv):
        l = 2 * np.pi * C_NM_FS / wv
        return wv * eps_ito_scaled(l, s)
    return ((we(w + dw) - we(w - dw)) / (2 * dw)).real


def modal_participation(rho_t, lam, s, n=48):
    """Driven-field proxy of the normalized modal ENZ participation."""
    sim = fwd.build_solved_sim(rho_t, lam, eps_ito_scaled(lam, s),
                               config.N_GLASS, eps_asi=fwd.eps_asi_of_lambda(lam))
    xs = torch.as_tensor((np.arange(n) + 0.5) / n * config.PX_NM,
                         dtype=config.GEO_DTYPE)
    ys = torch.as_tensor((np.arange(n) + 0.5) / n * config.PY_NM,
                         dtype=config.GEO_DTYPE)
    dA = (config.PX_NM / n) * (config.PY_NM / n)
    h_si, d_ito = config.ASI_THICKNESS_NM, config.ITO_THICKNESS_NM
    rho_np = rho_t.cpu().numpy()
    zoom = n / rho_np.shape[0]
    from scipy import ndimage
    geo = ndimage.zoom(rho_np, zoom, order=0)
    eps_si = complex(config.EPS_ASI).real
    eps_g = config.N_GLASS ** 2
    dwe_ito = d_omega_eps_ito(lam, s)
    layers = [   # (layer_num, z_prop list, dz, eps-weight callable, strip00)
        (0, (np.arange(8) + 0.5) * h_si / 8, h_si / 8,
         lambda: geo * eps_si + (1 - geo) * 1.0, False),
        (1, (np.arange(5) + 0.5) * d_ito / 5, d_ito / 5,
         lambda: np.full((n, n), dwe_ito), False),
        (-1, -((np.arange(12) + 0.5) * 40.0), 40.0,
         lambda: np.ones((n, n)), True),               # air, 0..-480 nm
        (sim.layer_N, (np.arange(12) + 0.5) * 40.0, 40.0,
         lambda: np.full((n, n), eps_g), True),        # glass, 0..+480 nm
    ]
    U_E = U_H = 0.0
    Iz_ito = It_ito = 0.0
    UE_ito = 0.0
    with torch.no_grad():
        for lay, zps, dz, wfun, strip in layers:
            w = wfun()
            for zp in zps:
                E, H = sim.field_xy(lay, xs, ys, float(zp))
                E = [c.cpu().numpy() for c in E]
                H = [c.cpu().numpy() for c in H]
                if strip:   # remove (0,0) plane-wave harmonic in claddings
                    E = [c - c.mean() for c in E]
                    H = [c - c.mean() for c in H]
                e2 = sum(np.abs(c) ** 2 for c in E)
                h2 = sum(np.abs(c) ** 2 for c in H)
                U_E += 0.25 * np.sum(w * e2) * dA * dz
                U_H += 0.25 * np.sum(h2) * dA * dz
                if lay == 1:
                    Iz_ito += np.sum(np.abs(E[2]) ** 2) * dA * dz
                    It_ito += np.sum(np.abs(E[0]) ** 2 + np.abs(E[1]) ** 2) * dA * dz
                    UE_ito += 0.25 * np.sum(w * e2) * dA * dz
    U = U_E + U_H
    return dict(eta_ENZ_z=Iz_ito / U, eta_z=Iz_ito / (Iz_ito + It_ito),
                ito_E_energy_fraction=UE_ito / U_E, U=U, U_E=U_E, U_H=U_H)


def analyze(name, rho_t, log):
    log(f"\n=== {name} ===")
    rows = []
    prev = None
    for s in S_LEVELS:
        r, t = rt_scan_scaled(rho_t, s)
        p = tracked_pole(SCAN, r, t, prev)
        if p is None:
            log(f"  s={s:4.2f}: no certified in-window pole")
            continue
        q = p["omega"]
        prev = q
        lam_p = 2 * np.pi * C_NM_FS / q.real
        rows.append(dict(s=s, omega_re=q.real, omega_im=q.imag,
                         lambda_pole=lam_p, Q=abs(q.real / (2 * q.imag)),
                         res_r=p["res_r"], res_t=p["res_t"]))
        log(f"  s={s:4.2f}: pole {lam_p:8.2f} nm, gamma=|Im w|="
            f"{abs(q.imag):.5f} rad/fs, Q={abs(q.real/(2*q.imag)):8.2f}, "
            f"|res_r|={p['res_r']:.3e} |res_t|={p['res_t']:.3e}")
    if len(rows) < 3:
        return dict(name=name, rows=rows, error="insufficient poles")
    S = np.array([r["s"] for r in rows]); G = np.array([abs(r["omega_im"]) for r in rows])
    slope, icpt = np.polyfit(S, G, 1)
    resid = np.max(np.abs(G - (slope * S + icpt))) / G.max()
    w0 = rows[0]["omega_re"]
    g_rad, g_nr = icpt, slope          # at s = 1: gamma = g_rad + g_nr
    Q_rad = w0 / (2 * g_rad) if g_rad > 0 else np.inf
    Q_nr = w0 / (2 * g_nr)
    Q_loaded = rows[0]["Q"]
    # air/glass split (approximate, port-power normalized residues)
    rr, rt = rows[0]["res_r"], rows[0]["res_t"] * np.sqrt(config.N_GLASS)
    f_glass = rt ** 2 / (rr ** 2 + rt ** 2)
    lossless = next((r for r in rows if r["s"] == 0.0), None)
    log(f"  linear fit gamma(s): gamma_rad = {g_rad:.5f}, gamma_nr(s=1) = "
        f"{g_nr:.5f} rad/fs, max linearity residual = {resid:.2e}")
    log(f"  Q_loaded = {Q_loaded:.2f}, Q_rad = {Q_rad:.1f}, Q_nr = {Q_nr:.2f}, "
        f"gamma_rad/gamma_nr = {g_rad/g_nr:.3f}; lossless-ITO pole Q = "
        + (f"{lossless['Q']:.1f} at {lossless['lambda_pole']:.1f} nm"
           if lossless else "n/a")
        + f"; pole shift s=1 -> s=0: {rows[0]['lambda_pole']-lossless['lambda_pole'] if lossless else np.nan:+.1f} nm")
    log(f"  gamma_rad split (residue-based, approx): glass fraction = "
        f"{f_glass:.2f}, air fraction = {1-f_glass:.2f}")
    part = {}
    for tag, s, lam_p in (("loaded", 1.0, rows[0]["lambda_pole"]),
                          ("lossless_aux", 0.0,
                           lossless["lambda_pole"] if lossless else rows[-1]["lambda_pole"])):
        part[tag] = modal_participation(rho_t, lam_p, s)
        log(f"  participation ({tag}, driven at its pole {lam_p:.1f} nm): "
            f"eta_ENZ,z = {part[tag]['eta_ENZ_z']:.4e} nm^-3... "
            f"(dimension 1/energy-density units; use ratios), eta_z = "
            f"{part[tag]['eta_z']:.3f}, ITO E-energy fraction = "
            f"{part[tag]['ito_E_energy_fraction']:.3f}")
    return dict(name=name, rows=rows, gamma_rad=g_rad, gamma_nr=g_nr,
                linearity_resid=resid, Q_loaded=Q_loaded, Q_rad=Q_rad,
                Q_nr=Q_nr, gamma_ratio=g_rad / g_nr, glass_fraction=f_glass,
                lossless_pole=lossless, participation=part)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    setup()
    lines = []
    def log(s):
        print(s); lines.append(s)
    cands = {
        "padded QNM winner": PAD / "outputs" / "geometries" / "rho_hard_binary.npy",
        "padded F_ENZ winner": EZC / "outputs" / "geometries" / "rho_hard_binary.npy",
    }
    results = {}
    for name, path in cands.items():
        rho_t = torch.as_tensor(np.load(path), dtype=config.GEO_DTYPE)
        results[name] = analyze(name, rho_t, log)
    with open(OUT / "stage_a_decomposition.json", "w") as f:
        json.dump(results, f, indent=1, default=lambda o: float(o)
                  if isinstance(o, (np.floating, np.integer)) else str(o))
    (OUT / "stage_a.log").write_text("\n".join(lines) + "\n")
    print(f"[saved] {OUT/'stage_a_decomposition.json'}")


if __name__ == "__main__":
    main()
