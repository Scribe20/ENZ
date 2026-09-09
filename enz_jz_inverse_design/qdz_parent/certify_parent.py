"""Exact post-hoc certification of ONE hard-binary parent candidate.

Computes, for the frozen geometry (no re-optimization anywhere):
    exact radiative poles of the parent (analysis.significant_poles on r/t, Rayleigh anomalies
      excluded, dense local rescan of the pole nearest lambda_E)  ->  lambda_r, Q_r
    eta_Dz, eta_Dz_P, U_mid, W_Si, W_layer, S_Dz at lambda_E and at lambda_r, at several orders
    R(lambda), T(lambda) of the parent
    parent field maps |E|^2, |Ez|^2, |Dz|^2 (interface plane and z stack)
    Cartesian multipole diagnostics of the a-Si:H polarization current (ED/MD/EQ/MQ/TD)
    geometry metrics (fill, minimum feature, pad leak, symmetry scores) + SHA256

    python certify_parent.py --rho path/to/rho_hard_binary.npy --h 525 --tag name [--P 825]
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import config, forward as fwd, materials as mat, analysis as an     # noqa: E402
import parent_fwd as pf, qproxy as qp, poles_parent as pp           # noqa: E402

GEO = config.GEO_DTYPE
EPS0 = 1.0            # torcwa normalized units


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def multipoles_parent(rho, P, h, lam, order, n_z=9, nx=64):
    """Cartesian current multipoles of the a-Si:H polarization current of the PARENT mode.
    P_pol = eps0 (eps_Si - 1) E on the Si cells, J = -i omega P_pol (exp(-i omega t) convention, the
    same one the RCWA phasors use).  Long-wavelength vacuum Cartesian forms; p_eff = p + i k T, so the
    toroidal term is a DIAGNOSTIC and is NOT added again to the ED+MD+EQ+MQ normalisation."""
    c_nm_fs = 299.792458
    w = 2 * np.pi * c_nm_fs / lam                       # rad/fs
    k0 = 2 * np.pi / lam                                # 1/nm
    eps_si = float(np.real(mat.eps_asi(lam)))
    with torch.no_grad():
        sim = fwd.build_sim(rho, P, h, lam, order, with_ito=False)
        zs = pf.asi_z_slices(h, n_z); dz = h / n_z
        step = config.NX // nx
        rr = rho[::step, ::step].numpy()
        x = (np.arange(nx) + 0.0) * P / nx - P / 2
        X, Y = np.meshgrid(x, x, indexing="ij")
        Z = np.asarray(zs) - h / 2
        E = np.zeros((3, nx, nx, n_z), complex)
        for iz, zp in enumerate(zs):
            Ex, Ey, Ez, _ = pf.layer_fourier(sim, float(zp))
            for c, mn in enumerate((Ex, Ey, Ez)):
                E[c, :, :, iz] = pf.to_real(mn, sim, config.NX).numpy()[::step, ::step]
    dV = (P / nx) ** 2 * dz * rr[:, :, None]
    J = (-1j * w * (eps_si - 1.0)) * E                  # J / eps0, units field * rad/fs
    r = np.stack([np.broadcast_to(X[:, :, None], E.shape[1:]), np.broadcast_to(Y[:, :, None], E.shape[1:]),
                  np.broadcast_to(Z[None, None, :], E.shape[1:])])
    ig = lambda f: (f * dV).sum()
    p = 1j / w * np.array([ig(J[a]) for a in range(3)])
    rxJ = np.cross(r, J, axis=0)
    m = 0.5 * np.array([ig(rxJ[a]) for a in range(3)])
    rdJ = (r * J).sum(0); r2 = (r ** 2).sum(0)
    T = 1.0 / (10 * c_nm_fs) * np.array([ig(rdJ * r[a] - 2 * r2 * J[a]) for a in range(3)])
    Qe = np.zeros((3, 3), complex); Qm = np.zeros((3, 3), complex)
    for i in range(3):
        for j in range(3):
            Qe[i, j] = 1j / w * ig(3 * (r[j] * J[i] + r[i] * J[j]) - 2 * rdJ * (i == j))
            Qm[i, j] = ig(rxJ[i] * r[j] + rxJ[j] * r[i]) / 3.0
    p_eff = p + 1j * k0 * T
    P_p = w ** 4 * np.sum(np.abs(p) ** 2)
    P_pe = w ** 4 * np.sum(np.abs(p_eff) ** 2)
    P_T = w ** 4 * k0 ** 2 * np.sum(np.abs(T) ** 2)
    P_m = w ** 4 * np.sum(np.abs(m) ** 2) / c_nm_fs ** 2
    P_Qe = w ** 6 * np.sum(np.abs(Qe) ** 2) / (120 * c_nm_fs ** 2)
    P_Qm = w ** 6 * np.sum(np.abs(Qm) ** 2) / (40 * c_nm_fs ** 4)
    tot = P_pe + P_m + P_Qe + P_Qm
    cx = lambda v: [[float(z.real), float(z.imag)] for z in np.atleast_1d(v).ravel()]
    return dict(lambda_nm=float(lam), p=cx(p), m=cx(m), T=cx(T), p_eff=cx(p_eff),
                Qe=cx(Qe), Qm=cx(Qm), P_ED_eff=float(P_pe), P_ED_p_only=float(P_p), P_TD_diag=float(P_T),
                P_MD=float(P_m), P_EQ=float(P_Qe), P_MQ=float(P_Qm), P_sum=float(tot),
                frac_ED_eff=float(P_pe / tot), frac_MD=float(P_m / tot), frac_EQ=float(P_Qe / tot),
                frac_MQ=float(P_Qm / tot), TD_diag_over_sum=float(P_T / tot), ED_p_only_over_sum=float(P_p / tot),
                convention="p_eff = p + i k T; fractions normalise ED_eff + MD + EQ + MQ, TD reported separately",
                grid=[nx, nx, n_z], eps_Si=eps_si)


def certify(rho, P, h, tag, out_dir, orders=([5, 5], [7, 7], [9, 9]), n_z=9, lam_lo=1160.0, lam_hi=1400.0,
            step=2.0, want_maps=True, log=print):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    lam_E = pf.lambda_E(); t0 = time.time()
    ng = float(mat.n_glass(1300.0))
    res = dict(tag=tag, P=float(P), h=float(h), lam_E=lam_E, sha256=sha(rho.numpy()),
               shape=list(rho.shape), fill_fraction=float(rho.mean()), n_z=int(n_z),
               orders=[list(o) for o in orders])
    # ---- exact parent poles at the certification order --------------------------
    order_c = list(orders[-1])
    pk = pp.parent_poles(rho, P, h, order_c, lam_lo, lam_hi, step, n_glass=ng)
    res["parent_poles"] = pk["poles"]
    res["parent_energy_resid"] = pk["energy_resid"]
    res["rayleigh_nm"] = pk["rayleigh"]
    near = pp.nearest_pole(pk["poles"], lam_E)
    res["pole_nearest_lambda_E"] = near
    if near:
        res["local_refine"] = pp.refine_pole(rho, P, h, order_c, near, ray=pk["rayleigh"])
        res["Q_r"] = near["Q"]; res["lambda_r"] = near["lambda_nm"]
        res["detuning_nm"] = near["lambda_nm"] - lam_E
        res["detuning_in_linewidths"] = (near["lambda_nm"] - lam_E) / max(near["fwhm_nm"], 1e-9)
    else:
        res["Q_r"] = None; res["lambda_r"] = None
    # pole convergence across orders
    res["pole_vs_order"] = []
    for o in orders[:-1]:
        pk_o = pp.parent_poles(rho, P, h, list(o), lam_lo, lam_hi, 3.0, n_glass=ng)
        n_o = pp.nearest_pole(pk_o["poles"], lam_E)
        res["pole_vs_order"].append(dict(order=list(o), lambda_r=(n_o["lambda_nm"] if n_o else None),
                                         Q_r=(n_o["Q"] if n_o else None), n_poles=len(pk_o["poles"])))
    # ---- metrics at lambda_E and at lambda_r, vs order ---------------------------
    res["metrics"] = {}
    for label, lam in (("lambda_E", lam_E), ("lambda_r", res.get("lambda_r"))):
        if lam is None:
            continue
        rows = []
        for o in orders:
            f = pf.to_floats(pf.parent_metrics(rho, P, h, float(lam), list(o), n_z=n_z))
            rows.append(dict(order=list(o), **{k: f[k] for k in
                        ("eta_Dz", "eta_Dz_P", "S_Dz", "W_Si", "W_layer", "U_mid", "R", "T", "S_Dz_ratio_glass_over_Si")}))
        res["metrics"][label] = dict(lambda_nm=float(lam), by_order=rows,
                                     order_convergence_rel=(abs(rows[-1]["eta_Dz"] - rows[-2]["eta_Dz"]) / rows[-2]["eta_Dz"]
                                                            if len(rows) > 1 else None))
    # ---- spectra ------------------------------------------------------------------
    lams = np.arange(lam_lo, lam_hi + 1e-9, step)
    sp = pf.spectrum_parent(rho, P, h, lams, order_c, n_z=5)
    np.savez_compressed(out / "parent_spectrum.npz", lam_nm=lams,
                        R=np.array([r["R"] for r in sp]), T=np.array([r["T"] for r in sp]),
                        W_Si=np.array([r["W_Si"] for r in sp]), W_layer=np.array([r["W_layer"] for r in sp]),
                        eta_Dz=np.array([r["eta_Dz"] for r in sp]), U_mid=np.array([r["U_mid"] for r in sp]),
                        r=np.array([r["r"] for r in sp]), t=np.array([r["t"] for r in sp]), order=order_c)
    res["spectrum_file"] = "parent_spectrum.npz"
    W = np.array([r["W_Si"] for r in sp])
    res["W_Si_peak_lambda_nm"] = float(lams[int(np.argmax(W))])
    res["W_Si_peak_over_edge"] = float(W.max() / max(W.min(), 1e-300))
    # ---- Q proxy at the certification point (for the calibration record) ----------
    lams_p, om_p, kt = qp.probe_wavelengths(lam_E, res["Q_r"] if res.get("Q_r") else 100.0)
    Wp = torch.stack([pf.parent_metrics(rho, P, h, float(l), order_c, n_z=n_z)["W_Si"] for l in lams_p])
    fit = qp.lorentz_fit(om_p, Wp, qp.omega_of(lam_E))
    res["q_proxy_at_certification"] = dict(Q_proxy=float(fit["Q"]), lambda_r_proxy=float(fit["lambda_r"]),
                                           kappa_proxy=float(fit["kappa"]), rel_resid=float(fit["rel_resid"]),
                                           well_posed=bool(fit["well_posed"]),
                                           rel_err_vs_pole=(abs(float(fit["Q"]) - res["Q_r"]) / res["Q_r"] if res.get("Q_r") else None))
    # ---- field maps ---------------------------------------------------------------
    if want_maps:
        for label, lam in (("lambda_E", lam_E), ("lambda_r", res.get("lambda_r"))):
            if lam is None:
                continue
            d = pf.parent_metrics(rho, P, h, float(lam), order_c, n_z=max(n_z, 15), want_maps=True)
            E2 = d["E2_slices"].numpy(); Ez = d["Ez_slices"].numpy(); Dz = d["Dz_map"].numpy()
            np.savez_compressed(out / f"parent_fields_{label}.npz", lam_nm=float(lam), order=order_c,
                                z_slices_nm=d["z_slices_nm"], E2=E2.astype(np.float32),
                                Ez=Ez.astype(np.complex64), Dz_interface=Dz.astype(np.complex64),
                                Ez2=(np.abs(Ez) ** 2).astype(np.float32), Dz2=(np.abs(Dz) ** 2).astype(np.float32),
                                rho=rho.numpy().astype(np.uint8), P=P, h=h,
                                note="E2 = |E|^2 on the a-Si:H z slices; Ez complex; Dz_interface = Dz/eps0 at the "
                                     "prospective Si/ITO interface (glass side); fields normalized to E_inc = 1")
    # ---- multipoles ----------------------------------------------------------------
    res["multipoles"] = {}
    for label, lam in (("lambda_E", lam_E), ("lambda_r", res.get("lambda_r"))):
        if lam is not None:
            res["multipoles"][label] = multipoles_parent(rho, P, h, float(lam), order_c, n_z=n_z)
    # ---- geometry metrics -----------------------------------------------------------
    M_np, _ = fwd.build_pad_mask(config.NX, P, 0.12)
    b = (rho.numpy() > 0.5)
    from optimizer import s_flip, s_flip_ud, s_inv
    res["geometry"] = dict(fill_fraction=float(b.mean()), n_material_px=int(b.sum()),
                           min_feature_px=an.min_feature_px(b), pixel_nm=float(P / config.NX),
                           symmetry_scores=dict(flip_lr=s_flip(rho), flip_ud=s_flip_ud(rho), inversion=s_inv(rho)),
                           pad_leak=float((rho * (1 - torch.as_tensor(M_np, dtype=GEO))).abs().max()),
                           fab=an.fab_metrics(rho, P, torch.as_tensor(M_np, dtype=GEO)))
    res["wall_s"] = time.time() - t0
    json.dump(res, open(out / "certification.json", "w"), indent=1, default=float)
    log(f"[cert {tag}] lambda_r={res.get('lambda_r')} Q_r={res.get('Q_r')} "
        f"eta_Dz(lambda_E)={res['metrics'].get('lambda_E', {}).get('by_order', [{}])[-1].get('eta_Dz')} "
        f"U_mid={res['metrics'].get('lambda_E', {}).get('by_order', [{}])[-1].get('U_mid')} ({res['wall_s']:.0f} s)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", required=True)
    ap.add_argument("--P", type=float, default=825.0)
    ap.add_argument("--h", type=float, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--orders", type=int, nargs="*", default=[5, 5, 7, 7, 9, 9])
    ap.add_argument("--n-z", type=int, default=9)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-maps", action="store_true")
    a = ap.parse_args()
    fwd.set_threads(a.threads)
    rho = torch.as_tensor(np.load(a.rho), dtype=GEO)
    orders = [a.orders[i:i + 2] for i in range(0, len(a.orders), 2)]
    out = Path(a.out) if a.out else HERE / "outputs" / "certified" / a.tag
    certify(rho, a.P, a.h, a.tag, out, orders=orders, n_z=a.n_z, want_maps=not a.no_maps)


if __name__ == "__main__":
    main()
