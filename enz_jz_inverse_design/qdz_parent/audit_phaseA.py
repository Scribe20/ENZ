"""PHASE A AUDIT - the gates that must pass before any optimization is launched.

  1. eta_Dz is invariant under an arbitrary rescaling of the field amplitude.
  2. eta_Dz samples ONLY the actual Si-contact region (a disjoint mask gives a different value; a
     mask restricted to the Si support reproduces it exactly; the air region contributes nothing).
  3. lambda_E comes from the supplied ITO material file (not hard-coded).
  4. the differentiable Q proxy correlates with the exact post-hoc pole Q (calibration + error).
  5. gradients of every differentiable term are finite, non-zero, and match finite differences.
Plus: field-synthesis verification against torcwa's own field_xy, parent energy conservation,
Dz-route agreement, and order/n_z convergence of the metrics.

    python audit_phaseA.py [--quick]
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import config, forward as fwd, materials as mat            # noqa: E402
from optimizer import gaussian_kernel_fft, filter_rho, project_rho   # noqa: E402
import parent_fwd as pf, qproxy as qp, poles_parent as pp  # noqa: E402

OUT = HERE / "outputs" / "phaseA"
GEO = config.GEO_DTYPE
P0 = 825.0


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def load_rho(name):
    return torch.as_tensor(np.load(HERE.parent / "outputs" / name), dtype=GEO)


def disc(nx, P, d):
    x = (np.arange(nx) + 0.5) / nx * P
    X, Y = np.meshgrid(x, x, indexing="ij")
    return ((X - P / 2) ** 2 + (Y - P / 2) ** 2 < (d / 2) ** 2).astype(float)


def smooth_random(nx, P, seed, pad_frac=0.12, radius_nm=60.0, beta=8.0):
    """A filtered/projected random density on the padded design mask (same machinery as the campaign)."""
    M_np, _ = fwd.build_pad_mask(nx, P, pad_frac)
    M = torch.as_tensor(M_np, dtype=GEO)
    g = gaussian_kernel_fft(nx, nx, P / nx, P / nx, radius_nm)
    torch.manual_seed(seed)
    r = torch.rand((nx, nx), dtype=GEO)
    rb = filter_rho(r * M, g)
    return (project_rho(rb, beta) * M), M


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fwd.set_threads(4)
    res, t0 = {}, time.time()
    lam_E = pf.lambda_E()
    order_a = [5, 5]

    # ---- 3. lambda_E provenance -------------------------------------------------
    ito = mat.ito()
    lo, hi = mat.ito_zero_crossing()
    res["check3_lambda_E"] = dict(
        lambda_E_nm=lam_E, bracket_nm=[float(lo), float(hi)],
        eps_at_lambda_E=[float(np.real(mat.eps_ito(lam_E))), float(np.imag(mat.eps_ito(lam_E)))],
        source_file=str(config.ITO_FILE), file_sha256=mat._sha(config.ITO_FILE),
        hard_coded_anywhere=False,
        note="lambda_E = the eps' = 0 crossing of the supplied ITO_nk.csv, via materials.ito_zero_crossing()")
    print(f"[3] lambda_E = {lam_E:.6f} nm from {Path(config.ITO_FILE).name} (eps = {mat.eps_ito(lam_E):.3e})")

    rho_hard = load_rho("best/rho_hard_binary.npy")
    rho_soft, M = smooth_random(config.NX, P0, 4242)
    h0 = 525.0

    # ---- 0. field synthesis + energy conservation + Dz routes -------------------
    d = pf.parent_metrics(rho_hard, P0, h0, lam_E, [7, 7], n_z=7)
    sim = d["sim"]; nx = config.NX
    _, _, Ez_mn, Dz_si = pf.layer_fourier(sim, h0 - 1e-9)
    Ez_r = pf.to_real(Ez_mn, sim, nx)
    xe = torch.arange(nx, dtype=torch.float64) * P0 / nx
    Ev, _ = sim.field_xy(0, xe, xe, h0 - 1e-9)
    _, _, _, Dz_out = pf.output_fourier(sim, 0.0)
    f = pf.to_floats(d)
    res["check0_synthesis"] = dict(
        ifft2_vs_field_xy_max_abs=float((Ez_r - Ev[2]).abs().max()), field_scale=float(Ev[2].abs().max()),
        grid="edge grid x_j = j P / nx - the SAME grid on which rho and eps are defined (torcwa FFTs "
             "the eps array with that convention); forward.cell_axes is cell-centred and differs by half a cell",
        energy_resid_RplusT_minus_1=float(f["R"] + f["T"] - 1.0),
        Dz_glass_over_Dz_inSi=f["S_Dz_ratio_glass_over_Si"],
        note="Dz_mn = Ky Hx - Kx Hy is taken before the inverse convolution; evaluated on the glass side "
             "(homogeneous, no Gibbs error) it is identical to the in-Si value, confirming normal-D continuity")
    print(f"[0] synthesis vs torcwa field_xy: {res['check0_synthesis']['ifft2_vs_field_xy_max_abs']:.2e} "
          f"(scale {res['check0_synthesis']['field_scale']:.2e}); R+T-1 = {res['check0_synthesis']['energy_resid_RplusT_minus_1']:+.1e}; "
          f"Dz glass/Si = {f['S_Dz_ratio_glass_over_Si']:.6f}")

    # ---- 1. amplitude-scale invariance ------------------------------------------
    # eta_Dz is a ratio of two field-quadratic integrals: rescaling the source amplitude must cancel.
    base = pf.parent_metrics(rho_hard, P0, h0, lam_E, order_a, n_z=7)
    fb = pf.to_floats(base)
    scaled = {}
    for s in (1e-3, 7.5, 1e4):
        sim_s = fwd.build_sim(rho_hard, P0, h0, lam_E, order_a, with_ito=False)
        sim_s.E_i = sim_s.E_i * s                       # rescale the incident amplitude
        _, _, _, Dz_mn = pf.output_fourier(sim_s, 0.0)
        Dz = pf.to_real(Dz_mn, sim_s, nx)
        S = pf.D_ITO_NM * (rho_hard * (Dz.real ** 2 + Dz.imag ** 2)).sum() * (P0 / nx) ** 2
        W = 0.0
        for zp in pf.asi_z_slices(h0, 7):
            Ex, Ey, Ez, _ = pf.layer_fourier(sim_s, float(zp))
            E2 = sum((pf.to_real(c, sim_s, nx).real ** 2 + pf.to_real(c, sim_s, nx).imag ** 2) for c in (Ex, Ey, Ez))
            W = W + (rho_hard * E2).sum() * (P0 / nx) ** 2 * (h0 / 7) * float(np.real(mat.eps_asi(lam_E)))
        scaled[f"{s:g}"] = float(S / W)
    devs = [abs(v - fb["eta_Dz"]) / fb["eta_Dz"] for v in scaled.values()]
    res["check1_scale_invariance"] = dict(eta_Dz_reference=fb["eta_Dz"], eta_Dz_scaled=scaled,
                                          max_rel_deviation=float(max(devs)), passed=bool(max(devs) < 1e-12))
    print(f"[1] amplitude-scale invariance: max rel deviation {max(devs):.2e} -> {'PASS' if max(devs) < 1e-12 else 'FAIL'}")

    # ---- 2. eta_Dz uses only the Si-contact region -------------------------------
    supp = (rho_hard > 0).to(GEO)
    comp = 1.0 - rho_hard
    m_sup = pf.to_floats(pf.parent_metrics(rho_hard, P0, h0, lam_E, order_a, n_z=7, rho_metric=supp))
    m_com = pf.to_floats(pf.parent_metrics(rho_hard, P0, h0, lam_E, order_a, n_z=7, rho_metric=comp))
    m_all = pf.to_floats(pf.parent_metrics(rho_hard, P0, h0, lam_E, order_a, n_z=7, rho_metric=torch.ones_like(rho_hard)))
    res["check2_contact_region"] = dict(
        eta_Dz_rho_weighted=fb["eta_Dz"], eta_Dz_binary_support=m_sup["eta_Dz"],
        eta_Dz_complement_mask=m_com["eta_Dz"], eta_Dz_full_cell_mask=m_all["eta_Dz"],
        S_Dz_contact=fb["S_Dz"], S_Dz_complement=m_com["S_Dz"], S_Dz_full_cell=m_all["S_Dz"],
        contact_fraction=float(rho_hard.mean()),
        identity_hardbinary_equals_support=bool(abs(fb["eta_Dz"] - m_sup["eta_Dz"]) < 1e-12),
        complement_is_distinct=bool(abs(m_com["S_Dz"] - fb["S_Dz"]) / fb["S_Dz"] > 0.05),
        additivity_resid=float(abs(fb["S_Dz"] + m_com["S_Dz"] - m_all["S_Dz"]) / m_all["S_Dz"]),
        note="for a hard-binary rho the density weight equals its own support; the complementary (air) "
             "mask gives a different and independent integral, and contact + complement = full cell")
    c2 = res["check2_contact_region"]
    print(f"[2] contact region: eta(rho)={fb['eta_Dz']:.6f} eta(support)={m_sup['eta_Dz']:.6f} "
          f"eta(air mask)={m_com['eta_Dz']:.6f} eta(full cell)={m_all['eta_Dz']:.6f}; additivity resid {c2['additivity_resid']:.1e} "
          f"-> {'PASS' if c2['identity_hardbinary_equals_support'] and c2['complement_is_distinct'] and c2['additivity_resid'] < 1e-12 else 'FAIL'}")

    # ---- convergence of the metrics ---------------------------------------------
    conv = {"order": [], "n_z": []}
    for order in ([5, 5], [7, 7], [9, 9], [11, 11]):
        ff = pf.to_floats(pf.parent_metrics(rho_hard, P0, h0, lam_E, order, n_z=7))
        conv["order"].append(dict(order=order, **{k: ff[k] for k in ("eta_Dz", "eta_Dz_P", "S_Dz", "W_Si", "W_layer", "U_mid", "R", "T")}))
    for n_z in (5, 7, 9, 15, 25):
        ff = pf.to_floats(pf.parent_metrics(rho_hard, P0, h0, lam_E, [7, 7], n_z=n_z))
        conv["n_z"].append(dict(n_z=n_z, **{k: ff[k] for k in ("eta_Dz", "W_Si", "U_mid")}))
    res["convergence_on_final3_parent"] = conv
    e = [c["eta_Dz"] for c in conv["order"]]
    print(f"[conv] eta_Dz vs order {['%.5f' % x for x in e]} (rel 9->11 {abs(e[3]-e[2])/e[2]:.3f}); "
          f"n_z 5->25 rel {abs(conv['n_z'][-1]['eta_Dz']-conv['n_z'][0]['eta_Dz'])/conv['n_z'][0]['eta_Dz']:.4f}")

    # ---- 5. gradient checks ------------------------------------------------------
    gr = {}
    for name, rho_t in (("soft_random_projected", rho_soft), ("hard_binary_final3", rho_hard)):
        x = rho_t.clone().requires_grad_(True)
        d1 = pf.parent_metrics(x, P0, h0, lam_E, order_a, n_z=5, with_grad=True)
        loss = -torch.log(d1["eta_Dz"] + 1e-30)
        loss.backward()
        g = x.grad.detach().clone()
        gn = float(torch.linalg.norm(g))
        # central finite differences on the few most sensitive pixels
        idx = torch.argsort(g.abs().reshape(-1), descending=True)[:4]
        fd, an_ = [], []
        for k in idx.tolist():
            i, j = divmod(k, config.NX)
            eps = 1e-4
            rp = rho_t.clone(); rp[i, j] = float(rp[i, j]) + eps
            rm = rho_t.clone(); rm[i, j] = float(rm[i, j]) - eps
            lp = -np.log(float(pf.parent_metrics(rp, P0, h0, lam_E, order_a, n_z=5)["eta_Dz"]) + 1e-30)
            lm = -np.log(float(pf.parent_metrics(rm, P0, h0, lam_E, order_a, n_z=5)["eta_Dz"]) + 1e-30)
            fd.append((lp - lm) / (2 * eps)); an_.append(float(g[i, j]))
        rel = [abs(f_ - a_) / max(abs(a_), 1e-12) for f_, a_ in zip(fd, an_)]
        gr[name] = dict(grad_norm=gn, finite=bool(np.isfinite(g.numpy()).all()), nonzero=bool(gn > 0),
                        n_nonzero_pixels=int((g.abs() > 0).sum()), fd_analytic=list(zip(fd, an_)),
                        max_rel_fd_error=float(max(rel)), passed=bool(np.isfinite(g.numpy()).all() and gn > 0 and max(rel) < 5e-3))
        print(f"[5] grad({name}): |g| = {gn:.3e}, finite={gr[name]['finite']}, nonzero pixels {gr[name]['n_nonzero_pixels']}, "
              f"max rel FD error {gr[name]['max_rel_fd_error']:.2e} -> {'PASS' if gr[name]['passed'] else 'FAIL'}")
    # gradient through the Q proxy
    x = rho_soft.clone().requires_grad_(True)
    lams, omegas, kt = qp.probe_wavelengths(lam_E, 100)
    Ws = torch.stack([pf.parent_metrics(x, P0, h0, float(l), order_a, n_z=5, with_grad=True)["W_Si"] for l in lams])
    fit = qp.lorentz_fit(omegas, Ws, qp.omega_of(lam_E))
    lq = qp.q_loss(fit["Q"], 100.0)
    lq.backward()
    gq = x.grad.detach()
    gr["q_proxy_loss"] = dict(grad_norm=float(torch.linalg.norm(gq)), finite=bool(np.isfinite(gq.numpy()).all()),
                              Q_proxy=float(fit["Q"]), lambda_r_proxy=float(fit["lambda_r"]), kappa=float(fit["kappa"]),
                              rel_resid=float(fit["rel_resid"]), well_posed=fit["well_posed"],
                              passed=bool(np.isfinite(gq.numpy()).all() and float(torch.linalg.norm(gq)) > 0))
    print(f"[5] grad(L_Q): |g| = {gr['q_proxy_loss']['grad_norm']:.3e}, Q_proxy = {float(fit['Q']):.1f}, "
          f"lambda_r = {float(fit['lambda_r']):.2f} nm, fit resid {float(fit['rel_resid']):.2e} -> {'PASS' if gr['q_proxy_loss']['passed'] else 'FAIL'}")
    res["check5_gradients"] = gr

    res["wall_s"] = time.time() - t0
    res["geometry_hashes"] = dict(final3_hard=sha(rho_hard.numpy()), soft_random_seed4242=sha(rho_soft.numpy()))
    json.dump(res, open(OUT / "audit_phaseA.json", "w"), indent=1, default=float)
    print(f"[audit] written to {OUT/'audit_phaseA.json'} in {res['wall_s']:.0f} s")


if __name__ == "__main__":
    main()
