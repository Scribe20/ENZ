"""Stage 0 of the JZ F_z campaign: materials audit + 10 hard preflight gates.

Writes outputs/stage0/preflight.json and PREFLIGHT.md.  Exits non-zero if
any hard gate fails (production search must not start in that case).

Gates (config.TOL):
  G1  material files parse; ITO zero crossing reproduced by interpolation
  G2  (n+ik)^2 agrees with the supplied eps1/eps2 columns
  G3  bare air/ITO/glass at normal incidence gives Ez = 0 in the ITO
  G4  lossless energy conservation R+T = 1 (all orders), order counts analytic
  G5  ITO volume-loss identity  Fx+Fy+Fz = 1-R-T  (predeclared tolerance)
  G6  Fx+Fy+Fz reproduces the independent real-space |E|^2 volume loss
  G7  autograd dF_z/drho agrees with central finite differences
  G8  the hard air ring is exactly empty after filter + projection
  G9  no hidden mirror-symmetry projection (dynamic S_flip + static AST check)
  G10 F_z converged in z-sampling and RCWA order (screen-level; re-certified
      for the finalists in Stage 4)
"""

import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import config
import forward as fwd
import materials as mat
import optimizer as opt

HERE = Path(__file__).resolve().parent
OUT = config.OUT / "stage0"
OUT.mkdir(parents=True, exist_ok=True)
TOL = config.TOL
LOG, FAILS = [], []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def gate(name, ok, detail):
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILS.append(name)
    return bool(ok)


def random_design(P, seed=1, pad=0.08, beta=8.0, nx=config.NX):
    M, _ = fwd.build_pad_mask(nx, P, pad)
    Mt = torch.as_tensor(M, dtype=config.GEO_DTYPE)
    g = opt.gaussian_kernel_fft(nx, nx, P / nx, P / nx, config.FILTER_RADIUS_NM)
    torch.manual_seed(seed)
    rho, _ = opt.preprocess(torch.rand((nx, nx), dtype=config.GEO_DTYPE), Mt, g, beta)
    return rho.detach(), Mt


def main():
    fwd.set_threads(config.N_THREADS_DEFAULT)
    rep = {}
    t_start = time.time()

    # ------------------------------------------------------------------ G1/G2
    log("== G1/G2: materials ==")
    ma = mat.audit(OUT / "materials_audit.json", verbose=False)
    lam = ma["ito"]["lambda_ZE_nm"]
    rep["materials"] = ma
    gate("G1 ITO zero crossing (interpolated)",
         abs(lam - 1302.3) < TOL["lambda_ze_sanity_nm"] and abs(ma["ito"]["eps2"] - 0.432) < TOL["eps2_ze_sanity"]
         and abs(lam - ma["ito"]["lambda_ZE_from_eps1_column_nm"]) < 1e-3,
         f"lambda_ZE = {lam:.4f} nm (eps1 column: {ma['ito']['lambda_ZE_from_eps1_column_nm']:.4f}); "
         f"n = k = {ma['ito']['n']:.5f}; eps = {ma['ito']['eps1']:+.2e} + {ma['ito']['eps2']:.5f}i")
    gate("G2 (n+ik)^2 vs eps1/eps2 columns",
         ma["ito"]["max_abs_err_eps_vs_nk2"] < TOL["ito_eps_columns_abs"],
         f"max |(n+ik)^2 - (eps1+i eps2)| = {ma['ito']['max_abs_err_eps_vs_nk2']:.2e} over {ma['ito']['n_rows']} rows")
    log(f"  a-Si:H at lambda_ZE: n = {ma['asi']['n']:.5f}, k = {ma['asi']['k']:.1e} "
        f"(row provenance: {ma['asi']['measured_rows_nm']}); glass n = {ma['glass']['n_used']:.5f}, k = 0")
    ng = mat.n_glass(lam)
    thr = dict(P_glass_first_order=lam / ng, P_air_first_order=lam)
    log(f"  diffraction thresholds at lambda_ZE: P = lambda/n_glass = {thr['P_glass_first_order']:.1f} nm, "
        f"P = lambda = {thr['P_air_first_order']:.1f} nm")
    rep["thresholds"] = thr
    rep["orders_by_period"] = {str(P): dict(glass=fwd.propagating_orders(P, lam, ng),
                                            air=fwd.propagating_orders(P, lam, ng, medium="air"))
                               for P in config.P_SCREEN + [900.0, 1000.0]}
    for P, d in rep["orders_by_period"].items():
        log(f"    P = {P}: propagating in glass {d['glass']}, in air {d['air']}")

    # -------------------------------------------------------------------- G3
    log("== G3: bare planar stack -> Ez = 0 ==")
    with torch.no_grad():
        sim = fwd.build_sim(None, 750.0, 200.0, lam, [5, 5])
        E = fwd.ito_fields(sim, 64, 5)
        R, T = fwd.rt_all_orders(sim)
        d = fwd.loss_components(sim, 64, 5)
    ez_max = float(E[:, 2].abs().max())
    rep["bare"] = dict(R=float(R), T=float(T), A=float(1 - R - T), max_abs_Ez=ez_max,
                       Fz=float(d["Fz"]), Ftot=float(d["Ftot"]),
                       identity_resid=float(d["Ftot"]) - float(1 - R - T))
    gate("G3 bare air/ITO/glass Ez", ez_max < TOL["ez_bare_max"],
         f"max|Ez| = {ez_max:.1e}; R = {float(R):.5f}, T = {float(T):.5f}, A = {float(1-R-T):.5f}, "
         f"Ftot(vol) = {float(d['Ftot']):.5f} (resid {rep['bare']['identity_resid']:+.1e})")

    # -------------------------------------------------------------------- G4
    log("== G4: lossless energy conservation (all orders) ==")
    ec = []
    for P in (550.0, 750.0, 850.0, 1000.0):
        rho, _ = random_design(P, seed=3)
        rho_b = (rho > 0.5).to(config.GEO_DTYPE)
        # NOTE: a lossless ITO layer exactly at lambda_ZE has eps = 0 + 0i to
        # machine precision, which makes the homogeneous-layer eigenproblem
        # singular (physical ENZ singularity, not a solver bug); the lossless-
        # ITO conservation case is therefore evaluated 2 nm off the crossing.
        for tag, kw, lam_c in (("no ITO", dict(with_ito=False), lam),
                               ("lossless ITO", dict(ito_loss_scale=0.0), lam + 2.0)):
            with torch.no_grad():
                sim = fwd.build_sim(rho_b, P, 200.0, lam_c, [5, 5], **kw)
                R, T, tab = fwd.rt_all_orders(sim, per_order=True)
            n_an = len(fwd.propagating_orders(P, lam_c, ng))
            ec.append(dict(P=P, case=tag, lam=lam_c, RT_minus_1=float(R + T) - 1, n_prop_torcwa=len(tab), n_prop_analytic_glass=n_an))
            log(f"    P={P:5.0f} {tag:12s} lam={lam_c:.1f}: R+T-1 = {float(R+T)-1:+.1e}; propagating orders torcwa {len(tab)} / analytic {n_an}")
    rep["energy_conservation"] = ec
    worst = max(abs(e["RT_minus_1"]) for e in ec)
    mism = [e for e in ec if e["n_prop_torcwa"] != e["n_prop_analytic_glass"]]
    gate("G4 R+T=1 lossless", worst < TOL["energy_abs"] and not mism,
         f"worst |R+T-1| = {worst:.1e} over {len(ec)} cases; order-count mismatches {len(mism)}")

    # ---------------------------------------------------------------- G5/G6
    log("== G5/G6: ITO volume-loss identity, component sum, z-convergence ==")
    refs = fwd.load_references()
    cases = [("EDR cuboid", refs["Karimi EDR cuboid (560x500x140, P850)"]),
             ("direct-Ez winner", refs["direct-Ez winner (P850,h140,pad86)"]),
             ("robust f0 P800", refs["robust A finalist0 (P800,h200,pad4%)"])]
    rho_r, _ = random_design(750.0, seed=1)
    cases.append(("random blob P750/h200", (rho_r, 750.0, 200.0, "random")))
    idn = []
    for name, (rho, P, h, _src) in cases:
        row = dict(name=name, P=P, h=h)
        with torch.no_grad():
            sim = fwd.build_sim(rho, P, h, lam, config.ORDER_FULL)
            R, T = fwd.rt_all_orders(sim)
            A = float(1 - R - T)
            for n_z in (7, 15, 31):
                d = fwd.loss_components(sim, n_z=n_z)
                row[f"Ftot_nz{n_z}"] = float(d["Ftot"]); row[f"Fz_nz{n_z}"] = float(d["Fz"])
            d7 = fwd.loss_components(sim, n_z=7)
            dr = fwd.loss_components(sim, nx=128, n_z=7, method="realspace")
            # independent real-space |E|^2 total (all components summed at once)
            E = fwd.ito_fields(sim, 128, 7)
            dV = (P / 128) ** 2 * (config.D_ITO_NM / 7)
            F_vol = 0.5 * (2 * np.pi / lam) * mat.eps_ito(lam).imag * float((E.abs() ** 2).sum()) * dV / fwd.p_inc_cell(P)
        row.update(A_rt=A, R=float(R), T=float(T), Fx=float(d7["Fx"]), Fy=float(d7["Fy"]), Fz=float(d7["Fz"]),
                   resid_nz7=float(d7["Ftot"]) - A, resid_nz15=row["Ftot_nz15"] - A, resid_nz31=row["Ftot_nz31"] - A,
                   sum_vs_realspace_vol=float(d7["Ftot"]) - F_vol, Fz_fourier_vs_realspace=float(d7["Fz"]) - float(dr["Fz"]),
                   eta_z=float(d7["Fz"] / d7["Ftot"]))
        idn.append(row)
        log(f"    {name:22s} P={P:.0f} h={h:.0f}: A=1-R-T={A:.5f} | Ftot(nz=7)={row['Ftot_nz7']:.5f} "
            f"({row['resid_nz7']:+.1e}) nz=15 ({row['resid_nz15']:+.1e}) nz=31 ({row['resid_nz31']:+.1e}) | "
            f"Fz={row['Fz']:.5f} Fx={row['Fx']:.5f} Fy={row['Fy']:.5f} eta_z={row['eta_z']:.3f} | "
            f"sum-vs-|E|^2 {row['sum_vs_realspace_vol']:+.1e}; Fz fourier-vs-real {row['Fz_fourier_vs_realspace']:+.1e}")
    rep["identity"] = idn
    w_abs = max(abs(r["resid_nz7"]) for r in idn)
    w_rel = max(abs(r["resid_nz7"]) / r["A_rt"] for r in idn)
    gate("G5 Fx+Fy+Fz = 1-R-T (n_z = 7)", w_abs < TOL["identity_abs"] and w_rel < TOL["identity_rel"],
         f"worst abs {w_abs:.1e} (tol {TOL['identity_abs']}), worst rel {w_rel:.1e} (tol {TOL['identity_rel']})")
    w6 = max(max(abs(r["sum_vs_realspace_vol"]), abs(r["Fz_fourier_vs_realspace"])) for r in idn)
    gate("G6 component sum = independent |E|^2 volume loss", w6 < TOL["components_abs"] * 1e3,
         f"worst |diff| = {w6:.1e} (Fourier-Parseval vs real-space field_xy)")

    # -------------------------------------------------------------------- G7
    log("== G7: autograd vs central finite differences ==")
    fd_rows = []
    for (nxs, order, npix, label) in ((32, [3, 3], 4, "32-grid [3,3]"), (128, [5, 5], 2, "128-grid [5,5] (production)")):
        P, h = 750.0, 200.0
        M, _ = fwd.build_pad_mask(nxs, P, 0.10)
        Mt = torch.as_tensor(M, dtype=config.GEO_DTYPE)
        g = opt.gaussian_kernel_fft(nxs, nxs, P / nxs, P / nxs, config.FILTER_RADIUS_NM)
        torch.manual_seed(7)
        rraw = torch.rand((nxs, nxs), dtype=config.GEO_DTYPE)

        def F_of(rv):
            rp, _ = opt.preprocess(rv, Mt, g, 8.0)
            return fwd.evaluate(rp, P, h, lam, order, nx=nxs, n_z=5, with_grad=rv.requires_grad)["Fz"]
        rv = rraw.clone().requires_grad_(True)
        F0 = F_of(rv); F0.backward(); g_ad = rv.grad.detach()
        rng = np.random.default_rng(0)
        act = np.argwhere(M > 0)
        for (i, j) in act[rng.choice(len(act), npix, replace=False)]:
            e = 1e-4
            rp_, rm_ = rraw.clone(), rraw.clone()
            rp_[i, j] += e; rm_[i, j] -= e
            with torch.no_grad():
                gfd = (float(F_of(rp_)) - float(F_of(rm_))) / (2 * e)
            rel = abs(float(g_ad[i, j]) - gfd) / max(abs(gfd), 1e-12)
            fd_rows.append(dict(case=label, pixel=[int(i), int(j)], autograd=float(g_ad[i, j]), fd=gfd, rel_err=rel))
            log(f"    {label:28s} pixel {(int(i), int(j))}: autograd {float(g_ad[i,j]):+.6e} FD {gfd:+.6e} rel {rel:.1e}")
    rep["gradient_check"] = fd_rows
    gate("G7 autograd vs FD", max(r["rel_err"] for r in fd_rows) < TOL["grad_rel"],
         f"worst rel err {max(r['rel_err'] for r in fd_rows):.1e} (tol {TOL['grad_rel']})")

    # ----------------------------------------------------------------- G8/G9
    log("== G8/G9: padding emptiness + no hidden symmetry (asymmetric seed through the full pipeline) ==")
    nx = config.NX
    P = 750.0
    M, minfo = fwd.build_pad_mask(nx, P, 0.08)
    Mt = torch.as_tensor(M, dtype=config.GEO_DTYPE)
    g = opt.gaussian_kernel_fft(nx, nx, P / nx, P / nx, config.FILTER_RADIUS_NM)
    torch.manual_seed(333)
    r0 = torch.rand((nx, nx), dtype=config.GEO_DTYPE)
    xg = torch.linspace(0, 1, nx, dtype=config.GEO_DTYPE)
    r_asym = torch.clamp(r0 * (0.4 + 0.6 * xg[None, :]) + 0.3 * (xg[:, None] > 0.6), 0, 1)   # deliberately asymmetric in x and y
    r_hist = (r0 + torch.fliplr(r0)) / 2                                                   # historical projection (reference only)
    rp, _ = opt.preprocess(r_asym, Mt, g, 8.0)
    sym = dict(raw_asym=dict(lr=opt.s_flip(r_asym), ud=opt.s_flip_ud(r_asym), c=opt.s_flip_centered(r_asym)),
               historical_fliplr_projected=opt.s_flip(r_hist),
               pipeline_filtered_projected=dict(lr=opt.s_flip(rp), ud=opt.s_flip_ud(rp), c=opt.s_flip_centered(rp), inv=opt.s_inv(rp)))
    leak = dict(after_pipeline=float((rp * (1 - Mt)).abs().max()))
    # 3 optimizer iterations at production grid (cheap order) from the asymmetric seed
    smoke = opt.optimize(P, 200.0, 0.08, 333, 3, [3, 3], OUT / "smoke_optimizer", lam,
                         rho_init=r_asym.numpy(), save_every=1, log=lambda *a: None, tag="smoke")
    sym["optimizer_3it_init"] = smoke["s_flip_init"]
    sym["optimizer_3it_final_hard"] = smoke["s_flip_final"]
    sym["optimizer_3it_history_s_flip"] = smoke["history"]["s_flip"]
    leak.update(smoke["pad_leak"])
    rep["symmetry"], rep["pad_leak"], rep["mask_info_P750_pad0.08"] = sym, leak, minfo
    # static AST check: torch.fliplr/flipud/rot90 only inside the s_* metric functions
    tree = ast.parse((HERE / "optimizer.py").read_text())
    bad = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for c in ast.walk(fn):
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr in ("fliplr", "flipud", "rot90"):
                if not fn.name.startswith("s_"):
                    bad.append(f"{fn.name}:{c.lineno}")
    for f in ("forward.py", "run_stage1.py"):
        p = HERE / f
        if p.exists() and ("fliplr" in p.read_text() or "flipud" in p.read_text()):
            bad.append(f)
    sym["static_check_offending"] = bad
    gate("G8 air ring exactly empty", max(leak.values()) <= TOL["pad_leak"],
         f"max density in ring after filter+projection {leak['after_pipeline']:.1e}, soft {leak['soft']:.1e}, hard {leak['hard']:.1e}")
    gate("G9 no hidden mirror symmetry",
         min(sym["pipeline_filtered_projected"]["lr"], sym["pipeline_filtered_projected"]["ud"]) > TOL["s_flip_min"]
         and min(smoke["s_flip_final"]["lr"], smoke["s_flip_final"]["ud"]) > TOL["s_flip_min"] and not bad,
         f"S_flip(lr,ud) raw asym {sym['raw_asym']['lr']:.3f},{sym['raw_asym']['ud']:.3f} -> pipeline "
         f"{sym['pipeline_filtered_projected']['lr']:.3f},{sym['pipeline_filtered_projected']['ud']:.3f} -> after 3 Adam steps (hard) "
         f"{smoke['s_flip_final']['lr']:.3f},{smoke['s_flip_final']['ud']:.3f}; historical projection would give {sym['historical_fliplr_projected']:.1e}; "
         f"static check offenders: {bad}")

    # ------------------------------------------------------------------- G10
    log("== G10: z-sampling and order convergence (screen level) ==")
    conv = []
    for name, (rho, P, h, _s) in [("direct-Ez winner", refs["direct-Ez winner (P850,h140,pad86)"]),
                                  ("robust f0 P800", refs["robust A finalist0 (P800,h200,pad4%)"]),
                                  ("random blob P750/h200", (rho_r, 750.0, 200.0, "random"))]:
        row = dict(name=name, P=P, h=h)
        for od in ([5, 5], [7, 7], [9, 9]):
            with torch.no_grad():
                sim = fwd.build_sim(rho, P, h, lam, od)
                R, T = fwd.rt_all_orders(sim)
                for n_z in (7, 15):
                    d = fwd.loss_components(sim, n_z=n_z)
                    row[f"Fz_{od[0]}_nz{n_z}"] = float(d["Fz"])
                row[f"A_{od[0]}"] = float(1 - R - T)
        conv.append(row)
        log(f"    {name:22s}: " + ", ".join(f"{k}={v:.4f}" for k, v in row.items() if k.startswith(("Fz_", "A_"))))
    rep["convergence"] = conv
    dz = max(abs(r["Fz_7_nz7"] - r["Fz_7_nz15"]) / r["Fz_7_nz15"] for r in conv)
    do = max(abs(r["Fz_7_nz7"] - r["Fz_9_nz7"]) / r["Fz_9_nz7"] for r in conv)
    gate("G10 z / order convergence (informational at screen level)", dz < 0.02 and do < 0.15,
         f"worst rel change nz 7->15 at [7,7]: {dz:.2e}; worst rel change [7,7]->[9,9]: {do:.2e} "
         "(finalists re-certified in Stage 4 with [9,9]/[11,11] and nz 7/15/31)")

    # --------------------------------------------------------- timing + refs
    log("== timing (fwd+bwd, 128 grid, Fourier objective) ==")
    tim = {}
    rho_t, _ = random_design(750.0, seed=5)
    for nthr in (1, 4):
        fwd.set_threads(nthr)
        for od in ([5, 5], [7, 7]):
            ts = []
            for _rep in range(2):
                rv = rho_t.clone().requires_grad_(True)
                t0 = time.time()
                d = fwd.evaluate(rv, 750.0, 200.0, lam, od, with_grad=True)
                (-d["Fz"]).backward()
                ts.append(time.time() - t0)
            tim[f"{nthr}thr_{od}"] = min(ts)
            log(f"    {nthr} thread(s), order {od}: {min(ts):.2f} s per iteration")
    fwd.set_threads(config.N_THREADS_DEFAULT)
    rep["timing_s_per_iter"] = tim
    n_runs = len(config.P_SCREEN) * len(config.H_SCREEN) * len(config.PAD_SCREEN) * len(config.SEEDS_SCREEN)
    rep["sizing"] = dict(n_runs_stage1=n_runs, s_per_iter_1thr_screen=tim["1thr_[5, 5]"],
                         est_hours_stage1_80it_4proc=n_runs * 80 * tim["1thr_[5, 5]"] / 4 / 3600)
    log(f"  Stage-1 sizing: {n_runs} runs x 80 it at [5,5] on 4 single-thread processes ~ "
        f"{rep['sizing']['est_hours_stage1_80it_4proc']:.1f} h")

    log("== reference designs under the NEW material model (lambda_ZE, order [7,7], n_z 7) ==")
    rrows = []
    for name, (rho, P, h, src) in refs.items():
        d = fwd.to_floats(fwd.evaluate(rho, P, h, lam, config.ORDER_FULL, want_maps=True))
        row = dict(name=name, source=src, P=P, h=h, fill=(float(rho.mean()) if rho is not None else 0.0),
                   Fz=d["Fz"], Fx=d["Fx"], Fy=d["Fy"], Ftot=d["Ftot"], A=d["A"], R=d["R"], T=d["T"],
                   eta_z=d["Fz"] / max(d["Ftot"], 1e-30), mean_Ez2=d["mean_Ez2"], max_Ez2=d["max_Ez2"],
                   identity_resid=d["Ftot"] - d["A"])
        rrows.append(row)
        log(f"    {name:40s} Fz={d['Fz']:.4f} Fx={d['Fx']:.4f} Fy={d['Fy']:.4f} A={d['A']:.4f} R={d['R']:.3f} T={d['T']:.3f} "
            f"eta_z={row['eta_z']:.3f} <|Ez|^2>={d['mean_Ez2']:.3f} max|Ez|^2={d['max_Ez2']:.1f}")
    rep["references_new_material"] = rrows

    rep["gates_failed"] = FAILS
    rep["wall_s"] = time.time() - t_start
    with open(OUT / "preflight.json", "w") as f:
        json.dump(rep, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    (OUT / "preflight.log").write_text("\n".join(LOG))
    write_md(rep)
    log(f"[done] preflight in {rep['wall_s']:.0f} s; failed gates: {FAILS or 'none'}")
    return 0 if not FAILS else 1


def write_md(rep):
    ma = rep["materials"]
    L = ["# PREFLIGHT - JZ freeform F_z campaign (Stage 0)", "",
         "Generated by `stage0_preflight.py`; every number is in `outputs/stage0/preflight.json` and `outputs/stage0/preflight.log`.",
         "", f"**Result: {'ALL HARD GATES PASSED' if not rep['gates_failed'] else 'FAILED: ' + ', '.join(rep['gates_failed'])}**", "",
         "## Materials (parsed directly from the supplied files; sha256 in materials_data/PROVENANCE.md)", "",
         f"- ITO `ITO_nk.csv`: {ma['ito']['range_nm'][0]:.0f}-{ma['ito']['range_nm'][1]:.0f} nm, {ma['ito']['n_rows']} rows; "
         f"max |(n+ik)^2 - (eps1 + i eps2)| = {ma['ito']['max_abs_err_eps_vs_nk2']:.1e}.",
         f"- **lambda_ZE = {ma['ito']['lambda_ZE_nm']:.3f} nm** (Re eps = 0 by cubic-spline interpolation; the eps1 column gives "
         f"{ma['ito']['lambda_ZE_from_eps1_column_nm']:.3f} nm). There: n = {ma['ito']['n']:.4f}, k = {ma['ito']['k']:.4f}, "
         f"eps = {ma['ito']['eps1']:+.1e} + {ma['ito']['eps2']:.4f} i.",
         f"- a-Si:H `aSi_H_measured_Postech.txt`: n = {ma['asi']['n']:.4f}, k = {ma['asi']['k']:.1e} at lambda_ZE. "
         f"PROVENANCE FLAG: rows 1000-1400 nm are the source workbook's Sellmeier extrapolation (k = 0), not ellipsometry.",
         f"- glass `SiO2_substrate_measured.txt` (soda-lime float glass, identical to the *_comsol.txt file: {ma['glass']['identical_to_comsol_file']}): "
         f"n = {ma['glass']['n_used']:.5f}, k = 0 at lambda_ZE.",
         f"- diffraction thresholds at lambda_ZE: P = lambda/n_glass = {rep['thresholds']['P_glass_first_order']:.1f} nm (glass), "
         f"P = lambda = {rep['thresholds']['P_air_first_order']:.1f} nm (air).", "",
         "| P (nm) | propagating orders in glass | in air |", "|---|---|---|"]
    for P, d in rep["orders_by_period"].items():
        L.append(f"| {P} | {d['glass']} | {d['air']} |")
    L += ["", "## Gates", ""]
    for s in LOG:
        if s.strip().startswith("[PASS]") or s.strip().startswith("[FAIL]"):
            L.append(f"- {s.strip()}")
    L += ["", "## Volume-loss identity table (order [7,7])", "",
          "| design | P | h | A = 1-R-T | Ftot nz=7 | resid nz=7 | resid nz=15 | resid nz=31 | Fz | Fx | Fy | eta_z |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep["identity"]:
        L.append(f"| {r['name']} | {r['P']:.0f} | {r['h']:.0f} | {r['A_rt']:.5f} | {r['Ftot_nz7']:.5f} | {r['resid_nz7']:+.1e} | "
                 f"{r['resid_nz15']:+.1e} | {r['resid_nz31']:+.1e} | {r['Fz']:.5f} | {r['Fx']:.5f} | {r['Fy']:.5f} | {r['eta_z']:.3f} |")
    L += ["", "## Gradient check", ""]
    for r in rep["gradient_check"]:
        L.append(f"- {r['case']} pixel {r['pixel']}: autograd {r['autograd']:+.6e}, FD {r['fd']:+.6e}, rel {r['rel_err']:.1e}")
    s = rep["symmetry"]
    L += ["", "## Symmetry preflight", "",
          "No `torch.fliplr`/`flipud` projection exists in the new optimizer (static AST check: offenders = "
          f"{s['static_check_offending']}); the historical Example6-derived path had two (`enz_inverse_design/optimize_enz_overlap.py` "
          "lines 128 and 276, `original_pixel_inverse_design.ipynb` cell 2).",
          f"- deliberately asymmetric seed: S_flip lr/ud = {s['raw_asym']['lr']:.3f}/{s['raw_asym']['ud']:.3f}; after mask+filter+projection "
          f"{s['pipeline_filtered_projected']['lr']:.3f}/{s['pipeline_filtered_projected']['ud']:.3f} (inversion {s['pipeline_filtered_projected']['inv']:.3f}); "
          f"after 3 Adam steps (hard binary) {s['optimizer_3it_final_hard']['lr']:.3f}/{s['optimizer_3it_final_hard']['ud']:.3f}. "
          f"The historical projection would give {s['historical_fliplr_projected']:.1e}.",
          f"- air-ring leakage after filter+projection: {rep['pad_leak']['after_pipeline']:.1e} (soft final {rep['pad_leak']['soft']:.1e}, hard {rep['pad_leak']['hard']:.1e}); "
          f"mask realization at P=750, pad 8%: {json.dumps(rep['mask_info_P750_pad0.08'])}",
          "", "## Convergence (screen level)", "",
          "| design | Fz [5,5] nz7 | Fz [7,7] nz7 | Fz [7,7] nz15 | Fz [9,9] nz7 | Fz [9,9] nz15 | A [5,5] | A [7,7] | A [9,9] |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rep["convergence"]:
        L.append(f"| {r['name']} | {r['Fz_5_nz7']:.4f} | {r['Fz_7_nz7']:.4f} | {r['Fz_7_nz15']:.4f} | {r['Fz_9_nz7']:.4f} | {r['Fz_9_nz15']:.4f} | "
                 f"{r['A_5']:.4f} | {r['A_7']:.4f} | {r['A_9']:.4f} |")
    L += ["", "## Timing and sizing", "", f"- {json.dumps(rep['timing_s_per_iter'])} s per iteration (fwd+bwd, 128 grid)",
          f"- {json.dumps(rep['sizing'])}", "",
          "## Reference designs under the NEW material model (lambda_ZE, [7,7], nz 7)", "",
          "| design | P | h | fill | Fz | Fx | Fy | Ftot | A | R | T | eta_z,abs | <|Ez/Einc|^2> | max|Ez/Einc|^2 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep["references_new_material"]:
        L.append(f"| {r['name']} | {r['P']:.0f} | {r['h']:.0f} | {r['fill']:.3f} | {r['Fz']:.4f} | {r['Fx']:.4f} | {r['Fy']:.4f} | {r['Ftot']:.4f} | "
                 f"{r['A']:.4f} | {r['R']:.3f} | {r['T']:.3f} | {r['eta_z']:.3f} | {r['mean_Ez2']:.3f} | {r['max_Ez2']:.1f} |")
    (HERE / "PREFLIGHT.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
