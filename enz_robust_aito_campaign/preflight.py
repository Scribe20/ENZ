"""Stage 0 (ingest audit) + Stage 1 (preflight) of the robust-A_ITO campaign.

Writes outputs/preflight.json, PREFLIGHT.md, figures/preflight_*.png.
No optimization is launched here (only a 2-iteration smoke run of the
optimizer path on a tiny problem to prove the no-symmetry code path).

Run:  python preflight.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import forward_multi as fm
import optimizer as opt
import references as refs
import robust_config as rc

HERE = Path(__file__).resolve().parent
OUT = rc.OUT
FIG = OUT / "figures"
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def angle_curve(rho, P, h, angles, order, pol="labx"):
    out = []
    with torch.no_grad():
        for th, ph in angles:
            sim = fm.build_sim(rho, P, h, theta_deg=th, phi_deg=ph,
                               order=order, pol=pol)
            A, R, T = fm.a_ito(sim)
            out.append(dict(theta=th, phi=ph, A=float(A), R=float(R),
                            T=float(T)))
    return out


def main():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    rep = {}
    t_start = time.time()

    # ------------------------------------------------------------------
    # Stage 0: ingest audit
    # ------------------------------------------------------------------
    log("== Stage 0: ingest audit ==")
    lam = rc.LAMBDA_E
    e_ito, e_asi = fm.eps_ito(lam), fm.eps_asi(lam)
    tgt = np.load(HERE.parent / "enz_target"
                  / "target_enz_mode_periodic_850_g10.npz")
    rep["materials"] = dict(
        lambda_E_nm=lam,
        lambda_E_origin=("Re(omega) of the bare air/ITO(23)/glass TM QNM at "
                         "K = G10(850 nm): omega = "
                         f"{float(tgt['omega_pole_rad_per_fs_real']):.6f}"
                         f"{float(tgt['omega_pole_rad_per_fs_imag']):+.6f}i "
                         f"rad/fs, Q = {float(tgt['pole_Q']):.3f}; inherited "
                         "from the frozen 850-nm benchmark and held FIXED for "
                         "all periods in this campaign (task statement)"),
        eps_ito_csv_at_lambda_E=[e_ito.real, e_ito.imag],
        eps_ito_target_npz=[float(tgt["eps_ito_real"]),
                            float(tgt["eps_ito_imag"])],
        eps_asi_postech_at_lambda_E=[e_asi.real, e_asi.imag],
        n_asi=float(np.sqrt(e_asi).real), d_ito_nm=fm.D_ITO_NM,
        n_glass=fm.N_GLASS, material_ENZ_crossing_nm=1419.59)
    log(json.dumps(rep["materials"], indent=1))
    R = refs.load_all()
    rep["references"] = {k: dict(P=v[1], h=v[2], source=v[3],
                                 fill=(float(v[0].mean()) if v[0] is not None
                                       else 0.0),
                                 S_flip=(opt.s_flip(v[0]) if v[0] is not None
                                         else 0.0),
                                 S_flipud=(opt.s_flip_ud(v[0]) if v[0] is not
                                           None else 0.0))
                         for k, v in R.items()}
    for k, v in rep["references"].items():
        log(f"  ref {k:22s}: fill={v['fill']:.3f} S_flip(lr)={v['S_flip']:.3f} "
            f"S_flip(ud)={v['S_flipud']:.3f}  <- {v['source']}")

    # ------------------------------------------------------------------
    # Stage 1a: symmetry preflight
    # ------------------------------------------------------------------
    log("== Stage 1a: symmetry preflight (S_flip = ||rho-fliplr(rho)||/||rho||) ==")
    nx = rc.NX
    torch.manual_seed(333)
    r0 = torch.rand((nx, nx), dtype=fm.GEO_DTYPE)
    r_old = (r0 + torch.fliplr(r0)) / 2               # historical projection
    g = opt.gaussian_kernel_fft(nx, nx, 850 / nx, 850 / nx, rc.FILTER_RADIUS_NM)
    M, _ = opt.build_pad_mask(nx, 850.0, 0.10)
    Mt = torch.as_tensor(M, dtype=fm.GEO_DTYPE)
    sym = dict(random_init_raw=opt.s_flip(r0),
               historical_fliplr_projected=opt.s_flip(r_old),
               new_path_filtered_masked=opt.s_flip(opt.filter_rho(r0 * Mt, g) * Mt),
               new_path_flipud=opt.s_flip_ud(opt.filter_rho(r0 * Mt, g) * Mt))
    # 2-iteration smoke run through the optimizer (tiny problem) to prove the
    # code path never symmetrizes
    smoke = opt.optimize(850.0, 140.0, 0.10, 333, 2, [2, 2],
                         [(0.0, 0.0), (20.0, 0.0)], 30.0,
                         OUT / "smoke_optimizer", nx=32, save_every=1,
                         log=lambda *a: None, tag="smoke")
    sym["optimizer_smoke_s_flip_init"] = smoke["s_flip_init"]
    sym["optimizer_smoke_s_flip_final"] = smoke["s_flip_final"]
    rep["symmetry"] = sym
    log(json.dumps(sym, indent=1))
    assert sym["historical_fliplr_projected"] < 1e-12
    assert sym["new_path_filtered_masked"] > 0.05
    assert smoke["s_flip_final"] > 0.05

    # ------------------------------------------------------------------
    # Stage 1b: energy conservation + diffraction-order audit (lossless)
    # ------------------------------------------------------------------
    log("== Stage 1b: energy conservation (no ITO, lossless) with ALL orders ==")
    ec = []
    rho_pq = R["padded QNM winner"][0]
    for P in (750.0, 850.0, 1050.0, 1300.0):
        for th, ph in ((0, 0), (20, 0), (20, 90), (25, 45), (40, 30)):
            for pol in ("labx", "p", "s"):
                with torch.no_grad():
                    sim = fm.build_sim(rho_pq, P, 140.0, theta_deg=th,
                                       phi_deg=ph, order=[5, 5],
                                       with_ito=False, pol=pol)
                    Rt, Tt, tab = fm.rt_all_orders(sim, per_order=True)
                ec.append(dict(P=P, theta=th, phi=ph, pol=pol,
                               RT_minus_1=float(Rt + Tt) - 1,
                               n_prop_torcwa=len(tab),
                               n_prop_analytic_glass=len(
                                   fm.propagating_orders(P, lam, th, ph)),
                               n_prop_analytic_air=len(
                                   fm.propagating_orders(P, lam, th, ph, "air"))))
    rep["energy_conservation"] = ec
    worst = max(abs(e["RT_minus_1"]) for e in ec)
    mism = [e for e in ec if e["n_prop_torcwa"] != e["n_prop_analytic_glass"]]
    log(f"  worst |R+T-1| = {worst:.2e} over {len(ec)} cases; order-count "
        f"mismatches: {len(mism)}")
    assert worst < 1e-10 and not mism

    # ------------------------------------------------------------------
    # Stage 1c: A_ITO identity vs volume integral at oblique incidence
    # ------------------------------------------------------------------
    log("== Stage 1c: A = 1-R-T (all orders) vs (w/2) Im eps int|E|^2 / P_inc ==")
    ai = []
    for name in ("EDR cuboid", "padded QNM winner"):
        rho, P, h, _ = R[name]
        for (Pc, th, ph) in ((850.0, 0, 0), (850.0, 20, 0), (850.0, 20, 90),
                             (1050.0, 25, 45)):
            with torch.no_grad():
                sim = fm.build_sim(rho, Pc, h, theta_deg=th, phi_deg=ph,
                                   order=rc.ORDER_FULL)
                A, Rt, Tt = fm.a_ito(sim)
                v = fm.a_ito_volume(sim, Pc, lam, th, n_xy=128, n_z=9)
                ps = fm.polarization_split(sim)
            ai.append(dict(ref=name, P=Pc, theta=th, phi=ph, A_rt=float(A),
                           A_vol=v["A_vol"], F_Ez=v["F_Ez"], eta_z=v["eta_z"],
                           R=float(Rt), T=float(Tt),
                           R00_cross_frac=ps["reflection"]["cross"] / max(
                               ps["reflection"]["co"] + ps["reflection"]["cross"], 1e-30),
                           T00_cross_frac=ps["transmission"]["cross"] / max(
                               ps["transmission"]["co"] + ps["transmission"]["cross"], 1e-30)))
            log(f"  {name:18s} P={Pc:5.0f} th={th:2d} ph={ph:2d}: A_rt="
                f"{float(A):.5f} A_vol={v['A_vol']:.5f} "
                f"(diff {float(A)-v['A_vol']:+.1e}) F_Ez={v['F_Ez']:.3f} "
                f"eta_z={v['eta_z']:.3f} xpol(R,T)="
                f"{ai[-1]['R00_cross_frac']:.3f},{ai[-1]['T00_cross_frac']:.3f}")
    rep["a_identity"] = ai
    assert max(abs(a["A_rt"] - a["A_vol"]) for a in ai) < 5e-4

    # ------------------------------------------------------------------
    # Stage 1d: gradient check of J_robust (finite differences)
    # ------------------------------------------------------------------
    log("== Stage 1d: finite-difference check of dJ_robust/drho ==")
    torch.manual_seed(7)
    nxs = 32
    Ms, _ = opt.build_pad_mask(nxs, 850.0, 0.10)
    Mst = torch.as_tensor(Ms, dtype=fm.GEO_DTYPE)
    gs = opt.gaussian_kernel_fft(nxs, nxs, 850 / nxs, 850 / nxs, 40.0)
    rraw = torch.rand((nxs, nxs), dtype=fm.GEO_DTYPE)
    angs = [(0.0, 0.0), (20.0, 0.0), (20.0, 90.0)]

    def J_of(rv):
        rp = opt.project_rho(opt.filter_rho(rv * Mst, gs), 8.0) * Mst
        J, _ = opt.evaluate(rp, 850.0, 140.0, angs, [3, 3], 30.0,
                            with_grad=rv.requires_grad)
        return J
    rv = rraw.clone().requires_grad_(True)
    J0 = J_of(rv); J0.backward(); g_ad = rv.grad.detach()
    fd = []
    rng = np.random.default_rng(0)
    act = np.argwhere(Ms > 0)
    for (i, j) in act[rng.choice(len(act), 4, replace=False)]:
        eps_ = 1e-4
        rp_ = rraw.clone(); rp_[i, j] += eps_
        rm_ = rraw.clone(); rm_[i, j] -= eps_
        with torch.no_grad():
            gfd = (float(J_of(rp_)) - float(J_of(rm_))) / (2 * eps_)
        fd.append(dict(pixel=[int(i), int(j)], autograd=float(g_ad[i, j]),
                       finite_diff=gfd,
                       rel_err=abs(float(g_ad[i, j]) - gfd) / max(abs(gfd), 1e-12)))
        log(f"  pixel {(int(i), int(j))}: autograd={float(g_ad[i,j]):+.6e} "
            f"FD={gfd:+.6e} rel={fd[-1]['rel_err']:.1e}")
    rep["gradient_check"] = fd
    assert max(f["rel_err"] for f in fd) < 1e-4

    # ------------------------------------------------------------------
    # Stage 1e: Fourier-order convergence of A at oblique incidence
    # ------------------------------------------------------------------
    log("== Stage 1e: order convergence of A (padded QNM winner) ==")
    conv = []
    for th, ph in ((0, 0), (20, 0), (20, 90)):
        row = dict(theta=th, phi=ph)
        for od in ([5, 5], [7, 7], [9, 9]):
            with torch.no_grad():
                sim = fm.build_sim(rho_pq, 850.0, 140.0, theta_deg=th,
                                   phi_deg=ph, order=od)
                A, _, _ = fm.a_ito(sim)
            row[str(od)] = float(A)
        conv.append(row)
        log(f"  th={th} ph={ph}: " + ", ".join(f"{k}: {v:.4f}" for k, v in
                                             row.items() if k[0] == "["))
    rep["order_convergence"] = conv

    # ------------------------------------------------------------------
    # Stage 1f: timing benchmark (forward + backward, 128 grid)
    # ------------------------------------------------------------------
    log("== Stage 1f: timing (fwd+bwd, one angle, 128x128) ==")
    tim = {}
    for od in ([5, 5], [7, 7]):
        rv = rho_pq.clone().requires_grad_(True)
        t0 = time.time()
        sim = fm.build_sim(rv, 850.0, 140.0, theta_deg=20.0, order=od)
        A, _, _ = fm.a_ito(sim); A.backward()
        tim[str(od)] = time.time() - t0
        log(f"  order {od}: {tim[str(od)]:.1f} s per angle (fwd+bwd)")
    rep["timing_s_per_angle"] = tim

    # ------------------------------------------------------------------
    # Stage 1g: angular response of the references -> beta calibration
    # ------------------------------------------------------------------
    log("== Stage 1g: reference angular curves (ORDER_FULL) + beta ==")
    curves = {}
    for name, (rho, P, h, _) in R.items():
        curves[name] = angle_curve(rho, P, h, rc.ANGLES_FULL, rc.ORDER_FULL)
        As = [c["A"] for c in curves[name]]
        log(f"  {name:22s}: A(FULL set) = {[round(a, 4) for a in As]} "
            f"spread={max(As)-min(As):.4f} min={min(As):.4f}")
    spreads = {k: max(c["A"] for c in v) - min(c["A"] for c in v)
               for k, v in curves.items() if k != "bare ITO"}
    med = float(np.median(list(spreads.values())))
    beta = float(np.clip(np.log(10.0) / max(med, 1e-6), *rc.BETA_RANGE))
    rep["beta_calibration"] = dict(rule=rc.BETA_RULE, spreads=spreads,
                                   median_spread=med, beta=beta,
                                   clip_range=rc.BETA_RANGE)
    log(f"  median reference spread = {med:.4f} -> beta = {beta:.2f}")
    # smooth-min vs hard min for the references at this beta
    for name, v in curves.items():
        As = torch.tensor([c["A"] for c in v])
        J = float(opt.smooth_min(As, opt.weights_for(rc.ANGLES_FULL), beta))
        rep["beta_calibration"].setdefault("J_vs_min", {})[name] = dict(
            J=J, A_min=float(As.min()), A_mean=float(As.mean()))
        log(f"  {name:22s}: J_robust={J:.4f} vs min={float(As.min()):.4f} "
            f"mean={float(As.mean()):.4f}")
    # dense evaluation curves (planes) for the figure, screen order for speed
    dense = {}
    for name, (rho, P, h, _) in R.items():
        dense[name] = {pl: angle_curve(rho, P, h, angs_, rc.ORDER_SCREEN)
                       for pl, angs_ in rc.ANGLES_EVAL_PLANES.items()}
    rep["reference_curves_full"] = curves
    rep["reference_curves_dense_screen_order"] = dense
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, pl in zip(axs, rc.ANGLES_EVAL_PLANES):
        for name, d in dense.items():
            ax.plot([c["theta"] for c in d[pl]], [c["A"] for c in d[pl]],
                    marker="o", ms=3, label=name)
        ax.set_title(f"A_ITO(lambda_E) vs theta, plane {pl} (lab-x pol, "
                     f"order {rc.ORDER_SCREEN})")
        ax.set_xlabel("theta (deg)"); ax.grid(alpha=.3)
    axs[0].set_ylabel("A_ITO"); axs[0].legend(fontsize=8)
    fig.savefig(FIG / "preflight_reference_angular.png", dpi=150,
                bbox_inches="tight"); plt.close(fig)

    # ------------------------------------------------------------------
    # Stage 1h: automatic run sizing from the budget and timing
    # ------------------------------------------------------------------
    t_scr = tim[str(rc.ORDER_SCREEN)] * len(rc.ANGLES_SCREEN)
    n_runs2 = len(rc.P_SCREEN) * len(rc.H_SCREEN) * len(rc.PAD_SCREEN) \
        * len(rc.SEEDS_SCREEN)
    n_iter2 = int(np.clip(rc.BUDGET_H["stage2"] * 3600 / (n_runs2 * t_scr),
                          *rc.N_ITER_SCREEN_RANGE))
    t_full = tim[str(rc.ORDER_FULL)] * len(rc.ANGLES_FULL)
    n_iter4 = int(np.clip(rc.BUDGET_H["stage4"] * 3600
                          / ((rc.N_FINALISTS + 1) * t_full), 60, rc.N_ITER_FULL))
    rep["sizing"] = dict(s_per_iter_screen=t_scr, n_runs_stage2=n_runs2,
                         n_iter_stage2=n_iter2, est_stage2_h=n_runs2 * n_iter2
                         * t_scr / 3600, s_per_iter_full=t_full,
                         n_iter_stage4=n_iter4,
                         est_stage4_h=(rc.N_FINALISTS + 1) * n_iter4 * t_full / 3600)
    log("  sizing: " + json.dumps(rep["sizing"]))
    rep["angular_domain"] = dict(
        authority="NONE found in repo (grep for NA / numerical aperture / "
                  "acceptance angle / high-NA over *.md,*.py,*.txt,*.csv,*.json)",
        assumption="modest +-30 deg cone (NA~0.5 in air), lab-frame x "
                   "polarization projected on the transverse plane; uniform "
                   "weights; screen set 3 angles, full set 5 angles; final "
                   "check on the phi=0, 90, 45 deg planes 0-40 deg",
        screen=rc.ANGLES_SCREEN, full=rc.ANGLES_FULL,
        eval_planes=rc.ANGLES_EVAL_PLANES)
    rep["wall_s"] = time.time() - t_start
    with open(OUT / "preflight.json", "w") as f:
        json.dump(rep, f, indent=1)
    write_md(rep)
    log(f"[done] preflight in {rep['wall_s']:.0f} s")


def write_md(rep):
    m = rep["materials"]; s = rep["symmetry"]; b = rep["beta_calibration"]
    L = ["# PREFLIGHT - robust ENZ energy-transfer inverse design", "",
         "Generated by `preflight.py` (reproducible; all numbers in "
         "`outputs/preflight.json`).", "",
         "## Stage 0 - ingest audit", "",
         f"- lambda_E = {m['lambda_E_nm']} nm. Origin: {m['lambda_E_origin']}.",
         f"- eps_ITO(lambda_E) from the measured CSV = {m['eps_ito_csv_at_lambda_E'][0]:.4f}"
         f" + {m['eps_ito_csv_at_lambda_E'][1]:.4f}i (target npz: "
         f"{m['eps_ito_target_npz'][0]:.4f} + {m['eps_ito_target_npz'][1]:.4f}i); "
         f"material ENZ crossing {m['material_ENZ_crossing_nm']} nm; d_ITO = {m['d_ito_nm']} nm; "
         f"n_glass = {m['n_glass']}.",
         f"- eps_aSi(lambda_E) (POSTECH measured file) = {m['eps_asi_postech_at_lambda_E'][0]:.4f}"
         f" (n = {m['n_asi']:.4f}, k = 0).",
         "- Reference structures (frozen, read-only):", ""]
    L += ["| reference | source | fill | S_flip(lr) | S_flip(ud) |",
          "|---|---|---|---|---|"]
    for k, v in rep["references"].items():
        L.append(f"| {k} | `{v['source']}` | {v['fill']:.3f} | {v['S_flip']:.3f} | {v['S_flipud']:.3f} |")
    L += ["", "## Stage 1a - symmetry preflight", "",
          "**Both historical Example6 fliplr symmetry projections were disabled "
          "in the new inverse-design path; no mirror symmetry is enforced.** "
          "(`optimizer.py`; the upstream notebook and "
          "`enz_inverse_design/optimize_enz_overlap.py` are untouched.)", "",
          f"- S_flip of the raw seed-333 random init: {s['random_init_raw']:.3f}",
          f"- after the historical projection `(rho+fliplr(rho))/2`: {s['historical_fliplr_projected']:.1e} (=0, symmetric)",
          f"- new path (mask + filter, no projection): {s['new_path_filtered_masked']:.3f} (lr), {s['new_path_flipud']:.3f} (ud)",
          f"- optimizer smoke run (2 iterations, tiny problem): S_flip init {s['optimizer_smoke_s_flip_init']:.3f} -> final {s['optimizer_smoke_s_flip_final']:.3f}",
          "", "## Stage 1b - energy conservation and diffraction-order audit", "",
          f"- {len(rep['energy_conservation'])} lossless (no-ITO) cases, P in {{750, 850, 1050, 1300}} nm, "
          "theta up to 40 deg, conical azimuths, lab-x / p / s inputs, ALL orders summed in TORCWA's p/s "
          f"basis: worst |R+T-1| = {max(abs(e['RT_minus_1']) for e in rep['energy_conservation']):.1e}; "
          "the number of propagating orders reported by TORCWA equals the analytic count in glass in every case "
          "(higher orders propagate in glass for P > lambda_E/n_glass = 992 nm and at oblique incidence).",
          "", "## Stage 1c - A_ITO identity at oblique incidence", "",
          "| reference | P | theta | phi | A = 1-R-T | volume integral | R | T | F_Ez | eta_z | x-pol frac R00 / T00 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in rep["a_identity"]:
        L.append(f"| {a['ref']} | {a['P']:.0f} | {a['theta']} | {a['phi']} | {a['A_rt']:.5f} | {a['A_vol']:.5f} | "
                 f"{a['R']:.4f} | {a['T']:.4f} | {a['F_Ez']:.3f} | {a['eta_z']:.3f} | {a['R00_cross_frac']:.3f} / {a['T00_cross_frac']:.3f} |")
    L += ["", "P_inc = 0.5 cos(theta) P^2 |E_inc|^2 with |E_inc| = 1 (p/s-notation source). "
          "Polarization conversion (TE/TM mixing) is included in R_total/T_total through the coherent p/s combination.",
          "", "## Stage 1d - gradient check", ""]
    for f in rep["gradient_check"]:
        L.append(f"- pixel {f['pixel']}: autograd {f['autograd']:+.6e}, central FD {f['finite_diff']:+.6e}, rel. err {f['rel_err']:.1e}")
    L += ["", "## Stage 1e - order convergence of A (padded QNM winner)", "",
          "| theta | phi | [5,5] | [7,7] | [9,9] |", "|---|---|---|---|---|"]
    for c in rep["order_convergence"]:
        L.append(f"| {c['theta']} | {c['phi']} | {c['[5, 5]']:.4f} | {c['[7, 7]']:.4f} | {c['[9, 9]']:.4f} |")
    L += ["", "## Stage 1f/1h - timing and run sizing", "",
          f"- {json.dumps(rep['timing_s_per_angle'])} s per angle (fwd+bwd, 128x128, 4 threads)",
          f"- {json.dumps(rep['sizing'])}",
          "", "## Stage 1g - angular domain and beta calibration", "",
          f"- Authority: {rep['angular_domain']['authority']}.",
          f"- ASSUMPTION: {rep['angular_domain']['assumption']}.",
          f"- screen set {rep['angular_domain']['screen']}; full set {rep['angular_domain']['full']}.",
          f"- beta rule: {b['rule']}; reference spreads {json.dumps({k: round(v, 4) for k, v in b['spreads'].items()})}; "
          f"median {b['median_spread']:.4f} -> **beta = {b['beta']:.2f}** (clip {b['clip_range']}).",
          "", "| reference | A(FULL set) | J_robust | min A | mean A |", "|---|---|---|---|---|"]
    for k, v in rep["reference_curves_full"].items():
        jv = b["J_vs_min"][k]
        L.append(f"| {k} | {[round(c['A'], 4) for c in v]} | {jv['J']:.4f} | {jv['A_min']:.4f} | {jv['A_mean']:.4f} |")
    L += ["", "![reference angular curves](outputs/figures/preflight_reference_angular.png)", ""]
    (HERE / "PREFLIGHT.md").write_text("\n".join(L))
    (OUT / "preflight.log").write_text("\n".join(LOG))


if __name__ == "__main__":
    main()
