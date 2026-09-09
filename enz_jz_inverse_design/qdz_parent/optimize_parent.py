"""Parent-mode topology optimization:   maximize eta_Dz  s.t.  Q_r ~ Q_target,  lambda_r ~ lambda_E.

LOSS (smooth unconstrained form; every term is logged separately each iteration)

    L = -log(eta_Dz + eps)  +  lambda_Q * [log(Q_proxy / Q_target)]^2
                            +  lambda_lam * [(lambda_r_proxy - lambda_E) / tol_nm]^2

SCALE AUDIT (why these weights are not arbitrary).  All three terms are expressed in the same
"factor-e / one-linewidth" units, so lambda_Q = lambda_lam = 1 makes them commensurate by construction:
    * the objective changes by 1.0 when eta_Dz changes by a factor e;
    * L_Q = 1.0 when Q_proxy is off target by a factor e (symmetric in over/under-shoot: exceeding the
      target is penalised exactly like falling short, so Q is never rewarded for being large);
    * L_lambda = 1.0 when the resonance is displaced by tol_nm, which is set to ONE TARGET LINEWIDTH
      lambda_E / Q_target - the constraint therefore tightens automatically as Q_target rises.
The magnitudes of all three terms and of their gradients are printed every iteration and stored in the
history, so it can be verified that eta_Dz never wins by abandoning the resonance.

GEOMETRY.  No extra penalty term is used: the geometry constraints are structural and identical to the
validated Jz/Fz campaign - a 128 x 128 design grid, a fixed hard air pad (forward.build_pad_mask), the
same Gaussian density filter (40 nm) and tanh projection with the same beta schedule, no imposed
symmetry, Adam with the same betas and a cosine learning-rate schedule.  lambda_geom is therefore 0
and is reported as such.

    python optimize_parent.py --h 525 --q-target 100 --tag pilot_h525_Q100 [--warm final3] [--iters 60]
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import config, forward as fwd, materials as mat                     # noqa: E402
from optimizer import (gaussian_kernel_fft, filter_rho, project_rho, preprocess,      # noqa: E402
                       binarization_metric, s_flip, s_flip_ud, s_inv)
import parent_fwd as pf, qproxy as qp                               # noqa: E402

GEO = config.GEO_DTYPE
DEV = config.DEVICE
WARM = {"final3": "final3_div_F1_P825_h500_pad0.12_s333_h525",
        "final1": "final1_F0_P825_h600_pad0.12_s1001_h575",
        "final0": "final0_F2_P850_h600_pad0.12_s333_P825",
        "final2": "final2_F0_P825_h600_pad0.12_s1001_pad0.08"}


def probe_metrics(rho_proj, P, h, lams, order, n_z, with_grad, want_center=True, ic=2):
    """W_Si at every probe wavelength (one RCWA solve each) + the full metric set at the centre."""
    Ws, center = [], None
    for k, lam in enumerate(lams):
        want = want_center and k == ic
        d = pf.parent_metrics(rho_proj, P, h, float(lam), order, n_z=n_z, with_grad=with_grad)
        Ws.append(d["W_Si"])
        if want:
            center = d
    return torch.stack(Ws), center


def run(P, h, Q_target, seed, n_iter, order, out_dir, *, pad_frac=0.12, nx=config.NX, n_z=5,
        rho_init=None, lam_Q=1.0, lam_lam=1.0, lam_shape=0.0, beta0=1.0, beta1=config.BETA_PROJ_MAX,
        lr0=config.LR_INITIAL, filter_nm=config.FILTER_RADIUS_NM, tol_factor=1.0,
        save_every=10, log=print, tag="", n_threads=4, n_probe=5, probe_span=0.25, span_acquire=1.0,
        acquire_frac=0.4):
    fwd.set_threads(n_threads)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    lam_E = pf.lambda_E(); w_E = qp.omega_of(lam_E)
    tol_nm = tol_factor * lam_E / float(Q_target)                # one target linewidth
    # probe-span schedule: start wide to ACQUIRE a resonance anywhere near lambda_E, then narrow to the
    # calibrated span (0.25 kappa) where the Lorentzian fit is accurate to ~5 % (qproxy docstring).
    n_acq = max(1, int(acquire_frac * n_iter))
    span_sched = np.concatenate([np.exp(np.linspace(np.log(span_acquire), np.log(probe_span), n_acq)),
                                 np.full(max(n_iter - n_acq, 0), probe_span)])
    lams, omegas, kappa_t = qp.probe_wavelengths(lam_E, Q_target, n=n_probe, span=probe_span)
    ic = int(np.argmin(np.abs(omegas - w_E)))
    g_fft = gaussian_kernel_fft(nx, nx, P / nx, P / nx, filter_nm)
    beta_sched = np.exp(np.linspace(np.log(beta0), np.log(beta1), n_iter))
    lr_sched = lr0 * 0.5 * (1 + np.cos(np.arange(n_iter) * np.pi / n_iter))
    M_np, mask_info = fwd.build_pad_mask(nx, P, pad_frac)
    M = torch.as_tensor(M_np, dtype=GEO, device=DEV)
    if rho_init is None:
        torch.manual_seed(int(seed))
        rho = filter_rho(torch.rand((nx, nx), dtype=GEO, device=DEV) * M, g_fft) * M
    else:
        rho = torch.as_tensor(np.asarray(rho_init), dtype=GEO, device=DEV) * M
    np.save(out / "rho_initial.npy", rho.cpu().numpy())
    momentum = torch.zeros_like(rho); velocity = torch.zeros_like(rho)
    keys = ("eta_Dz", "eta_Dz_P", "Q_proxy", "lambda_r_proxy", "kappa_proxy", "fit_resid", "well_posed",
            "U_mid", "W_Si", "W_layer", "S_Dz", "R", "T", "L_obj", "L_Q", "L_lam", "L_total",
            "grad_norm", "binarization", "beta_proj", "lr", "shape_loss", "probe_span",
            "curvature_u", "x0_robust", "t")
    hist = {k: [] for k in keys}
    t0 = time.time()
    for it in range(n_iter):
        rho.requires_grad_(True)
        rho_proj, _ = preprocess(rho, M, g_fft, beta_sched[it])
        lams, omegas, kappa_t = qp.probe_wavelengths(lam_E, Q_target, n=n_probe, span=float(span_sched[it]))
        ic = int(np.argmin(np.abs(omegas - w_E)))
        W, cen = probe_metrics(rho_proj, P, h, lams, order, n_z, True, ic=ic)
        fit = qp.lorentz_fit(omegas, W, w_E)                       # reporting / diagnostics
        L_Q, L_lam, diag = qp.constraint_losses(omegas, W, w_E, Q_target, tol_nm, lam_E)   # optimized form
        L_shape, _, _ = qp.shape_loss(omegas, W, w_E, kappa_t)
        L_obj = -torch.log(cen["eta_Dz"] + 1e-30)
        loss = L_obj + lam_Q * L_Q + lam_lam * L_lam + lam_shape * L_shape
        loss.backward()
        with torch.no_grad():
            grad = rho.grad; rho.grad = None
            gnorm = float(torch.linalg.norm(grad))
            sh = L_shape.detach()
            f = pf.to_floats(cen)
            for k in ("eta_Dz", "eta_Dz_P", "U_mid", "W_Si", "W_layer", "S_Dz", "R", "T"):
                hist[k].append(f[k])
            Qp = float(fit["Q"]) if fit["well_posed"] else float("nan")
            lrp = float(fit["lambda_r"]) if fit["well_posed"] else float("nan")
            hist["Q_proxy"].append(Qp); hist["lambda_r_proxy"].append(lrp)
            hist["curvature_u"].append(float(diag["u"])); hist["x0_robust"].append(float(diag["x0_robust"]))
            hist["kappa_proxy"].append(float(fit["kappa"])); hist["fit_resid"].append(float(fit["rel_resid"]))
            hist["well_posed"].append(bool(fit["well_posed"])); hist["shape_loss"].append(float(sh))
            hist["L_obj"].append(float(L_obj)); hist["L_Q"].append(float(L_Q)); hist["L_lam"].append(float(L_lam))
            hist["L_total"].append(float(loss)); hist["grad_norm"].append(gnorm)
            hist["binarization"].append(binarization_metric(rho_proj))
            hist["beta_proj"].append(float(beta_sched[it])); hist["lr"].append(float(lr_sched[it]))
            hist["probe_span"].append(float(span_sched[it]))
            hist["t"].append(time.time() - t0)
            momentum = config.ADAM_B1 * momentum + (1 - config.ADAM_B1) * (-grad)
            velocity = config.ADAM_B2 * velocity + (1 - config.ADAM_B2) * grad ** 2
            rho = rho.detach() + lr_sched[it] * (momentum / (1 - config.ADAM_B1 ** (it + 1))) \
                / torch.sqrt(velocity / (1 - config.ADAM_B2 ** (it + 1)) + config.ADAM_EPS)
            rho.clamp_(0, 1); rho = rho * M
            if (it % save_every == 0) or it == n_iter - 1:
                np.save(out / f"rho_proj_it{it:04d}.npy", rho_proj.detach().cpu().numpy())
            log(f"[{tag}] it {it:3d} eta={f['eta_Dz']:.5f} Q={Qp:8.1f} lam_r={lrp:8.2f} u={float(diag['u']):6.3f} "
                f"| L_obj={float(L_obj):+.3f} L_Q={float(L_Q):.3f} L_lam={float(L_lam):.3f} L_sh={float(L_shape):.4f} L={float(loss):+.3f} "
                f"| U_mid={f['U_mid']:.2f} R={f['R']:.3f} fitres={float(fit['rel_resid']):.2e} |g|={gnorm:.2e} "
                f"bin={hist['binarization'][-1]:.3f} t={time.time()-t0:.0f}s")
        del cen, W
    # ---- final soft and hard-binary evaluation --------------------------------
    with torch.no_grad():
        rho_proj, rho_bar = preprocess(rho, M, g_fft, beta_sched[-1])
        rho_hard = (rho_proj > 0.5).to(GEO) * M
        fin = {}
        for name, r_ in (("soft", rho_proj), ("hard", rho_hard)):
            lf, of, _ = qp.probe_wavelengths(lam_E, Q_target, n=n_probe, span=probe_span)
            Wf, cf = probe_metrics(r_, P, h, lf, order, 9, False, ic=int(np.argmin(np.abs(of - w_E))))
            ff = qp.lorentz_fit(of, Wf, w_E)
            LQf, LLf, dgf = qp.constraint_losses(of, Wf, w_E, Q_target, tol_nm, lam_E)
            fin[name] = dict(pf.to_floats(cf),
                             Q_proxy=(float(ff["Q"]) if ff["well_posed"] else None),
                             lambda_r_proxy=(float(ff["lambda_r"]) if ff["well_posed"] else None),
                             kappa_proxy=float(ff["kappa"]), fit_rel_resid=float(ff["rel_resid"]),
                             well_posed=bool(ff["well_posed"]), curvature_u=float(dgf["u"]),
                             L_Q=float(LQf), L_lam=float(LLf))
    np.save(out / "rho_raw_final.npy", rho.cpu().numpy())
    np.save(out / "rho_filtered_final.npy", rho_bar.cpu().numpy())
    np.save(out / "rho_proj_final.npy", rho_proj.cpu().numpy())
    np.save(out / "rho_hard_binary.npy", rho_hard.cpu().numpy())
    np.save(out / "design_mask.npy", M_np)
    res = dict(tag=tag, campaign="qdz_parent", P=float(P), h=float(h), Q_target=float(Q_target),
               lam_E=lam_E, tol_nm=float(tol_nm), pad_frac=float(pad_frac), seed=int(seed), n_iter=int(n_iter),
               order=list(order), nx=nx, n_z=int(n_z), n_probe=int(n_probe), probe_span=float(probe_span),
               probe_lams=[float(x) for x in lams], probe_omegas=[float(x) for x in omegas], kappa_target=float(kappa_t),
               lambda_Q=float(lam_Q), lambda_lambda=float(lam_lam), lambda_shape=float(lam_shape), lambda_geom=0.0,
               weights_rationale="all three loss terms are unity at a factor-e error (objective, Q) or at one "
                                 "target linewidth (wavelength); geometry constraints are structural (filter, "
                                 "projection, hard pad mask), so no geometry penalty term is used",
               mask=mask_info, filter_nm=float(filter_nm), lr0=float(lr0), beta_proj=[beta0, beta1],
               warm_start=(rho_init is not None), soft=fin["soft"], hard=fin["hard"],
               fill_fraction=float(rho_hard.mean()), fill_fraction_active=float(rho_hard.sum() / M.sum()),
               pad_leak=dict(soft=float((rho_proj * (1 - M)).abs().max()), hard=float((rho_hard * (1 - M)).abs().max())),
               s_flip_final=dict(lr=s_flip(rho_hard), ud=s_flip_ud(rho_hard), inv=s_inv(rho_hard)),
               wall_s=time.time() - t0, history=hist)
    json.dump(res, open(out / "result.json", "w"), indent=1, default=float)
    log(f"[{tag}] done: eta_soft={fin['soft']['eta_Dz']:.5f} eta_hard={fin['hard']['eta_Dz']:.5f} "
        f"Q_hard={fin['hard']['Q_proxy']} lam_hard={fin['hard']['lambda_r_proxy']} u={fin['hard']['curvature_u']:.3f} "
        f"U_mid={fin['hard']['U_mid']:.2f} fill={res['fill_fraction']:.3f} wall={res['wall_s']:.0f}s")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=float, default=825.0)
    ap.add_argument("--h", type=float, required=True)
    ap.add_argument("--q-target", type=float, required=True)
    ap.add_argument("--pad", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=333)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--order", type=int, nargs=2, default=[5, 5])
    ap.add_argument("--n-z", type=int, default=5)
    ap.add_argument("--n-probe", type=int, default=5)
    ap.add_argument("--warm", default=None, help="final3|final1|final0|final2 or a path to a .npy density")
    ap.add_argument("--lam-Q", type=float, default=1.0)
    ap.add_argument("--lam-lam", type=float, default=1.0)
    ap.add_argument("--lam-shape", type=float, default=0.0)
    ap.add_argument("--tol-factor", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=config.LR_INITIAL)
    ap.add_argument("--beta1", type=float, default=config.BETA_PROJ_MAX)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--span", type=float, default=0.25)
    ap.add_argument("--span-acquire", type=float, default=1.0)
    a = ap.parse_args()
    tag = a.tag or f"h{a.h:.0f}_Q{a.q_target:.0f}_{'warm_' + a.warm if a.warm else 's' + str(a.seed)}"
    out = Path(a.out) if a.out else HERE / "pilot" / tag
    rho_init = None
    if a.warm:
        p = Path(a.warm) if a.warm.endswith(".npy") else HERE.parent / "outputs" / "stage3" / "runs" / WARM[a.warm] / "rho_hard_binary.npy"
        rho_init = np.load(p)
        print(f"[warm] {p}")
    logf = open(HERE / "logs" / f"{tag}.log", "a")
    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()
    run(a.P, a.h, a.q_target, a.seed, a.iters, a.order, out, pad_frac=a.pad, n_z=a.n_z, rho_init=rho_init,
        lam_Q=a.lam_Q, lam_lam=a.lam_lam, lam_shape=a.lam_shape, tol_factor=a.tol_factor, lr0=a.lr, beta1=a.beta1,
        probe_span=a.span, span_acquire=a.span_acquire, n_probe=a.n_probe, log=log, tag=tag, n_threads=a.threads)


if __name__ == "__main__":
    main()
