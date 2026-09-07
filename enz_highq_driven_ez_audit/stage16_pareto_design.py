"""Stage 16 (reached because the diagnostics found no coexistence):
focused epsilon-constraint Pareto inverse design.

    maximize  F_Ez(lambda_E; real ITO loss)                    [driven field]
    s.t.      Q_rad >= Q_target                                [radiative Q]
              |lambda_pole - lambda_E| <= lambda_E / (2 Q_target)
              positive air padding (hard mask), NO mirror symmetry.

Differentiable Q_rad proxy: with the ITO loss switched off (s = 0) the only
loss channel is radiation, so the LOSSLESS structure's Lorentzian linewidth
IS the radiative linewidth.  The resonance indicator is the driven ITO Ez
intensity I0(lambda) = <|Ez|^2>_ITO at s = 0, sampled at lambda_E and
lambda_E +- delta, delta = lambda_E / (2 Q_target) (half-width of a
Q_target Lorentzian).  Constraint (relu^2 penalty on the log ratio):
        I0(lambda_E +- delta) <= 0.5 I0(lambda_E)
i.e. the lossless ITO-field resonance is centred within +-delta of lambda_E
and has half-width <= delta  ->  Q_rad >= Q_target (proxy; certified
afterwards with the field-overlap loss-scaling tracker at order [7,7]).
Using the ITO Ez intensity as the indicator selects ENZ-participating
modes by construction.
Seeds: P925 robust (925/240, pad 4 %) and padded QNM (850/140, 86 nm);
Q_target in {30, 100, 300} chosen from the measured Q_rad distribution
(20-206 for the accessible branches, ~10^2-10^3 for the dark ones).
Architecture: Example6 filter/projection/Adam (enz_robust_aito_campaign/
optimizer.py), both fliplr projections absent.
"""
import json
import sys
import time

import numpy as np
import torch

import common as cm
import forward_multi as fm
import poles as pl
import optimizer as opt          # enz_robust_aito_campaign (no fliplr)

torch.set_num_threads(4)
OUT = cm.OUT / "stage16"
OUT.mkdir(parents=True, exist_ok=True)
LOGF = OUT / "stage16.log"


def log(s):
    print(s, flush=True)
    with open(LOGF, "a") as f:
        f.write(s + "\n")


SEEDS = {"P925": ("P925 robust", 0.04), "padQNM": ("padded QNM", 86.33 / 850.0)}
Q_TARGETS = [30.0, 100.0, 300.0]
ORDER_OPT, N_ITER, MU, NXY, NZ = [5, 5], 80, 30.0, 32, 3


def f_ez_diff(rho_proj, P, h, lam, s):
    sim = fm.build_sim(rho_proj, P, h, lam=lam, order=ORDER_OPT, ito_loss_scale=s)
    x, y = fm.cell_axes(P, NXY)
    acc = 0.0
    for zp in (np.arange(NZ) + 0.5) * cm.D_ITO / NZ:
        E, _ = sim.field_xy(1, x, y, float(zp))
        acc = acc + torch.mean(E[2].real ** 2 + E[2].imag ** 2)
    return acc / NZ


def run(seed_key, Q_t):
    name, pad_frac = SEEDS[seed_key]
    rho0, P, h = cm.load_candidate(name)
    tag = f"{seed_key}_Qt{Q_t:.0f}"
    d = OUT / tag
    if (d / "result.json").exists():
        return cm.jload(d / "result.json")
    d.mkdir(exist_ok=True)
    nx = 128
    M_np, minfo = opt.build_pad_mask(nx, P, pad_frac)
    M = torch.as_tensor(M_np, dtype=cm.GEO)
    g = opt.gaussian_kernel_fft(nx, nx, P / nx, P / nx, 40.0)
    delta = cm.LAMBDA_E / (2 * Q_t)
    lams_side = (cm.LAMBDA_E - delta, cm.LAMBDA_E + delta)
    beta_sched = np.exp(np.linspace(np.log(8.0), np.log(1000.0), N_ITER))
    lr_sched = 0.02 * 0.5 * (1 + np.cos(np.arange(N_ITER) * np.pi / N_ITER))
    rho = (rho0 * M).clone()                       # warm start, no symmetrization
    mom = torch.zeros_like(rho); vel = torch.zeros_like(rho)
    hist = []
    t0 = time.time()
    for it in range(N_ITER):
        rho.requires_grad_(True)
        rp = opt.project_rho(opt.filter_rho(rho * M, g), beta_sched[it]) * M
        F1 = f_ez_diff(rp, P, h, cm.LAMBDA_E, 1.0)
        I0c = f_ez_diff(rp, P, h, cm.LAMBDA_E, 0.0)
        pen = 0.0
        sides = []
        for ls in lams_side:
            I0s = f_ez_diff(rp, P, h, ls, 0.0)
            sides.append(float(I0s / I0c))
            pen = pen + torch.clamp(torch.log(I0s / I0c) - np.log(0.5), min=0.0) ** 2   # log form (conditioning)
        J = torch.log(F1) - MU * pen
        (-J).backward()
        with torch.no_grad():
            grad = rho.grad; rho.grad = None
            mom = 0.9 * mom + 0.1 * (-grad); vel = 0.999 * vel + 0.001 * grad ** 2
            rho = rho.detach() + lr_sched[it] * (mom / (1 - 0.9 ** (it + 1))) / torch.sqrt(vel / (1 - 0.999 ** (it + 1)) + 1e-8)
            rho[rho > 1] = 1; rho[rho < 0] = 0; rho = rho * M
        hist.append(dict(it=it, F_Ez=float(F1), I0_center=float(I0c), side_ratios=sides, penalty=float(pen)))
        if it % 10 == 0 or it == N_ITER - 1:
            log(f"[{tag}] it {it:3d} F_Ez(s=1)={float(F1):.3f} I0(s=0)={float(I0c):.1f} sides={[round(v, 3) for v in sides]} pen={float(pen):.4f} t={time.time()-t0:.0f}s")
    with torch.no_grad():
        rp = opt.project_rho(opt.filter_rho(rho * M, g), beta_sched[-1]) * M
        hard = (rp > 0.5).to(cm.GEO) * M
    np.save(d / "rho_hard_binary.npy", hard.numpy())
    # ---- certification at order [7,7]: harness + reconnaissance + tracking --
    m = cm.driven_metrics(hard, P, h)
    r, t = pl.rt_scan(hard, P, h, 1.0, pl.SCAN_COARSE)
    sig = pl.significant_poles(pl.SCAN_COARSE, r, t)
    near = min(sig, key=lambda p: abs(p["lambda_nm"] - cm.LAMBDA_E)) if sig else None
    fit, rows, en = {}, [], None
    if near:
        rows = pl.track_branch(hard, P, h, near["omega"], log=log, tag=tag)
        fit = pl.fit_gamma(rows)
        if fit.get("lossless_pole"):
            en = cm.energy_participation(hard, P, h, fit["lossless_pole"]["lambda_nm"], 0.0)
    res = dict(tag=tag, seed=name, Q_target=Q_t, P=P, h=h, pad_frac=pad_frac, mask=minfo,
               F_Ez=m["F_Ez"], A=m["A_rt"], eta_z=m["eta_z"], peak_Ez2=m["peak_Ez2"],
               near_pole=(dict(lambda_nm=near["lambda_nm"], Q=near["Q"]) if near else None),
               Q_rad=fit.get("Q_rad"), Q_nr=fit.get("Q_nr"), gamma_ratio=fit.get("gamma_ratio"),
               linearity_resid=fit.get("linearity_resid"), rows=rows,
               eta_ENZ_z_modal=(en["eta_ENZ_z"] if en else None), ito_E_frac_modal=(en["ito_E_energy_fraction"] if en else None),
               s_flip=opt.s_flip(hard), fill=float(hard.mean()), history=hist, wall_s=time.time() - t0)
    cm.jdump(res, d / "result.json")
    log(f"[{tag}] DONE F_Ez={m['F_Ez']:.3f} A={m['A_rt']:.3f} near pole {res['near_pole']} Q_rad={fit.get('Q_rad')} "
        f"Q_nr={fit.get('Q_nr')} ratio={fit.get('gamma_ratio')} etaENZ={res['eta_ENZ_z_modal']} S_flip={res['s_flip']:.2f}")
    return res


results = []
for seed_key in SEEDS:
    for Q_t in Q_TARGETS:
        results.append(run(seed_key, Q_t))
cm.jdump(results, OUT / "stage16_results.json")

# Pareto frontier figure incl. existing branches
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
t3 = pd.read_csv(cm.OUT / "stage3_branches_table.csv")
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.scatter(t3["Q_rad"], t3["F_Ez_lambdaE"], c="gray", s=40, label="existing branches (Stage 3)")
for _, r in t3.iterrows():
    ax.annotate(r["branch"].replace(" robust", "").replace("padded ", "pad "), (r["Q_rad"], r["F_Ez_lambdaE"]), fontsize=6, textcoords="offset points", xytext=(3, 3))
for r in results:
    if r["Q_rad"]:
        ax.scatter(r["Q_rad"], r["F_Ez"], c="tab:red", s=70, marker="*")
        ax.annotate(r["tag"], (r["Q_rad"], r["F_Ez"]), fontsize=7, color="tab:red", textcoords="offset points", xytext=(3, -8))
ax.set_xscale("log"); ax.set_xlabel("certified Q_rad (field-overlap loss scaling, order [7,7])"); ax.set_ylabel("driven F_Ez(lambda_E), real ITO loss")
ax.set_title("Stage 16 - epsilon-constraint designs (red stars) vs existing branches"); ax.grid(alpha=.3); ax.legend()
fig.savefig(cm.FIG / "stage16_pareto_frontier.png", dpi=150, bbox_inches="tight")
log("[stage16] done")
