"""Single angle-aware wide-FOV R-type optimization run (both methods),
checkpointed and idempotent.

usage: python wf_optimize.py <A|B> <P> <H> <seed> <iters> <stage>
Angular minibatches (5 states) from iteration 0; hard-angle mining every
10 iters; curriculum never drops >=60 deg; robust objective
(0.4 mean + 0.35 softmin + 0.25 tail) - leakage/absorption penalties +
smooth PB phase term at theta=0. Method A adds soft exact-multipole
ED/MD gates at theta=0 only. Coarse solves at order [7,7], mining pool
at [5,5], hard-binary finalization + full-pool eval at [9,9].
"""
import csv
import hashlib
import json
import sys
import time

import numpy as np
import torch

import wf_core as wf
import rt_core as rc
from rt_optimize import fab_metrics

LEDGER = wf.HERE / 'results' / 'widefov_master_ledger.csv'
ORDER_OPT = (7, 7)
ORDER_POOL = (5, 5)
ORDER_FINAL = (9, 9)


def seed_latent(method, P, H, seed, n_grid):
    g = torch.Generator().manual_seed(seed)
    rgen = np.random.default_rng(seed * 1000 + int(P) + int(H))
    ax = (torch.arange(n_grid) + 0.5) / n_grid * P - P / 2
    X, Y = torch.meshgrid(ax, ax, indexing='ij')
    rd = rc.r_design(P)
    blob = torch.rand(n_grid, n_grid, generator=g)
    blob = torch.nn.functional.conv2d(
        blob[None, None], rc.conic_filter_kernel(n_grid, P, 30.0),
        padding='same')[0, 0]
    blob = (blob - blob.mean()) / (blob.std() + 1e-9)
    if method == 'A':
        # paper-inspired anisotropic rectangle seed, shrunk to fit the
        # rotation-safe envelope, + smooth perturbation
        scale = min(1.0, 0.92 * rd / 93.3)
        wx, wy = 160.0 * scale, 96.0 * scale
        rect = ((X.abs() <= wx / 2) & (Y.abs() <= wy / 2)).float()
        rect = torch.nn.functional.conv2d(
            rect[None, None], rc.conic_filter_kernel(n_grid, P, 20.0),
            padding='same')[0, 0]
        x = 3.0 * (2 * rect - 0.6) + 0.8 * blob
    else:
        # smooth D2-compatible freeform seed; weak isotropic centering
        # bump + weak quadrupolar degeneracy-breaker (both signs across
        # seeds); does NOT encode the paper rectangle
        bump = torch.exp(-((X ** 2 + Y ** 2) / (0.75 * rd) ** 2) ** 2)
        qsign = 1.0 if rgen.random() < 0.5 else -1.0
        quad = qsign * (X ** 2 - Y ** 2) / rd ** 2
        x = 2.2 * blob + 1.2 * (2 * bump - 0.75) + 0.6 * quad
    return x.clone().requires_grad_(True)


def mode_penalty(rho_bar, P, H):
    fx = rc.moments_families(rho_bar, P, H, wf.LAM0, ORDER_OPT, 'x')
    fy = rc.moments_families(rho_bar, P, H, wf.LAM0, ORDER_OPT, 'y')
    pen = (rc.softgate(fx['f_ED'], 0.50) + rc.softgate(fx['px_in_ED'], 0.80)
           + rc.softgate(fy['f_MD'], 0.50)
           + rc.softgate(fy['mx_in_MD'], 0.80))
    return 0.6 * pen, {'f_ED_x': float(fx['f_ED']),
                       'f_EQ_x': float(fx['f_EQ']),
                       'f_MD_y': float(fy['f_MD'])}


def run(method, P, H, seed, iters, stage, src_tag=None):
    tag = f'{method}_P{P:.0f}_H{H:.0f}_s{seed}_wf'
    outdir = wf.HERE / stage / tag
    outdir.mkdir(parents=True, exist_ok=True)
    final_p = outdir / 'final.json'
    if final_p.exists():
        print(f'{tag}: already complete', flush=True)
        return
    order_opt = (9, 9) if stage == 'refinement' else ORDER_OPT
    order_pool = (7, 7) if stage == 'refinement' else ORDER_POOL
    n_grid = 96
    mask = rc.design_mask(n_grid, P)
    kern = rc.conic_filter_kernel(n_grid, P, 15.0)
    ck = outdir / 'checkpoint.npz'
    pool_rows = []
    worst = []
    if ck.exists():
        z = np.load(ck, allow_pickle=True)
        x = torch.tensor(z['x'], requires_grad=True)
        it0 = int(z['it'])
        hist = list(z['hist'])
        worst = [tuple(w) for w in z['worst']] if len(z['worst']) else []
        pool_rows = list(z['pool']) if 'pool' in z else []
        print(f'{tag}: resume at iter {it0}', flush=True)
    elif (outdir / 'warm.npy').exists() or src_tag:
        if not (outdir / 'warm.npy').exists():
            src = wf.HERE / 'coarse' / src_tag / 'checkpoint.npz'
            np.save(outdir / 'warm.npy', np.load(src)['x'])
        x = torch.tensor(np.load(outdir / 'warm.npy')).clone() \
            .requires_grad_(True)
        it0, hist = 0, []
        print(f'{tag}: warm start ({src_tag})', flush=True)
    else:
        x = seed_latent(method, P, H, seed, n_grid)
        it0, hist = 0, []
    opt = torch.optim.Adam([x], lr=0.04)
    betas = [(0.0, 2.0), (0.4, 4.0), (0.65, 8.0), (0.85, 16.0)]
    t_start = time.time()
    for it in range(it0, iters):
        frac = it / iters
        beta = [b for f, b in betas if frac >= f][-1]
        rho_bar = rc.filt_project(torch.sigmoid(x), kern, beta, mask=mask)
        batch = wf.minibatch(it, iters, seed, worst)
        scores, R0 = [], None
        for th, ph in batch:
            R, T = wf.jones_angle(rho_bar, P, H, th, ph, order=order_opt)
            scores.append(wf.angle_scores(R, T))
            if th == 0.0 and R0 is None:
                R0 = R
        L = wf.robust_loss(scores, R0)
        mstats = {}
        if method == 'A':
            pen, mstats = mode_penalty(rho_bar, P, H)
            L = L + pen
        opt.zero_grad()
        L.backward()
        opt.step()
        Rcs = [float(s['Rc']) for s in scores]
        hist.append([it, float(L), float(np.mean(Rcs)), float(np.min(Rcs)),
                     float(np.mean([float(s['co']) for s in scores])),
                     float(np.mean([float(s['A']) for s in scores]))])
        if it % 10 == 9:
            with torch.no_grad():
                rho_e = rc.filt_project(torch.sigmoid(x), kern, beta,
                                        mask=mask)
                rows, worst = wf.full_pool_eval(rho_e, P, H,
                                                order=order_pool)
            ps = wf.pool_summary(rows)
            binar = float((2 * rho_e - 1).abs().mean())
            pool_rows.append([it, ps['Rc_mean'], ps['Rc_min'],
                              ps['Rc_tail25'], ps['Rc_omega'],
                              ps['co_mean'], ps['co_max'], ps['T_mean'],
                              ps['A_mean'], ps['theta_worst'],
                              ps['phi_worst'], binar])
            np.savez(ck, x=x.detach().numpy(), it=it + 1,
                     hist=np.array(hist), worst=np.array(worst),
                     pool=np.array(pool_rows))
            print(f'{tag} it{it} L={float(L):+.4f} pool: mean='
                  f'{ps["Rc_mean"]:.3f} min={ps["Rc_min"]:.3f} '
                  f'omega={ps["Rc_omega"]:.3f} worst=('
                  f'{ps["theta_worst"]:.0f},{ps["phi_worst"]:.0f}) '
                  f'{mstats}', flush=True)
    # hard-binary finalization; full-pool eval at ORDER_FINAL
    with torch.no_grad():
        rho_bar = rc.filt_project(torch.sigmoid(x), kern, 16.0, mask=mask)
        b = rho_bar.numpy() > 0.5
        rho_bin = torch.tensor(b.astype(np.float32)) * mask
        b = rho_bin.numpy() > 0.5
        rows, _ = wf.full_pool_eval(rho_bin, P, H, order=ORDER_FINAL)
        ps = wf.pool_summary(rows)
        R, T = wf.jones_angle(rho_bin, P, H, 0.0, 0.0, order=ORDER_FINAL)
        m0 = rc.device_metrics(R, T)
        fams = {}
        for pol in ('x', 'y'):
            f = rc.moments_families(rho_bin, P, H, wf.LAM0, (9, 9), pol,
                                    n_xy=48, nz=7)
            fams.update({f'{k}_{pol}': float(f[k]) for k in
                         ('f_ED', 'f_MD', 'f_EQ', 'f_MQ', 'px_in_ED',
                          'mx_in_MD', 'my_in_MD')})
    fm = fab_metrics(b, P)
    np.save(outdir / 'rho_binary.npy', rho_bin.numpy())
    rec = {'tag': tag, 'method': method, 'P': P, 'H': H, 'seed': seed,
           'iters': iters, 'n_grid': n_grid, 'order_opt': order_opt[0],
           'order_final': ORDER_FINAL[0], 'stage': stage,
           'padding_nm': rc.padding(P), 'design_radius_nm': rc.r_design(P),
           'runtime_s': time.time() - t_start,
           'sha256_rho': hashlib.sha256(
               (outdir / 'rho_binary.npy').read_bytes()).hexdigest(),
           'R_cross0': m0['R_cross'], 'R_co0': m0['R_co'],
           'pb_phase_err_deg0': m0['pb_phase_err_deg'],
           'abs_rx0': m0['abs_rx'], 'abs_ry0': m0['abs_ry'],
           **ps, **fams, **fm}
    final_p.write_text(json.dumps(rec, indent=1))
    wf.write_rows(outdir / 'pool_final.csv', rows)
    with open(outdir / 'history.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['it', 'L', 'batch_Rc_mean', 'batch_Rc_min',
                    'batch_co_mean', 'batch_A_mean'])
        w.writerows(hist)
    with open(outdir / 'pool_history.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['it', 'Rc_mean', 'Rc_min', 'Rc_tail25', 'Rc_omega',
                    'co_mean', 'co_max', 'T_mean', 'A_mean', 'theta_worst',
                    'phi_worst', 'binarity'])
        w.writerows(pool_rows)
    new = not LEDGER.exists()
    with open(LEDGER, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rec.keys()))
        if new:
            w.writeheader()
        w.writerow(rec)
    print(f'{tag}: DONE Rc0={m0["R_cross"]:.3f} pool mean='
          f'{ps["Rc_mean"]:.3f} min={ps["Rc_min"]:.3f} omega='
          f'{ps["Rc_omega"]:.3f} co={ps["co_mean"]:.3f} '
          f'({rec["runtime_s"]:.0f}s)', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    run(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]),
        int(sys.argv[4]), int(sys.argv[5]), sys.argv[6],
        sys.argv[7] if len(sys.argv) > 7 else None)
