"""Single R-type topology-optimization run (both methods), checkpointed
and idempotent.

usage: python rt_optimize.py <method A|B> <P> <H> <seed> <iters> <n_grid>
                             <order> <stage coarse|refine>
Writes <stage>/<tag>/ {checkpoint.npz, final.json, rho_binary.npy,
history.csv}; appends results/rtype_master_ledger.csv.
"""
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import torch

import rt_core as rc

LEDGER = rc.HERE / 'results' / 'rtype_master_ledger.csv'


def fab_metrics(b, P):
    n = b.shape[0]
    px = P / n
    from scipy.ndimage import label
    ncomp = label(b)[1]
    ncomp_air = label(~b)[1]

    def min_width(m):
        k, mm = 0, m.copy()
        n0 = label(m)[1]
        if n0 == 0:
            return 0.0
        while True:
            m2 = ndi.binary_erosion(mm)
            if label(m2)[1] < n0 or m2.sum() == 0:
                break
            mm = m2
            k += 1
        return (2 * k + 1) * px
    # edge clearance / radial extent
    ax = (np.arange(n) + 0.5) / n * P - P / 2
    X, Y = np.meshgrid(ax, ax, indexing='ij')
    rmax = float(np.sqrt(X[b] ** 2 + Y[b] ** 2).max()) if b.any() else 0.0
    return {'n_components': int(ncomp), 'n_air_components': int(ncomp_air),
            'min_si_linewidth_nm': min_width(b),
            'min_air_gap_nm': min_width(~b & (X ** 2 + Y ** 2
                                              <= rc.r_design(P) ** 2)),
            'radial_extent_nm': rmax,
            'edge_clearance_nm': rc.r_design(P) - rmax,
            'fill': float(b.mean())}


def mode_penalty(rho_bar, P, H, order):
    fx = rc.moments_families(rho_bar, P, H, rc.LAM0, order, 'x')
    fy = rc.moments_families(rho_bar, P, H, rc.LAM0, order, 'y')
    pen = (rc.softgate(fx['f_ED'], 0.55) + rc.softgate(fx['px_in_ED'], 0.80)
           + rc.softgate(fy['f_MD'], 0.55) + rc.softgate(fy['mx_in_MD'], 0.80))
    stats = {'f_ED_x': float(fx['f_ED']), 'f_MD_x': float(fx['f_MD']),
             'f_EQ_x': float(fx['f_EQ']), 'f_MQ_x': float(fx['f_MQ']),
             'px_in_ED_x': float(fx['px_in_ED']),
             'f_ED_y': float(fy['f_ED']), 'f_MD_y': float(fy['f_MD']),
             'f_EQ_y': float(fy['f_EQ']), 'f_MQ_y': float(fy['f_MQ']),
             'mx_in_MD_y': float(fy['mx_in_MD'])}
    return 0.6 * pen, stats


def run(method, P, H, seed, iters, n_grid, order_n, stage):
    order = (order_n, order_n)
    tag = f'{method}_P{P:.0f}_H{H:.0f}_s{seed}_g{n_grid}_o{order_n}'
    outdir = rc.HERE / stage / tag
    outdir.mkdir(parents=True, exist_ok=True)
    final_p = outdir / 'final.json'
    if final_p.exists():
        print(f'{tag}: already complete', flush=True)
        return
    mask = rc.design_mask(n_grid, P)
    kern = rc.conic_filter_kernel(n_grid, P, 15.0)
    g = torch.Generator().manual_seed(seed)
    ck = outdir / 'checkpoint.npz'
    if ck.exists():
        z = np.load(ck)
        x = torch.tensor(z['x'], requires_grad=True)
        it0 = int(z['it'])
        hist = list(np.atleast_2d(z['hist']))
        print(f'{tag}: resume at iter {it0}', flush=True)
    else:
        blob = torch.rand(n_grid, n_grid, generator=g)
        blob = torch.nn.functional.conv2d(
            blob[None, None], rc.conic_filter_kernel(n_grid, P, 30.0),
            padding='same')[0, 0]
        blob = (blob - blob.mean()) / (blob.std() + 1e-9)
        # seeded anisotropic elliptical bias (D2-compatible): breaks the
        # isotropic rx=ry saddle without prescribing a mechanism
        rgen = np.random.default_rng(seed * 1000 + int(P) + int(H))
        aspect = float(rgen.uniform(1.4, 2.4))
        ax_, ay_ = (aspect, 1.0) if rgen.random() < 0.5 else (1.0, aspect)
        r0 = rc.r_design(P) * float(rgen.uniform(0.55, 0.8))
        axv = (torch.arange(n_grid) + 0.5) / n_grid * P - P / 2
        X, Y = torch.meshgrid(axv, axv, indexing='ij')
        ell = torch.exp(-((X / (r0 * ax_)) ** 2
                          + (Y / (r0 * ay_)) ** 2) ** 2)
        x = (1.2 * blob + 4.0 * (2 * ell - 0.7)).clone() \
            .requires_grad_(True)
        it0, hist = 0, []
    opt = torch.optim.Adam([x], lr=0.05)
    betas = [(0.0, 2.0), (0.4, 4.0), (0.65, 8.0), (0.85, 16.0)]
    t_start = time.time()
    for it in range(it0, iters):
        frac = it / iters
        beta = [b for f, b in betas if frac >= f][-1]
        rho_bar = rc.filt_project(torch.sigmoid(x), kern, beta, mask=mask)
        sim = rc.build_sim(rho_bar, P, H, order=order)
        R, T = rc.jones(sim, 'backward')
        pens, mstats = (None, {})
        if method == 'A':
            pens, mstats = mode_penalty(rho_bar, P, H, order)
        L = rc.objective(R, T, method, pens)
        opt.zero_grad()
        L.backward()
        opt.step()
        m = rc.device_metrics(R.detach(), T.detach())
        hist.append([it, float(L), m['R_cross'], m['R_co'],
                     m['T_total_x'] + m['T_total_y'],
                     m['pb_phase_err_deg']])
        if it % 25 == 24 or it == iters - 1:
            np.savez(ck, x=x.detach().numpy(), it=it + 1,
                     hist=np.array(hist))
        if it % 20 == 0:
            print(f'{tag} it{it} L={float(L):+.4f} Rc={m["R_cross"]:.3f} '
                  f'co={m["R_co"]:.3f} dphi_err={m["pb_phase_err_deg"]:.0f}',
                  flush=True)
    # hard-binary finalization at order [9,9]
    with torch.no_grad():
        rho_bar = rc.filt_project(torch.sigmoid(x), kern, 16.0, mask=mask)
        b = (rho_bar.numpy() > 0.5)
        rho_bin = torch.tensor(b.astype(np.float32)) * mask
        b = rho_bin.numpy() > 0.5
        sim = rc.build_sim(rho_bin, P, H, order=(9, 9))
        R, T = rc.jones(sim, 'backward')
        m = rc.device_metrics(R, T)
        fams = {}
        for pol in ('x', 'y'):
            f = rc.moments_families(rho_bin, P, H, rc.LAM0, (9, 9), pol)
            fams.update({f'{k}_{pol}': float(f[k]) for k in
                         ('f_ED', 'f_MD', 'f_EQ', 'f_MQ', 'px_in_ED',
                          'py_in_ED', 'mx_in_MD', 'my_in_MD')})
    fm = fab_metrics(b, P)
    np.save(outdir / 'rho_binary.npy', rho_bin.numpy())
    rec = {'tag': tag, 'method': method, 'P': P, 'H': H, 'seed': seed,
           'iters': iters, 'n_grid': n_grid, 'order_opt': order_n,
           'order_final': 9, 'stage': stage,
           'padding_nm': rc.padding(P), 'design_radius_nm': rc.r_design(P),
           'runtime_s': time.time() - t_start,
           'sha256_rho': hashlib.sha256(
               (outdir / 'rho_binary.npy').read_bytes()).hexdigest(),
           **m, **fams, **fm}
    final_p.write_text(json.dumps(rec, indent=1))
    with open(outdir / 'history.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['it', 'L', 'R_cross', 'R_co', 'T_tot', 'pb_err_deg'])
        w.writerows(hist)
    new = not LEDGER.exists()
    with open(LEDGER, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rec.keys()))
        if new:
            w.writeheader()
        w.writerow(rec)
    print(f'{tag}: DONE Rc={m["R_cross"]:.3f} co={m["R_co"]:.3f} '
          f'err={m["pb_phase_err_deg"]:.0f}deg comps={fm["n_components"]} '
          f'({rec["runtime_s"]:.0f}s)', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    run(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]),
        int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]),
        int(sys.argv[7]), sys.argv[8])
