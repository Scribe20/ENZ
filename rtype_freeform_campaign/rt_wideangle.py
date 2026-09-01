"""Stage-2 wide-angle refinement (spec sections 19-20): re-optimize the
top finalist of each method with a multi-angle worst-case-weighted
objective. Angles are GLASS-side (input-layer) angles; the air-side
device angle is theta_air = asin(n_glass sin theta_glass):
  glass 0/15/25/33 deg -> air 0/22.2/38.0/52.5 deg.
Warm-started from the finalist latent. usage:
  python rt_wideangle.py <method> <P> <H> <src_tag> <iters>
"""
import csv
import hashlib
import json
import math
import sys
import time

import numpy as np
import torch

import rt_core as rc
from rt_optimize import fab_metrics, LEDGER

THETAS = [0.0, 15.0, 25.0, 33.0]
W = [1.0, 1.0, 1.0, 1.0]


def sim_at(rho, P, H, theta, order=(9, 9)):
    import torcwa
    e = rc.eps_asi()
    sim = torcwa.rcwa(freq=1.0 / rc.LAM0, order=list(order),
                      L=[float(P), float(P)], dtype=rc.SIM_DTYPE,
                      device=rc.DEVICE)
    sim.add_input_layer(eps=rc.EPS_GLASS)
    sim.set_incident_angle(inc_ang=math.radians(theta), azi_ang=0.0)
    sim.add_layer(thickness=float(H), eps=rho * (e - 1.0) + 1.0)
    sim.solve_global_smatrix()
    return sim


def run(method, P, H, src_tag, iters):
    tag = f'{method}_P{P:.0f}_H{H:.0f}_wideangle'
    outdir = rc.HERE / 'finalists' / tag
    outdir.mkdir(parents=True, exist_ok=True)
    final_p = outdir / 'final.json'
    if final_p.exists():
        print(f'{tag}: complete', flush=True)
        return
    n_grid = 96
    mask = rc.design_mask(n_grid, P)
    kern = rc.conic_filter_kernel(n_grid, P, 15.0)
    ck = outdir / 'checkpoint.npz'
    if ck.exists():
        z = np.load(ck)
        x = torch.tensor(z['x'], requires_grad=True)
        it0 = int(z['it'])
    else:
        src = rc.HERE / 'refinement' / src_tag / 'checkpoint.npz'
        x = torch.tensor(np.load(src)['x']).clone().requires_grad_(True)
        it0 = 0
    opt = torch.optim.Adam([x], lr=0.03)
    t0 = time.time()
    for it in range(it0, iters):
        beta = 8.0 if it < iters * 0.5 else 16.0
        rho_bar = rc.filt_project(torch.sigmoid(x), kern, beta, mask=mask)
        losses = []
        for th, w in zip(THETAS, W):
            sim = sim_at(rho_bar, P, H, th)
            R, T = rc.jones(sim, 'backward')
            losses.append(w * rc.objective(R, T))
        Ls = torch.stack(losses)
        L = Ls.mean() + 0.8 * torch.logsumexp(Ls * 6.0, 0) / 6.0
        opt.zero_grad()
        L.backward()
        opt.step()
        if it % 10 == 9 or it == iters - 1:
            np.savez(ck, x=x.detach().numpy(), it=it + 1)
        if it % 10 == 0:
            print(f'{tag} it{it} L={float(L):+.4f} per-angle '
                  f'{[round(float(v), 3) for v in Ls]}', flush=True)
    with torch.no_grad():
        rho_bar = rc.filt_project(torch.sigmoid(x), kern, 16.0, mask=mask)
        rho_bin = torch.tensor((rho_bar.numpy() > 0.5)
                               .astype(np.float32)) * mask
        b = rho_bin.numpy() > 0.5
        rows = []
        for th in THETAS:
            sim = sim_at(rho_bin, P, H, th)
            R, T = rc.jones(sim, 'backward')
            m = rc.device_metrics(R, T)
            th_air = math.degrees(math.asin(min(
                1.0, rc.N_GLASS * math.sin(math.radians(th)))))
            rows.append({'theta_glass': th, 'theta_air': th_air, **m})
            print(f'{tag} final th_air={th_air:.0f}: Rc={m["R_cross"]:.3f} '
                  f'co={m["R_co"]:.3f} err={m["pb_phase_err_deg"]:.0f}',
                  flush=True)
    fm = fab_metrics(b, P)
    np.save(outdir / 'rho_binary.npy', rho_bin.numpy())
    m0 = rows[0]
    rec = {'tag': tag, 'method': method, 'P': P, 'H': H, 'seed': 11,
           'iters': iters, 'n_grid': n_grid, 'order_opt': 9,
           'order_final': 9, 'stage': 'finalists',
           'padding_nm': rc.padding(P), 'design_radius_nm': rc.r_design(P),
           'runtime_s': time.time() - t0,
           'sha256_rho': hashlib.sha256(
               (outdir / 'rho_binary.npy').read_bytes()).hexdigest(),
           **{k: v for k, v in m0.items() if k not in ('theta_glass',
                                                       'theta_air')},
           **fm}
    final_p.write_text(json.dumps({'rec': rec, 'angles': rows}, indent=1))
    with open(outdir / 'angle_table.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'{tag}: WIDEANGLE_DONE', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    run(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4],
        int(sys.argv[5]))
