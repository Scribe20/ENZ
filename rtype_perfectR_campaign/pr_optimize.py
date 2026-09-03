"""Single perfect-R optimization run (theta = 0 unless stage says
otherwise), checkpointed + idempotent.

usage: python pr_optimize.py <branch D2|C2|FULL> <P> <H> <seedspec>
                             <iters> <stage> [lossless]
seedspec:
  rand<k>        random blob + random-amplitude quadrupolar bias
  multi<k>       2-3 symmetric gaussian islands + blob (multi-island)
  warm:<path>    binarized-history warm start (rho .npy, any grid)
  mix:<pathA>+<pathB>  latent average of two historical rhos + noise
  mut<k>:<path>  smooth latent mutation of a parent rho (basin hop)
Loss: staged continuation to the ideal reflective PB operator
(pr_core.pr_loss); augmented multipliers on T/co caps grown when
violated at 15-iter checkpoints. Absorption never penalized directly.
Writes <stage>/<tag>/{checkpoint.npz,final.json,rho_binary.npy} and
appends results/<stage>_ledger.csv.
"""
import csv
import os
import hashlib
import json
import sys
import time

import numpy as np
import scipy.ndimage as ndi
import torch
from scipy.ndimage import label, binary_erosion

import pr_core as pr
import rt_core as rc


def resize96(a):
    n = a.shape[0]
    if n == 96:
        return a
    idx = (np.arange(96) * n / 96).astype(int)
    return a[np.ix_(idx, idx)]


def latent_from_rho(rho_np, k=3.0):
    sm = ndi.gaussian_filter(resize96(rho_np.astype(np.float32)), 1.5)
    return torch.tensor(k * (2.0 * sm - 1.0))


def build_seed(spec, P, H, n_grid=96):
    if spec.startswith('warm:'):
        return latent_from_rho(np.load(spec[5:]))
    if spec.startswith('mix:'):
        a, b = spec[4:].split('+')
        la = latent_from_rho(np.load(a))
        lb = latent_from_rho(np.load(b))
        g = torch.Generator().manual_seed(int(P) + int(H))
        return 0.5 * (la + lb) + 0.4 * torch.randn(96, 96, generator=g)
    if spec.startswith('mut'):
        head, path = spec.split(':', 1)
        k = int(head[3:])
        rgen = np.random.default_rng(k * 7919 + int(P) + int(H))
        base = latent_from_rho(np.load(path))
        g = torch.Generator().manual_seed(k)
        noise = torch.nn.functional.conv2d(
            torch.randn(96, 96, generator=g)[None, None],
            rc.conic_filter_kernel(96, P, 25.0), padding='same')[0, 0]
        noise = noise / (noise.std() + 1e-9)
        return base + float(rgen.uniform(0.8, 2.4)) * noise
    k = int(spec[4:] if spec.startswith('rand') else spec[5:])
    g = torch.Generator().manual_seed(k)
    rgen = np.random.default_rng(k * 1000 + int(P) + int(H))
    ax = (torch.arange(n_grid) + 0.5) / n_grid * P - P / 2
    X, Y = torch.meshgrid(ax, ax, indexing='ij')
    rd = pr.r_design(P)
    blob = torch.rand(n_grid, n_grid, generator=g)
    blob = torch.nn.functional.conv2d(
        blob[None, None], rc.conic_filter_kernel(n_grid, P, 30.0),
        padding='same')[0, 0]
    blob = (blob - blob.mean()) / (blob.std() + 1e-9)
    if spec.startswith('multi'):
        n_isl = int(rgen.integers(2, 4))
        bump = torch.zeros(n_grid, n_grid)
        for _ in range(n_isl):
            cx = float(rgen.uniform(0.15, 0.6)) * rd
            cy = float(rgen.uniform(-0.5, 0.5)) * rd
            s = float(rgen.uniform(0.2, 0.4)) * rd
            b1 = torch.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / s ** 2))
            bump = bump + b1 + torch.flip(b1, [0])   # symmetric partner
        return 1.6 * blob + 3.0 * (bump - 0.45)
    qamp = float(rgen.uniform(0.8, 2.2))
    qsign = 1.0 if rgen.random() < 0.5 else -1.0
    bump = torch.exp(-((X ** 2 + Y ** 2) / (0.75 * rd) ** 2) ** 2)
    quad = qsign * (X ** 2 - Y ** 2) / rd ** 2
    return 2.2 * blob + 1.2 * (2 * bump - 0.75) + qamp * quad


def fab_metrics(b, P):
    n = b.shape[0]
    px = P / n

    def min_width(m):
        k, mm = 0, m.copy()
        n0 = label(m)[1]
        if n0 == 0:
            return 0.0
        while True:
            m2 = binary_erosion(mm)
            if label(m2)[1] < n0 or m2.sum() == 0:
                break
            mm = m2
            k += 1
        return (2 * k + 1) * px
    ax = (np.arange(n) + 0.5) / n * P - P / 2
    X, Y = np.meshgrid(ax, ax, indexing='ij')
    inside = X ** 2 + Y ** 2 <= pr.r_design(P) ** 2
    rmax = float(np.sqrt(X[b] ** 2 + Y[b] ** 2).max()) if b.any() else 0.0
    return {'n_islands': int(label(b)[1]),
            'min_si_nm': min_width(b),
            'min_gap_nm': min_width(~b & inside),
            'edge_clear_nm': pr.r_design(P) - rmax,
            'fill': float(b.mean())}


def ck_exists(outdir):
    return (outdir / 'checkpoint.npz').exists()


def caps_for(frac, stage):
    if stage.startswith('cont') or frac >= 0.8:
        return 0.10, 0.05
    if frac >= 0.5:
        return 0.10, 0.05
    return 0.15, 0.08


def run(branch, P, H, seedspec, iters, stage, lossless=False):
    tagseed = seedspec.replace(':', '~').replace('/', '-').replace('+', 'X')
    if len(tagseed) > 40:
        tagseed = tagseed[:12] + hashlib.sha256(
            tagseed.encode()).hexdigest()[:8]
    tag = f'{branch}_P{P:.0f}_H{H:.0f}_{tagseed}'
    outdir = pr.HERE / stage / tag
    outdir.mkdir(parents=True, exist_ok=True)
    final_p = outdir / 'final.json'
    if final_p.exists():
        print(f'{tag}: already complete', flush=True)
        return
    eps = complex(rc.EPS_ASI_633.real, 0.0) if lossless else None
    order_opt = (9, 9) if P >= 300 else (7, 7)
    ovr = pr.HERE / 'iters_override.txt'          # budget control for
    if ovr.exists() and not ck_exists(outdir):    # not-yet-started runs
        iters = min(iters, int(ovr.read_text().strip()))
    n_grid = 96
    mask = pr.design_mask(n_grid, P)
    kern = rc.conic_filter_kernel(n_grid, P, float(os.environ.get("PR_FILTER_NM", "15")))  # fab-robust reopt: PR_FILTER_NM=25
    ck = outdir / 'checkpoint.npz'
    lamT, lamCo = 2.0, 2.0
    if ck.exists():
        z = np.load(ck)
        x = torch.tensor(z['x'], requires_grad=True)
        it0 = int(z['it'])
        lamT, lamCo = float(z['lamT']), float(z['lamCo'])
        hist = list(z['hist'])
    else:
        x = build_seed(seedspec, P, H).clone().requires_grad_(True)
        it0, hist = 0, []
    opt = torch.optim.Adam([x], lr=0.04)
    betas = [(0.0, 2.0), (0.35, 4.0), (0.6, 8.0), (0.8, 16.0)]
    t0 = time.time()
    for it in range(it0, iters):
        frac = it / iters
        beta = [b for f, b in betas if frac >= f][-1]
        rho = pr.filt_project(torch.sigmoid(x), kern, beta, mask, branch)
        Rj, Tj = pr.jones_theta0(rho, P, H, order_opt, eps)
        m = pr.port_metrics(Rj, Tj)
        capT, capCo = caps_for(frac, stage)
        L = pr.pr_loss(m, frac, lamT, lamCo, capT, capCo, branch)
        opt.zero_grad()
        L.backward()
        opt.step()
        hist.append([it, float(L), float(m['F']), float(m['T']),
                     float(m['co']), float(m['A'])])
        if it % 15 == 14:
            if float(m['T']) > capT:
                lamT = min(40.0, lamT * 1.6)
            if float(m['co']) > capCo:
                lamCo = min(40.0, lamCo * 1.6)
            np.savez(ck, x=x.detach().numpy(), it=it + 1, lamT=lamT,
                     lamCo=lamCo, hist=np.array(hist))
        if it % 30 == 0:
            Fv, Tv = float(m['F']), float(m['T'])
            cv, Av = float(m['co']), float(m['A'])
            print(f'{tag} it{it} F={Fv:.3f} T={Tv:.3f} co={cv:.3f} '
                  f'A={Av:.3f} lam=({lamT:.1f},{lamCo:.1f})', flush=True)
    with torch.no_grad():
        rho = pr.filt_project(torch.sigmoid(x), kern, 16.0, mask, branch)
        b = rho.numpy() > 0.5
        rho_bin = torch.tensor(b.astype(np.float32)) * mask
        b = rho_bin.numpy() > 0.5
    mfin = pr.eval_full(rho_bin, P, H, order=(9, 9), eps_override=eps)
    fm = fab_metrics(b, P)
    np.save(outdir / 'rho_binary.npy', rho_bin.numpy())
    rec = {'tag': tag, 'branch': branch, 'P': P, 'H': H,
           'seed': seedspec, 'iters': iters, 'stage': stage,
           'lossless': bool(lossless), 'order_opt': order_opt[0],
           'runtime_s': time.time() - t0, 'lamT': lamT, 'lamCo': lamCo,
           'sha256_rho': hashlib.sha256(
               (outdir / 'rho_binary.npy').read_bytes()).hexdigest(),
           **mfin, **fm}
    final_p.write_text(json.dumps(rec, indent=1))
    led = pr.HERE / 'results' / f'{stage}_ledger.csv'
    new = not led.exists()
    with open(led, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rec.keys()))
        if new:
            w.writeheader()
        w.writerow(rec)
    print(f"{tag}: DONE F={mfin['F']:.3f} Rc={mfin['Rcross']:.3f} "
          f"T={mfin['T']:.3f} co={mfin['co']:.3f} A={mfin['A']:.3f} "
          f"err={mfin['phase_err_deg']:.0f} isl={fm['n_islands']} "
          f"({rec['runtime_s']:.0f}s)", flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    run(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4],
        int(sys.argv[5]), sys.argv[6], len(sys.argv) > 7)
