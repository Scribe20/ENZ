"""Angular continuation of a perfect-R candidate (spec secs 22-25).

usage: python pr_continue.py <branch> <P> <H> <parent_rho.npy> <iters>
                             <stage cont30|cont55> [L]
cont30: theta pool {0,10,20,30} x phi {0,45,90}; 4-state minibatches.
cont55: theta pool {0,15,30,40,50,55} x phi {0,45,90}; 5-state
        minibatches; in the second half, PB-ROTATION states are added:
        the geometry is physically rotated by alpha in {30,60,90} at
        theta in {0,30,45,50} and scored with the rotated ideal
        operator U_alpha (principal-axis offset theta0 estimated from
        the unrotated Jones for non-D2 motifs).
Multi-failure hard mining every 12 iters over the full pool: the worst
F, the largest T, the largest R_co (and the worst rotated-fidelity
state) are oversampled. Constraints via grown multipliers; no
absorption penalty. Final: hard-binary, [9,9] full pool + PB fidelity.
"""
import csv
import hashlib
import json
import math
import sys
import time

import numpy as np
import torch

import pr_core as pr
import rt_core as rc
import wf_core as wf
from pr_optimize import latent_from_rho, fab_metrics

POOLS = {'cont30': ([0.0, 10.0, 20.0, 30.0], 4),
         'cont55': ([0.0, 15.0, 30.0, 40.0, 50.0, 55.0], 5)}
PHIS = [0.0, 45.0, 90.0]
ROT_TH = [0.0, 30.0, 45.0, 50.0]
ROT_AL = [30.0, 60.0, 90.0]


def rotate_rho(rho, alpha):
    import scipy.ndimage as ndi
    if alpha == 0.0:
        return rho
    rr = ndi.rotate(rho.detach().numpy(), alpha, reshape=False, order=1,
                    mode='constant', cval=0.0)
    return torch.tensor(rr, dtype=rho.dtype)


def rotated_density(x, kern, beta, mask, branch, alpha):
    """Differentiable rotation of the latent via grid_sample, then the
    usual filter/projection chain (rotation of a D2/C2 latent keeps
    the symmetric envelope; mask reapplied)."""
    # symmetrize BEFORE rotating (a rotated motif is no longer D2/C2-
    # symmetric about the cell axes; symmetrizing after rotation would
    # fold it back onto its mirror images and destroy the rotation)
    lat = pr.symmetrize(torch.sigmoid(x), branch)
    if alpha != 0.0:
        a = math.radians(alpha)
        c, s = math.cos(a), math.sin(a)
        theta = torch.tensor([[c, -s, 0.0], [s, c, 0.0]],
                             dtype=torch.float32)[None]
        grid = torch.nn.functional.affine_grid(theta, (1, 1) + lat.shape,
                                               align_corners=False)
        lat = torch.nn.functional.grid_sample(lat[None, None], grid,
                                              mode='bilinear',
                                              padding_mode='zeros',
                                              align_corners=False)[0, 0]
    return pr.filt_project(lat, kern, beta, mask, 'FULL')


def theta0_offset(Rj):
    Rc = rc.circular(Rj)
    return float(torch.angle(Rc[1, 0]) - torch.angle(Rc[0, 1])) / 4.0


def state_loss(Rj, Tj, alpha, th0, phi=0.0):
    m = pr.port_metrics(Rj, Tj, phi)
    Fa = pr.fidelity_state(Rj, alpha, phi, th0)
    return Fa, m


def full_pool(x, kern, mask, branch, P, H, thetas, order, eps, rot=False):
    rows = []
    with torch.no_grad():
        rho = pr.filt_project(torch.sigmoid(x), kern, 16.0, mask, branch)
        Rj0, _ = wf.jones_angle(rho, P, H, 0.0, 0.0, order=order)
        th0 = theta0_offset(Rj0) if branch != 'D2' else 0.0
        for th in thetas:
            for ph in PHIS:
                Rj, Tj = wf.jones_angle(rho, P, H, th, ph, order=order)
                Fa, m = state_loss(Rj, Tj, 0.0, th0, ph)
                rows.append({'theta': th, 'phi': ph, 'alpha': 0.0,
                             'F': float(Fa), 'T': float(m['T']),
                             'co': float(m['co']), 'A': float(m['A'])})
        if rot:
            for th in ROT_TH:
                for al in ROT_AL:
                    rr = rotated_density(x, kern, 16.0, mask, branch, al)
                    Rj, Tj = wf.jones_angle(rr, P, H, th, 0.0, order=order)
                    Fa, m = state_loss(Rj, Tj, al, th0)
                    rows.append({'theta': th, 'phi': 0.0, 'alpha': al,
                                 'F': float(Fa), 'T': float(m['T']),
                                 'co': float(m['co']), 'A': float(m['A'])})
    return rows, th0


def mined(rows):
    r = sorted(rows, key=lambda q: q['F'])
    worst = [(q['theta'], q['phi'], q['alpha']) for q in r[:3]]
    worst.append(max(rows, key=lambda q: q['T'])
                 and tuple(max(rows, key=lambda q: q['T'])[k]
                           for k in ('theta', 'phi', 'alpha')))
    worst.append(tuple(max(rows, key=lambda q: q['co'])[k]
                       for k in ('theta', 'phi', 'alpha')))
    return worst


def run(branch, P, H, parent, iters, stage, lossless=False):
    thetas, K = POOLS[stage]
    ptag = parent.split('/')[-2] if parent.endswith('rho_binary.npy') \
        else parent.split('/')[-1].replace('.npy', '')
    tag = f'{branch}_P{P:.0f}_H{H:.0f}_{stage}_{ptag[:28]}'
    outdir = pr.HERE / 'continuation' / tag
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / 'final.json').exists():
        print(f'{tag}: already complete', flush=True)
        return
    eps = complex(rc.EPS_ASI_633.real, 0.0) if lossless else None
    order_opt = (7, 7)
    mask = pr.design_mask(96, P)
    kern = rc.conic_filter_kernel(96, P, 15.0)
    ck = outdir / 'checkpoint.npz'
    lamT, lamCo, worst = 3.0, 3.0, []
    if ck.exists():
        z = np.load(ck, allow_pickle=True)
        x = torch.tensor(z['x'], requires_grad=True)
        it0 = int(z['it'])
        lamT, lamCo = float(z['lamT']), float(z['lamCo'])
        worst = [tuple(w) for w in z['worst']] if len(z['worst']) else []
    else:
        x = latent_from_rho(np.load(parent)).clone().requires_grad_(True)
        it0 = 0
    opt = torch.optim.Adam([x], lr=0.03)
    rng = np.random.default_rng(int(P) * 31 + int(H))
    capT, capCo = (0.10, 0.05) if stage == 'cont30' else (0.08, 0.04)
    t0 = time.time()
    for it in range(it0, iters):
        frac = it / iters
        beta = 8.0 if frac < 0.5 else 16.0
        rho = pr.filt_project(torch.sigmoid(x), kern, beta, mask, branch)
        # minibatch: theta=0 anchor + sampled states (+ mined + rotations)
        batch = [(0.0, float(PHIS[it % 3]), 0.0)]
        while len(batch) < K:
            if worst and rng.random() < 0.5:
                batch.append(worst[rng.integers(len(worst))])
            else:
                batch.append((float(rng.choice(thetas[1:])),
                              float(rng.choice(PHIS)), 0.0))
        if stage == 'cont55' and frac >= 0.5:
            for _ in range(2):
                batch.append((float(rng.choice(ROT_TH)), 0.0,
                              float(rng.choice(ROT_AL))))
        with torch.no_grad():
            Rj0, _ = wf.jones_angle(rho, P, H, 0.0, 0.0, order=order_opt)
            th0 = theta0_offset(Rj0) if branch != 'D2' else 0.0
        Fs, Ts, Cs = [], [], []
        for th, ph, al in batch:
            rr = rho if al == 0.0 else rotated_density(x, kern, beta, mask,
                                                       branch, al)
            Rj, Tj = wf.jones_angle(rr, P, H, th, ph, order=order_opt)
            Fa, m = state_loss(Rj, Tj, al, th0, ph)
            Fs.append(Fa)
            Ts.append(m['T'])
            Cs.append(m['co'])
        Fs = torch.stack(Fs)
        Ts = torch.stack(Ts)
        Cs = torch.stack(Cs)
        softmin = -torch.logsumexp(-8.0 * Fs, 0) / 8.0
        L = -(0.5 * Fs.mean() + 0.5 * softmin) \
            + lamT * (pr.soft_over(Ts.mean(), capT)
                      + 0.5 * pr.soft_over(Ts.max(), capT + 0.05)) \
            + lamCo * (pr.soft_over(Cs.mean(), capCo)
                       + 0.5 * pr.soft_over(Cs.max(), capCo + 0.03))
        opt.zero_grad()
        L.backward()
        opt.step()
        if it % 12 == 11:
            rows, _ = full_pool(x, kern, mask, branch, P, H, thetas,
                                order_opt, eps,
                                rot=(stage == 'cont55' and frac >= 0.5))
            worst = mined(rows)
            if np.mean([r['T'] for r in rows]) > capT:
                lamT = min(40.0, lamT * 1.5)
            if np.mean([r['co'] for r in rows]) > capCo:
                lamCo = min(40.0, lamCo * 1.5)
            np.savez(ck, x=x.detach().numpy(), it=it + 1, lamT=lamT,
                     lamCo=lamCo, worst=np.array(worst))
            Fp = [r['F'] for r in rows]
            print(f'{tag} it{it} pool F mean={np.mean(Fp):.3f} '
                  f'min={min(Fp):.3f} T={np.mean([r["T"] for r in rows]):.3f}'
                  f' co={np.mean([r["co"] for r in rows]):.3f} worst={worst[0]}',
                  flush=True)
    # finalize hard-binary at [9,9]
    with torch.no_grad():
        rho = pr.filt_project(torch.sigmoid(x), kern, 16.0, mask, branch)
        b = rho.numpy() > 0.5
        rho_bin = torch.tensor(b.astype(np.float32)) * mask
    np.save(outdir / 'rho_binary.npy', rho_bin.numpy())
    xb = torch.logit(rho_bin.clamp(0.02, 0.98))
    rows, th0 = full_pool(xb, kern, mask, branch, P, H, thetas, (9, 9),
                          eps, rot=True)
    wf.write_rows(outdir / 'pool_final.csv', rows)
    m0 = pr.eval_full(rho_bin, P, H, (9, 9), eps)
    fm = fab_metrics(rho_bin.numpy() > 0.5, P)
    unrot = [r for r in rows if r['alpha'] == 0.0]
    rot = [r for r in rows if r['alpha'] != 0.0]
    rec = {'tag': tag, 'branch': branch, 'P': P, 'H': H, 'parent': parent,
           'stage': stage, 'iters': iters, 'lossless': bool(lossless),
           'theta0_offset_deg': math.degrees(th0),
           'runtime_s': time.time() - t0,
           'F0': m0['F'], 'T0': m0['T'], 'co0': m0['co'], 'A0': m0['A'],
           'F_mean': float(np.mean([r['F'] for r in unrot])),
           'F_min': float(min(r['F'] for r in unrot)),
           'T_mean': float(np.mean([r['T'] for r in unrot])),
           'T_max': float(max(r['T'] for r in unrot)),
           'co_mean': float(np.mean([r['co'] for r in unrot])),
           'co_max': float(max(r['co'] for r in unrot)),
           'A_mean': float(np.mean([r['A'] for r in unrot])),
           'Frot_mean': float(np.mean([r['F'] for r in rot])),
           'Frot_min': float(min(r['F'] for r in rot)),
           'sha256_rho': hashlib.sha256(
               (outdir / 'rho_binary.npy').read_bytes()).hexdigest(),
           **fm}
    (outdir / 'final.json').write_text(json.dumps(rec, indent=1))
    led = pr.HERE / 'results' / f'{stage}_ledger.csv'
    new = not led.exists()
    with open(led, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rec.keys()))
        if new:
            w.writeheader()
        w.writerow(rec)
    print(f"{tag}: DONE F0={rec['F0']:.3f} Fmean={rec['F_mean']:.3f} "
          f"Fmin={rec['F_min']:.3f} Tmean={rec['T_mean']:.3f} "
          f"co={rec['co_mean']:.3f} Frot_min={rec['Frot_min']:.3f} "
          f"({rec['runtime_s']:.0f}s)", flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    run(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4],
        int(sys.argv[5]), sys.argv[6], len(sys.argv) > 7)
