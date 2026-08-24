"""
ed_eq_pilot.py — Stage-A pilot discovery (contract-bound)
=========================================================
Primary FoM: F_ED_EQ = 0.5*[log(S_px+1e-12) + log(S_Qxz+1e-12)]  — exact
current multipoles only. Q / linewidth / T / R / phase NEVER appear here
(SCIENTIFIC_CONTRACT.md). Unwanted channels are MONITORED, not penalized.

Pilot grid: P in {550,650,750} x h in {150,250,350} x seeds {11,29}
(18 runs), order [7,7], 48x48x7 objective grid, 120 Adam-ascent iters,
tanh-projection continuation beta 1->1000, blur radius 20 nm (physical),
NO symmetry operations anywhere. Per-run checkpoint/resume; idempotent.
"""

import argparse
import csv
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_HERE.parent))
import ed_eq_core as core                      # noqa: E402

RESULTS = _HERE / 'results' / 'pilot'
LAM0 = 1332.5
PERIODS = [550.0, 650.0, 750.0]
HEIGHTS = [150.0, 250.0, 350.0]
SEEDS = [11, 29]
ORDER = [7, 7]
N_XY, NZ = 48, 7
ITERS = 120
GAR0, B1, B2, AEPS = 0.02, 0.9, 0.999, 1e-8
BLUR_NM = 20.0
DX_NM = 5.0


def run_id(P, h, seed):
    return f'P{int(P):04d}_H{int(h):04d}_seed{seed:03d}'


def build_filter(n, P):
    dx = P / n
    ax = (torch.arange(n, dtype=torch.float32) - (n - 1) / 2) * dx
    xg, yg = torch.meshgrid(ax, ax, indexing='ij')
    g = torch.exp(-(xg ** 2 + yg ** 2) / BLUR_NM ** 2)
    g = g / g.sum()
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(g)))


def filt_proj(rho, g_fft, beta):
    rf = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(rho)))
    bar = torch.real(torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(rf * g_fft))))
    return 0.5 + torch.tanh(2 * beta * bar - beta) / (2 * math.tanh(beta))


def monitor_row(mo):
    row = {}
    for t in ('px', 'py', 'pz', 'mx', 'my', 'mz',
              'Qxx', 'Qyy', 'Qzz', 'Qxy', 'Qxz', 'Qyz', 'Tx'):
        v = complex(mo[t].detach())
        row[t + '_re'], row[t + '_im'] = v.real, v.imag
    Cp, Cm, CQe = core.family_weights({k: (v.detach() if torch.is_tensor(v) else v)
                                       for k, v in mo.items()})
    row['Cp'], row['Cm'], row['CQe'] = float(Cp), float(Cm), float(CQe)
    return row


def optimize_run(P, h, seed, out):
    out.mkdir(parents=True, exist_ok=True)
    cfgp = out / 'config.json'
    if cfgp.exists() and json.loads(cfgp.read_text()).get('status') in (
            'completed',) or (cfgp.exists() and
            json.loads(cfgp.read_text()).get('status', '').startswith('failed')):
        print(f'{out.name}: done/failed - skip', flush=True)
        return
    n = int(round(P / DX_NM))
    n += n % 2
    g_fft = build_filter(n, P)
    eps_si = core.si_eps(LAM0)
    beta_s = np.exp(np.arange(ITERS) * math.log(1000) / ITERS)
    gar = GAR0 * 0.5 * (1 + np.cos(np.arange(ITERS) * np.pi / ITERS))

    ck = out / 'checkpoint.pt'
    hist = []
    it0 = 0
    if ck.exists():
        c = torch.load(ck, weights_only=False)
        rho, mom, vel, it0, hist = c['rho'], c['m'], c['v'], c['it'] + 1, c['hist']
        print(f'{out.name}: resume at {it0}', flush=True)
    else:
        torch.manual_seed(seed); np.random.seed(seed)
        rho = torch.rand(n, n)
        rho = filt_proj(rho, g_fft, 1.0).detach()
        mom = torch.zeros_like(rho); vel = torch.zeros_like(rho)

    cfg = {'run_id': out.name, 'P': P, 'h': h, 'seed': seed, 'lam0': LAM0,
           'order': ORDER, 'n_mask': n, 'n_xy': N_XY, 'nz': NZ, 'iters': ITERS,
           'material': 'aSi_Franta2013', 'eps_si': [eps_si.real, eps_si.imag],
           'objective': 'F_ED_EQ=0.5[log S_px + log S_Qxz]; Q excluded',
           'symmetry': 'NONE', 'status': 'running'}
    cfgp.write_text(json.dumps(cfg, indent=1))

    t0 = time.time()
    status = 'completed'
    for it in range(it0, ITERS):
        rho.requires_grad_(True)
        rt = filt_proj(rho, g_fft, float(beta_s[it]))
        try:
            F, S_px, S_Q, mo = core.eval_objective(rt, P, h, LAM0, ORDER,
                                                   n_xy=N_XY, nz=NZ,
                                                   eps_si_val=eps_si)
        except Exception as e:
            status = f'failed_solver_it{it}: {e}'
            break
        if not torch.isfinite(F):
            status = f'failed_nonfinite_it{it}'
            break
        F.backward()
        with torch.no_grad():
            g = rho.grad; rho.grad = None
            if g is None or not torch.all(torch.isfinite(g)):
                status = f'failed_grad_it{it}'
                break
            row = {'it': it, 'F': float(F.detach()), 'S_px': float(S_px.detach()),
                   'S_Qxz': float(S_Q.detach()), 'beta': float(beta_s[it]),
                   'lr': float(gar[it]), 't': time.time() - t0}
            if it % 10 == 0 or it == ITERS - 1:
                row.update(monitor_row(mo))
                print(f"{out.name}: it{it:4d} F={row['F']:+.3f} "
                      f"S_px={row['S_px']:.3e} S_Qxz={row['S_Qxz']:.3e}",
                      flush=True)
            hist.append(row)
            mom = B1 * mom + (1 - B1) * g
            vel = B2 * vel + (1 - B2) * g * g
            rho = rho.detach() + gar[it] * (mom / (1 - B1 ** (it + 1))) / \
                torch.sqrt(vel / (1 - B2 ** (it + 1)) + AEPS)
            rho.clamp_(0, 1)
            if (it + 1) % 20 == 0 or it == ITERS - 1:
                torch.save({'rho': rho, 'm': mom, 'v': vel, 'it': it,
                            'hist': hist}, ck)

    with torch.no_grad():
        rt = filt_proj(rho.detach(), g_fft, float(beta_s[-1]))
        rb = (rt >= 0.5).float()
        final = {}
        for tag, dens in [('projected', rt), ('binary', rb)]:
            F, S_px, S_Q, mo = core.eval_objective(dens, P, h, LAM0, ORDER,
                                                   n_xy=N_XY, nz=NZ,
                                                   eps_si_val=eps_si)
            final[tag] = {'F': float(F), 'S_px': float(S_px),
                          'S_Qxz': float(S_Q), **monitor_row(mo)}
        np.save(out / 'rho_final.npy', rho.detach().numpy())
        np.save(out / 'rho_projected.npy', rt.numpy())
        np.save(out / 'rho_binary.npy', rb.numpy())
        keys = sorted({k for r in hist for k in r})
        with open(out / 'history.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in hist:
                w.writerow(r)
        cfg['status'] = status
        cfg['runtime_s'] = time.time() - t0
        cfg['final'] = final
        cfg['fill'] = float(rb.mean())
        cfgp.write_text(json.dumps(cfg, indent=1))
    print(f"{out.name}: {status} F_bin={final['binary']['F']:+.3f} "
          f"S_px={final['binary']['S_px']:.3e} "
          f"S_Qxz={final['binary']['S_Qxz']:.3e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, nargs=2, default=[0, 1])
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--max-runs', type=int, default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    runs = [(P, h, s) for P in PERIODS for h in HEIGHTS for s in SEEDS]
    runs = runs[args.shard[0]::args.shard[1]]
    done = 0
    for (P, h, s) in runs:
        if args.max_runs is not None and done >= args.max_runs:
            print('SHARD_CONTINUE', flush=True)
            return
        out = RESULTS / run_id(P, h, s)
        if (out / 'config.json').exists() and \
                json.loads((out / 'config.json').read_text()).get('status') != 'running':
            continue
        try:
            optimize_run(P, h, s, out)
            done += 1
        except Exception:
            (out).mkdir(parents=True, exist_ok=True)
            (out / 'fatal.log').write_text(traceback.format_exc())
            print(f'{out.name}: FATAL, continuing', flush=True)
            done += 1
    print('SHARD_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
