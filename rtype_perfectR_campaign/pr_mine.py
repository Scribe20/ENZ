"""Stage 0: mine the ENTIRE historical workspace (spec secs 5-7).

Enumerates every run directory of rtype_freeform_campaign and
rtype_widefov_campaign (coarse/refinement/finalists), including
incomplete ones: where rho_binary.npy is missing but checkpoint.npz
exists, the latent is recovered and binarized with the original
campaign's filter chain (beta=16). Everything re-evaluated at theta=0,
order [9,9], under the current validated conventions; F_ideal +
channels + principal amplitudes/phases + island count.

Writes results/perfect_r_workspace_candidates.csv
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import label

import pr_core as pr
import rt_core as rc
from wf_preflight import rect_rho, P_R, H_R

RES = pr.HERE / 'results'
CAMPS = [('rtfree', pr.HERE.parent / 'rtype_freeform_campaign'),
         ('widefov', pr.HERE.parent / 'rtype_widefov_campaign')]


def recover_checkpoint(d, P):
    z = np.load(d / 'checkpoint.npz', allow_pickle=True)
    x = torch.tensor(z['x'])
    n = x.shape[0]
    mask = rc.design_mask(n, P)
    kern = rc.conic_filter_kernel(n, P, 15.0)
    rho = rc.filt_project(torch.sigmoid(x), kern, 16.0, mask=mask)
    return (torch.tensor((rho.numpy() > 0.5).astype(np.float32)) * mask,
            int(z['it']))


def parse_ph(tag):
    P = H = None
    for part in tag.split('_'):
        if part.startswith('P') and part[1:].replace('.', '').isdigit():
            P = float(part[1:])
        if part.startswith('H') and part[1:].replace('.', '').isdigit():
            H = float(part[1:])
    return P, H


def main():
    torch.set_num_threads(4)
    rows = []

    def add(tag, camp, stage, source, rho, P, H, extra=None):
        m = pr.eval_full(rho, P, H, order=(9, 9))
        b = rho.numpy() > 0.5
        m['n_islands'] = int(label(b)[1])
        m['fill'] = float(b.mean())
        row = {'tag': tag, 'campaign': camp, 'stage': stage,
               'source': source, 'P': P, 'H': H, **m, **(extra or {})}
        rows.append(row)
        print(f"{camp}/{stage}/{tag}: F={m['F']:.3f} Rc={m['Rcross']:.3f} "
              f"T={m['T']:.3f} co={m['co']:.3f} A={m['A']:.3f} "
              f"err={m['phase_err_deg']:.0f} [{source}]", flush=True)

    add('rectangle', 'paper', 'baseline', 'analytic', rect_rho(), P_R, H_R)
    for camp, base in CAMPS:
        for stage in ('coarse', 'refinement', 'finalists'):
            sd = base / stage
            if not sd.exists():
                continue
            for d in sorted(sd.iterdir()):
                if not d.is_dir():
                    continue
                tag = d.name
                P, H = parse_ph(tag)
                if P is None or H is None:
                    continue
                try:
                    if (d / 'rho_binary.npy').exists():
                        rho = torch.tensor(np.load(d / 'rho_binary.npy'))
                        src = 'final'
                        extra = {}
                    elif (d / 'checkpoint.npz').exists():
                        rho, it = recover_checkpoint(d, P)
                        src = f'checkpoint@{it}'
                        extra = {}
                    else:
                        continue
                    add(tag, camp, stage, src, rho, P, H, extra)
                except Exception as ex:
                    print(f'{tag}: SKIP ({ex})', flush=True)
    with open(RES / 'perfect_r_workspace_candidates.csv', 'w',
              newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'MINING_DONE {len(rows)} candidates', flush=True)


if __name__ == '__main__':
    main()
