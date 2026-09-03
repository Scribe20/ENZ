"""Stage-I job generation (spec secs 8, 14-16, 18, 20).

Historical seed rhos are frozen as .npy under mining/seeds/. Base set
on every device P/H point: rand, multi-island, warm:newA (wide-FOV
dipolar), warm:oldB (EQ bow-tie) -> 4 distinct basins x 36 points.
Extra families (rect, oldA, p258, mix, mut, lowT, lowA) are added
later at the top basins by pr_jobs_extra.py.
usage: python pr_jobs.py
"""
import numpy as np
import torch

import pr_core as pr
from wf_preflight import rect_rho

SEEDS = pr.HERE / 'mining' / 'seeds'
SEEDS.mkdir(parents=True, exist_ok=True)
OLD = pr.HERE.parent
HIST = {
    'rect': None,
    'oldA': OLD / 'rtype_freeform_campaign/refinement/A_P271_H200_s11_g96_o9/rho_binary.npy',
    'oldB': OLD / 'rtype_freeform_campaign/refinement/B_P271_H215_s11_g96_o9/rho_binary.npy',
    'newA': OLD / 'rtype_widefov_campaign/refinement/A_P239_H200_s11_wf/rho_binary.npy',
    'p258': OLD / 'rtype_widefov_campaign/refinement/A_P258_H200_s11_wf/rho_binary.npy',
    'newB': OLD / 'rtype_widefov_campaign/refinement/B_P252_H185_s47_wf/rho_binary.npy',
}


def main():
    for k, p in HIST.items():
        arr = rect_rho().numpy() if p is None else np.load(p)
        np.save(SEEDS / f'{k}.npy', arr.astype(np.float32))
    jobs = []
    for P in pr.DEVICE_P:
        for H in pr.HEIGHTS:
            for s in ('rand11', 'multi7', f'warm:{SEEDS}/newA.npy',
                      f'warm:{SEEDS}/oldB.npy'):
                jobs.append(f'D2 {P:.0f} {H:.0f} {s} 150 ceiling_theta0')
    (pr.HERE / 'stage1_jobs.txt').write_text('\n'.join(jobs) + '\n')
    ceil = []
    for P in pr.CEIL_P:
        for H in (170.0, 230.0, 290.0):
            for s in ('rand11', 'multi7', f'warm:{SEEDS}/oldB.npy'):
                ceil.append(f'D2 {P:.0f} {H:.0f} {s} 150 ceiling_largeP')
    (pr.HERE / 'ceiling_jobs.txt').write_text('\n'.join(ceil) + '\n')
    print(len(jobs), 'stage-1 jobs;', len(ceil), 'large-P jobs')


if __name__ == '__main__':
    main()
