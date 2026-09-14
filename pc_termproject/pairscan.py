"""Screen which (TM, TE) band pairs can host a complete gap at the target: gray-start SLP at Mmax=7 (fast),
then binarise and evaluate at the exact grading basis."""
import sys, json, time, os
import numpy as np
from pwem import *
from topopt import *
pairs = [(int(a), int(b)) for a in sys.argv[1].split(',') for b in sys.argv[2].split(',')]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
Mfast = int(sys.argv[4]) if len(sys.argv) > 4 else 7
iters = int(sys.argv[5]) if len(sys.argv) > 5 else 30
rng = np.random.default_rng(seed)
os.makedirs('runs/pairscan', exist_ok=True)
log = open(f'runs/pairscan/pairscan_seed{seed}_M{Mfast}.jsonl', 'a')
for (ntm, nte) in pairs:
    t0 = time.time()
    rho0 = 0.5 + 0.05 * rng.standard_normal((96, 96)); rho0 = np.clip(ConicFilter(4.0)(rho0), 0, 1); rho0 = symmetrize_c4v(rho0)
    opt = ScoreOptimizer(rho0, ntm - 1, nte - 1, Mmax=Mfast, symmetric=True, filter_R=2.0, beta=1.0)
    per = max(1, iters // 3)
    x, cur = opt.run(iters=iters, step0=0.2, step_min=0.004, beta_schedule=[(0, 1.0), (per, 4.0), (2 * per, 16.0)], verbose=False)
    mask = opt.binary_design()
    r = evaluate(mask_to_eps(mask), Mmax=9, nbands=12)
    rec = dict(pair=(ntm, nte), seed=seed, Mfast=Mfast, cont_margin=float(cur), cont_score=200 * cur / TARGET, bin_score=100 * r['score'],
               gap=r['gap'], fill=r['fill'], t=time.time() - t0)
    log.write(json.dumps(rec) + "\n"); log.flush()
    np.save(f'runs/pairscan/mask_TM{ntm}_TE{nte}_s{seed}_M{Mfast}.npy', mask)
    g = r['gap']
    print(f"pair TM{ntm}-{ntm+1}/TE{nte}-{nte+1} seed{seed}: continuous score {200*cur/TARGET:7.3f}%  -> binary official {100*r['score']:6.3f}%  fill={r['fill']:.3f}"
          f"  {'gap [%.4f,%.4f] TM%d-%d/TE%d-%d' % (g['w_low'], g['w_high'], g['n_tm']+1, g['n_tm']+2, g['n_te']+1, g['n_te']+2) if g else ''}  ({time.time()-t0:.0f}s)", flush=True)
