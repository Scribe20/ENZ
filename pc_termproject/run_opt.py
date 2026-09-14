"""Driver for Stage B optimisation runs.
usage: python3 run_opt.py NAME START N_TM N_TE [options as key=value]
  START: 'gray' | 'file:path.npy' | 'rod_veins:r,w' | 'circle_hole:r' | 'hole_rod:rh,rr' | 'frame_rod:w,rr' | 'circle_rod:r' | 'diamond_rod:s' | ...
  N_TM, N_TE: 1-based index of the band BELOW the gap for each polarisation.
options: iters=60 step0=0.15 R=2.0 betas=1,2,4,8,16,32 seed=0 sym=1 Mmax=9 noise=0.02
"""
import sys, json, time, os
import numpy as np
from pwem import *
from topopt import *
import stageA as SA
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

name, start, n_tm, n_te = sys.argv[1], sys.argv[2], int(sys.argv[3]) - 1, int(sys.argv[4]) - 1
opts = dict(step0=0.15, R=3.0, betas='1:10,2:8,4:8,8:8,16:10,32:12,64:14,128:16,256:16', seed=0, sym=1, Mmax=9, noise=0.02, stepmin=0.004, refine=1, refine_rounds=120, depth=2)
for a in sys.argv[5:]:
    k, v = a.split('='); opts[k] = type(opts[k])(v) if k in opts else v
rng = np.random.default_rng(int(opts['seed']))
N = 96
outdir = f"/home/user/ENZ/pc_termproject/runs/{name}"; os.makedirs(outdir, exist_ok=True)

# ---- starting density
if start == 'gray':
    rho0 = 0.5 + float(opts['noise']) * rng.standard_normal((N, N))
    rho0 = ConicFilter(4.0)(rho0); rho0 = np.clip(rho0, 0, 1)
elif start.startswith('file:'):
    rho0 = np.load(start[5:]).astype(float)
else:
    fam, ps = start.split(':'); ps = [float(p) for p in ps.split(',')]
    fn = getattr(SA, 'fam_' + fam)
    rho0 = fn(*ps).astype(float)
    # soften the binary start slightly so gradients see both phases at the boundary
    rho0 = 0.9 * rho0 + 0.05
if int(opts['sym']): rho0 = symmetrize_c4v(rho0)

bspec = [(float(s.split(':')[0]), int(s.split(':')[1])) for s in str(opts['betas']).split(',')]
betas = [b for b, n in bspec]
schedule = []; acc = 0
for b, n in bspec:
    schedule.append((acc, b)); acc += n
iters = acc

opt = ScoreOptimizer(rho0, n_tm, n_te, Mmax=int(opts['Mmax']), symmetric=bool(int(opts['sym'])), filter_R=float(opts['R']), beta=betas[0], name=name)
print(f"run {name}: start={start} bands TM{n_tm+1}-{n_tm+2} TE{n_te+1}-{n_te+2} nvar={opt.nvar} opts={opts}", flush=True)
t0 = time.time()
x, cur = opt.run(iters=iters, step0=float(opts['step0']), step_min=float(opts['stepmin']), beta_schedule=schedule)
print(f"continuous optimisation done in {time.time()-t0:.0f}s, final min margin {cur:.5f} (score {200*cur/TARGET:.3f}%)", flush=True)

# ---- binarise and evaluate at the exact grading settings (wedge is exact for C4v; full grid otherwise)
mask = opt.binary_design()
eps = mask_to_eps(mask)
r = evaluate(eps, Mmax=9, nbands=10)
print(summarize(r, name + ' BINARY official'), flush=True)
rp, _ = opt.physical(x)
np.save(f"{outdir}/mask_prerefine.npy", mask)
score_prerefine = r['score']
if int(opts['refine']):
    print("=== discrete refinement of the binary design", flush=True)
    rlog = []
    mask, cur_b = discrete_refine(mask, n_tm, n_te, Mmax=9, symmetric=bool(int(opts['sym'])), max_rounds=int(opts['refine_rounds']), log=rlog, depth=int(opts['depth']))
    eps = mask_to_eps(mask)
    r = evaluate(eps, Mmax=9, nbands=10)
    print(summarize(r, name + ' REFINED official'), flush=True)
np.save(f"{outdir}/rho_final.npy", x); np.save(f"{outdir}/rho_phys.npy", rp); np.save(f"{outdir}/mask.npy", mask)
json.dump(dict(name=name, start=start, n_tm=n_tm, n_te=n_te, opts=opts, score_binary=r['score'], score_prerefine=score_prerefine, gap=r['gap'], fill=r['fill'],
               tm_gaps=r['tm_gaps'], te_gaps=r['te_gaps'], history=opt.history, cont_min_margin=float(cur)),
          open(f"{outdir}/result.json", 'w'), indent=1)
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(rho0, origin='lower', cmap='gray_r', vmin=0, vmax=1); ax[0].set_title('start')
ax[1].imshow(rp, origin='lower', cmap='gray_r', vmin=0, vmax=1); ax[1].set_title('final continuous')
ax[2].imshow(mask, origin='lower', cmap='gray_r', vmin=0, vmax=1); ax[2].set_title(f"binary (refined): {100*r['score']:.2f}%")
plt.tight_layout(); plt.savefig(f"{outdir}/design.png", dpi=100)
