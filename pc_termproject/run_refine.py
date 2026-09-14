import sys, json, time, os, numpy as np
from pwem import *; from topopt import *; import stageA as SA
name, start, n_tm, n_te = sys.argv[1], sys.argv[2], int(sys.argv[3]) - 1, int(sys.argv[4]) - 1
rounds = int(sys.argv[5]) if len(sys.argv) > 5 else 100
depth = int(sys.argv[6]) if len(sys.argv) > 6 else 1
sym = int(sys.argv[7]) if len(sys.argv) > 7 else 1
if start.startswith('file:'): mask = np.load(start[5:]).astype(bool)
else:
    fam, ps = start.split(':'); mask = getattr(SA, 'fam_' + fam)(*[float(p) for p in ps.split(',')])
outdir = f"runs/{name}"; os.makedirs(outdir, exist_ok=True)
r0 = evaluate(mask_to_eps(mask)); print(summarize(r0, name + ' START'), flush=True)
log = []
mask, cur = discrete_refine(mask, n_tm, n_te, Mmax=9, symmetric=bool(sym), max_rounds=rounds, log=log, depth=depth, K=None if sym else kgrid_official(wedge=False))
r = evaluate(mask_to_eps(mask), nbands=12); print(summarize(r, name + ' REFINED'), flush=True)
np.save(f"{outdir}/mask.npy", mask)
json.dump(dict(name=name, start=start, n_tm=n_tm, n_te=n_te, score_start=r0['score'], score=r['score'], gap=r['gap'], fill=r['fill'], log=log), open(f"{outdir}/result.json", 'w'), indent=1)
