"""Basin hopping on the exact objective: random boundary kicks of the current best binary C4v design, each followed by
LP-guided exact discrete refinement; keep the kick only if the refined score beats the incumbent."""
import sys, json, time, os, numpy as np
from scipy import ndimage
from pwem import *; from topopt import *
name, start, n_tm, n_te = sys.argv[1], sys.argv[2], int(sys.argv[3]) - 1, int(sys.argv[4]) - 1
hops = int(sys.argv[5]) if len(sys.argv) > 5 else 20
seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
rng = np.random.default_rng(seed)
outdir = f"runs/{name}"; os.makedirs(outdir, exist_ok=True)
best = np.load(start).astype(bool)
orb_ids, nvar = c4v_orbit_ids()
r = evaluate(mask_to_eps(best)); best_score = r['score']
print(f"incumbent {100*best_score:.3f}%", flush=True)
hist = []
for h in range(hops):
    t0 = time.time()
    m = best.copy()
    kind = rng.choice(['orbits', 'erode_patch', 'dilate_patch', 'shift_vein', 'shift_block'])
    pm = np.pad(m, 4, mode='wrap')
    bd = (m ^ ndimage.binary_erosion(pm, iterations=2)[4:-4, 4:-4]) | (m ^ ndimage.binary_dilation(pm, iterations=2)[4:-4, 4:-4])
    if kind == 'orbits':
        cand = np.unique(orb_ids[bd]); sel = rng.choice(cand, size=int(rng.integers(6, 25)), replace=False)
        flip = np.isin(orb_ids, sel); m = m ^ flip
    elif kind in ('erode_patch', 'dilate_patch'):
        # erode/dilate within a random C4v-symmetric annulus of radii (r1, r2) about the centre
        X, Y = meshgrid_xy(); R = np.sqrt(X**2 + Y**2); r1 = rng.uniform(0, 0.6); r2 = r1 + rng.uniform(0.03, 0.12)
        region = (R >= r1) & (R < r2)
        op = ndimage.binary_erosion if kind == 'erode_patch' else ndimage.binary_dilation
        m2 = op(pm, iterations=1)[4:-4, 4:-4]; m = np.where(region, m2, m)
    elif kind == 'shift_vein':
        # widen or narrow the veins (rows/cols near the axes) by one pixel
        X, Y = meshgrid_xy(); vein = (np.abs(X) < 0.08) | (np.abs(Y) < 0.08); far = (np.abs(X) > 0.25) & (np.abs(Y) > 0.25) | ((np.abs(X) > 0.25) & (np.abs(Y) < 0.08)) | ((np.abs(Y) > 0.25) & (np.abs(X) < 0.08))
        op = ndimage.binary_erosion if rng.random() < 0.5 else ndimage.binary_dilation
        m2 = op(pm, iterations=1)[4:-4, 4:-4]; m = np.where(vein & far, m2, m)
    else:
        X, Y = meshgrid_xy(); blk = (np.abs(X) < 0.32) & (np.abs(Y) < 0.32)
        op = ndimage.binary_erosion if rng.random() < 0.5 else ndimage.binary_dilation
        m2 = op(pm, iterations=1)[4:-4, 4:-4]; m = np.where(blk, m2, m)
    m = symmetrize_c4v(m, 'majority')
    m, cur = discrete_refine(m, n_tm, n_te, symmetric=True, max_rounds=60, verbose=False, depth=2)
    r = evaluate(mask_to_eps(m)); s = r['score']
    improved = s > best_score + 1e-6
    print(f"hop {h:2d} [{kind:12s}] refined {100*s:.3f}% {'IMPROVED' if improved else ''} (incumbent {100*best_score:.3f}%, fill {m.mean():.4f}, {time.time()-t0:.0f}s)", flush=True)
    hist.append(dict(hop=h, kind=str(kind), score=s, improved=bool(improved)))
    if improved:
        best, best_score = m, s; np.save(f"{outdir}/mask.npy", best)
        json.dump(dict(name=name, score=best_score, gap=r['gap'], fill=r['fill'], hist=hist), open(f"{outdir}/result.json", 'w'), indent=1)
np.save(f"{outdir}/mask.npy", best)
r = evaluate(mask_to_eps(best), nbands=12); print(summarize(r, name + ' FINAL'))
json.dump(dict(name=name, score=best_score, gap=r['gap'], fill=r['fill'], hist=hist), open(f"{outdir}/result.json", 'w'), indent=1)
