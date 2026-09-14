"""Fine (r, w) map of the rods+veins family around the optimum (ablation figure), official settings, wedge."""
import numpy as np, json, itertools, time
from pwem import *; import stageA as SA
rs = np.round(np.arange(0.27, 0.371, 0.01), 3); ws = np.round(np.arange(0.03, 0.131, 0.01), 3)
out = []
t0 = time.time()
for r, w in itertools.product(rs, ws):
    m = SA.fam_rod_veins(float(r), float(w)); res = evaluate(mask_to_eps(m), nbands=10)
    g = res['gap']; btm, bte = res['bands_tm'], res['bands_te']
    rec = dict(r=float(r), w=float(w), fill=res['fill'], score=res['score'],
               tm3max=float(btm[:, 2].max()), tm4min=float(btm[:, 3].min()), te2max=float(bte[:, 1].max()), te3min=float(bte[:, 2].min()),
               gap=g)
    out.append(rec)
    print(f"r={r:.2f} w={w:.2f} fill={res['fill']:.3f} score={100*res['score']:6.2f}% TM3max={rec['tm3max']:.4f} TM4min={rec['tm4min']:.4f} TE2max={rec['te2max']:.4f} TE3min={rec['te3min']:.4f} ({time.time()-t0:.0f}s)", flush=True)
json.dump(out, open('logs/fine_sweep_rod_veins.json', 'w'), indent=1)
