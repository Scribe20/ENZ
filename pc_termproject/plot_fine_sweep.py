"""Ablation figure: score and the four limiting band edges vs rod radius r and vein width w (rods+veins family)."""
import json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pwem import TARGET
recs = json.load(open('logs/fine_sweep_rod_veins.json'))
rs = sorted(set(r['r'] for r in recs)); ws = sorted(set(r['w'] for r in recs))
def grid(key):
    Z = np.full((len(ws), len(rs)), np.nan)
    for r in recs: Z[ws.index(r['w']), rs.index(r['r'])] = r[key]
    return Z
fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
Z = 100 * grid('score')
im = axs[0].imshow(Z, origin='lower', aspect='auto', extent=[rs[0]-0.005, rs[-1]+0.005, ws[0]-0.005, ws[-1]+0.005], cmap='viridis', vmin=0)
axs[0].set_xlabel('rod radius r/a'); axs[0].set_ylabel('vein width w/a'); axs[0].set_title('score (%) — rods + veins'); plt.colorbar(im, ax=axs[0])
# slice at w = 0.08: edges vs r ; slice at r = 0.32: edges vs w
for ax, (key, vals, xlabel, fixed) in zip(axs[1:], [('r', rs, 'rod radius r/a', ('w', 0.08)), ('w', ws, 'vein width w/a', ('r', 0.32))]):
    sel = [r for r in recs if abs(r[fixed[0]] - fixed[1]) < 1e-6]; sel.sort(key=lambda r: r[key])
    x = [r[key] for r in sel]
    for edge, lab, c, ls in [('tm3max', 'TM band 3 max', 'tab:blue', '-'), ('tm4min', 'TM band 4 min', 'tab:blue', '--'), ('te2max', 'TE band 2 max', 'tab:red', '-'), ('te3min', 'TE band 3 min', 'tab:red', '--')]:
        ax.plot(x, [r[edge] for r in sel], color=c, ls=ls, marker='o', ms=3, label=lab)
    ax.axhline(TARGET, color='k', ls=':', label='target 0.409')
    ax.set_xlabel(xlabel); ax.set_ylabel('a/λ'); ax.set_title(f'band edges vs {key} ({fixed[0]} = {fixed[1]})'); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig('figs/ablation_rod_veins.png', dpi=110)
best = max(recs, key=lambda r: r['score']); print("best in fine sweep:", best['r'], best['w'], round(100 * best['score'], 2), "fill", round(best['fill'], 3))
