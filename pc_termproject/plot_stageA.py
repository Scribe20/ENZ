"""Stage A summary figures from logs/stageA.jsonl: best score per family, and (r,w) maps for 2-parameter families."""
import json, numpy as np, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
recs = [json.loads(l) for l in open('logs/stageA.jsonl')]
fams = {}
for r in recs: fams.setdefault(r['family'], []).append(r)
print(f"{len(recs)} records, {len(fams)} families")
rows = []
for name, rs in fams.items():
    b = max(rs, key=lambda r: r['score'])
    g = b['gap']
    rows.append((name, len(rs), 100 * b['score'], b['params'], b['fill'], f"TM{g['n_tm']+1}-{g['n_tm']+2}/TE{g['n_te']+1}-{g['n_te']+2}" if g else '-', sum(1 for r in rs if r['score'] > 0)))
rows.sort(key=lambda x: -x[2])
print(f"{'family':24s} {'n':>4s} {'best %':>7s} {'params':22s} {'fill':>6s} {'pair':12s} {'#>0':>4s}")
for row in rows: print(f"{row[0]:24s} {row[1]:4d} {row[2]:7.2f} {str(row[3]):22s} {row[4]:6.3f} {row[5]:12s} {row[6]:4d}")
# bar chart
fig, ax = plt.subplots(figsize=(9, 4))
ax.barh([r[0] for r in rows][::-1], [r[2] for r in rows][::-1], color=['tab:green' if r[2] > 0 else 'lightgray' for r in rows][::-1])
ax.set_xlabel('best official-setting score in family (%)'); ax.set_title('Stage A: best score per topology family (all C4v, Mmax=9)')
plt.tight_layout(); plt.savefig('figs/stageA_families.png', dpi=110)
# 2-parameter maps
two = [n for n, rs in fams.items() if len(rs[0]['params']) == 2 and len(rs) > 6]
if two:
    fig, axs = plt.subplots(2, (len(two) + 1) // 2, figsize=(4.2 * ((len(two) + 1) // 2), 7.5)); axs = np.atleast_1d(axs).ravel()
    for ax, name in zip(axs, two):
        rs = fams[name]; p0 = sorted(set(r['params'][0] for r in rs)); p1 = sorted(set(r['params'][1] for r in rs))
        Z = np.full((len(p1), len(p0)), np.nan)
        for r in rs: Z[p1.index(r['params'][1]), p0.index(r['params'][0])] = 100 * r['score']
        im = ax.imshow(Z, origin='lower', aspect='auto', extent=[p0[0], p0[-1], p1[0], p1[-1]], cmap='viridis', vmin=0)
        ax.set_title(name); ax.set_xlabel('param 1'); ax.set_ylabel('param 2'); plt.colorbar(im, ax=ax, label='score (%)')
    for ax in axs[len(two):]: ax.axis('off')
    plt.tight_layout(); plt.savefig('figs/stageA_maps.png', dpi=100)
