"""Collect all optimisation runs (runs/*/result.json, runs/*/mask.npy) into one comparison table (re-evaluated at
the exact grading settings, full 231-point grid, so the numbers are SURROGATE-OFFICIAL regardless of symmetry)."""
import json, glob, os, numpy as np
from pwem import *
rows = []
for d in sorted(glob.glob('runs/*/')):
    mp = os.path.join(d, 'mask.npy')
    if not os.path.exists(mp): continue
    m = np.load(mp).astype(bool)
    r = evaluate(mask_to_eps(m), nbands=12, wedge=False)
    g = r['gap']
    rows.append(dict(run=os.path.basename(d.rstrip('/')), score=100 * r['score'], fill=r['fill'], c4v=is_c4v(m),
                     gap=(g['w_low'], g['w_high']) if g else None, pair=f"TM{g['n_tm']+1}-{g['n_tm']+2}/TE{g['n_te']+1}-{g['n_te']+2}" if g else '-',
                     lo=(TARGET - g['w_low']) if g else None, hi=(g['w_high'] - TARGET) if g else None))
rows.sort(key=lambda x: -x['score'])
print(f"{'run':16s} {'score%':>7s} {'fill':>6s} {'C4v':>4s} {'gap':24s} {'pair':12s} {'lo':>7s} {'hi':>7s}")
for x in rows:
    print(f"{x['run']:16s} {x['score']:7.3f} {x['fill']:6.4f} {str(x['c4v']):>4s} {('[%.5f, %.5f]' % x['gap']) if x['gap'] else '-':24s} {x['pair']:12s} {x['lo'] if x['lo'] is None else round(x['lo'],5)!s:>7} {x['hi'] if x['hi'] is None else round(x['hi'],5)!s:>7}")
json.dump(rows, open('logs/candidate_table.json', 'w'), indent=1)
