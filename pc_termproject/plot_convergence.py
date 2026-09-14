"""Convergence / robustness figure for a design: score and gap edges vs Mmax for the three formulations."""
import sys, json, numpy as np
from pwem import *; from verify import basis_convergence
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
mask = np.load(sys.argv[1]); eps = mask_to_eps(mask); out_png = sys.argv[2]
r0 = evaluate(eps, nbands=12); g0 = r0['gap']
rows = basis_convergence(eps, Ms=(5, 7, 9, 11, 13, 15), forms=('official', 'eta', 'eps'), pair_bands=(g0['n_tm'], g0['n_te']))
json.dump(rows, open(out_png.replace('.png', '.json'), 'w'), indent=1)
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for form, c in (('official', 'k'), ('eta', 'tab:red'), ('eps', 'tab:blue')):
    rr = [x for x in rows if x['form'] == form]
    axs[0].plot([x['Mmax'] for x in rr], [100 * x['score'] for x in rr], 'o-', color=c, label=form)
    axs[1].plot([x['Mmax'] for x in rr], [x['tm_pair'][1] for x in rr], 's-', color=c, label=f'{form}: TM{g0["n_tm"]+2} min')
    axs[1].plot([x['Mmax'] for x in rr], [x['te_pair'][0] for x in rr], '^--', color=c, label=f'{form}: TE{g0["n_te"]+1} max')
axs[0].axvline(9, color='gray', ls=':'); axs[0].set_xlabel('Mmax'); axs[0].set_ylabel('score (%)'); axs[0].legend(); axs[0].set_title('score vs basis size')
axs[1].axhline(TARGET, color='k', ls='--', lw=0.8); axs[1].axvline(9, color='gray', ls=':'); axs[1].set_xlabel('Mmax'); axs[1].set_ylabel('a/λ'); axs[1].legend(fontsize=7); axs[1].set_title('limiting band edges vs basis size')
plt.tight_layout(); plt.savefig(out_png, dpi=110)
