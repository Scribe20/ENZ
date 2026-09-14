"""Physics analysis figure for a binary design: geometry, band diagram, band-edge mode fields."""
import sys, numpy as np
from pwem import *; from plots import *
import matplotlib.pyplot as plt

def band_edge_kpoints(res, n_tm, n_te):
    """k-points (on the official grid) where the four gap-defining band extrema occur"""
    K = kgrid_official(wedge=True)
    btm, bte = res['bands_tm'], res['bands_te']
    return dict(tm_lo=(K[np.argmax(btm[:, n_tm])], btm[:, n_tm].max()), tm_hi=(K[np.argmin(btm[:, n_tm + 1])], btm[:, n_tm + 1].min()),
                te_lo=(K[np.argmax(bte[:, n_te])], bte[:, n_te].max()), te_hi=(K[np.argmin(bte[:, n_te + 1])], bte[:, n_te + 1].min()))

def kname(k):
    names = {(0, 0): 'Γ', (1, 0): 'X', (1, 1): 'M'}
    key = (round(k[0] / np.pi, 2), round(k[1] / np.pi, 2))
    return names.get(key, f"({key[0]:.1f},{key[1]:.1f})π/a")

def analyze(mask, out_png, title=''):
    eps = mask_to_eps(mask)
    r = evaluate(eps, nbands=12)
    g = r['gap']; n_tm, n_te = g['n_tm'], g['n_te']
    edges = band_edge_kpoints(r, n_tm, n_te)
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(gs[0, 0]); plot_geometry(eps, ax, title=f"{title} fill={r['fill']:.3f}")
    ax = fig.add_subplot(gs[0, 1:3]); band_diagram(eps, ax, gap=g, title=f"TE red, TM blue | score {100*r['score']:.2f}%  gap [{g['w_low']:.4f},{g['w_high']:.4f}]", ylim=(0, 0.7))
    ax = fig.add_subplot(gs[0, 3]); ax.axis('off')
    txt = (f"complete gap: [{g['w_low']:.5f}, {g['w_high']:.5f}]\nTM bands {n_tm+1}-{n_tm+2}: [{r['tm_gap_at_target'][0]:.5f}, {r['tm_gap_at_target'][1]:.5f}]\n"
           f"TE bands {n_te+1}-{n_te+2}: [{r['te_gap_at_target'][0]:.5f}, {r['te_gap_at_target'][1]:.5f}]\n"
           f"margin lo = {TARGET-g['w_low']:.5f}\nmargin hi = {g['w_high']-TARGET:.5f}\nscore = {100*r['score']:.3f}%\n\n"
           f"band-edge k-points:\n TM{n_tm+1} max at {kname(edges['tm_lo'][0])}: {edges['tm_lo'][1]:.4f}\n TM{n_tm+2} min at {kname(edges['tm_hi'][0])}: {edges['tm_hi'][1]:.4f}\n"
           f" TE{n_te+1} max at {kname(edges['te_lo'][0])}: {edges['te_lo'][1]:.4f}\n TE{n_te+2} min at {kname(edges['te_hi'][0])}: {edges['te_hi'][1]:.4f}")
    ax.text(0, 1, txt, va='top', family='monospace', fontsize=9)
    # field plots of the four band-edge modes
    panels = [('tm', n_tm, edges['tm_lo'][0], f"TM band {n_tm+1} (gap bottom) at {kname(edges['tm_lo'][0])}"),
              ('tm', n_tm + 1, edges['tm_hi'][0], f"TM band {n_tm+2} (gap top) at {kname(edges['tm_hi'][0])}"),
              ('te', n_te, edges['te_lo'][0], f"TE band {n_te+1} (gap bottom) at {kname(edges['te_lo'][0])}"),
              ('te', n_te + 1, edges['te_hi'][0], f"TE band {n_te+2} (gap top) at {kname(edges['te_hi'][0])}")]
    conc = {}
    for i, (pol, n, k, lab) in enumerate(panels):
        w, F = mode_field(eps, k, pol, n)
        ax = fig.add_subplot(gs[1, i])
        if pol == 'tm':
            fld = np.real(F['Ez'] * np.exp(-1j * np.angle(F['Ez'][np.unravel_index(np.argmax(np.abs(F['Ez'])), F['Ez'].shape)])))
            vmax = np.abs(fld).max(); ax.imshow(fld, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=[0, 1, 0, 1])
            what = 'Ez'
        else:
            fld = F['energy']; ax.imshow(fld, origin='lower', cmap='inferno', extent=[0, 1, 0, 1]); what = '|D|²/ε (electric energy)'
        en = F['energy']; conc[(pol, n)] = float(en[mask].sum() / en.sum())
        ax.contour(np.linspace(0, 1, 96), np.linspace(0, 1, 96), mask.astype(float), levels=[0.5], colors='lime', linewidths=0.7)
        ax.set_title(f"{lab}\nω={w:.4f}, {what}, energy in Si: {100*conc[(pol,n)]:.0f}%", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(out_png, dpi=100)
    print(summarize(r, title)); print("energy concentration in Si (electric energy):", {f"{p}{n+1}": round(v, 3) for (p, n), v in conc.items()})
    print("band-edge k:", {k: (kname(v[0]), round(v[1], 5)) for k, v in edges.items()})
    return r, conc, edges

if __name__ == '__main__':
    m = np.load(sys.argv[1]); analyze(m, sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else '')
