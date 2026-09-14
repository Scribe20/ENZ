"""Stage C: verification / robustness analysis of a binary design at the exact grading settings and beyond."""
import numpy as np, json, time, sys
from pwem import *

def official_check(eps, label=''):
    """exact grading settings: Mmax=9, 21x11 half-BZ grid (231 points, no symmetry shortcut), official formulation"""
    r = evaluate(eps, Mmax=9, nbands=12, wedge=False)
    print(summarize(r, label + ' OFFICIAL-SETTINGS (231 k, 361 PW)'))
    return r

def wedge_vs_full(eps):
    rw = evaluate(eps, Mmax=9, nbands=12, wedge=True); rf = evaluate(eps, Mmax=9, nbands=12, wedge=False)
    d = max(np.max(np.abs(rw['bands_tm'].max(0) - rf['bands_tm'].max(0))), np.max(np.abs(rw['bands_tm'].min(0) - rf['bands_tm'].min(0))),
            np.max(np.abs(rw['bands_te'].max(0) - rf['bands_te'].max(0))), np.max(np.abs(rw['bands_te'].min(0) - rf['bands_te'].min(0))))
    print(f"  wedge(66) vs full(231) band-extrema difference: {d:.2e}   scores {100*rw['score']:.4f}% vs {100*rf['score']:.4f}%")
    return d

def basis_convergence(eps, Ms=(5, 7, 9, 11, 13, 15), forms=('official',), nbands=12, pair_bands=(2, 1)):
    out = []
    for M in Ms:
        for form in forms:
            t0 = time.time()
            r = evaluate(eps, Mmax=M, nbands=nbands, formulation=form, wedge=is_c4v(eps_to_mask(eps)))
            g = r['gap']
            # gap edges of the specific band pairs (even when the complete gap is closed): TM n_tm..n_tm+1, TE n_te..n_te+1
            btm, bte = r['bands_tm'], r['bands_te']
            pair = pair_bands
            tm_lo, tm_hi = btm[:, pair[0]].max(), btm[:, pair[0] + 1].min(); te_lo, te_hi = bte[:, pair[1]].max(), bte[:, pair[1] + 1].min()
            rec = dict(Mmax=M, form=form, score=r['score'], w_low=g['w_low'] if g else None, w_high=g['w_high'] if g else None,
                       n_tm=g['n_tm'] if g else None, n_te=g['n_te'] if g else None, t=time.time() - t0,
                       tm_pair=(float(tm_lo), float(tm_hi)), te_pair=(float(te_lo), float(te_hi)))
            out.append(rec)
            print(f"  Mmax={M:2d} {form:8s}: score={100*r['score']:6.3f}%  gap={'[%.5f, %.5f] TM%d-%d/TE%d-%d' % (g['w_low'], g['w_high'], g['n_tm']+1, g['n_tm']+2, g['n_te']+1, g['n_te']+2) if g else 'none'}"
                  f"  | TM{pair[0]+1}max={tm_lo:.5f} TM{pair[0]+2}min={tm_hi:.5f}  TE{pair[1]+1}max={te_lo:.5f} TE{pair[1]+2}min={te_hi:.5f}  ({rec['t']:.0f}s)", flush=True)
    return out

def fine_kgrid_check(eps, n=41):
    """band extrema on a fine full-BZ grid (n x n over [-pi,pi]^2) to check the score is not a k-sampling artefact"""
    kx = np.linspace(-np.pi, np.pi, n); KX, KY = np.meshgrid(kx, kx, indexing='xy')
    K = np.stack([KX.ravel(), KY.ravel()], axis=1)
    if is_c4v(eps_to_mask(eps)):
        K = K[(K[:, 1] >= -1e-12) & (K[:, 1] <= K[:, 0] + 1e-12)]
    r = evaluate(eps, Mmax=9, nbands=12, K=K)
    print(summarize(r, f'fine k-grid {n}x{n} (nk={len(K)})'))
    return r

def pixel_robustness(eps, n_trials=8, flip_frac=0.01, seed=0, erode_dilate=True):
    """score under random pixel flips (fraction of boundary pixels) and under 1-pixel erosion/dilation of Si"""
    rng = np.random.default_rng(seed)
    mask = eps_to_mask(eps)
    out = {}
    if erode_dilate:
        from scipy import ndimage
        for name, m in [('erode1', ndimage.binary_erosion(mask, iterations=1, border_value=0)),
                        ('dilate1', ndimage.binary_dilation(mask, iterations=1, border_value=0))]:
            # periodic: use wrap
            pass
        m_er = ndimage.binary_erosion(np.pad(mask, 2, mode='wrap'), iterations=1)[2:-2, 2:-2]
        m_di = ndimage.binary_dilation(np.pad(mask, 2, mode='wrap'), iterations=1)[2:-2, 2:-2]
        for name, m in [('erode1', m_er), ('dilate1', m_di)]:
            r = evaluate(mask_to_eps(m), Mmax=9, nbands=12)
            out[name] = r['score']; print(f"  {name}: fill={fill_fraction(m):.4f} score={100*r['score']:.3f}%")
    # random boundary flips
    from scipy import ndimage
    bd = mask ^ ndimage.binary_erosion(np.pad(mask, 2, mode='wrap'), iterations=1)[2:-2, 2:-2]
    bd |= mask ^ ndimage.binary_dilation(np.pad(mask, 2, mode='wrap'), iterations=1)[2:-2, 2:-2]
    idx = np.flatnonzero(bd)
    scores = []
    for t in range(n_trials):
        m = mask.copy().ravel()
        sel = rng.choice(idx, size=max(1, int(flip_frac * len(idx))), replace=False)
        m[sel] = ~m[sel]; m = m.reshape(mask.shape)
        r = evaluate(mask_to_eps(m), Mmax=9, nbands=12)   # no symmetry -> full grid
        scores.append(r['score'])
    print(f"  random flips of {100*flip_frac:.0f}% of boundary pixels ({len(idx)} boundary px): scores {[round(100*s,2) for s in scores]}")
    out['random_flips'] = scores
    return out

if __name__ == '__main__':
    path = sys.argv[1]
    eps = load_mat(path) if path.endswith('.mat') else mask_to_eps(np.load(path))
    print("design check:", check_design(eps), "fill", fill_fraction(eps_to_mask(eps)), "C4v", is_c4v(eps_to_mask(eps)))
    r0 = official_check(eps, path)
    wedge_vs_full(eps)
    g0 = r0['gap']
    basis_convergence(eps, Ms=(5, 7, 9, 11, 13, 15, 17), forms=('official', 'eta', 'eps'), pair_bands=(g0['n_tm'], g0['n_te']) if g0 else (2, 1))
    fine_kgrid_check(eps)
    pixel_robustness(eps)
