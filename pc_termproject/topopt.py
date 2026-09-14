"""
Stage B: gradient-based (Hellmann-Feynman / adjoint-free) max-min optimisation of the official score.

Design variable: density rho in [0,1] per pixel (optionally reduced to C4v orbits),
    eps_p = 1 + rho_phys_p (EPS_SI - 1),  rho_phys = project(filter(rho))   (periodic conic filter, tanh projection)
Objective: maximise  t = min_i f_i,   f_i in { w_hi^pol(k) - w_t ,  w_t - w_lo^pol(k) }  over the k-set and both polarisations
   -> exactly the official score (x 2/w_t) for the chosen pair of gaps (TM bands n_tm/n_tm+1, TE bands n_te/n_te+1).
Method: sequential linear programming with a trust region (max-min LP), gradients from first-order perturbation theory:
   TM (official, eps-matrix form):  dlam/deps_p = -(1/N^2) |u(r_p)|^2 ,  u = E^{-1} D^{1/2} v          (~ |E_z|^2)
   TE (official, 1/eps form):       dlam/deta_p = +(1/N^2) |sum_G (k+G) v_G e^{iG.r_p}|^2 ,  deta/deps = -1/eps^2   (~ |D_t|^2/eps^2 ~ |E_t|^2)
"""
import numpy as np, time, json, os
from scipy import linalg as sla
from scipy.optimize import linprog
from pwem import *

# ----------------------------------------------------------------------------- C4v orbits
def c4v_orbit_ids(N=N_GRID):
    idx = np.arange(N * N).reshape(N, N)
    imgs = c4v_orbit(idx)                       # 8 images of the index array
    stack = np.stack(imgs, axis=0)              # (8,N,N): stack[g, iy, ix] = index of pixel that maps here
    orb = stack.min(axis=0)                     # orbit id = min index over the group images
    uniq, inv = np.unique(orb.ravel(), return_inverse=True)
    return inv.reshape(N, N), len(uniq)         # pixel -> variable id, number of variables

def reduce_to_orbits(field, orb_ids, nvar):
    """sum a per-pixel gradient over each orbit -> gradient wrt the orbit variable"""
    return np.bincount(orb_ids.ravel(), weights=field.ravel(), minlength=nvar)

def expand_from_orbits(x, orb_ids):
    return x[orb_ids]

# ----------------------------------------------------------------------------- filter / projection
class ConicFilter:
    def __init__(self, R, N=N_GRID):
        self.R = R
        if R <= 0.5:
            self.kern_fft = None; return
        ix = np.arange(N); d = np.minimum(ix, N - ix)       # periodic distances
        DX, DY = np.meshgrid(d, d, indexing='ij')
        w = np.clip(R - np.sqrt(DX**2 + DY**2), 0, None)
        w /= w.sum()
        self.kern_fft = np.fft.fft2(w)
    def __call__(self, x):
        if self.kern_fft is None: return x
        return np.real(np.fft.ifft2(np.fft.fft2(x) * self.kern_fft))
    # symmetric kernel -> adjoint == forward

def project(x, beta, eta=0.5):
    if beta <= 0: return x
    return (np.tanh(beta * eta) + np.tanh(beta * (x - eta))) / (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))

def project_deriv(x, beta, eta=0.5):
    if beta <= 0: return np.ones_like(x)
    return beta * (1 - np.tanh(beta * (x - eta))**2) / (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))

# ----------------------------------------------------------------------------- fields on the pixel grid
def field_on_grid(coeffs, basis, N=N_GRID):
    """u(r_p) = sum_G c_G exp(i G.r_p) on the pixel-centred grid; returns array [iy, ix] (complex).
    coeffs: (n,) or (n,2) (vector field -> returns (2,N,N))"""
    m1, m2 = basis.m1, basis.m2
    ph = np.exp(1j * np.pi * (m1 + m2) / N) * np.exp(-1j * np.pi * (m1 + m2))
    c = np.asarray(coeffs)
    if c.ndim == 1:
        U = np.zeros((N, N), complex)
        np.add.at(U, (m1 % N, m2 % N), c * ph)
        return (N * N * np.fft.ifft2(U)).T
    out = []
    for j in range(c.shape[1]):
        U = np.zeros((N, N), complex)
        np.add.at(U, (m1 % N, m2 % N), c[:, j] * ph)
        out.append((N * N * np.fft.ifft2(U)).T)
    return np.array(out)

def band_values_and_grads(st, k, bands_tm, bands_te, nb_solve=None):
    """For one k: eigenvalues and d(omega)/d(eps_p) fields for the requested band indices (0-based).
    Returns dict pol -> list of (omega, grad_field[N,N]) in the order of the requested band lists."""
    from pwem import _matrices
    A_tm, A_te, kG, mag = _matrices(st, k, 'official')
    N = st.N
    out = {'tm': [], 'te': []}
    if bands_tm:
        nb = max(bands_tm) + 1 if nb_solve is None else nb_solve
        lam, V = sla.eigh(A_tm, subset_by_index=[0, nb - 1], driver='evr')
        Einv = st.eps_inv_mat
        for n in bands_tm:
            v = V[:, n]; l = max(lam[n], 0.0); w = np.sqrt(l) / (2 * np.pi)
            u = Einv @ (mag * v)
            uf = field_on_grid(u, st.basis, N)
            dl_deps = -(np.abs(uf)**2) / N**2                  # dlambda / d eps_p
            dw = dl_deps / (8 * np.pi**2 * w) if w > 1e-9 else np.zeros_like(dl_deps)
            out['tm'].append((w, dw))
    if bands_te:
        nb = max(bands_te) + 1 if nb_solve is None else nb_solve
        lam, V = sla.eigh(A_te, subset_by_index=[0, nb - 1], driver='evr')
        eps2 = st.eps**2
        for n in bands_te:
            v = V[:, n]; l = max(lam[n], 0.0); w = np.sqrt(l) / (2 * np.pi)
            wf = field_on_grid(kG * v[:, None], st.basis, N)    # (2,N,N)
            dl_deta = (np.abs(wf[0])**2 + np.abs(wf[1])**2) / N**2
            dl_deps = dl_deta * (-1.0 / eps2)
            dw = dl_deps / (8 * np.pi**2 * w) if w > 1e-9 else np.zeros_like(dl_deps)
            out['te'].append((w, dw))
    return out

# ----------------------------------------------------------------------------- the optimiser
class ScoreOptimizer:
    def __init__(self, rho0, n_tm, n_te, K=None, Mmax=9, symmetric=True, filter_R=2.0, beta=1.0,
                 log_path=None, name='opt'):
        """n_tm, n_te: 0-based index of the band BELOW the gap for each polarisation."""
        self.N = rho0.shape[0]
        self.n_tm, self.n_te = n_tm, n_te
        self.K = kgrid_official(wedge=True) if K is None else K
        self.Mmax = Mmax
        self.symmetric = symmetric
        self.filt = ConicFilter(filter_R, self.N)
        self.beta = beta
        self.orb_ids, self.nvar = c4v_orbit_ids(self.N) if symmetric else (np.arange(self.N**2).reshape(self.N, self.N), self.N**2)
        self.rho = self.to_var(rho0)
        self.log_path = log_path; self.name = name
        self.history = []

    def to_var(self, field):
        if self.symmetric:
            # average over orbit -> variable
            s = np.bincount(self.orb_ids.ravel(), weights=field.ravel(), minlength=self.nvar)
            c = np.bincount(self.orb_ids.ravel(), minlength=self.nvar)
            return s / c
        return field.ravel().astype(float)

    def to_field(self, x):
        return x[self.orb_ids] if self.symmetric else x.reshape(self.N, self.N)

    def physical(self, x):
        """design variable -> filtered -> projected density field, plus the chain-rule factor"""
        rf = self.filt(self.to_field(x))
        rp = project(rf, self.beta)
        return rp, project_deriv(rf, self.beta)

    def eps_of(self, rp):
        return EPS_AIR + rp * (EPS_SI - EPS_AIR)

    def evaluate_constraints(self, x, with_grads=True):
        """returns f (array of margins), G (array nconstr x nvar of gradients d f / d x), extra info"""
        rp, dproj = self.physical(x)
        eps = self.eps_of(rp)
        st = Structure(eps, self.Mmax)
        st.eps_inv_mat
        f = []; G = []; info = []
        for k in self.K:
            res = band_values_and_grads(st, k, [self.n_tm, self.n_tm + 1], [self.n_te, self.n_te + 1])
            for pol, n0 in (('tm', self.n_tm), ('te', self.n_te)):
                (w_lo, g_lo), (w_hi, g_hi) = res[pol]
                # margins
                f.append(TARGET - w_lo); f.append(w_hi - TARGET)
                info.append((pol, n0, 'lo', tuple(k), w_lo)); info.append((pol, n0 + 1, 'hi', tuple(k), w_hi))
                if with_grads:
                    for sgn, g in ((-1.0, g_lo), (+1.0, g_hi)):
                        # d f / d rho_phys = sgn * dw/deps * (EPS_SI-1); chain: through projection and filter, then orbits
                        gp = sgn * g * (EPS_SI - EPS_AIR) * dproj
                        gf = self.filt(gp)                       # filter adjoint (symmetric kernel)
                        G.append(reduce_to_orbits(gf, self.orb_ids, self.nvar) if self.symmetric else gf.ravel())
        f = np.array(f); G = np.array(G) if with_grads else None
        return f, G, info, rp

    def lp_step(self, f, G, x, step):
        """max t  s.t.  f_i + G_i . d >= t ,  -step <= d <= step ,  0 <= x + d <= 1"""
        nv = self.nvar
        # variables: [d (nv), t (1)] ; minimise -t
        c = np.zeros(nv + 1); c[-1] = -1.0
        A_ub = np.hstack([-G, np.ones((G.shape[0], 1))])       # -G d + t <= f
        b_ub = f
        lo = np.maximum(-step, -x); hi = np.minimum(step, 1 - x)
        bounds = [(lo[i], hi[i]) for i in range(nv)] + [(None, None)]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not res.success:
            raise RuntimeError("LP failed: " + res.message)
        return res.x[:nv], res.x[-1]

    def run(self, iters=50, step0=0.15, step_min=0.005, beta_schedule=None, verbose=True, tol=1e-5):
        """beta_schedule: list of (iteration, beta) to raise projection sharpness"""
        x = self.rho.copy(); step = step0
        f, G, info, rp = self.evaluate_constraints(x)
        cur = f.min()
        t_start = time.time()
        for it in range(iters):
            if beta_schedule:
                for (i0, b) in beta_schedule:
                    if it == i0 and b != self.beta:
                        self.beta = b
                        f, G, info, rp = self.evaluate_constraints(x); cur = f.min()
                        if verbose: print(f"  [beta -> {b}] current min margin {cur:.5f} (score {200*cur/TARGET:.3f}%)", flush=True)
            d, t_pred = self.lp_step(f, G, x, step)
            x_new = np.clip(x + d, 0, 1)
            f_new, G_new, info_new, rp_new = self.evaluate_constraints(x_new)
            new = f_new.min()
            improved = new > cur + tol
            imin = int(np.argmin(f_new))
            if verbose:
                print(f"  it {it:3d} beta={self.beta:5.1f} step={step:.4f} pred={t_pred:.5f} actual={new:.5f} (score {200*new/TARGET:6.3f}%) "
                      f"{'ACCEPT' if improved else 'reject'} | active: {info_new[imin][0]} band{info_new[imin][1]+1} {info_new[imin][2]} k=({info_new[imin][3][0]/np.pi:.1f},{info_new[imin][3][1]/np.pi:.1f})pi "
                      f"| fill(phys)={rp_new.mean():.3f} gray={np.mean((rp_new>0.05)&(rp_new<0.95)):.3f} | {time.time()-t_start:.0f}s", flush=True)
            self.history.append(dict(it=it, beta=self.beta, step=step, pred=float(t_pred), actual=float(new), accepted=bool(improved), fill=float(rp_new.mean())))
            if improved:
                x, f, G, info, rp, cur = x_new, f_new, G_new, info_new, rp_new, new
                if new - (f.min() if False else 0) >= 0: pass
                # expand trust region if the prediction was good
                if t_pred > 0 and (new - cur) >= 0: step = min(step * 1.3, 0.5)
            else:
                step *= 0.5
                if step < step_min:
                    if verbose: print("  trust region exhausted", flush=True)
                    break
        self.rho = x
        return x, cur

    def binary_design(self, x=None, thresh=0.5):
        rp, _ = self.physical(self.rho if x is None else x)
        mask = rp >= thresh
        if self.symmetric:
            mask = symmetrize_c4v(mask, 'majority')
        return mask

# ----------------------------------------------------------------------------- discrete (binary) refinement
def discrete_refine(mask0, n_tm, n_te, K=None, Mmax=9, symmetric=True, max_rounds=60, k_init=8, verbose=True,
                    log=None, boundary_only=True, min_gain=1e-6):
    """Gradient-guided flips of pixel orbits on a BINARY design, accepted only if the exact min margin improves.
    Each round: linearised margins f_i + sum_p G_ip * d_p (d_p = +1 for air->Si, -1 for Si->air) rank the candidate
    orbit flips; a greedy set of up to k flips is chosen so that the predicted min margin keeps increasing; the set is
    evaluated exactly; accepted if better, else k is halved."""
    from scipy import ndimage
    N = mask0.shape[0]
    K = kgrid_official(wedge=True) if K is None else K
    orb_ids, nvar = c4v_orbit_ids(N) if symmetric else (np.arange(N * N).reshape(N, N), N * N)
    mask = mask0.copy()

    def margins_and_grads(mask):
        eps = mask_to_eps(mask)
        st = Structure(eps, Mmax); st.eps_inv_mat
        f = []; G = []
        for k in K:
            res = band_values_and_grads(st, k, [n_tm, n_tm + 1], [n_te, n_te + 1])
            for pol, n0 in (('tm', n_tm), ('te', n_te)):
                (w_lo, g_lo), (w_hi, g_hi) = res[pol]
                f.append(TARGET - w_lo); G.append(reduce_to_orbits(-g_lo * (EPS_SI - EPS_AIR), orb_ids, nvar))
                f.append(w_hi - TARGET); G.append(reduce_to_orbits(+g_hi * (EPS_SI - EPS_AIR), orb_ids, nvar))
        return np.array(f), np.array(G)

    f, G = margins_and_grads(mask)
    cur = f.min(); k = k_init
    if verbose: print(f"  refine start: min margin {cur:.5f} (score {200*cur/TARGET:.3f}%)", flush=True)
    xo = np.bincount(orb_ids.ravel(), weights=mask.ravel().astype(float), minlength=nvar) / np.bincount(orb_ids.ravel(), minlength=nvar)
    xo = (xo > 0.5).astype(float)
    for rnd in range(max_rounds):
        # candidate orbits: boundary pixels only (interior flips are never first-order optimal for a gap)
        if boundary_only:
            pm = np.pad(mask, 2, mode='wrap')
            bd = (mask ^ ndimage.binary_erosion(pm, iterations=1)[2:-2, 2:-2]) | (mask ^ ndimage.binary_dilation(pm, iterations=1)[2:-2, 2:-2])
            cand = np.unique(orb_ids[bd])
        else:
            cand = np.arange(nvar)
        d = 1.0 - 2.0 * xo[cand]                       # +1 if currently air (0) -> Si, -1 if Si -> air
        # predicted min margin after each single flip
        pred_single = np.min(f[:, None] + G[:, cand] * d[None, :], axis=0)
        order = np.argsort(-pred_single)
        # greedy accumulation
        chosen = []; acc = np.zeros(len(f))
        for j in order[:max(4 * k, 20)]:
            trial = acc + G[:, cand[j]] * d[j]
            if np.min(f + trial) > np.min(f + acc) + min_gain and len(chosen) < k:
                chosen.append(j); acc = trial
        if not chosen:
            if verbose: print("  no improving flip predicted; stop", flush=True)
            break
        pred = np.min(f + acc)
        # apply
        xo_new = xo.copy(); xo_new[cand[chosen]] = 1.0 - xo_new[cand[chosen]]
        mask_new = xo_new[orb_ids] > 0.5
        f_new, G_new = margins_and_grads(mask_new)
        new = f_new.min()
        if verbose:
            print(f"  round {rnd:3d}: k={len(chosen):2d} flips pred {pred:.5f} actual {new:.5f} (score {200*new/TARGET:.3f}%) {'ACCEPT' if new > cur + min_gain else 'reject'} fill={mask_new.mean():.4f}", flush=True)
        if log is not None:
            log.append(dict(round=rnd, k=len(chosen), pred=float(pred), actual=float(new), accepted=bool(new > cur + min_gain)))
        if new > cur + min_gain:
            mask, xo, f, G, cur = mask_new, xo_new, f_new, G_new, new
            k = min(k * 2, 64)
        else:
            k = k // 2
            if k < 1:
                if verbose: print("  no single flip improves; stop", flush=True)
                break
    return mask, cur
