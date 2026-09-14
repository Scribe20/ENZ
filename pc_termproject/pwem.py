"""
Independent plane-wave expansion (PWEM) solver for 2D square-lattice photonic crystals.

Written for the Nanophotonics term project (Si/air, a = 634 nm, 96x96 pixel-centred grid).

Conventions (photonic-crystal convention, Lecture "Photonic Crystals III", p.14):
    TM : (Ez, Hx, Hy)   ->  scalar Ez problem
    TE : (Hz, Ex, Ey)   ->  scalar Hz problem
Units: lattice constant a = 1, frequencies reported as  omega*a/(2*pi*c) = a/lambda.

Matrix eigenvalue equation (Lecture "Photonic Crystals II", p.8-9, 13), symmetrised form:
    TM :  sum_G' |k+G| eta_{G-G'} |k+G'| b_G' = k0^2 b_G
    TE :  sum_G' (k+G).(k+G') eta_{G-G'} c_G'  = k0^2 c_G
with eta = 1/eps and
    eta_{G} = (1/|Omega|) sum_{p,q} eps_{p,q}^{-1} exp(-i G.r_{p,q}) dx dy      (pixel-centred sum)

A second, independent formulation ('eps') uses the Fourier coefficients of eps itself and the
inverse of the truncated eps-matrix (Ho-Chan-Soukoulis).  Both converge to the same bands as the
basis grows; their difference at finite Mmax is a truncation artefact that we monitor.
"""
import os, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy import linalg as sla
from concurrent.futures import ThreadPoolExecutor
try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)          # one BLAS thread per eigen-solve; parallelism is over k-points instead
except Exception:
    pass

EPS_SI = 12.1104
EPS_AIR = 1.0
A_NM = 634.0
LAMBDA_NM = 1550.0
N_GRID = 96
TARGET = A_NM / LAMBDA_NM          # 0.409032...  (spec quotes 0.40903)
MMAX_OFFICIAL = 9

# ----------------------------------------------------------------------------- grid / geometry
def pixel_centres(N=N_GRID):
    """pixel-centre coordinates in units of a, cell = [-0.5, 0.5)"""
    return (np.arange(N) + 0.5) / N - 0.5

def meshgrid_xy(N=N_GRID):
    x = pixel_centres(N)
    X, Y = np.meshgrid(x, x, indexing='xy')   # X[iy, ix], Y[iy, ix]   (MATLAB imagesc orientation: row=y)
    return X, Y

def mask_to_eps(mask):
    return np.where(mask, EPS_SI, EPS_AIR).astype(np.float64)

def eps_to_mask(eps):
    return eps > 0.5 * (EPS_SI + EPS_AIR)

def fill_fraction(mask):
    return float(np.mean(mask))

# ----------------------------------------------------------------------------- symmetry (C4v about cell centre)
def c4v_orbit(arr):
    """all 8 images of a [iy, ix] array under C4v about the cell centre (grid is pixel-centred, so
    mirror i -> N-1-i and transpose are exact symmetries of the grid)"""
    outs = []
    a = arr
    for r in range(4):
        outs.append(a)
        outs.append(a.T)
        a = np.rot90(a)
    return outs

def symmetrize_c4v(arr, mode='mean'):
    imgs = c4v_orbit(arr.astype(np.float64))
    m = np.mean(imgs, axis=0)
    if mode == 'mean':
        return m
    elif mode == 'majority':
        return m >= 0.5
    elif mode == 'or':
        return m > 0
    elif mode == 'and':
        return m >= 1 - 1e-12
    raise ValueError(mode)

def is_c4v(mask):
    return all(np.array_equal(mask, im) for im in c4v_orbit(mask))

# ----------------------------------------------------------------------------- reciprocal lattice / basis
class Basis:
    def __init__(self, Mmax):
        self.Mmax = Mmax
        m = np.arange(-Mmax, Mmax + 1)
        M1, M2 = np.meshgrid(m, m, indexing='ij')
        self.m1 = M1.ravel()
        self.m2 = M2.ravel()
        self.n = self.m1.size                       # (2Mmax+1)^2 plane waves
        # square lattice: a1=(1,0), a2=(0,1)  ->  b1=(2pi,0), b2=(0,2pi)   (a_p . b_q = 2 pi delta_pq)
        self.G = 2 * np.pi * np.stack([self.m1, self.m2], axis=1)      # (n,2)
        # index tables for eta_{G-G'}
        self.d1 = (self.m1[:, None] - self.m1[None, :]) + 2 * Mmax
        self.d2 = (self.m2[:, None] - self.m2[None, :]) + 2 * Mmax

def fourier_table(field, Mmax):
    """
    Fourier coefficients  c(m1,m2) = (1/N^2) sum_{ix,iy} field[iy,ix] exp(-i 2pi (m1 x_ix + m2 y_iy))
    with pixel centres x_ix = (ix+0.5)/N - 0.5, for m1,m2 in [-2Mmax, 2Mmax].
    Returns table T[m1+2Mmax, m2+2Mmax].
    field is indexed [iy, ix] (MATLAB imagesc orientation).
    """
    N = field.shape[0]
    assert field.shape == (N, N)
    f = np.asarray(field, dtype=np.float64).T           # -> [ix, iy]
    F = np.fft.fft2(f) / N**2                           # F[m1, m2] = (1/N^2) sum f[ix,iy] e^{-2pi i (m1 ix + m2 iy)/N}
    idx = np.arange(-2 * Mmax, 2 * Mmax + 1)
    # pixel-centre offset x = (ix + 0.5)/N - 0.5  ->  phase exp(-2pi i m (0.5/N - 0.5)) = exp(-i pi m/N) * exp(+i pi m)
    ph = np.exp(-1j * np.pi * idx / N) * np.exp(1j * np.pi * idx)
    T = F[np.ix_(idx % N, idx % N)] * ph[:, None] * ph[None, :]
    return T

class Structure:
    """Pre-computed Fourier tables of a permittivity map for a given basis size."""
    def __init__(self, eps, Mmax=MMAX_OFFICIAL):
        self.eps = np.asarray(eps, dtype=np.float64)
        self.N = self.eps.shape[0]
        self.basis = Basis(Mmax)
        self.eta_tab = fourier_table(1.0 / self.eps, Mmax)
        self.eps_tab = fourier_table(self.eps, Mmax)
        b = self.basis
        self.eta_mat = self.eta_tab[b.d1, b.d2]        # eta_{G-G'}
        self.eps_mat = self.eps_tab[b.d1, b.d2]        # eps_{G-G'}
        self._eps_inv_mat = None

    @property
    def eps_inv_mat(self):
        if self._eps_inv_mat is None:
            self._eps_inv_mat = np.linalg.inv(self.eps_mat)
        return self._eps_inv_mat

# ----------------------------------------------------------------------------- k-point sets
def kgrid_official(wedge=False):
    """21 x 11 = 231 points over half the Brillouin zone: kx in [-pi, pi] (21), ky in [0, pi] (11), units 1/a.
    wedge=True keeps only the irreducible wedge 0<=ky<=kx<=pi (66 pts): identical band extrema for C4v structures."""
    kx = np.linspace(-np.pi, np.pi, 21)
    ky = np.linspace(0.0, np.pi, 11)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    K = np.stack([KX.ravel(), KY.ravel()], axis=1)
    if wedge:
        sel = (K[:, 1] <= K[:, 0] + 1e-12)
        K = K[sel]
    return K

def kgrid_coarse(nx=11, ny=6, wedge=True):
    kx = np.linspace(-np.pi, np.pi, nx)
    ky = np.linspace(0.0, np.pi, ny)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    K = np.stack([KX.ravel(), KY.ravel()], axis=1)
    if wedge:
        K = K[K[:, 1] <= K[:, 0] + 1e-12]
    return K

def kpath_GXMG(npts=40):
    """Gamma-X-M-Gamma path for band diagrams, returns (K, s, ticks)"""
    G_ = np.array([0, 0.]); X = np.array([np.pi, 0.]); M = np.array([np.pi, np.pi])
    segs = [(G_, X), (X, M), (M, G_)]
    K = []; s = []; ticks = [0.0]; acc = 0.0
    for p, q in segs:
        L = np.linalg.norm(q - p)
        for t in np.linspace(0, 1, npts, endpoint=False):
            K.append(p + t * (q - p)); s.append(acc + t * L)
        acc += L; ticks.append(acc)
    K.append(G_); s.append(acc)
    return np.array(K), np.array(s), ticks

# ----------------------------------------------------------------------------- eigen-solvers
def _matrices(st, k, formulation='official'):
    b = st.basis
    kG = k[None, :] + b.G                    # (n,2)
    mag = np.linalg.norm(kG, axis=1)
    # 'official' : TM with the eps Fourier matrix (inverse of truncated eps matrix == generalised problem
    #              |k+G|^2 c = k0^2 sum eps_{G-G'} c_G'), TE with the Fourier coefficients of 1/eps.
    #              This reproduces the evaluator's Figure-1 band values (digitised) to ~1e-3.
    # 'eta'      : both polarisations with the 1/eps coefficients (lecture PC-II p.13 form for both)
    # 'eps'      : both polarisations with the inverse of the truncated eps matrix
    if formulation == 'official':
        eta_tm = st.eps_inv_mat; eta_te = st.eta_mat
    elif formulation == 'eta':
        eta_tm = eta_te = st.eta_mat
    elif formulation == 'eps':
        eta_tm = eta_te = st.eps_inv_mat
    else:
        raise ValueError(formulation)
    A_tm = (mag[:, None] * mag[None, :]) * eta_tm
    A_te = (kG @ kG.T) * eta_te
    return A_tm, A_te, kG, mag

def solve_k(st, k, nbands=10, formulation='official', vectors=False):
    """returns (w_tm, w_te) arrays of normalised frequencies a/lambda (ascending), optionally eigenvectors"""
    A_tm, A_te, kG, mag = _matrices(st, k, formulation)
    n = A_tm.shape[0]
    nb = min(nbands, n)
    if vectors:
        l_tm, v_tm = sla.eigh(A_tm, subset_by_index=[0, nb - 1], driver='evr')
        l_te, v_te = sla.eigh(A_te, subset_by_index=[0, nb - 1], driver='evr')
    else:
        l_tm = sla.eigh(A_tm, eigvals_only=True, subset_by_index=[0, nb - 1], driver='evr')
        l_te = sla.eigh(A_te, eigvals_only=True, subset_by_index=[0, nb - 1], driver='evr')
    l_tm = np.clip(l_tm, 0, None); l_te = np.clip(l_te, 0, None)
    w_tm = np.sqrt(l_tm) / (2 * np.pi)
    w_te = np.sqrt(l_te) / (2 * np.pi)
    if vectors:
        return w_tm, w_te, v_tm, v_te
    return w_tm, w_te

def compute_bands(eps, K, Mmax=MMAX_OFFICIAL, nbands=10, formulation='official', nproc=None, st=None):
    """bands for all k in K. returns (bands_tm, bands_te) with shape (nk, nbands).
    Parallel over k-points with threads (LAPACK releases the GIL; BLAS threads are pinned to 1)."""
    if st is None:
        st = Structure(eps, Mmax)
    if nproc is None:
        nproc = min(os.cpu_count() or 1, 4)
    if formulation in ('eps', 'official'):
        st.eps_inv_mat  # build once before threading
    f = lambda k: solve_k(st, k, nbands, formulation)
    if nproc <= 1 or len(K) < 4:
        res = [f(k) for k in K]
    else:
        with ThreadPoolExecutor(nproc) as ex:
            res = list(ex.map(f, list(K)))
    btm = np.array([r[0] for r in res]); bte = np.array([r[1] for r in res])
    return btm, bte

# ----------------------------------------------------------------------------- gap analysis / official score
def band_gaps(bands, tol=0.0):
    """list of (w_low, w_high, n) : gap between sorted band n and n+1 (0-based) over the given k set"""
    mx = bands.max(axis=0); mn = bands.min(axis=0)
    out = []
    for n in range(bands.shape[1] - 1):
        if mn[n + 1] - mx[n] > tol:
            out.append((float(mx[n]), float(mn[n + 1]), n))
    return out

def score_from_edges(w_low, w_high, target=TARGET):
    """official metric: Score = 2/w_t * min(w_high - w_t, w_t - w_low)  (0 if target outside)"""
    m_hi = w_high - target; m_lo = target - w_low
    if m_hi <= 0 or m_lo <= 0:
        return 0.0
    return 2.0 / target * min(m_hi, m_lo)

def analyze(btm, bte, target=TARGET):
    """Full analysis dictionary: per-polarisation gaps, complete gaps, the complete gap containing the
    target (if any), official score (fraction; multiply by 100 for %)."""
    gtm = band_gaps(btm); gte = band_gaps(bte)
    complete = []
    for (l1, h1, n1) in gtm:
        for (l2, h2, n2) in gte:
            lo = max(l1, l2); hi = min(h1, h2)
            if hi > lo:
                complete.append(dict(w_low=lo, w_high=hi, n_tm=n1, n_te=n2,
                                     width=hi - lo, mid=(hi + lo) / 2, ratio=(hi - lo) / ((hi + lo) / 2)))
    best = None; score = 0.0
    for c in complete:
        s = score_from_edges(c['w_low'], c['w_high'], target)
        if s > score:
            score = s; best = c
    # gap containing the target, per polarisation (for diagnostics even when not complete)
    def containing(gaps):
        for (l, h, n) in gaps:
            if l < target < h:
                return (l, h, n)
        return None
    return dict(score=score, gap=best, tm_gaps=gtm, te_gaps=gte, complete_gaps=complete,
                tm_gap_at_target=containing(gtm), te_gap_at_target=containing(gte),
                target=target)

def evaluate(eps, Mmax=MMAX_OFFICIAL, K=None, nbands=10, formulation='official', nproc=None, wedge=None):
    """One-call evaluation. Uses the official 21x11 half-BZ grid unless K is given.
    wedge=None -> automatically use the 66-point irreducible wedge iff the structure is exactly C4v."""
    eps = np.asarray(eps, dtype=np.float64)
    if K is None:
        if wedge is None:
            wedge = is_c4v(eps_to_mask(eps))
        K = kgrid_official(wedge=wedge)
    t0 = time.time()
    btm, bte = compute_bands(eps, K, Mmax, nbands, formulation, nproc)
    res = analyze(btm, bte)
    res['runtime_s'] = time.time() - t0
    res['nk'] = len(K); res['Mmax'] = Mmax; res['formulation'] = formulation
    res['fill'] = fill_fraction(eps_to_mask(eps))
    res['bands_tm'] = btm; res['bands_te'] = bte; res['K'] = K
    return res

def summarize(res, label=''):
    g = res['gap']
    s = f"[{label}] score = {100*res['score']:.3f}%  fill = {res['fill']:.4f}  Mmax={res['Mmax']} nk={res['nk']} form={res['formulation']} t={res.get('runtime_s',0):.1f}s"
    if g:
        s += (f"\n    complete gap [{g['w_low']:.5f}, {g['w_high']:.5f}]  width={g['width']:.5f} ratio={100*g['ratio']:.2f}%"
              f"  TM bands {g['n_tm']+1}-{g['n_tm']+2}, TE bands {g['n_te']+1}-{g['n_te']+2}"
              f"  margins: lo={TARGET-g['w_low']:.5f} hi={g['w_high']-TARGET:.5f}")
    tm = res['tm_gap_at_target']; te = res['te_gap_at_target']
    s += f"\n    TM gap at target: {None if tm is None else (round(tm[0],5), round(tm[1],5), tm[2]+1)}"
    s += f"   TE gap at target: {None if te is None else (round(te[0],5), round(te[1],5), te[2]+1)}"
    return s

# ----------------------------------------------------------------------------- geometry generators (all C4v)
def geom_circle(r, N=N_GRID):
    X, Y = meshgrid_xy(N); return (X**2 + Y**2) <= r**2

def geom_square(s, N=N_GRID, rot45=False):
    X, Y = meshgrid_xy(N)
    if rot45:
        return (np.abs(X) + np.abs(Y)) <= s / np.sqrt(2)
    return (np.abs(X) <= s / 2) & (np.abs(Y) <= s / 2)

def geom_veins(w, N=N_GRID, diag=False):
    """dielectric veins of width w through the cell centre along x and y (connect to neighbours);
    diag=True: veins along the diagonals through the corners (the 45-degree grid)"""
    X, Y = meshgrid_xy(N)
    if not diag:
        return (np.abs(X) <= w / 2) | (np.abs(Y) <= w / 2)
    # diagonal veins passing through the cell corners: lines x+y = +-1... in periodic cell |x+y| = 1 mod...
    # equivalently veins through the centre rotated by 45 deg: |x - y|/sqrt2 <= w/2  or |x + y|/sqrt2 <= w/2
    return (np.abs(X - Y) <= w / np.sqrt(2)) | (np.abs(X + Y) <= w / np.sqrt(2))

def geom_corner_circle(r, N=N_GRID):
    """circles centred at the cell corners (periodic images) of radius r"""
    X, Y = meshgrid_xy(N)
    Xc = 0.5 - np.abs(X); Yc = 0.5 - np.abs(Y)
    return (Xc**2 + Yc**2) <= r**2

def geom_edge_veins(w, N=N_GRID):
    """veins along the cell boundary (x=+-0.5 and y=+-0.5), i.e. the dielectric frame of a square air hole"""
    X, Y = meshgrid_xy(N)
    return (np.abs(X) >= 0.5 - w / 2) | (np.abs(Y) >= 0.5 - w / 2)

def geom_ring(r_out, r_in, N=N_GRID):
    X, Y = meshgrid_xy(N); R2 = X**2 + Y**2
    return (R2 <= r_out**2) & (R2 >= r_in**2)

def geom_cross(L, w, N=N_GRID, rot45=False):
    X, Y = meshgrid_xy(N)
    if rot45:
        U = (X + Y) / np.sqrt(2); V = (X - Y) / np.sqrt(2)
    else:
        U, V = X, Y
    return ((np.abs(U) <= L / 2) & (np.abs(V) <= w / 2)) | ((np.abs(V) <= L / 2) & (np.abs(U) <= w / 2))

# ----------------------------------------------------------------------------- IO
def save_mat(path, eps):
    import scipy.io as sio
    eps = np.asarray(eps, dtype=np.float64)
    assert eps.shape == (96, 96)
    assert set(np.unique(eps).tolist()) <= {EPS_AIR, EPS_SI}
    sio.savemat(path, {'epsr': eps}, do_compression=False)

def load_mat(path):
    import scipy.io as sio
    return np.asarray(sio.loadmat(path)['epsr'], dtype=np.float64)

def check_design(eps):
    eps = np.asarray(eps)
    ok = dict(shape=eps.shape == (96, 96),
              binary=set(np.unique(eps).tolist()) <= {EPS_AIR, EPS_SI},
              finite=bool(np.all(np.isfinite(eps))))
    return ok
