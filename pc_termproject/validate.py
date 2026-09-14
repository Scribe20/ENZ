"""Validation of the independent PWEM solver (no MATLAB available)."""
import numpy as np, time, sys
from pwem import *

UP = "/root/.claude/uploads/0544bbbf-a4a5-500d-b67c-8cecae91e9bb/"
np.set_printoptions(precision=5, suppress=True, linewidth=150)

print("=== Test 1: homogeneous medium (eps = 12.1104 everywhere) -> omega = |k+G| / (2 pi sqrt(eps)) exactly")
eps = np.full((96, 96), EPS_SI)
st = Structure(eps, 9)
for k in [np.array([0.3, 0.1]), np.array([np.pi, np.pi]), np.array([0., 0.])]:
    for form in ['eta', 'eps']:
        w_tm, w_te = solve_k(st, k, nbands=12, formulation=form)
        kG = k[None, :] + st.basis.G
        exact = np.sort(np.linalg.norm(kG, axis=1) / (2 * np.pi * np.sqrt(EPS_SI)))[:12]
        print(f"  k={k}, {form}: max|TM-exact|={np.max(np.abs(w_tm-exact)):.2e}  max|TE-exact|={np.max(np.abs(w_te-exact)):.2e}")

print("\n=== Test 2: 1D limit (stripes along y, Si width d1 = 0.25 a) vs analytic Bloch/transfer-matrix dispersion")
# structure: Si for |x| <= d1/2
d1 = 0.25; d2 = 1 - d1
X, Y = meshgrid_xy()
mask = np.abs(X) <= d1 / 2
eps1d = mask_to_eps(mask)
print("  actual Si fill:", fill_fraction(mask), " (24/96 = 0.25 expected)")
st = Structure(eps1d, 9)
def analytic_1d(kx, wmax=0.8, n=20000):
    """solve cos(kx a) = cos(k1 d1) cos(k2 d2) - 0.5 (k1/k2 + k2/k1) sin(k1 d1) sin(k2 d2) for omega (normal incidence)"""
    w = np.linspace(1e-4, wmax, n)
    n1, n2 = np.sqrt(EPS_SI), 1.0
    k1 = 2 * np.pi * w * n1; k2 = 2 * np.pi * w * n2
    f = np.cos(k1 * d1) * np.cos(k2 * d2) - 0.5 * (k1 / k2 + k2 / k1) * np.sin(k1 * d1) * np.sin(k2 * d2) - np.cos(kx)
    roots = []
    for i in range(n - 1):
        if f[i] * f[i + 1] < 0:
            roots.append(w[i] - f[i] * (w[i + 1] - w[i]) / (f[i + 1] - f[i]))
    return np.array(roots)
for kx in [0.3 * np.pi, 0.7 * np.pi, np.pi]:
    k = np.array([kx, 0.0])
    w_tm, w_te = solve_k(st, k, nbands=6, formulation='eta')
    w_tm2, w_te2 = solve_k(st, k, nbands=6, formulation='eps')
    ex = analytic_1d(kx)[:6]
    # in the 1D limit with k along x, bands of a 2D solver include modes with G_y != 0 (transverse) -> select those
    # by comparing to the analytic set: report nearest-match errors for the analytic roots
    def nearest_err(w): return np.array([np.min(np.abs(w - e)) for e in ex[:4]])
    print(f"  kx={kx/np.pi:.1f}pi: analytic {ex[:4]}\n     eta: TM err {nearest_err(w_tm)}  TE err {nearest_err(w_te)}\n     eps: TM err {nearest_err(w_tm2)}  TE err {nearest_err(w_te2)}")
# Higher M for the 1D case to show convergence of the residual error
for M in [9, 15, 25]:
    st2 = Structure(eps1d, M)
    k = np.array([0.7 * np.pi, 0.0]); ex = analytic_1d(0.7 * np.pi)[:4]
    w_tm, w_te = solve_k(st2, k, nbands=8, formulation='eta')
    w_tm2, w_te2 = solve_k(st2, k, nbands=8, formulation='eps')
    e = lambda w: np.array([np.min(np.abs(w - x)) for x in ex])
    print(f"  Mmax={M}: eta TM err {e(w_tm)} | eps TM err {e(w_tm2)} | eta TE err {e(w_te)} | eps TE err {e(w_te2)}")

print("\n=== Test 3: alumina rods eps=8.9, r=0.2a (Lecture PC-II p.10 / PC-III p.15): expect TM gap ~0.32-0.44, no TE gap")
eps89 = np.where(geom_circle(0.2), 8.9, 1.0)
t0 = time.time()
r = evaluate(eps89, Mmax=9, nbands=8)
print(f"  TM gaps: {[(round(a,4), round(b,4), n+1) for a,b,n in r['tm_gaps']]}")
print(f"  TE gaps: {[(round(a,4), round(b,4), n+1) for a,b,n in r['te_gaps']]}   ({time.time()-t0:.1f}s, nk={r['nk']})")

print("\n=== Test 4: Term-project rod example (example_2_rod.mat): expect score 0.00%, TM gap top just below 0.409")
eps_rod = load_mat(UP + "15485546-example_2_rod.mat")
print("  check_design:", check_design(eps_rod), " C4v:", is_c4v(eps_to_mask(eps_rod)))
for form in ['eta', 'eps']:
    t0 = time.time()
    r_w = evaluate(eps_rod, Mmax=9, nbands=8, formulation=form, wedge=True)
    r_f = evaluate(eps_rod, Mmax=9, nbands=8, formulation=form, wedge=False)
    print(summarize(r_w, f"rod wedge66 {form}"))
    print(summarize(r_f, f"rod full231 {form}"))
    print("  wedge vs full: max|band extrema diff| TM:",
          np.max(np.abs(r_w['bands_tm'].max(0) - r_f['bands_tm'].max(0))), np.max(np.abs(r_w['bands_tm'].min(0) - r_f['bands_tm'].min(0))),
          " TE:", np.max(np.abs(r_w['bands_te'].max(0) - r_f['bands_te'].max(0))), np.max(np.abs(r_w['bands_te'].min(0) - r_f['bands_te'].min(0))))
    print(f"  TM gaps: {[(round(a,4), round(b,4), n+1) for a,b,n in r_f['tm_gaps']]}")
    print(f"  TE gaps: {[(round(a,4), round(b,4), n+1) for a,b,n in r_f['te_gaps']]}")

print("\n=== Test 5: Term-project random example (example_1_random.mat)")
eps_rnd = load_mat(UP + "615dea0b-example_1_random.mat")
print("  check_design:", check_design(eps_rnd), " C4v:", is_c4v(eps_to_mask(eps_rnd)))
for form in ['eta', 'eps']:
    r = evaluate(eps_rnd, Mmax=9, nbands=8, formulation=form)
    print(summarize(r, f"random {form}"))
    print(f"  TM gaps: {[(round(a,4), round(b,4), n+1) for a,b,n in r['tm_gaps']]}")
    print(f"  TE gaps: {[(round(a,4), round(b,4), n+1) for a,b,n in r['te_gaps']]}")

print("\n=== Test 6: basis convergence for the rod example (TM gap edges, TE band-1 max), both formulations")
for M in [5, 7, 9, 11, 13, 15, 19, 23]:
    t0 = time.time()
    out = []
    for form in ['eta', 'eps']:
        r = evaluate(eps_rod, Mmax=M, nbands=6, formulation=form, wedge=True)
        g = r['tm_gaps'][0] if r['tm_gaps'] else (np.nan, np.nan, -1)
        out.append(f"{form}: TM gap1 [{g[0]:.5f},{g[1]:.5f}] TE b1max {r['bands_te'][:,0].max():.5f} TE b2min {r['bands_te'][:,1].min():.5f}")
    print(f"  Mmax={M:2d} ({(2*M+1)**2:4d} PW, {time.time()-t0:5.1f}s): " + " | ".join(out))

print("\n=== Test 7: timing of one official-setting evaluation (231 k, 361 PW, 10 bands, both pol)")
t0 = time.time(); r = evaluate(eps_rnd, Mmax=9, nbands=10, wedge=False); print(f"  full 231-k evaluation: {time.time()-t0:.2f} s with {min(os.cpu_count(),4)} processes")
t0 = time.time(); r = evaluate(eps_rod, Mmax=9, nbands=10, wedge=True); print(f"  wedge 66-k evaluation: {time.time()-t0:.2f} s")
t0 = time.time(); r = evaluate(eps_rod, Mmax=5, nbands=10, K=kgrid_coarse(11, 6, wedge=True)); print(f"  coarse Mmax=5, {r['nk']}-k evaluation: {time.time()-t0:.2f} s")
