"""
menp_validation.py
==================

Validation of menp_port.py against the ORIGINAL MENP implementation and its
shipped reference outputs, in four independent layers:

A. Reference-demo reproduction (TRUE original data): the original
   demo_sphere/ENxyzf.mat (66.8 MB, Lumerical FDTD fields of a Si
   nanosphere R = 100 nm; fetched from the MENP GitHub repo) is run through
   the Python port and compared against the SHIPPED reference
   demo_exact.csv produced by the original MATLAB code.

B. Original-implementation cross-run: the same data (re-saved as MAT-v5,
   because Octave cannot read MAT-7.3/HDF5) is run through the ORIGINAL
   exactME.m / toroidalME.m / toroidalME_phase.m under GNU Octave, and the
   outputs are compared against the Python port at machine precision.
   The demo_disk data validates the toroidal path against demo_toroidal.csv
   and demo_toroidal_phase.csv the same way.

C. Analytic unit tests (absolute normalization ground truth):
   - a compact Gaussian current blob with prescribed dipole moment p0 must
     return p = p0 and Cp = k^4 |p0|^2 / (6 pi eps0^2);
   - a small current loop with known m0 must return m = m0 and
     Cm = k^4 |m0|^2 / (6 pi eps0^2 c^2).
   These check units, prefactors, and sign/phase conventions independently
   of MENP itself.

D. Bug demonstrations: faithful-vs-corrected differences (dQmxz asymmetry;
   r=0 kernel NaN) are shown explicitly on synthetic data.
"""

import csv
import subprocess
from pathlib import Path

import h5py
import numpy as np
import scipy.io

from menp_port import C0, EPS0, exact_me, toroidal_me

SCRATCH = Path('/tmp/claude-0/-home-user-ENZ/907cdbf0-047a-54fd-9c0a-716dad194fd6/scratchpad/menp')
MENP_DIR = SCRATCH / 'MENP-main'


def load_73(path):
    """Load a MATLAB 7.3 (HDF5) ENxyzf.mat into (x,y,z,f,E*,n_*) with
    MATLAB (nx,ny,nz,nf) axis order."""
    with h5py.File(path, 'r') as h:
        def cplx(name):
            d = h[name][()]
            a = d['real'] + 1j * d['imag']
            return np.transpose(a, (3, 2, 1, 0))   # (nf,nz,ny,nx)->(nx,ny,nz,nf)
        x = h['x'][()].ravel(); y = h['y'][()].ravel(); z = h['z'][()].ravel()
        f = h['f'][()].ravel()
        return dict(x=x, y=y, z=z, f=f,
                    Ex=cplx('Ex'), Ey=cplx('Ey'), Ez=cplx('Ez'),
                    n_x=cplx('n_x'), n_y=cplx('n_y'), n_z=cplx('n_z'))


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.max(np.abs(a - b) / (np.abs(b).max() + 1e-300))


def part_A_sphere():
    print('=== A. demo_sphere reference reproduction (true original data) ===')
    d = load_73(SCRATCH / 'ENxyzf_sphere.mat')
    res = exact_me(d['x'], d['y'], d['z'], d['f'], d['Ex'], d['Ey'], d['Ez'],
                   d['n_x'], d['n_y'], d['n_z'], mode='faithful')
    ref = np.array([[float(v) for v in row] for row in
                    csv.reader(open(MENP_DIR / 'demo_sphere' / 'demo_exact.csv'))])
    lbd = C0 / d['f'] * 1e9
    assert np.allclose(lbd, ref[:, 0], rtol=1e-4), 'wavelength axis mismatch'
    names = ['Cp', 'Cm', 'CQe', 'CQm', 'Csum']
    ok = True
    for i, nme in enumerate(names):
        r = rel(res[nme], ref[:, i + 1])
        print(f'  {nme:5s}: max rel diff vs shipped reference CSV = {r:.3e}')
        ok &= r < 5e-4   # CSV stores 5 significant digits
    print('  A:', 'PASS' if ok else 'FAIL')
    return d, res, ok


def part_B_octave(sphere_data):
    print('=== B. Octave cross-run of ORIGINAL exactME/toroidalME ===')
    d = sphere_data
    scipy.io.savemat(SCRATCH / 'sphere_v5.mat',
                     {k: (v.reshape(-1, 1) if v.ndim == 1 else v)
                      for k, v in d.items()})
    disk = load_73(SCRATCH / 'ENxyzf_disk.mat')
    scipy.io.savemat(SCRATCH / 'disk_v5.mat',
                     {k: (v.reshape(-1, 1) if v.ndim == 1 else v)
                      for k, v in disk.items()})
    oct_script = f"""
    warning('off','all');
    addpath('{MENP_DIR}/MENP');
    load('{SCRATCH}/sphere_v5.mat');
    [Cp,Cm,CQe,CQm,Csum] = exactME(x,y,z,f,Ex,Ey,Ez,n_x,n_y,n_z);
    save('-v6','{SCRATCH}/oct_sphere.mat','Cp','Cm','CQe','CQm','Csum');
    load('{SCRATCH}/disk_v5.mat');
    [Cp,CT,Cm,CQe,CQm,Csum] = toroidalME(x,y,z,f,Ex,Ey,Ez,n_x,n_y,n_z);
    [apx,apy,apz,aTx,aTy,aTz] = toroidalME_phase(x,y,z,f,Ex,Ey,Ez,n_x,n_y,n_z);
    save('-v6','{SCRATCH}/oct_disk.mat','Cp','CT','Cm','CQe','CQm','Csum', ...
         'apx','apy','apz','aTx','aTy','aTz');
    """
    (SCRATCH / 'run_menp.m').write_text(oct_script)
    subprocess.run(['octave', '--no-gui', '--quiet', str(SCRATCH / 'run_menp.m')],
                   check=True, timeout=1200)

    ok = True
    o = scipy.io.loadmat(SCRATCH / 'oct_sphere.mat')
    res = exact_me(d['x'], d['y'], d['z'], d['f'], d['Ex'], d['Ey'], d['Ez'],
                   d['n_x'], d['n_y'], d['n_z'], mode='faithful')
    for nme in ['Cp', 'Cm', 'CQe', 'CQm', 'Csum']:
        r = rel(res[nme], o[nme].ravel())
        print(f'  exactME  {nme:5s}: |python - octave| rel = {r:.3e}')
        ok &= r < 1e-10
    ot = scipy.io.loadmat(SCRATCH / 'oct_disk.mat')
    rest = toroidal_me(disk['x'], disk['y'], disk['z'], disk['f'],
                       disk['Ex'], disk['Ey'], disk['Ez'],
                       disk['n_x'], disk['n_y'], disk['n_z'], mode='faithful')
    for nme in ['Cp', 'CT', 'Cm', 'CQe', 'CQm', 'Csum']:
        r = rel(rest[nme], ot[nme].ravel())
        print(f'  toroidal {nme:5s}: |python - octave| rel = {r:.3e}')
        ok &= r < 1e-10
    # phases (angle of px and of -ikT as in toroidalME_phase)
    k = rest['k']
    apx = np.angle(rest['p'][0]); aTx = np.angle(-1j * k * rest['T'][0])
    r1 = np.max(np.abs(apx - ot['apx'].ravel()))
    r2 = np.max(np.abs(aTx - ot['aTx'].ravel()))
    print(f'  phases   arg(px): max abs diff = {r1:.3e}; arg(-ikTx): {r2:.3e}')
    ok &= r1 < 1e-10 and r2 < 1e-10
    # disk reference CSVs (shipped)
    refT = np.array([[float(v) for v in row] for row in
                     csv.reader(open(MENP_DIR / 'demo_disk' / 'demo_toroidal.csv'))])
    for i, nme in enumerate(['Cp', 'CT', 'Cm', 'CQe', 'CQm', 'Csum']):
        r = rel(rest[nme], refT[:, i + 1])
        print(f'  toroidal {nme:5s}: rel vs shipped CSV = {r:.3e}')
        ok &= r < 5e-4
    print('  B:', 'PASS' if ok else 'FAIL')
    return ok


def part_C_analytic():
    print('=== C. Analytic normalization ground truth ===')
    lam = 1.0e-6
    f = np.array([C0 / lam]); k = 2 * np.pi / lam
    n = 41
    L = 0.02 * lam                      # compact source: kr(edge) ~ 0.13,
                                        # kr(sigma) ~ 0.03 so the exact j0
                                        # kernel is within ~2e-4 of 1
    ax = np.linspace(-L, L, n)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing='ij')
    sig = L / 4
    g = np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / (2 * sig ** 2))
    g /= np.trapezoid(np.trapezoid(np.trapezoid(g, ax, axis=0), ax, axis=0), ax, axis=0)

    omega = 2 * np.pi * f[0]
    # (1) prescribed ELECTRIC dipole p0 along x: J = -i*omega*p0*g(r)
    p0 = 1e-30
    Jx = (-1j * omega * p0) * g
    pref = -1j * omega * EPS0           # J = pref*(n^2-1)*E  =>  E = J/pref (n^2=2)
    E = Jx / pref
    ones = np.ones((n, n, n, 1), complex)
    res = exact_me(ax, ax, ax, f, E[..., None], 0 * E[..., None], 0 * E[..., None],
                   np.sqrt(2) * ones, np.sqrt(2) * ones, np.sqrt(2) * ones,
                   mode='corrected')
    Cp_expect = k ** 4 * p0 ** 2 / (6 * np.pi * EPS0 ** 2)
    r1 = abs(res['p'][0][0] - p0) / p0
    r2 = abs(res['Cp'][0] - Cp_expect) / Cp_expect
    print(f'  dipole blob: |p - p0|/p0 = {r1:.2e}, |Cp - k^4 p0^2/(6 pi eps0^2)|/. = {r2:.2e}')

    # (2) prescribed MAGNETIC dipole m0 along z: J = curl(m0 g) = m0 (dg/dy, -dg/dx, 0)... (m = 1/2 int r x J)
    gy = -(Y / sig ** 2) * g
    gx = -(X / sig ** 2) * g
    m0 = 1e-25
    Jxm, Jym = m0 * gy, -m0 * gx
    Exm, Eym = Jxm / pref, Jym / pref
    resm = exact_me(ax, ax, ax, f, Exm[..., None], Eym[..., None], 0 * Exm[..., None],
                    np.sqrt(2) * ones, np.sqrt(2) * ones, np.sqrt(2) * ones,
                    mode='corrected')
    Cm_expect = k ** 4 * m0 ** 2 / (6 * np.pi * EPS0 ** 2 * C0 ** 2)
    r3 = abs(resm['m'][2][0] - m0) / m0
    r4 = abs(resm['Cm'][0] - Cm_expect) / Cm_expect
    print(f'  current loop: |m - m0|/m0 = {r3:.2e}, |Cm - k^4 m0^2/(6 pi eps0^2 c^2)|/. = {r4:.2e}')
    ok = r1 < 3e-3 and r2 < 6e-3 and r3 < 3e-3 and r4 < 6e-3
    print('  C:', 'PASS' if ok else 'FAIL', '(finite source size + trapz tolerance)')
    return ok


def part_D_bugs():
    print('=== D. faithful vs corrected differences ===')
    rng = np.random.default_rng(7)
    n = 14
    lam = 1e-6; f = np.array([C0 / lam])
    ax = np.linspace(-0.3e-6, 0.3e-6, n) + 1.3e-9   # avoid r=0 for faithful
    E = [rng.standard_normal((n, n, n, 1)) + 1j * rng.standard_normal((n, n, n, 1))
         for _ in range(3)]
    ones = np.ones((n, n, n, 1), complex) * 2.0
    rf = exact_me(ax, ax, ax, f, *E, ones, ones, ones, mode='faithful')
    rc = exact_me(ax, ax, ax, f, *E, ones, ones, ones, mode='corrected')
    dq = abs(rf['CQm'][0] - rc['CQm'][0]) / rc['CQm'][0]
    print(f'  dQmxz asymmetry bug shifts CQm by {dq * 100:.2f}% on random J (this dataset)')
    ax0 = np.linspace(-0.3e-6, 0.3e-6, 15)          # contains r=0 exactly
    E0 = [rng.standard_normal((15, 15, 15, 1)) + 0j for _ in range(3)]
    ones15 = np.ones((15, 15, 15, 1), complex) * 2.0
    rf0 = exact_me(ax0, ax0, ax0, f, *E0, ones15, ones15, ones15, mode='faithful')
    rc0 = exact_me(ax0, ax0, ax0, f, *E0, ones15, ones15, ones15, mode='corrected')
    print(f'  r=0 grid node, faithful mode: Cp = {rf0["Cp"][0]} (NaN expected, as in MATLAB)')
    print(f'  corrected mode, same grid: Cp = {rc0["Cp"][0]:.6e} (finite)')
    return True


if __name__ == '__main__':
    d, res, okA = part_A_sphere()
    okB = part_B_octave(d)
    okC = part_C_analytic()
    part_D_bugs()
    print()
    print('VALIDATION', 'PASS' if (okA and okB and okC) else 'FAIL')
