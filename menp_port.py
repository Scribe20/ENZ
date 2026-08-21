"""
menp_port.py
============

Python port of MENP (Multipole Expansion for NanoPhotonics, T. Hinamoto,
Kobe University; MIT license; OSA Continuum 4, 1640 (2021)), audited from
the supplied MATLAB sources:

    MENP/PhysConst.m  MENP/E2J.m  MENP/trapz4Dto1D.m
    MENP/exactME.m    MENP/approxME.m
    MENP/toroidalME.m MENP/toroidalME_phase.m

AUDIT SUMMARY (source-level, before porting)
--------------------------------------------
* Formulation: induced polarization current J(r) = -i*omega*eps0*(n^2-1)*E(r)
  (E2J.m; SI units, exp(-i*omega*t) convention - matching TORCWA's exp(-jwt)).
  exactME implements the EXACT (beyond long-wavelength) Cartesian multipoles
  of Alaee, Rockstuhl, Fernandez-Corbaton, Opt. Commun. 407, 17 (2018),
  Table 2 (spherical-Bessel kernels j0..j3); approxME implements Table 1;
  toroidalME implements the long-wavelength split p, T (Baryshnikova 2019)
  with C_pT = const*|p + i*k*T|^2 and Csum = CpT+Cm+CQe+CQm.
* Scattering-cross-section constants assume |E0| = 1 (SI, V/m) and a VACUUM
  background: const = k^4/(6*pi*eps0^2), Cm and CQm carry extra 1/c^2, k is
  the vacuum wavenumber, and the material contrast is (n^2 - 1). For our
  metasurface application this means: (i) moments are per-unit-cell moments
  of the induced current computed from the actual (substrate-aware) fields;
  (ii) the "cross sections" are formal isolated-particle vacuum radiation
  weights used to compare relative multipole content - documented in the
  campaign report, not silently reinterpreted.
* Findings (kept verbatim in mode='faithful', fixed in mode='corrected'):
  1. dQmxz asymmetry: all three ME files build the Qm_xz integrand as
     x*(rxJ)_z + x*(rxJ)_z (the same term twice) while Qm_zx uses the
     correctly symmetrized z*(rxJ)_x + x*(rxJ)_z. Corrected mode uses the
     symmetric form for both. (Qm is defined symmetric in the references.)
  2. r -> 0 singularity: exactME evaluates sqrt(pi/(2*k*r))*J_{n+1/2}(k*r)
     which is NaN at r = 0 (0/0), silently poisoning the integral if a grid
     node sits exactly at the multipole origin. Faithful mode reproduces
     this; corrected mode evaluates the regular kernels j0(x), j1(x)/x,
     j2(x)/x^2, j3(x)/x^3 with their finite x->0 limits (1, 1/3, 1/15,
     1/105) via series switchover.
  3. In toroidalME, CT = const*|(i*k)^2|*|T|^2 = const*k^2*|T|^2 and the
     total is Csum = CpT + Cm + CQe + CQm (interference of p and ikT kept
     inside CpT). Ported verbatim.
* trapz4Dto1D integrates x (axis 0), then y (axis 1), then z (axis 2) by
  trapezoidal rule - ported with numpy.trapz over identical axes.

Validation: see menp_validation.py - (a) machine-precision comparison
against the ORIGINAL MATLAB implementation executed under GNU Octave on
synthetic 4D datasets; (b) reproduction of the demo_sphere reference example
with analytic Mie internal fields, compared against the shipped
demo_exact.csv and against analytic Mie partial cross sections.

All returns preserve COMPLEX moments (px..pz, mx..mz, Qe, Qm, T) so that
relative phases (e.g. arg(px) - arg(mz)) can be computed downstream.
"""

import numpy as np
from scipy.special import spherical_jn

C0 = 299792458.0                 # PhysConst.m
EPS0 = 8.854187817e-12           # PhysConst.m


def _grids(x, y, z, f):
    x4, y4, z4, f4 = np.meshgrid(x, y, z, f, indexing='ij')
    return x4, y4, z4, f4


def _trapz4Dto1D(F, x, y, z):
    """MATLAB: trapz(z,trapz(y,trapz(x,F,1),2),3) then squeeze."""
    out = np.trapezoid(F, x, axis=0)
    out = np.trapezoid(out, y, axis=0)
    out = np.trapezoid(out, z, axis=0)
    return np.atleast_1d(out)


def e2j(Ex, Ey, Ez, n_x, n_y, n_z, f4):
    """E2J.m: J = -i*2*pi*f*eps0*(n^2-1)*E."""
    pref = -1j * 2 * np.pi * f4 * EPS0
    return (pref * (n_x ** 2 - 1) * Ex,
            pref * (n_y ** 2 - 1) * Ey,
            pref * (n_z ** 2 - 1) * Ez)


def _bessel_kernels(kr, mode):
    """Return (j0, j1/(kr), j2/(kr)^2, j3/(kr)^3).

    faithful: plain evaluation (NaN at kr=0, like the MATLAB source).
    corrected: series switchover below x=1e-3 for the regular limits."""
    if mode == 'faithful':
        j0 = spherical_jn(0, kr)
        k1 = spherical_jn(1, kr) / kr
        k2 = spherical_jn(2, kr) / kr ** 2
        k3 = spherical_jn(3, kr) / kr ** 3
        return j0, k1, k2, k3
    x = kr
    x2 = x * x
    small = x < 1e-3
    xs = np.where(small, 1.0, x)  # avoid 0-division in the large branch
    j0 = spherical_jn(0, x)
    k1 = np.where(small, 1 / 3 - x2 / 30 + x2 * x2 / 840,
                  spherical_jn(1, xs) / xs)
    k2 = np.where(small, 1 / 15 - x2 / 210 + x2 * x2 / 7560,
                  spherical_jn(2, xs) / xs ** 2)
    k3 = np.where(small, 1 / 105 - x2 / 1890 + x2 * x2 / 83160,
                  spherical_jn(3, xs) / xs ** 3)
    return j0, k1, k2, k3


def exact_me(x, y, z, f, Ex, Ey, Ez, n_x, n_y, n_z, mode='corrected'):
    """Port of exactME.m (Alaee 2018 Table 2, exact kernels).

    Returns dict with complex moments p, m (vectors), Qe, Qm (3x3), and
    cross sections Cp, Cm, CQe, CQm, Csum (arrays over f).
    mode='faithful' reproduces the MATLAB source bit-for-bit (including the
    dQmxz asymmetry and the r=0 NaN); mode='corrected' fixes both."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    z = np.asarray(z, float); f = np.atleast_1d(np.asarray(f, float))
    x4, y4, z4, f4 = _grids(x, y, z, f)
    k4 = 2 * np.pi * f4 / C0
    omega = 2 * np.pi * f
    k = omega / C0
    const = k ** 4 / (6 * np.pi * EPS0 ** 2 * 1.0)

    Jx, Jy, Jz = e2j(Ex, Ey, Ez, n_x, n_y, n_z, f4)

    rJ = x4 * Jx + y4 * Jy + z4 * Jz
    rr = x4 * x4 + y4 * y4 + z4 * z4
    r = np.sqrt(rr)
    rxJx = y4 * Jz - z4 * Jy
    rxJy = z4 * Jx - x4 * Jz
    rxJz = x4 * Jy - y4 * Jx

    kr = k4 * r
    with np.errstate(divide='ignore', invalid='ignore'):
        j0, K1, K2, K3 = _bessel_kernels(kr, mode)

    T = lambda F: _trapz4Dto1D(F, x, y, z)

    # electric dipole (kernel j0 + k^2/2 * j2/(kr)^2 correction)
    px = -1 / (1j * omega) * (T(Jx * j0) + k ** 2 / 2 * T((3 * rJ * x4 - rr * Jx) * K2))
    py = -1 / (1j * omega) * (T(Jy * j0) + k ** 2 / 2 * T((3 * rJ * y4 - rr * Jy) * K2))
    pz = -1 / (1j * omega) * (T(Jz * j0) + k ** 2 / 2 * T((3 * rJ * z4 - rr * Jz) * K2))
    Cp = const * (np.abs(px) ** 2 + np.abs(py) ** 2 + np.abs(pz) ** 2)

    # magnetic dipole (kernel j1/(kr))
    mx = 1.5 * T(rxJx * K1)
    my = 1.5 * T(rxJy * K1)
    mz = 1.5 * T(rxJz * K1)
    Cm = const * (np.abs(mx) ** 2 + np.abs(my) ** 2 + np.abs(mz) ** 2) / C0 ** 2

    # electric quadrupole
    def Qe(a4, b4, Ja, Jb):
        d1 = (3 * (b4 * Ja + a4 * Jb) - (2 * rJ if a4 is b4 else 0)) * K1
        d2 = (5 * a4 * b4 * rJ - rr * (a4 * Jb + b4 * Ja) - (rr * rJ if a4 is b4 else 0)) * K3
        return -3 / (1j * omega) * (T(d1) + 2 * k ** 2 * T(d2))
    Qexx = Qe(x4, x4, Jx, Jx); Qeyy = Qe(y4, y4, Jy, Jy); Qezz = Qe(z4, z4, Jz, Jz)
    Qexy = Qe(x4, y4, Jx, Jy); Qexz = Qe(x4, z4, Jx, Jz); Qeyz = Qe(y4, z4, Jy, Jz)
    # (matrix symmetric: xy=yx etc., exactly as in the MATLAB source)
    norm2_Qe = (np.abs(Qexx) ** 2 + np.abs(Qeyy) ** 2 + np.abs(Qezz) ** 2
                + 2 * np.abs(Qexy) ** 2 + 2 * np.abs(Qexz) ** 2 + 2 * np.abs(Qeyz) ** 2)
    CQe = const / 120 * k ** 2 * norm2_Qe

    # magnetic quadrupole (kernel j2/(kr)^2)
    Qmxx = 15 * T(2 * x4 * rxJx * K2)
    Qmyy = 15 * T(2 * y4 * rxJy * K2)
    Qmzz = 15 * T(2 * z4 * rxJz * K2)
    Qmxy = 15 * T((x4 * rxJy + y4 * rxJx) * K2)
    Qmyx = 15 * T((y4 * rxJx + x4 * rxJy) * K2)   # == Qmxy
    if mode == 'faithful':
        Qmxz = 15 * T((x4 * rxJz + x4 * rxJz) * K2)   # MATLAB bug kept
    else:
        Qmxz = 15 * T((x4 * rxJz + z4 * rxJx) * K2)   # symmetrized
    Qmzx = 15 * T((z4 * rxJx + x4 * rxJz) * K2)
    Qmyz = 15 * T((y4 * rxJz + z4 * rxJy) * K2)
    Qmzy = 15 * T((z4 * rxJy + y4 * rxJz) * K2)
    norm2_Qm = (np.abs(Qmxx) ** 2 + np.abs(Qmxy) ** 2 + np.abs(Qmxz) ** 2
                + np.abs(Qmyy) ** 2 + np.abs(Qmyx) ** 2 + np.abs(Qmyz) ** 2
                + np.abs(Qmzz) ** 2 + np.abs(Qmzx) ** 2 + np.abs(Qmzy) ** 2)
    CQm = const / 120 * (k / C0) ** 2 * norm2_Qm

    return {
        'p': np.stack([px, py, pz]), 'm': np.stack([mx, my, mz]),
        'Qe': {'xx': Qexx, 'yy': Qeyy, 'zz': Qezz,
               'xy': Qexy, 'xz': Qexz, 'yz': Qeyz},
        'Qm': {'xx': Qmxx, 'yy': Qmyy, 'zz': Qmzz, 'xy': Qmxy, 'yx': Qmyx,
               'xz': Qmxz, 'zx': Qmzx, 'yz': Qmyz, 'zy': Qmzy},
        'Cp': Cp, 'Cm': Cm, 'CQe': CQe, 'CQm': CQm,
        'Csum': Cp + Cm + CQe + CQm,
        'const': const, 'k': k,
    }


def toroidal_me(x, y, z, f, Ex, Ey, Ez, n_x, n_y, n_z, mode='corrected'):
    """Port of toroidalME.m (long-wavelength split p / T; Csum keeps the
    p+ikT interference inside CpT). Complex p, T, m preserved."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    z = np.asarray(z, float); f = np.atleast_1d(np.asarray(f, float))
    x4, y4, z4, f4 = _grids(x, y, z, f)
    omega = 2 * np.pi * f
    k = omega / C0
    const = k ** 4 / (6 * np.pi * EPS0 ** 2 * 1.0)

    Jx, Jy, Jz = e2j(Ex, Ey, Ez, n_x, n_y, n_z, f4)
    rJ = x4 * Jx + y4 * Jy + z4 * Jz
    rr = x4 * x4 + y4 * y4 + z4 * z4
    rxJx = y4 * Jz - z4 * Jy
    rxJy = z4 * Jx - x4 * Jz
    rxJz = x4 * Jy - y4 * Jx
    T = lambda F: _trapz4Dto1D(F, x, y, z)

    px = -1 / (1j * omega) * T(Jx)
    py = -1 / (1j * omega) * T(Jy)
    pz = -1 / (1j * omega) * T(Jz)
    Cp = const * (np.abs(px) ** 2 + np.abs(py) ** 2 + np.abs(pz) ** 2)

    Tx = 1 / (10 * C0) * T(rJ * x4 - 2 * rr * Jx)
    Ty = 1 / (10 * C0) * T(rJ * y4 - 2 * rr * Jy)
    Tz = 1 / (10 * C0) * T(rJ * z4 - 2 * rr * Jz)
    CT = const * np.abs((1j * k) ** 2) * (np.abs(Tx) ** 2 + np.abs(Ty) ** 2 + np.abs(Tz) ** 2)

    pTx, pTy, pTz = px + 1j * k * Tx, py + 1j * k * Ty, pz + 1j * k * Tz
    CpT = const * (np.abs(pTx) ** 2 + np.abs(pTy) ** 2 + np.abs(pTz) ** 2)

    mx = 0.5 * T(rxJx)
    my = 0.5 * T(rxJy)
    mz = 0.5 * T(rxJz)
    Cm = const * (np.abs(mx) ** 2 + np.abs(my) ** 2 + np.abs(mz) ** 2) / C0 ** 2

    # Qe with k^2/14 correction (identical structure to approxME)
    def Qe(a4, b4, Ja, Jb, diag):
        d1 = 3 * (b4 * Ja + a4 * Jb) - (2 * rJ if diag else 0)
        d2 = 4 * a4 * b4 * rJ - 5 * rr * (a4 * Jb + b4 * Ja) + (2 * rr * rJ if diag else 0)
        return -1 / (1j * omega) * (T(d1) + k ** 2 / 14 * T(d2))
    Qexx = Qe(x4, x4, Jx, Jx, True); Qeyy = Qe(y4, y4, Jy, Jy, True)
    Qezz = Qe(z4, z4, Jz, Jz, True)
    Qexy = Qe(x4, y4, Jx, Jy, False); Qexz = Qe(x4, z4, Jx, Jz, False)
    Qeyz = Qe(y4, z4, Jy, Jz, False)
    norm2_Qe = (np.abs(Qexx) ** 2 + np.abs(Qeyy) ** 2 + np.abs(Qezz) ** 2
                + 2 * np.abs(Qexy) ** 2 + 2 * np.abs(Qexz) ** 2 + 2 * np.abs(Qeyz) ** 2)
    CQe = const / 120 * k ** 2 * norm2_Qe

    if mode == 'faithful':
        Qmxz = T(x4 * rxJz + x4 * rxJz)
    else:
        Qmxz = T(x4 * rxJz + z4 * rxJx)
    Qmxx = T(2 * x4 * rxJx); Qmyy = T(2 * y4 * rxJy); Qmzz = T(2 * z4 * rxJz)
    Qmxy = T(x4 * rxJy + y4 * rxJx); Qmyx = T(y4 * rxJx + x4 * rxJy)
    Qmzx = T(z4 * rxJx + x4 * rxJz)
    Qmyz = T(y4 * rxJz + z4 * rxJy); Qmzy = T(z4 * rxJy + y4 * rxJz)
    norm2_Qm = (np.abs(Qmxx) ** 2 + np.abs(Qmxy) ** 2 + np.abs(Qmxz) ** 2
                + np.abs(Qmyy) ** 2 + np.abs(Qmyx) ** 2 + np.abs(Qmyz) ** 2
                + np.abs(Qmzz) ** 2 + np.abs(Qmzx) ** 2 + np.abs(Qmzy) ** 2)
    CQm = const / 120 * (k / C0) ** 2 * norm2_Qm

    return {'p': np.stack([px, py, pz]), 'T': np.stack([Tx, Ty, Tz]),
            'pT': np.stack([pTx, pTy, pTz]), 'm': np.stack([mx, my, mz]),
            'Cp': Cp, 'CT': CT, 'CpT': CpT, 'Cm': Cm, 'CQe': CQe, 'CQm': CQm,
            'Csum': CpT + Cm + CQe + CQm, 'k': k, 'const': const}
