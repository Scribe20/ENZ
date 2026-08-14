"""TM (p-polarized) modal physics for the air / ITO / glass slab.

Everything here follows the conventions declared in config.py:

    time dependence      exp(-i*omega*t)
    in-plane dependence  exp(+i*K*x),  K = K' + i*K''
    kz_j = sqrt(eps_j*k0^2 - K^2)

Derivation of the dispersion relation (source-free TM mode)
-----------------------------------------------------------
Layer indexing (Vassant Fig. 1): medium 1 = air (z > d), medium 2 = ITO
(0 < z < d), medium 3 = glass (z < 0).  With H = (0, Hy, 0),

    Hy(z) =  A1 * exp(+i*kz1*(z-d))              z > d      (up-going / decaying)
             A  * exp(+i*kz2*z) + B*exp(-i*kz2*z)  0 < z < d
             A3 * exp(-i*kz3*z)                  z < 0      (down-going / decaying)

From curl H = -i*omega*eps0*eps*E (exp(-i*omega*t) convention):

    Ex = (1/(i*omega*eps0*eps)) * dHy/dz    ->  for exp(+i*kz*z):  Ex = (kz/(omega*eps0*eps)) * Hy
    Ez = -(K /(omega*eps0*eps)) * Hy

Continuity of Hy and Ex at z=0 and z=d gives four equations; eliminating the
amplitudes yields the pole condition

    D(K, omega) = 1 + r12 * r23 * exp(2i*kz2*d) = 0                     (Fresnel form)

with TM Fresnel reflection coefficients (in the p_j = kz_j/eps_j notation)

    r_ij = (p_i - p_j)/(p_i + p_j) = (eps_j*kz_i - eps_i*kz_j)/(eps_j*kz_i + eps_i*kz_j).

Note 1 + r12*r23*e^{2i kz2 d} = 1 - r21*r23*e^{2i kz2 d} because r21 = -r12;
this is exactly the pole of the three-layer reflection coefficient

    r_123 = (r12 + r23 e^{2i kz2 d}) / (1 + r12 r23 e^{2i kz2 d}).

The same condition, rearranged, is Eq. (1) of Vassant et al. (2012) (also
Eq. (1) of the Karimi Supporting Information):

    1 + (eps1*kz3)/(eps3*kz1)
        = i * tan(kz2*d) * ( (eps2*kz3)/(eps3*kz2) + (eps1*kz2)/(eps2*kz1) )

Both forms are implemented and cross-checked numerically in validate_mode.py.
The equation is even in kz2, so the branch of kz2 inside the (lossy) film is
irrelevant; only the cladding sheets (kz1, kz3) matter.
"""

import numpy as np

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# kz branch selection
# ---------------------------------------------------------------------------
def kz_branch(eps, k0, K, sheet=+1):
    """kz = sqrt(eps*k0^2 - K^2) with explicit sheet control.

    sheet = +1  : "proper" determination, Re(kz) + Im(kz) > 0 (Vassant's
                  prescription).  For an evanescent cladding wave this gives
                  Im(kz) > 0, i.e. exp(+i*kz1*(z-d)) decays for z -> +inf and
                  exp(-i*kz3*z) decays for z -> -inf: a bound mode.
    sheet = -1  : opposite (improper / growing) determination, used to search
                  for leaky (Berreman-type) solutions honestly labeled as such.
    """
    kz = np.sqrt(eps * k0**2 - K**2 + 0j)
    flip = (kz.real + kz.imag) < 0
    kz = np.where(flip, -kz, kz)
    return sheet * kz


# ---------------------------------------------------------------------------
# Dispersion functions
# ---------------------------------------------------------------------------
def _kzs(K, k0, eps1, eps2, eps3, sheet1=+1, sheet3=+1):
    kz1 = kz_branch(eps1, k0, K, sheet1)
    kz2 = kz_branch(eps2, k0, K, +1)          # film sheet irrelevant (even in kz2)
    kz3 = kz_branch(eps3, k0, K, sheet3)
    return kz1, kz2, kz3


def r_tm(eps_i, kz_i, eps_j, kz_j):
    """TM Fresnel reflection coefficient for incidence from i onto j."""
    return (eps_j * kz_i - eps_i * kz_j) / (eps_j * kz_i + eps_i * kz_j)


def D_fresnel(K, k0, eps2, d, eps1=1.0, eps3=1.0, sheet1=+1, sheet3=+1):
    """Pole denominator D = 1 + r12*r23*exp(2i*kz2*d).  Mode <=> D = 0."""
    kz1, kz2, kz3 = _kzs(K, k0, eps1, eps2, eps3, sheet1, sheet3)
    r12 = r_tm(eps1, kz1, eps2, kz2)
    r23 = r_tm(eps2, kz2, eps3, kz3)
    return 1.0 + r12 * r23 * np.exp(2j * kz2 * d)


def D_vassant(K, k0, eps2, d, eps1=1.0, eps3=1.0, sheet1=+1, sheet3=+1):
    """Vassant et al. Eq. (1), written as LHS - RHS (zero at a mode)."""
    kz1, kz2, kz3 = _kzs(K, k0, eps1, eps2, eps3, sheet1, sheet3)
    lhs = 1.0 + (eps1 * kz3) / (eps3 * kz1)
    rhs = 1j * np.tan(kz2 * d) * ((eps2 * kz3) / (eps3 * kz2)
                                  + (eps1 * kz2) / (eps2 * kz1))
    return lhs - rhs


# ---------------------------------------------------------------------------
# Driven response (independent check of pole location)
# ---------------------------------------------------------------------------
def r123_tm(K, k0, eps2, d, eps1=1.0, eps3=1.0, sheet1=+1, sheet3=+1):
    """Three-layer TM reflection coefficient for incidence from medium 1."""
    kz1, kz2, kz3 = _kzs(K, k0, eps1, eps2, eps3, sheet1, sheet3)
    r12 = r_tm(eps1, kz1, eps2, kz2)
    r23 = r_tm(eps2, kz2, eps3, kz3)
    ph = np.exp(2j * kz2 * d)
    return (r12 + r23 * ph) / (1.0 + r12 * r23 * ph)


# ---------------------------------------------------------------------------
# Field reconstruction
# ---------------------------------------------------------------------------
class ModeField:
    """Analytic TM mode field of the three-layer slab at a pole (K, omega).

    Amplitudes are fixed by the boundary conditions at z=0 (ITO/glass) with
    A3 = Hy(0) = 1 as the raw gauge; the z=d interface conditions are then
    satisfied automatically iff D(K)=0, which is what validate_mode.py checks.

    Physical fields (SI-like, with the constant 1/(omega*eps0) absorbed into
    the arbitrary mode amplitude):
        Ex =  (kz/eps) * Hy_component_wise
        Ez = -(K /eps) * Hy
    """

    def __init__(self, K, k0, eps2, d, eps1=1.0, eps3=1.0, sheet1=+1, sheet3=+1):
        # k0 may be complex: evaluating the mode profile at a complex-omega
        # pole (real-K formulation) uses k0 = omega_tilde / c.
        self.K, self.k0, self.d = complex(K), complex(k0), float(d)
        self.eps = (complex(eps1), complex(eps2), complex(eps3))
        self.sheet1, self.sheet3 = sheet1, sheet3
        kz1, kz2, kz3 = _kzs(self.K, k0, eps1, eps2, eps3, sheet1, sheet3)
        self.kz = (complex(kz1), complex(kz2), complex(kz3))

        # amplitudes from the z=0 interface, gauge Hy(0)=1:
        #   A + B = A3 = 1
        #   (kz2/eps2)*(A - B) = -(kz3/eps3)*(A + B)
        p2 = kz2 / eps2
        p3 = kz3 / eps3
        self.A3 = 1.0 + 0j
        self.A = 0.5 * (1.0 - p3 / p2)
        self.B = 0.5 * (1.0 + p3 / p2)
        # air amplitude from Hy continuity at z=d
        self.A1 = self.A * np.exp(1j * kz2 * self.d) + self.B * np.exp(-1j * kz2 * self.d)

    # -- region masks ---------------------------------------------------
    def _region(self, z):
        z = np.asarray(z, float)
        return (z > self.d).astype(int) * 1 + ((z >= 0) & (z <= self.d)).astype(int) * 2 \
            + (z < 0).astype(int) * 3

    def Hy(self, z):
        z = np.asarray(z, float)
        kz1, kz2, kz3 = self.kz
        out = np.empty(z.shape, complex)
        m1 = z > self.d
        m2 = (z >= 0) & (z <= self.d)
        m3 = z < 0
        out[m1] = self.A1 * np.exp(1j * kz1 * (z[m1] - self.d))
        out[m2] = self.A * np.exp(1j * kz2 * z[m2]) + self.B * np.exp(-1j * kz2 * z[m2])
        out[m3] = self.A3 * np.exp(-1j * kz3 * z[m3])
        return out

    def eps_of_z(self, z):
        z = np.asarray(z, float)
        eps1, eps2, eps3 = self.eps
        out = np.empty(z.shape, complex)
        out[z > self.d] = eps1
        out[(z >= 0) & (z <= self.d)] = eps2
        out[z < 0] = eps3
        return out

    def Ex(self, z):
        """Ex = (1/(i*eps)) dHy/dz, with 1/(omega*eps0) absorbed."""
        z = np.asarray(z, float)
        kz1, kz2, kz3 = self.kz
        eps1, eps2, eps3 = self.eps
        out = np.empty(z.shape, complex)
        m1 = z > self.d
        m2 = (z >= 0) & (z <= self.d)
        m3 = z < 0
        out[m1] = (kz1 / eps1) * self.A1 * np.exp(1j * kz1 * (z[m1] - self.d))
        out[m2] = (kz2 / eps2) * (self.A * np.exp(1j * kz2 * z[m2])
                                  - self.B * np.exp(-1j * kz2 * z[m2]))
        out[m3] = -(kz3 / eps3) * self.A3 * np.exp(-1j * kz3 * z[m3])
        return out

    def Ez(self, z):
        return -(self.K / self.eps_of_z(z)) * self.Hy(z)

    def Dz(self, z):
        """Normal displacement (eps0 absorbed): Dz = eps*Ez = -K*Hy. Continuous."""
        return self.eps_of_z(z) * self.Ez(z)

    # -- diagnostics ----------------------------------------------------
    def interface_residuals(self, h=1e-9):
        """Relative mismatch of Hy, Ex, Dz across both interfaces.

        Evaluated with the analytic layer expressions a distance h (nm) on
        either side of each interface; h is negligible vs. any length scale.
        """
        res = {}
        for name, z0 in (("air/ITO", self.d), ("ITO/glass", 0.0)):
            za, zb = z0 + h, z0 - h
            for fname, f in (("Hy", self.Hy), ("Ex", self.Ex), ("Dz", self.Dz)):
                fa, fb = complex(f(np.array([za]))[0]), complex(f(np.array([zb]))[0])
                denom = max(abs(fa), abs(fb))
                res[f"{name}:{fname}"] = abs(fa - fb) / denom if denom else 0.0
        return res

    def decay_lengths_nm(self):
        """1/e amplitude decay lengths into air and glass (None if growing)."""
        kz1, _, kz3 = self.kz
        La = 1.0 / kz1.imag if kz1.imag > 0 else None
        Lg = 1.0 / kz3.imag if kz3.imag > 0 else None
        return La, Lg

    def ez_localization(self, n_clad_lengths=60.0, n_pts=200001):
        """Fraction of integral |Ez|^2 dz residing in the ITO film.

        Cladding tails are integrated analytically (pure exponentials), the
        film numerically.  Only meaningful when both claddings decay.
        """
        kz1, kz2, kz3 = self.kz
        if kz1.imag <= 0 or kz3.imag <= 0:
            return None
        zf = np.linspace(0.0, self.d, n_pts)
        I_ito = np.trapezoid(np.abs(self.Ez(zf))**2, zf)
        # air: |Ez(z)|^2 = |K/eps1 * A1|^2 exp(-2 Im kz1 (z-d)), z in (d, inf)
        I_air = np.abs(self.K / self.eps[0] * self.A1)**2 / (2 * kz1.imag)
        I_glass = np.abs(self.K / self.eps[2] * self.A3)**2 / (2 * kz3.imag)
        return float(I_ito / (I_ito + I_air + I_glass))
