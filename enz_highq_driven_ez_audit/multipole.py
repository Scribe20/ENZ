"""Cartesian multipole moments of the a-Si polarization current (diagnostic
only).  Complex moments p, m, Q^e, Q^m (origin = cell centre, LH units,
c = 1, J = -i w (eps-1) E in the a-Si layer) and the specular +-z channel
amplitudes of each multipole for an array at normal incidence in the
free-space, single-cell approximation (Evlyukhin/Terekhov-type expression):
    E_x(+z)  ~  p_x + m_y + (ik/6) Q^e_xz + (ik/2) Q^m_yz
    E_x(-z)  ~  p_x - m_y - (ik/6) Q^e_xz + (ik/2) Q^m_yz
The overall prefactor is dropped; only the relative complex amplitudes and
their coherent sums are meaningful.  Sign conventions of the quadrupole
terms differ between references; this is the Kerker-type convention in
which p_x = m_y cancels backscattering.  The substrate/ITO environment is
NOT included, so this is an indicator, not a proof, of channel cancellation.
Direct S-matrix evidence (r, t amplitudes and residues) must be used for
actual open-channel statements.
"""
import numpy as np
import torch
from scipy import ndimage

import common as cm
import forward_multi as fm


def moments(rho, P, h, lam, s=1.0, n=64, nz=8, order=cm.ORDER):
    sim = fm.build_sim(rho, P, h, lam=lam, order=order, ito_loss_scale=s)
    x, y = fm.cell_axes(P, n)
    geo = ndimage.zoom(rho.numpy(), n / rho.shape[0], order=0)
    k = 2 * np.pi / lam
    chi = (fm.eps_asi(lam) - 1.0) * geo
    xs = x.numpy() - P / 2
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    dV = (P / n) ** 2 * (h / nz)
    p = np.zeros(3, complex); m = np.zeros(3, complex)
    Qe = np.zeros((3, 3), complex); Qm = np.zeros((3, 3), complex)
    with torch.no_grad():
        for zp in (np.arange(nz) + 0.5) * h / nz:
            E, _ = sim.field_xy(0, x, y, float(zp))
            E = [c.numpy() for c in E]
            Z = np.full_like(X, zp - h / 2)
            J = [-1j * k * chi * c for c in E]
            r = [X, Y, Z]
            rxJ = [r[(i + 1) % 3] * J[(i + 2) % 3] - r[(i + 2) % 3] * J[(i + 1) % 3] for i in range(3)]
            rJ = sum(r[i] * J[i] for i in range(3))
            for a in range(3):
                p[a] += 1j / k * np.sum(J[a]) * dV
                m[a] += 0.5 * np.sum(rxJ[a]) * dV
                for b in range(3):
                    Qe[a, b] += 1j / k * np.sum(3 * (r[a] * J[b] + r[b] * J[a]) - 2 * (a == b) * rJ) * dV
                    Qm[a, b] += np.sum(r[a] * rxJ[b] + r[b] * rxJ[a]) * dV / 3
    terms_f = dict(ED=p[0], MD=m[1], EQ=1j * k / 6 * Qe[0, 2], MQ=1j * k / 2 * Qm[1, 2])
    terms_b = dict(ED=p[0], MD=-m[1], EQ=-1j * k / 6 * Qe[0, 2], MQ=1j * k / 2 * Qm[1, 2])
    powers = dict(ED=k ** 4 * np.sum(np.abs(p) ** 2) / (12 * np.pi),
                  MD=k ** 4 * np.sum(np.abs(m) ** 2) / (12 * np.pi),
                  EQ=k ** 6 * np.sum(np.abs(Qe) ** 2) / (1440 * np.pi),
                  MQ=k ** 6 * np.sum(np.abs(Qm) ** 2) / (160 * np.pi))
    tot = sum(powers.values())

    def chan(terms):
        tot_amp = sum(terms.values())
        mx = max(abs(v) for v in terms.values())
        return dict(amplitudes={k_: [v.real, v.imag] for k_, v in terms.items()},
                    magnitudes={k_: abs(v) for k_, v in terms.items()},
                    phases_deg={k_: float(np.degrees(np.angle(v))) for k_, v in terms.items()},
                    coherent_sum_abs=abs(tot_amp), max_term_abs=mx,
                    cancellation_ratio=abs(tot_amp) / mx if mx > 0 else None)
    return dict(power_fractions={k_: v / tot for k_, v in powers.items()},
                forward=chan(terms_f), backward=chan(terms_b),
                p=[[c.real, c.imag] for c in p], m=[[c.real, c.imag] for c in m])
