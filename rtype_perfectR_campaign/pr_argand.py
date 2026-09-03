"""Finalist complex-port forensics (spec secs 29-30): at theta = 0,
for BOTH principal channels (x, y), reconstruct
    t_total = t_bg + t_ED + t_MD + t_EQ + residual      (into glass)
    r_total = r_bg + r_ED + r_MD + r_EQ + residual      (into air)
with the validated first-order ladder (rt_targand / wf_argand
prefactors; each channel's overall coupling g calibrated by the exact
current-overlap integral, so the ladder is exact in sum and the pieces
carry the multipole ratios). The MQ family is NOT given an analytic
amplitude here (its far-field prefactor was never validated in this
stack); it is reported through its exact POWER fraction (family_weights4)
and left inside the residual arrow - stated, not hidden.

usage: python pr_argand.py <rho.npy> <P> <H> <label>
Writes results/transmission_zero_forensics.csv (+ reflection rows) and
figures/pr_argand_<label>.png
"""
import csv
import math
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import pr_core as pr
import rt_core as rc

sys.path.insert(0, str(pr.HERE.parent / 'ed_eq_causality_campaign'))
import ed_eq_core as core                                   # noqa: E402
import torcwa                                               # noqa: E402

Z0 = 376.730313668
RES = pr.HERE / 'results'
FIG = pr.HERE / 'figures'


def bare(P, H):
    sim = torcwa.rcwa(freq=1.0 / pr.LAM0, order=[9, 9], L=[P, P],
                      dtype=rc.SIM_DTYPE, device=rc.DEVICE)
    sim.add_input_layer(eps=rc.EPS_GLASS)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    sim.add_layer(thickness=float(H), eps=1.0 + 0j)
    sim.solve_global_smatrix()
    R, T = rc.jones(sim, 'backward')
    return complex(R[0, 0]), complex(T[0, 0])


def ladder(rho, P, H, label):
    e = rc.eps_asi()
    rb, tb = bare(P, H)
    sim = torcwa.rcwa(freq=1.0 / pr.LAM0, order=[9, 9], L=[P, P],
                      dtype=rc.SIM_DTYPE, device=rc.DEVICE)
    sim.add_input_layer(eps=rc.EPS_GLASS)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    sim.add_layer(thickness=float(H), eps=rho * (e - 1.0) + 1.0)
    sim.solve_global_smatrix()
    Rj, Tj = rc.jones(sim, 'backward')
    k0 = 2 * math.pi / (pr.LAM0 * 1e-9)
    om = 2 * math.pi * core.C0 / (pr.LAM0 * 1e-9)
    A = (P * 1e-9) ** 2
    rows, panels = [], []
    for pol, amp, j in (('x', [1.0, 0.0], 0), ('y', [0.0, 1.0], 1)):
        sim.source_planewave(amplitude=amp, direction='backward')
        with torch.no_grad():
            x_ax, z_ax, E, _ = core.fields_3d(sim, float(P), float(H), 48, 7)
        n = rho.shape[0]
        idx = (torch.floor(x_ax / P * n).long()) % n
        eps3 = (rho[idx][:, idx] * (e - 1.0) + 1.0)[:, :, None].expand(48, 48, 7)
        mo = core.torch_moments(E, eps3, x_ax, z_ax, pr.LAM0)
        Cp, Cm, CQe, CQm = core.family_weights4(mo)
        tot = float(Cp + Cm + CQe + CQm)
        chi = (eps3 - 1.0).to(torch.complex128)
        Jv = [(-1j * om * core.EPS0) * chi * E[i].to(torch.complex128)
              for i in range(3)]
        xm = ((x_ax - P / 2) * 1e-9).to(torch.float64)
        zm = ((z_ax - H / 2) * 1e-9).to(torch.float64)
        Zg = zm.reshape(1, 1, -1)
        ev = np.array([1.0, 0.0, 0.0]) if pol == 'x' else np.array([0.0, 1.0, 0.0])
        p_v = np.array([complex(mo['px']), complex(mo['py']), complex(mo['pz'])])
        m_v = np.array([complex(mo['mx']), complex(mo['my']), complex(mo['mz'])])
        Q = np.array([[complex(mo['Qxx']), complex(mo['Qxy']), complex(mo['Qxz'])],
                      [complex(mo['Qxy']), complex(mo['Qyy']), complex(mo['Qyz'])],
                      [complex(mo['Qxz']), complex(mo['Qyz']), complex(mo['Qzz'])]])
        pref = -(Z0 / (2 * A))

        def tz(Fv):
            Fv = torch.trapezoid(Fv, xm, dim=0)
            Fv = torch.trapezoid(Fv, xm, dim=0)
            return torch.trapezoid(Fv, zm, dim=0)
        for port, khat, kmed, bg, full in (
                ('t', np.array([0.0, 0.0, -1.0]), k0 * rc.N_GLASS, tb,
                 complex(Tj[j, j])),
                ('r', np.array([0.0, 0.0, 1.0]), k0, rb, complex(Rj[j, j]))):
            Je = (Jv[0] * ev[0] + Jv[1] * ev[1]) * torch.exp(
                -1j * kmed * khat[2] * Zg)
            E_int = complex(pref * tz(Je))
            ED = pref * (-1j * om) * np.dot(p_v, ev)
            MD = pref * (-1j * kmed) * np.dot(np.cross(m_v, khat), ev)
            EQ = pref * (-1j * kmed) * (-1j * om / 6) * np.dot(Q @ khat, ev)
            g = E_int / (full - bg)
            pieces = {'ED': ED / g, 'MD': MD / g, 'EQ': EQ / g}
            lad = bg + sum(pieces.values())
            resid = full - lad
            for nm, v in [('bg', bg)] + list(pieces.items()) + \
                    [('ladder', lad), ('residual', resid), ('full', full)]:
                rows.append({'label': label, 'port': port, 'pol': pol,
                             'term': nm, 're': v.real, 'im': v.imag,
                             'abs': abs(v),
                             'f_ED': float(Cp) / tot, 'f_MD': float(Cm) / tot,
                             'f_EQ': float(CQe) / tot,
                             'f_MQ': float(CQm) / tot})
            panels.append((port, pol, bg, pieces, full))
            print(f'{label} {port}{pol}: |full|^2={abs(full)**2:.3f} '
                  f'ladder^2={abs(lad)**2:.3f} resid={abs(resid):.3f} '
                  f'|ED|={abs(pieces["ED"]):.2f} |MD|={abs(pieces["MD"]):.2f} '
                  f'|EQ|={abs(pieces["EQ"]):.2f} fMQ={float(CQm)/tot:.2f}',
                  flush=True)
    fig, axs = plt.subplots(2, 2, figsize=(9.5, 9))
    for (port, pol, bg, pieces, full) in panels:
        ax = axs[0 if port == 't' else 1][0 if pol == 'x' else 1]
        cum, z = [('bg', bg)], bg
        for nm in ('ED', 'MD', 'EQ'):
            z = z + pieces[nm]
            cum.append((f'+{nm}', z))
        prev = 0j
        for (nm, z2), c in zip(cum, ['#666', 'tab:blue', 'tab:green', 'tab:red']):
            ax.annotate('', xy=(z2.real, z2.imag), xytext=(prev.real, prev.imag),
                        arrowprops=dict(arrowstyle='->', color=c, lw=1.7))
            ax.text(((prev + z2) / 2).real, ((prev + z2) / 2).imag, nm,
                    fontsize=7, color=c)
            prev = z2
        ax.annotate('', xy=(full.real, full.imag), xytext=(prev.real, prev.imag),
                    arrowprops=dict(arrowstyle='->', color='k', lw=1.0,
                                    linestyle=':'))
        ax.plot([full.real], [full.imag], 'k*', ms=10)
        ax.plot([0], [0], 'ko', ms=5, mfc='none')
        lim = 1.15 * max(max(abs(p2.real), abs(p2.imag))
                         for p2 in [c for _, c in cum] + [full, 0.3 + 0.3j])
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect('equal'); ax.grid(alpha=0.2)
        ax.set_title(f'{port}_{pol}: |{port}|^2={abs(full)**2:.3f}', fontsize=9)
    fig.suptitle(f'{label}: t (top) and r (bottom) ladders, theta=0 '
                 '(bg + ED + MD + EQ; dotted = residual incl. MQ)', fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / f'pr_argand_{label}.png', dpi=170)
    plt.close(fig)
    out = RES / 'transmission_zero_forensics.csv'
    new = not out.exists()
    with open(out, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'PR_ARGAND_DONE {label}', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    rho = torch.tensor(np.load(sys.argv[1]).astype(np.float32))
    ladder(rho, float(sys.argv[2]), float(sys.argv[3]), sys.argv[4])
