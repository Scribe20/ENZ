"""Section 24: complex-t cancellation analysis for the best finalists.

Air-side illumination; transmission into glass. t_total = t_bg + ladder
(ED + MD + EQ terms of the z-parity pair for each polarization) with the
per-row exact port coupling g = (exact substrate-side channel integral) /
(t_full - t_bg); removal combos; Argand figure.
Pieces per pol: x-pol -> (px | my, Qe_xz); y-pol -> (py | mx, Qe_yz).
Ladder is 1st-order (2nd-order integral not stored here); the residual
arrow to full TORCWA t shows the truncation honestly.
"""
import csv
import json
import math
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import rt_core as rc
import ed_eq_core as core
from rt_qualify import load_tag

Z0 = 376.730313668
F = rc.HERE / 'figures'
R = rc.HERE / 'results'


def t_bare(P, H, order=(9, 9)):
    import torcwa
    sim = torcwa.rcwa(freq=1.0 / rc.LAM0, order=list(order), L=[P, P],
                      dtype=rc.SIM_DTYPE, device=rc.DEVICE)
    sim.add_input_layer(eps=rc.EPS_GLASS)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    sim.add_layer(thickness=float(H), eps=1.0 + 0j)
    sim.solve_global_smatrix()
    sim.source_planewave(amplitude=[1.0, 0.0], direction='backward')
    return complex(sim.S_parameters(orders=[0, 0], direction='backward',
                                    port='transmission', polarization='xx',
                                    ref_order=[0, 0]))


def analyze(tag):
    rec, rho = load_tag(tag)
    P, H = rec['P'] if 'P' in rec else rec['rec']['P'], \
        rec['H'] if 'H' in rec else rec['rec']['H']
    tb = t_bare(P, H)
    out_rows = []
    figs = []
    for pol in ('x', 'y'):
        import torcwa
        e = rc.eps_asi()
        sim = torcwa.rcwa(freq=1.0 / rc.LAM0, order=[9, 9], L=[P, P],
                          dtype=rc.SIM_DTYPE, device=rc.DEVICE)
        sim.add_input_layer(eps=rc.EPS_GLASS)
        sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
        sim.add_layer(thickness=float(H), eps=rho * (e - 1.0) + 1.0)
        sim.solve_global_smatrix()
        amp = [1.0, 0.0] if pol == 'x' else [0.0, 1.0]
        sim.source_planewave(amplitude=amp, direction='backward')
        polstr = ('xx', 'yx') if pol == 'x' else ('yy', 'xy')
        t_full = complex(sim.S_parameters(orders=[0, 0],
                                          direction='backward',
                                          port='transmission',
                                          polarization=polstr[0],
                                          ref_order=[0, 0]))
        with torch.no_grad():
            x_ax, z_ax, E, _ = core.fields_3d(sim, float(P), float(H),
                                              48, 7)
        n = rho.shape[0]
        idx = (torch.floor(x_ax / P * n).long()) % n
        eps3 = (rho[idx][:, idx] * (e - 1.0) + 1.0)[:, :, None] \
            .expand(48, 48, 7)
        mo = core.torch_moments(E, eps3, x_ax, z_ax, rc.LAM0)
        k = 2 * math.pi / (rc.LAM0 * 1e-9)
        om = 2 * math.pi * core.C0 / (rc.LAM0 * 1e-9)
        A = (P * 1e-9) ** 2
        # exact channel integral toward the GLASS (downward, e^{+ikz} in
        # the local frame used by the ENZ campaign machinery)
        chi = (eps3 - 1.0).to(torch.complex128)
        Jc = (-1j * om * core.EPS0) * chi \
            * E[0 if pol == 'x' else 1].to(torch.complex128)
        xm = ((x_ax - P / 2) * 1e-9).to(torch.float64)
        zm = ((z_ax - H / 2) * 1e-9).to(torch.float64)
        Zc = zm.reshape(1, 1, -1)

        def tz(Fv):
            Fv = torch.trapezoid(Fv, xm, dim=0)
            Fv = torch.trapezoid(Fv, xm, dim=0)
            return torch.trapezoid(Fv, zm, dim=0)
        E_dn = complex(-(Z0 / (2 * A)) * tz(Jc * torch.exp(1j * k * Zc)))
        pref = -(Z0 / (2 * A))
        p_ = complex(mo['px' if pol == 'x' else 'py'])
        m_ = complex(mo['my' if pol == 'x' else 'mx'])
        Q_ = complex(mo['Qxz' if pol == 'x' else 'Qyz'])
        sgn = 1.0 if pol == 'x' else -1.0     # (r x J) parity bookkeeping
        even = pref * (-1j * om * p_)
        odd_m = pref * (1j * k) * m_ * sgn
        odd_Q = pref * (1j * k) * (-(1j * om / 6) * Q_)
        g = E_dn / (t_full - tb)
        pieces = {'ED': even / g, 'MD': odd_m / g, 'EQ': odd_Q / g}
        combos = {'bg_only': tb,
                  'bg+ED': tb + pieces['ED'],
                  'bg+MD': tb + pieces['MD'],
                  'bg+EQ': tb + pieces['EQ'],
                  'bg+ED+MD': tb + pieces['ED'] + pieces['MD'],
                  'bg+ED+EQ': tb + pieces['ED'] + pieces['EQ'],
                  'bg+MD+EQ': tb + pieces['MD'] + pieces['EQ'],
                  'ladder_all': tb + sum(pieces.values()),
                  'full': t_full}
        for name, v in combos.items():
            out_rows.append({'tag': tag, 'pol': pol, 'model': name,
                             't_re': v.real, 't_im': v.imag,
                             'T_model': abs(v) ** 2})
        figs.append((pol, tb, pieces, t_full))
        print(f'{tag} {pol}: |t_full|^2={abs(t_full)**2:.4f} '
              f'ladder={abs(combos["ladder_all"])**2:.3f} '
              f'|ED|={abs(pieces["ED"]):.2f} |MD|={abs(pieces["MD"]):.2f} '
              f'|EQ|={abs(pieces["EQ"]):.2f}', flush=True)
    # figure
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 5))
    for ax, (pol, tb2, pieces, t_full) in zip(axs, figs):
        cum = [('bg', tb2)]
        z = tb2
        for nm in ('ED', 'MD', 'EQ'):
            z += pieces[nm]
            cum.append((f'+{nm}', z))
        colors = ['#666', 'tab:blue', 'tab:green', 'tab:red']
        prev = 0j
        for (nm, z2), c in zip(cum, colors):
            ax.annotate('', xy=(z2.real, z2.imag),
                        xytext=(prev.real, prev.imag),
                        arrowprops=dict(arrowstyle='->', color=c, lw=1.8))
            ax.text(((prev + z2) / 2).real, ((prev + z2) / 2).imag, nm,
                    fontsize=7.5, color=c)
            prev = z2
        ax.annotate('', xy=(t_full.real, t_full.imag),
                    xytext=(prev.real, prev.imag),
                    arrowprops=dict(arrowstyle='->', color='k', lw=1.1,
                                    linestyle=':'))
        ax.plot([t_full.real], [t_full.imag], 'k*', ms=11,
                label=f'full t, |t|^2={abs(t_full)**2:.3f}')
        ax.plot([0], [0], 'ko', ms=6, mfc='none', label='t = 0')
        pts = [c for _, c in cum] + [t_full, 0j]
        lim = 1.15 * max(max(abs(p.real), abs(p.imag)) for p in pts)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.grid(alpha=0.2)
        ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(0, color='gray', lw=0.5)
        ax.legend(fontsize=7, loc='lower left')
        ax.set_title(f'{pol}-pol', fontsize=9)
        ax.set_xlabel('Re t')
        ax.set_ylabel('Im t')
    fig.suptitle(f'{tag}: forward-transmission cancellation (633 nm, '
                 'air-side; 1st-order ladder + residual)', fontsize=10)
    fig.tight_layout()
    fig.savefig(F / f'targand_{tag}.png', dpi=180)
    plt.close(fig)
    return out_rows


if __name__ == '__main__':
    torch.set_num_threads(2)
    rows = []
    for tag in sys.argv[1:]:
        rows += analyze(tag)
    with open(R / 't_argand_finalists.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('RT_TARGAND_DONE')
