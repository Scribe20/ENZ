"""Complex-t cancellation vs incident angle (spec section 40).

Air-side illumination, transmission into glass, phi = 0 (so p = x-like,
s = y). Generalized first-order ladder into the oblique specular
channel (k_hat, e_hat in glass):
    piece_ED = -i w (p . e*)
    piece_MD = -i k_g ((m x k_hat) . e*)
    piece_EQ = -i k_g (-i w / 6) ((Q k_hat) . e*)
(reduces exactly to the validated normal-incidence ladder of
rt_targand.py at theta = 0). Per-row exact port coupling
g = E_channel_integral / (t_full - t_bg); the dotted residual arrow to
the full TORCWA t shows the truncation (MQ + higher orders) honestly.

usage: python wf_argand.py <name> [...]
Writes results/angle_argand.csv + figures/wf_argand_<name>.png
"""
import csv
import math
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import wf_core as wf
import rt_core as rc
from wf_anglemap import load_geometry

sys.path.insert(0, str(wf.HERE.parent / 'ed_eq_causality_campaign'))
import ed_eq_core as core                                   # noqa: E402
import torcwa                                               # noqa: E402

Z0 = 376.730313668
R = wf.HERE / 'results'
F = wf.HERE / 'figures'
F.mkdir(exist_ok=True)
THETAS = [0.0, 30.0, 60.0, 75.0]


def t_bare(P, H, th_air, order=(9, 9)):
    sim = torcwa.rcwa(freq=1.0 / wf.LAM0, order=list(order),
                      L=[float(P), float(P)], dtype=rc.SIM_DTYPE,
                      device=rc.DEVICE)
    sim.add_input_layer(eps=rc.EPS_GLASS)
    sim.set_incident_angle(inc_ang=wf.glass_angle(th_air), azi_ang=0.0)
    sim.add_layer(thickness=float(H), eps=1.0 + 0j)
    sim.solve_global_smatrix()
    _, T = wf.jones_dev(sim)
    return complex(T[0, 0]), complex(T[1, 1])


def ladder(name):
    rho, P, H, e_ovr = load_geometry(name)
    e = rc.eps_asi() if e_ovr is None else e_ovr
    out_rows = []
    panels = []
    for th in THETAS:
        th_g = wf.glass_angle(th)
        k0 = 2 * math.pi / (wf.LAM0 * 1e-9)
        k_g = k0 * rc.N_GLASS
        om = 2 * math.pi * core.C0 / (wf.LAM0 * 1e-9)
        A = (P * 1e-9) ** 2
        khat = np.array([math.sin(th_g), 0.0, -math.cos(th_g)])
        evec = {'p': np.array([math.cos(th_g), 0.0, math.sin(th_g)]),
                's': np.array([0.0, 1.0, 0.0])}
        tb_p, tb_s = t_bare(P, H, th)
        sim = wf.build_sim_angle(rho, P, H, th, 0.0, order=(9, 9))
        _, Tdev = wf.jones_dev(sim)
        tfull = {'p': complex(Tdev[0, 0]), 's': complex(Tdev[1, 1])}
        tb = {'p': tb_p, 's': tb_s}
        for pol, amp in (('p', [1.0, 0.0]), ('s', [0.0, 1.0])):
            sim.source_planewave(amplitude=amp, direction='backward')
            with torch.no_grad():
                x_ax, z_ax, E, _ = core.fields_3d(sim, float(P), float(H),
                                                  48, 7)
            n = rho.shape[0]
            idx = (torch.floor(x_ax / P * n).long()) % n
            eps3 = (rho[idx][:, idx] * (e - 1.0) + 1.0)[:, :, None] \
                .expand(48, 48, 7)
            mo = core.torch_moments(E, eps3, x_ax, z_ax, wf.LAM0)
            chi = (eps3 - 1.0).to(torch.complex128)
            J = [(-1j * om * core.EPS0) * chi * E[i].to(torch.complex128)
                 for i in range(3)]
            xm = ((x_ax - P / 2) * 1e-9).to(torch.float64)
            zm = ((z_ax - H / 2) * 1e-9).to(torch.float64)
            Xg = xm.reshape(-1, 1, 1)
            Zg = zm.reshape(1, 1, -1)
            ph_fac = torch.exp(-1j * k_g * (khat[0] * Xg + khat[2] * Zg))
            ev = evec[pol]
            Je = (J[0] * ev[0] + J[1] * ev[1] + J[2] * ev[2]) * ph_fac

            def tz(Fv):
                Fv = torch.trapezoid(Fv, xm, dim=0)
                Fv = torch.trapezoid(Fv, xm, dim=0)
                return torch.trapezoid(Fv, zm, dim=0)
            pref = -(Z0 / (2 * A))
            E_int = complex(pref * tz(Je))
            p_v = np.array([complex(mo['px']), complex(mo['py']),
                            complex(mo['pz'])])
            m_v = np.array([complex(mo['mx']), complex(mo['my']),
                            complex(mo['mz'])])
            Q = np.array([[complex(mo['Qxx']), complex(mo['Qxy']),
                           complex(mo['Qxz'])],
                          [complex(mo['Qxy']), complex(mo['Qyy']),
                           complex(mo['Qyz'])],
                          [complex(mo['Qxz']), complex(mo['Qyz']),
                           complex(mo['Qzz'])]])
            ED = pref * (-1j * om) * np.dot(p_v, ev)
            MD = pref * (-1j * k_g) * np.dot(np.cross(m_v, khat), ev)
            EQ = pref * (-1j * k_g) * (-1j * om / 6) * np.dot(Q @ khat, ev)
            g = E_int / (tfull[pol] - tb[pol])
            pieces = {'ED': ED / g, 'MD': MD / g, 'EQ': EQ / g}
            combos = {'bg_only': tb[pol]}
            for nm, v in pieces.items():
                combos[f'bg+{nm}'] = tb[pol] + v
            combos['ladder_all'] = tb[pol] + sum(pieces.values())
            combos['full'] = tfull[pol]
            for nm, v in combos.items():
                out_rows.append({'tag': name, 'theta': th, 'pol': pol,
                                 'model': nm, 't_re': v.real,
                                 't_im': v.imag, 'T_model': abs(v) ** 2})
            panels.append((th, pol, tb[pol], pieces, tfull[pol]))
            t2 = abs(tfull[pol]) ** 2
            l2 = abs(combos['ladder_all']) ** 2
            print(f'{name} th={th:.0f} {pol}: |t|^2={t2:.3f} '
                  f'ladder={l2:.3f} |ED|={abs(pieces["ED"]):.2f} '
                  f'|MD|={abs(pieces["MD"]):.2f} '
                  f'|EQ|={abs(pieces["EQ"]):.2f}', flush=True)
    # figure: rows = pol, cols = theta
    fig, axs = plt.subplots(2, len(THETAS), figsize=(4.2 * len(THETAS),
                                                     8.6))
    for (th, pol, tb2, pieces, tfl) in panels:
        ax = axs[0 if pol == 'p' else 1][THETAS.index(th)]
        cum, z = [('bg', tb2)], tb2
        for nm in ('ED', 'MD', 'EQ'):
            z = z + pieces[nm]
            cum.append((f'+{nm}', z))
        colors = ['#666', 'tab:blue', 'tab:green', 'tab:red']
        prev = 0j
        for (nm, z2), c in zip(cum, colors):
            ax.annotate('', xy=(z2.real, z2.imag),
                        xytext=(prev.real, prev.imag),
                        arrowprops=dict(arrowstyle='->', color=c, lw=1.7))
            ax.text(((prev + z2) / 2).real, ((prev + z2) / 2).imag, nm,
                    fontsize=7, color=c)
            prev = z2
        ax.annotate('', xy=(tfl.real, tfl.imag),
                    xytext=(prev.real, prev.imag),
                    arrowprops=dict(arrowstyle='->', color='k', lw=1.0,
                                    linestyle=':'))
        ax.plot([tfl.real], [tfl.imag], 'k*', ms=10)
        ax.plot([0], [0], 'ko', ms=5, mfc='none')
        pts = [c for _, c in cum] + [tfl, 0j]
        lim = 1.15 * max(max(abs(p2.real), abs(p2.imag)) for p2 in pts)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.grid(alpha=0.2)
        ax.set_title(f'{pol}-pol theta={th:.0f} |t|^2={abs(tfl)**2:.3f}',
                     fontsize=8.5)
    fig.suptitle(f'{name}: forward-transmission ladder vs angle (633 nm, '
                 'phi=0; 1st-order ED/MD/EQ + residual)', fontsize=10)
    fig.tight_layout()
    fig.savefig(F / f'wf_argand_{name}.png', dpi=170)
    plt.close(fig)
    return out_rows


if __name__ == '__main__':
    torch.set_num_threads(2)
    rows = []
    for nm in sys.argv[1:]:
        rows += ladder(nm)
    out = R / 'angle_argand.csv'
    new = not out.exists()
    with open(out, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print('WF_ARGAND_DONE', flush=True)
