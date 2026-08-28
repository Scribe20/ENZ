"""Shared core for the clean-ED-EQ exploitation stage (Stage B).

Reference structures (frozen, from the audited Stage-A pilot):
  P0550_H0250_seed011  - clean balanced ED-EQ composition reference
  P0750_H0250_seed011  - MD-majority my/Qxz dark-state contrast reference

All solvers/settings inherit the audited conventions:
  order [9,9], 48x48x9 moment grid, canonical cell-center origin,
  Franta-2013 a-Si on silica, x-polarized normal incidence from the
  substrate, exp(-j w t), TORCWA power-normalized S-parameters.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CAMP = HERE.parent / 'ed_eq_causality_campaign'
sys.path.insert(0, str(CAMP))
sys.path.insert(0, str(CAMP.parent))

import ed_eq_core as core                                   # noqa: E402
from ed_eq_audit import family_row, classify, ORDER_A       # noqa: E402

RESULTS = HERE / 'results'
PILOT = CAMP / 'results' / 'pilot'
LAM0 = 1332.5
ORDER = list(ORDER_A)           # [9, 9]
N_XY, NZ = 48, 9
EPS0, C0 = core.EPS0, core.C0
Z0 = 376.730313668
N_SUB = math.sqrt(float(core.SUBSTRATE_EPS.real)) if hasattr(
    core, 'SUBSTRATE_EPS') else 1.444

MO_KEYS = ('px', 'py', 'pz', 'mx', 'my', 'mz',
           'Qxx', 'Qyy', 'Qzz', 'Qxy', 'Qxz', 'Qyz',
           'Qmxx', 'Qmyy', 'Qmzz', 'Qmxy', 'Qmxz', 'Qmyz',
           'Tx', 'Ty', 'Tz')


def load_ref(name='P0550_H0250_seed011'):
    cfg = json.loads((PILOT / name / 'config.json').read_text())
    rho = torch.tensor(np.load(PILOT / name / 'rho_binary.npy'),
                       dtype=torch.float32)
    return rho, float(cfg['P']), float(cfg['h'])


def scan_point_full(rho, P, h, lam, order=None, lossless=False,
                    eps_scale=None, h_override=None, inc_ang=0.0):
    """One wavelength: full observables row.

    Returns dict with complex moments, families+purities, exact channel
    integrals E_up/E_dn, ladder integrals I0/I1/I2 (= int Jx z^k dV),
    TORCWA t/r (xx and yx), bare-stack background t_bg/r_bg, T/R and
    energy residual. eps_scale: optional complex multiplier applied to
    (n+ik) for the index-tuning diagnostic. h_override for thickness
    perturbations. inc_ang in DEGREES (theta in x-z plane).
    """
    order = order or ORDER
    hh = h_override if h_override is not None else h
    eps_si = core.si_eps(float(lam), lossless=lossless)
    if eps_scale is not None:
        n_c = complex(eps_si) ** 0.5 * eps_scale
        eps_si = n_c ** 2
    n = rho.shape[0]
    with torch.no_grad():
        sim = core.build_sim(rho, P, hh, float(lam), order, eps_si=eps_si)
        if inc_ang:
            sim.set_incident_angle(inc_ang=inc_ang * math.pi / 180.0,
                                   azi_ang=0.0)
            sim.solve_global_smatrix()
            sim.source_planewave(amplitude=[1.0, 0.0], direction='forward')
        amps = core.channel_amplitudes(sim)
        x_ax, z_ax, E, _ = core.fields_3d(sim, P, hh, N_XY, NZ)
        idx = (torch.floor(x_ax / P * n).long()) % n
        eps3 = (rho[idx][:, idx] * (eps_si - 1.0) + 1.0)[:, :, None] \
            .expand(N_XY, N_XY, NZ)
        mo = core.torch_moments(E, eps3, x_ax, z_ax, float(lam))
        k = 2 * math.pi / (lam * 1e-9)
        omega = 2 * math.pi * C0 / (lam * 1e-9)
        A_cell = (P * 1e-9) ** 2
        chi = (eps3 - 1.0).to(torch.complex128)
        Jx = (-1j * omega * EPS0) * chi * E[0].to(torch.complex128)
        xm = ((x_ax - P / 2) * 1e-9).to(torch.float64)
        zm = ((z_ax - hh / 2) * 1e-9).to(torch.float64)
        Zc = zm.reshape(1, 1, -1)

        def tz(F):
            F = torch.trapezoid(F, xm, dim=0)
            F = torch.trapezoid(F, xm, dim=0)
            return torch.trapezoid(F, zm, dim=0)

        E_up = complex(-(Z0 / (2 * A_cell)) * tz(Jx * torch.exp(-1j * k * Zc)))
        E_dn = complex(-(Z0 / (2 * A_cell)) * tz(Jx * torch.exp(+1j * k * Zc)))
        I0 = complex(tz(Jx))
        I1 = complex(tz(Jx * Zc))
        I2 = complex(tz(Jx * Zc ** 2))
        U = float(torch.mean(eps3.real * sum(torch.abs(Ei) ** 2 for Ei in E)))
    # bare-stack background (empty patterned layer, same thickness)
    bare = core.bare_stack_amplitudes(P, hh, float(lam), order)
    row = {'lam_nm': float(lam)}
    fr = family_row(mo, lam)
    row.update({kk: float(fr[kk]) for kk in
                ('f_ED', 'f_MD', 'f_EQ', 'f_MQ', 'C_total_exact',
                 'px_given_ED', 'py_given_ED', 'pz_given_ED',
                 'mx_given_MD', 'my_given_MD', 'mz_given_MD',
                 'Qxx_given_EQ', 'Qyy_given_EQ', 'Qzz_given_EQ',
                 'Qxy_given_EQ', 'Qxz_given_EQ', 'Qyz_given_EQ',
                 'Cpx_total_fraction', 'Cmy_total_fraction',
                 'CQxz_total_fraction', 'ED_EQ_balance',
                 'CT_diag_over_CED')})
    row['class'] = classify(fr)
    for t in MO_KEYS:
        v = complex(mo[t])
        row[t + '_re'], row[t + '_im'] = v.real, v.imag
    for key in ('txx', 'rxx', 'tyx', 'ryx'):
        v = complex(amps[key])
        row[key + '_re'], row[key + '_im'] = v.real, v.imag
    for key, v in (('tbg', complex(bare['txx'])), ('rbg', complex(bare['rxx'])),
                   ('E_up', E_up), ('E_dn', E_dn),
                   ('I0', I0), ('I1', I1), ('I2', I2)):
        row[key + '_re'], row[key + '_im'] = v.real, v.imag
    row['T'] = abs(complex(amps['txx'])) ** 2 + abs(complex(amps['tyx'])) ** 2
    row['R'] = abs(complex(amps['rxx'])) ** 2 + abs(complex(amps['ryx'])) ** 2
    row['en_res'] = abs(row['T'] + row['R'] - 1.0) if not lossless else \
        abs(row['T'] + row['R'] - 1.0)
    row['U_proxy'] = U
    # ladder channel terms in the same (field-amplitude) units as E_up/E_dn:
    px, my, Qxz = complex(mo['px']), complex(mo['my']), complex(mo['Qxz'])
    pref = -(Z0 / (2 * A_cell))
    even_px = pref * (-1j * omega * px)
    odd_Q = pref * (-1j * k) * (-(1j * omega / 6) * Qxz)
    odd_m = pref * (-1j * k) * my
    for key, v in (('even_px', even_px), ('odd_Q', odd_Q), ('odd_m', odd_m)):
        row[key + '_re'], row[key + '_im'] = v.real, v.imag
    return row


def append_row(csv_path, row, fieldnames=None):
    import csv as _csv
    new = not Path(csv_path).exists()
    fn = fieldnames or list(row.keys())
    with open(csv_path, 'a', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=fn)
        if new:
            w.writeheader()
        w.writerow(row)


def done_keys(csv_path, key='lam_nm', nd=3):
    import csv as _csv
    p = Path(csv_path)
    if not p.exists():
        return set()
    with open(p) as f:
        return {round(float(r[key]), nd) for r in _csv.DictReader(f)}
