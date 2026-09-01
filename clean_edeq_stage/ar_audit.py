"""Antireflection / baseline audit of the P0550 family (Stage-B addendum).

Subcommand `rt`: R/T-only spectra (no field reconstruction - fast) for
  cases: bare stack, uniform a-Si film, fill-matched centered disk,
  P0550 freeform; both incidence directions; multiple Fourier orders.
Sharded + per-point checkpointed.
usage: python ar_audit.py rt <shard> <nshards>
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

import stage_core as sc
import ed_eq_core as core

NAME = 'P0550_H0250_seed011'
H_STAR = 227.2
HS = [225.0, H_STAR, 235.0, 250.0]
FILL = 0.6176859736442566


def disk_rho(n=110, fill=FILL):
    r_px = n * math.sqrt(fill / math.pi)
    y, x = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2
    return torch.tensor(((x - c) ** 2 + (y - c) ** 2 <= r_px ** 2)
                        .astype(np.float32))


def build_case(case, h, lam, order):
    """Returns solved sim for the case. Homogeneous layers use scalar eps."""
    P = 550.0
    eps_si = core.si_eps(float(lam))
    if case in ('bare', 'uniform'):
        sim = __import__('torcwa').rcwa(freq=1.0 / float(lam),
                                        order=list(order), L=[P, P],
                                        dtype=core.SIM_DTYPE,
                                        device=core.DEVICE)
        sim.add_input_layer(eps=core.SUBSTRATE_EPS)
        sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
        sim.add_layer(thickness=float(h),
                      eps=(1.0 + 0j) if case == 'bare' else eps_si)
        sim.solve_global_smatrix()
        return sim
    rho = disk_rho() if case == 'simple' else sc.load_ref(NAME)[0]
    return core.build_sim(rho, P, float(h), float(lam), list(order),
                          eps_si=eps_si)


def rt_both(sim):
    out = {}
    for direction, pre in (('forward', 'f'), ('backward', 'b')):
        sim.source_planewave(amplitude=[1.0, 0.0], direction=direction)
        T = R = 0.0
        for pol in ('xx', 'yx'):
            t = complex(sim.S_parameters(orders=[0, 0], direction=direction,
                                         port='transmission',
                                         polarization=pol, ref_order=[0, 0]))
            r = complex(sim.S_parameters(orders=[0, 0], direction=direction,
                                         port='reflection',
                                         polarization=pol, ref_order=[0, 0]))
            T += abs(t) ** 2
            R += abs(r) ** 2
            if pol == 'xx':
                out[pre + '_t_re'], out[pre + '_t_im'] = t.real, t.imag
                out[pre + '_r_re'], out[pre + '_r_im'] = r.real, r.imag
        out[pre + '_T'], out[pre + '_R'] = T, R
    return out


def jobs_list():
    lam1 = np.arange(1260.0, 1420.0 + 0.1, 1.0)
    lam2 = np.arange(1260.0, 1420.0 + 0.1, 2.0)
    lam4 = np.arange(1260.0, 1420.0 + 0.1, 4.0)
    J = []
    for lam in lam1:
        J.append(('bare', 250.0, lam, 9))
    for h in HS:
        for lam in lam1:
            J.append(('uniform', h, lam, 9))
            J.append(('p0550', h, lam, 9))
    for h in (H_STAR, 235.0):
        for lam in lam1:
            J.append(('simple', h, lam, 9))
    for o, lams in ((11, lam2), (13, lam2), (15, lam4)):
        for h in HS:
            for lam in lams:
                J.append(('p0550', h, lam, o))
    return J


def cmd_rt(shard, nsh):
    J = jobs_list()[shard::nsh]
    out = sc.RESULTS / f'ar_rt_shard{shard}.csv'
    done = set()
    if out.exists():
        with open(out) as f:
            done = {(r['case'], round(float(r['h_nm']), 2),
                     round(float(r['lam_nm']), 3), int(r['order']))
                    for r in csv.DictReader(f)}
    fields = None
    for case, h, lam, o in J:
        key = (case, round(float(h), 2), round(float(lam), 3), o)
        if key in done:
            continue
        with torch.no_grad():
            sim = build_case(case, h, lam, [o, o])
            row = {'case': case, 'h_nm': float(h), 'lam_nm': float(lam),
                   'order': o, **rt_both(sim)}
        if fields is None:
            fields = list(row.keys())
        sc.append_row(out, row, fieldnames=fields)
        if abs(lam - 1332.0) < 0.6:
            print(f's{shard} {case} h={h} o={o} {lam}: fR={row["f_R"]:.4f} '
                  f'bR={row["b_R"]:.4f}', flush=True)
    print(f'AR_RT_SHARD_DONE {shard}', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    cmd_rt(int(sys.argv[2]), int(sys.argv[3]))
