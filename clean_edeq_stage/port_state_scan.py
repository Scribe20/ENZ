"""Port-state audit: new compute (all reuse-aware, sharded, checkpointed).

Job types:
  p0750_fine  - full observable rows, 1328-1340 nm step 0.2 (resolves the
                T-minimum; existing 1-nm rows remain in the master scan)
  p0750_ord   - full rows at 1334.0 nm, orders [11,11] and [13,13]
  p0550_href  - full rows at lam0 for h = 216.25..238.75 step 2.5 (fills
                the fine sweep to 1.25-nm spacing in the knob region)
  smx_h       - complex S-matrix (both directions) vs h for P0550, [9,9]
  smx_p0750   - complex S-matrix vs lambda for P0750 (1-nm + 0.2-nm core)
  overlap     - mid-slab complex Ex snapshot for P0550 vs h (mode identity)
usage: python port_state_scan.py <shard> <nshards>
"""
import csv
import sys

import numpy as np
import torch

import stage_core as sc
from ar_audit import rt_both, build_case
import ed_eq_core as core

LAM0 = 1332.5
H_FINE2 = np.arange(216.25, 238.75 + 0.1, 2.5)
LAM_P0750_FINE = np.round(np.arange(1328.0, 1340.0 + 0.01, 0.2), 3)
H_SMX = np.arange(200.0, 300.0 + 0.1, 2.5)
H_OVL = np.arange(200.0, 300.0 + 0.1, 5.0)


def smx_point(name, h, lam, order=(9, 9)):
    rho, P, h0 = sc.load_ref(name)
    eps_si = core.si_eps(float(lam))
    sim = core.build_sim(rho, P, float(h), float(lam), list(order),
                         eps_si=eps_si)
    return rt_both(sim)


def main():
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    jobs = []
    for lam in LAM_P0750_FINE:
        jobs.append(('p0750_fine', 'P0750_H0250_seed011', 250.0, float(lam),
                     (9, 9)))
    for o in (11, 13):
        jobs.append(('p0750_ord', 'P0750_H0250_seed011', 250.0, 1334.0,
                     (o, o)))
    for h in H_FINE2:
        jobs.append(('p0550_href', 'P0550_H0250_seed011', float(h), LAM0,
                     (9, 9)))
    for h in H_SMX:
        jobs.append(('smx_h', 'P0550_H0250_seed011', float(h), LAM0, (9, 9)))
    for lam in np.arange(1300.0, 1360.0 + 0.1, 1.0):
        jobs.append(('smx_p0750', 'P0750_H0250_seed011', 250.0, float(lam),
                     (9, 9)))
    for lam in LAM_P0750_FINE:
        jobs.append(('smx_p0750', 'P0750_H0250_seed011', 250.0, float(lam),
                     (9, 9)))
    for h in H_OVL:
        jobs.append(('overlap', 'P0550_H0250_seed011', float(h), LAM0,
                     (9, 9)))
    jobs = jobs[shard::nsh]

    full_csv = sc.RESULTS / f'ps_full_shard{shard}.csv'
    smx_csv = sc.RESULTS / f'ps_smx_shard{shard}.csv'
    ovl_dir = sc.RESULTS / 'audit' / 'overlap'
    ovl_dir.mkdir(parents=True, exist_ok=True)
    done_full, done_smx = set(), set()
    for p, s in ((full_csv, done_full), (smx_csv, done_smx)):
        if p.exists():
            with open(p) as f:
                for r in csv.DictReader(f):
                    s.add((r['kind'], r['name'], round(float(r['h_nm']), 3),
                           round(float(r['lam_nm']), 3), int(r['order'])))
    f_fields = s_fields = None
    for kind, name, h, lam, order in jobs:
        key = (kind, name, round(h, 3), round(lam, 3), order[0])
        if kind in ('p0750_fine', 'p0750_ord', 'p0550_href'):
            if key in done_full:
                continue
            rho, P, h0 = sc.load_ref(name)
            row = sc.scan_point_full(rho, P, h0, lam, order=list(order),
                                     h_override=h)
            row = {'kind': kind, 'name': name, 'h_nm': h, 'order': order[0],
                   **row}
            if f_fields is None:
                f_fields = list(row.keys())
            sc.append_row(full_csv, row, fieldnames=f_fields)
            print(f's{shard} {kind} h={h:g} lam={lam:g} o={order[0]} '
                  f'T={row["T"]:.4f}', flush=True)
        elif kind in ('smx_h', 'smx_p0750'):
            if key in done_smx:
                continue
            with torch.no_grad():
                rr = smx_point(name, h, lam, order)
            row = {'kind': kind, 'name': name, 'h_nm': h, 'lam_nm': lam,
                   'order': order[0], **rr}
            if s_fields is None:
                s_fields = list(row.keys())
            sc.append_row(smx_csv, row, fieldnames=s_fields)
        elif kind == 'overlap':
            out = ovl_dir / f'ex_mid_h{h:07.2f}.npz'
            if out.exists():
                continue
            rho, P, h0 = sc.load_ref(name)
            eps_si = core.si_eps(lam)
            with torch.no_grad():
                sim = core.build_sim(rho, P, h, lam, list(order),
                                     eps_si=eps_si)
                x_ax, z_ax, E, _ = core.fields_3d(sim, P, h, 64, 5)
            np.savez(out, Ex=E[0][:, :, 2].numpy().astype(np.complex64),
                     h=h)
            print(f's{shard} overlap h={h:g} saved', flush=True)
    print(f'PS_SHARD_DONE {shard}', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
