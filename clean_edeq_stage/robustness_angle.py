"""Phases 16-17: robustness suite and angle/polarization scan for P0550.

Perturbations (Phase 16): thickness +-5/+-10 nm, 1-pixel erosion/dilation
(lateral etch bias ~ P/64 = 8.6 nm), corner rounding (gaussian sigma=1px,
re-binarized), index scale +-1.4% (~ +-0.05 in n). Each evaluated on a
10-wavelength grid -> p0550_robustness.csv (full observable rows).

Angle scan (Phase 17): theta in {2,4,6,8,10} deg, same grid ->
p0550_angle_scan.csv. Diffraction safety: n_sub(P)(1+sin theta) stays
below lambda for all cases (first substrate order opens at
lambda < n_sub P (1+sin th) = 932 nm at 10 deg << 1300 nm band).
Polarization: y-pol R/T at the lambda grid (multipole targets are
x-pol-specific; recorded as R/T only).

usage: python robustness_angle.py <shard> <nshards>
"""
import sys

import numpy as np
import scipy.ndimage as ndi
import torch

import stage_core as sc
import ed_eq_core as core

NAME = 'P0550_H0250_seed011'
LAMS = [1300.0, 1315.0, 1330.0, 1332.5, 1345.0, 1360.0, 1375.0, 1390.0,
        1405.0, 1414.0]


def perturbed_geometries(rho):
    b = rho.numpy() > 0.5
    out = {'baseline': rho}
    out['erode_1px'] = torch.tensor(
        ndi.binary_erosion(b).astype(np.float32))
    out['dilate_1px'] = torch.tensor(
        ndi.binary_dilation(b).astype(np.float32))
    out['round_s1'] = torch.tensor(
        (ndi.gaussian_filter(b.astype(float), 1.0) > 0.5).astype(np.float32))
    return out


def ypol_rt(rho, P, h, lam):
    eps_si = core.si_eps(float(lam))
    sim = core.build_sim(rho, P, h, float(lam), sc.ORDER, eps_si=eps_si)
    sim.source_planewave(amplitude=[0.0, 1.0], direction='forward')
    t = {}
    for pol in ('yy', 'xy'):
        t['t' + pol] = complex(sim.S_parameters(
            orders=[0, 0], direction='forward', port='transmission',
            polarization=pol, ref_order=[0, 0]))
        t['r' + pol] = complex(sim.S_parameters(
            orders=[0, 0], direction='forward', port='reflection',
            polarization=pol, ref_order=[0, 0]))
    T = abs(t['tyy']) ** 2 + abs(t['txy']) ** 2
    R = abs(t['ryy']) ** 2 + abs(t['rxy']) ** 2
    return T, R


def main():
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    rho, P, h = sc.load_ref(NAME)
    geos = perturbed_geometries(rho)
    jobs = []
    for tag in ('baseline', 'erode_1px', 'dilate_1px', 'round_s1'):
        for lam in LAMS:
            jobs.append(('rob', tag, lam, {}))
    for dh in (-10, -5, 5, 10):
        for lam in LAMS:
            jobs.append(('rob', f'h{dh:+d}nm', lam, {'h_override': h + dh}))
    for ds in (-0.014, 0.014):
        for lam in LAMS:
            jobs.append(('rob', f'n{ds:+.3f}', lam, {'eps_scale': 1 + ds}))
    for th in (2.0, 4.0, 6.0, 8.0, 10.0):
        for lam in LAMS:
            jobs.append(('ang', f'th{th:.0f}', lam, {'inc_ang': th}))
    jobs = jobs[shard::nsh]

    import csv
    outs = {'rob': sc.RESULTS / f'robustness_shard{shard}.csv',
            'ang': sc.RESULTS / f'angle_shard{shard}.csv'}
    done = {}
    for kind, p in outs.items():
        done[kind] = set()
        if p.exists():
            with open(p) as f:
                done[kind] = {(r['tag'], round(float(r['lam_nm']), 3))
                              for r in csv.DictReader(f)}
    fields = {}
    for kind, tag, lam, kw in jobs:
        if (tag, round(float(lam), 3)) in done[kind]:
            continue
        g = geos.get(tag, rho)
        row = sc.scan_point_full(g, P, h, float(lam), **kw)
        row = {'tag': tag, **row}
        if kind not in fields:
            fields[kind] = list(row.keys())
        sc.append_row(outs[kind], row, fieldnames=fields[kind])
        print(f's{shard} {tag} {lam:.1f} B={row["ED_EQ_balance"]:.3f} '
              f'R={row["R"]:.3f}', flush=True)
    # y-pol quick rows (shard 0 only)
    if shard == 0:
        out = sc.RESULTS / 'p0550_ypol_rt.csv'
        if not out.exists():
            with open(out, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['lam_nm', 'T_ypol', 'R_ypol'])
                for lam in LAMS:
                    with torch.no_grad():
                        T, R = ypol_rt(rho, P, h, lam)
                    w.writerow([lam, T, R])
                    print(f'ypol {lam} T={T:.3f} R={R:.3f}', flush=True)
    print(f'ROBUST_SHARD_DONE {shard}', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
