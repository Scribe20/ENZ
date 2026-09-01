"""Section 4 pre-flight: reproduce the paper rectangle, validate
conventions (illumination direction, circular basis, PB rotation law,
ED/MD multipole assignment), and emit the padding table.
"""
import csv
import json
import math

import numpy as np
import torch

import rt_core as rc

P, H, WX, WY = 226.0, 170.0, 160.0, 96.0
N = 128
ORDER = (9, 9)


def rect_rho(n=N, P=P, wx=WX, wy=WY, theta_deg=0.0):
    ax = (np.arange(n) + 0.5) / n * P - P / 2
    X, Y = np.meshgrid(ax, ax, indexing='ij')
    th = math.radians(theta_deg)
    Xr = X * math.cos(th) + Y * math.sin(th)
    Yr = -X * math.sin(th) + Y * math.cos(th)
    return torch.tensor(((np.abs(Xr) <= wx / 2) & (np.abs(Yr) <= wy / 2))
                        .astype(np.float32))


def main():
    print(f'a-Si eps(633) = {rc.EPS_ASI_633:.4f} '
          f'(n+ik = {rc.EPS_ASI_633**0.5:.4f}); glass n = {rc.N_GLASS}',
          flush=True)
    rho = rect_rho()
    with torch.no_grad():
        sim = rc.build_sim(rho, P, H, order=ORDER)
        rows = {}
        for direction in ('backward', 'forward'):
            R, T = rc.jones(sim, direction)
            m = rc.device_metrics(R, T)
            rows[direction] = m
            print(f"\n== {direction} ({'air-side' if direction=='backward' else 'glass-side'} incidence) ==")
            for k in ('abs_rx', 'abs_ry', 'dphi_r_deg', 'R_cross', 'R_co',
                      'T_total_x', 'T_total_y', 'R_total_x', 'R_total_y',
                      'A_x', 'A_y', 'abs_rxy'):
                print(f'  {k:12s} {m[k]:+.4f}')
            print(f'  dphi/pi      {m["dphi_r_deg"]/180:+.3f}')
        # circular-basis identity check (rectangle is diagonal):
        R, T = rc.jones(sim, 'backward')
        rx, ry = complex(R[0, 0]), complex(R[1, 1])
        rc_manual = abs((rx - ry) / 2) ** 2
        m = rows['backward']
        print(f'\ncircular check: |{m["R_cross"]:.5f} - {rc_manual:.5f}| '
              f'= {abs(m["R_cross"]-rc_manual):.2e}')
    # multipole assignment under device illumination
    for pol in ('x', 'y'):
        fam = rc.moments_families(rho, P, H, rc.LAM0, ORDER, pol,
                                  direction='backward')
        print(f'pol {pol}: f_ED={float(fam["f_ED"]):.3f} '
              f'f_MD={float(fam["f_MD"]):.3f} f_EQ={float(fam["f_EQ"]):.3f} '
              f'f_MQ={float(fam["f_MQ"]):.3f} '
              f'px|ED={float(fam["px_in_ED"]):.2f} '
              f'py|ED={float(fam["py_in_ED"]):.2f} '
              f'mx|MD={float(fam["mx_in_MD"]):.2f} '
              f'my|MD={float(fam["my_in_MD"]):.2f}', flush=True)
    # PB rotation smoke: 0/15/30/45 deg
    print('\nPB rotation smoke (backward, phase of circular cross term):')
    ph0 = None
    for th in (0.0, 15.0, 30.0, 45.0):
        rho_r = rect_rho(theta_deg=th)
        with torch.no_grad():
            sim = rc.build_sim(rho_r, P, H, order=ORDER)
            R, T = rc.jones(sim, 'backward')
            Rc = rc.circular(R)
        phc = math.degrees(np.angle(complex(Rc[0, 1])))
        if ph0 is None:
            ph0 = phc
        d = ((phc - ph0 + 180) % 360) - 180
        print(f'  theta={th:5.1f}: |r_cross|^2={abs(complex(Rc[0,1]))**2:.4f}'
              f'  phase={phc:+8.2f} deg  shift={d:+8.2f} (expect ~{-2*th:+.0f} or {2*th:+.0f})',
              flush=True)
    # padding table
    with open(rc.HERE / 'results' / 'padding_by_period.csv', 'w',
              newline='') as f:
        w = csv.writer(f)
        w.writerow(['period_nm', 'fixed_padding_nm', 'design_radius_nm'])
        for Pv in rc.PERIODS:
            w.writerow([Pv, rc.padding(Pv), rc.r_design(Pv)])
    # diffraction thresholds at 60 deg
    print('\ndiffraction check at theta=60 deg (P*(n+sin60) vs 633):')
    for Pv in rc.PERIODS:
        thr_air = Pv * (1 + math.sin(math.radians(60)))
        thr_glass = Pv * (rc.N_GLASS + math.sin(math.radians(60)))
        print(f'  P={Pv:.0f}: air {thr_air:.0f} nm, glass {thr_glass:.0f} nm '
              f'-> {"SAFE" if thr_glass < 633 else "OPENS"}')
    json.dump(rows, open(rc.HERE / 'results' / 'rectangle_baseline.json',
                         'w'), indent=1)
    with open(rc.HERE / 'results' / 'rectangle_baseline.csv', 'w',
              newline='') as f:
        w = csv.DictWriter(f, fieldnames=['direction'] +
                           list(rows['backward'].keys()))
        w.writeheader()
        for d, m in rows.items():
            w.writerow({'direction': d, **m})
    print('\nBASELINE_DONE')


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
