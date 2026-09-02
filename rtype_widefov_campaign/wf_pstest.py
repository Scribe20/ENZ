"""Validate the p/s-basis oblique Jones extraction:
  1. energy closure (lossless) at extreme angle/azimuth
  2. theta->0 reduction to the validated xy Jones (find sign matrices)
  3. phi-covariance of helicity powers
"""
import math

import numpy as np
import torch

import wf_core as wf
import rt_core as rc
import torcwa
from wf_preflight import rect_rho, P_R, H_R


def jones_ps(sim):
    r = [[None, None], [None, None]]
    t = [[None, None], [None, None]]
    for i, po in ((0, 'p'), (1, 's')):
        for j, pi in ((0, 'p'), (1, 's')):
            r[i][j] = sim.S_parameters(orders=[0, 0], direction='backward',
                                       port='reflection',
                                       polarization=po + pi,
                                       ref_order=[0, 0]).reshape(())
            t[i][j] = sim.S_parameters(orders=[0, 0], direction='backward',
                                       port='transmission',
                                       polarization=po + pi,
                                       ref_order=[0, 0]).reshape(())
    R = torch.stack([torch.stack(r[0]), torch.stack(r[1])])
    T = torch.stack([torch.stack(t[0]), torch.stack(t[1])])
    return R, T


def build(rho, P, H, th_air, phi, eps_override=None, order=(9, 9)):
    e = eps_override if eps_override is not None else rc.eps_asi()
    sim = torcwa.rcwa(freq=1.0 / 633.0, order=list(order), L=[P, P],
                      dtype=rc.SIM_DTYPE, device=rc.DEVICE)
    sim.add_input_layer(eps=rc.EPS_GLASS)
    sim.set_incident_angle(inc_ang=wf.glass_angle(th_air),
                           azi_ang=math.radians(phi))
    sim.add_layer(thickness=float(H), eps=rho * (e - 1.0) + 1.0)
    sim.solve_global_smatrix()
    return sim


def main():
    torch.set_num_threads(2)
    rho = rect_rho()
    e_l = complex(rc.EPS_ASI_633.real, 0.0)

    print('== lossless closure in ps basis ==', flush=True)
    for th, ph in ((80.0, 45.0), (80.0, 17.0), (60.0, 45.0), (85.0, 30.0)):
        sim = build(rho, P_R, H_R, th, ph, eps_override=e_l)
        R, T = jones_ps(sim)
        for j, name in ((0, 'p-in'), (1, 's-in')):
            tot = float((torch.abs(R[:, j]) ** 2).sum()
                        + (torch.abs(T[:, j]) ** 2).sum())
            print(f'  th={th} phi={ph} {name}: R+T = {tot:.5f}', flush=True)

    print('== theta->0 reduction vs xy ==', flush=True)
    sim = build(rho, P_R, H_R, 0.01, 0.0)
    Rps, Tps = jones_ps(sim)
    Rxy, Txy = rc.jones(sim, 'backward')
    np.set_printoptions(precision=4, suppress=True)
    print('R_xy:\n', Rxy.numpy())
    print('R_ps:\n', Rps.numpy())
    print('T_xy:\n', Txy.numpy())
    print('T_ps:\n', Tps.numpy())

    print('== phi=90 reduction (p ~ +y, s ~ -x) ==', flush=True)
    sim = build(rho, P_R, H_R, 0.01, 90.0)
    Rps90, Tps90 = jones_ps(sim)
    print('R_ps(phi=90):\n', Rps90.numpy())

    print('== helicity powers, lossy, th=50 phi sweep ==', flush=True)
    for ph in (0.0, 22.5, 45.0, 67.5, 90.0):
        sim = build(rho, P_R, H_R, 50.0, ph)
        Rps, Tps = jones_ps(sim)
        Rdev = torch.diag(torch.tensor([-1.0 + 0j, 1.0 + 0j])) @ Rps
        Rc = rc.circular(Rdev)
        print(f'  phi={ph:5.1f}: Rcross={float(torch.abs(Rc[0,1])**2):.4f}'
              f'/{float(torch.abs(Rc[1,0])**2):.4f} '
              f'Rco={float(torch.abs(Rc[0,0])**2):.4f}'
              f'/{float(torch.abs(Rc[1,1])**2):.4f}', flush=True)


if __name__ == '__main__':
    main()
