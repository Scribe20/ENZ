"""Validate F_ideal normalization + rotated-operator sign (spec sec 1-2)."""
import math

import numpy as np
import scipy.ndimage as ndi
import torch

import pr_core as pr
import rt_core as rc
import wf_core as wf
from wf_preflight import rect_rho, P_R, H_R


def main():
    torch.set_num_threads(2)
    # 1. synthetic normalization checks
    for rx, ry, name in ((1, -1, 'ideal'), (1j, -1j, 'ideal*e^{i pi/2}'),
                         (0.8, -0.8, '|r|=0.8'), (1, 1, 'mirror (no HW)'),
                         (1, -1j, '90-deg retarder')):
        R = torch.tensor([[rx, 0], [0, ry]], dtype=torch.complex64)
        F = float(pr.fidelity(R))
        print(f'{name:22s} F = {F:.4f}')
    # 2. rectangle at theta=0: F vs known R_cross (diagonal identity)
    rho = rect_rho()
    Rj, Tj = pr.jones_theta0(rho, P_R, H_R, order=(9, 9))
    m = pr.scalars(pr.port_metrics(Rj, Tj))
    print(f"rectangle: F={m['F']:.4f} Rcross={m['Rcross']:.4f} "
          f"T={m['T']:.4f} co={m['co']:.4f} A={m['A']:.4f}")
    # 3. rotated-operator sign: physically rotate by +15 deg; correct
    # U_alpha must give F(alpha=+15) >> F(alpha=-15) relative change...
    # use coherent fidelity ratio and the phase shift of Rc[0,1].
    rr = ndi.rotate(rho.numpy(), 15.0, reshape=False, order=0,
                    mode='constant', cval=0.0)
    rr = torch.tensor((rr > 0.5).astype(np.float32))
    Rj15, _ = pr.jones_theta0(rr, P_R, H_R, order=(9, 9))
    Rc0 = rc.circular(Rj)
    Rc15 = rc.circular(Rj15)
    shift = math.degrees(float(torch.angle(Rc15[0, 1])
                               - torch.angle(Rc0[0, 1])))
    shift = ((shift + 180) % 360) - 180
    print(f'angle(Rc01) shift under +15 deg rotation: {shift:+.1f} deg '
          '(expect ~ -30)')
    Fp = float(pr.fidelity(Rj15, +15.0))
    Fm = float(pr.fidelity(Rj15, -15.0))
    F0 = float(pr.fidelity(Rj15, 0.0))
    print(f'F(U_+15)={Fp:.4f}  F(U_0)={F0:.4f}  F(U_-15)={Fm:.4f} '
          '(U_+15 must be the largest)')
    print('PR_VALIDATE_DONE')


if __name__ == '__main__':
    main()
