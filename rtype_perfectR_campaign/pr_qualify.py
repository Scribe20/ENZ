"""Finalist qualification (spec secs 31-34): eigenchannel analysis,
spectral scan 620-645 nm, fabrication robustness (boundary bias and
height), Fourier-order convergence on complex eigenvalues + closure.

usage: python pr_qualify.py <what> <rho.npy> <P> <H> <label>
  what in {eigen, spectra, fab, conv, all}
Appends to results/{eigenchannels,spectra_finalists,fab_robustness,
convergence}.csv
"""
import csv
import math
import sys

import numpy as np
import scipy.ndimage as ndi
import torch

import pr_core as pr
import rt_core as rc

RES = pr.HERE / 'results'


def append(name, rows):
    out = RES / f'{name}.csv'
    new = not out.exists()
    with open(out, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def eigen(rho, P, H, label, order=(9, 9)):
    with torch.no_grad():
        Rj, Tj = pr.jones_theta0(rho, P, H, order)
    R = Rj.numpy().astype(complex)
    T = Tj.numpy().astype(complex)
    ev, vec = np.linalg.eig(R)
    # principal-axis angle of each eigenpolarization (linear if the
    # eigenvector has (near) real ratio)
    rows = []
    for k in range(2):
        v = vec[:, k] / (np.linalg.norm(vec[:, k]) + 1e-12)
        ang = math.degrees(math.atan2(abs(v[1]), abs(v[0])))
        # bounded ellipticity measure: 0 for linear, 0.5 for circular
        ellip = abs(np.imag(np.conj(v[0]) * v[1])) / (abs(v[0]) ** 2
                                                     + abs(v[1]) ** 2)
        rows.append({'label': label, 'k': k, 'eig_abs': abs(ev[k]),
                     'eig_phase_deg': math.degrees(np.angle(ev[k])),
                     'axis_angle_deg': ang, 'ellipticity': ellip})
    dphi = math.degrees(np.angle(ev[0]) - np.angle(ev[1]))
    err = abs(((dphi - 180.0 + 180.0) % 360.0) - 180.0)
    Rc = rc.circular(Rj)
    rows.append({'label': label, 'k': -1, 'eig_abs': min(abs(ev)),
                 'eig_phase_deg': dphi, 'axis_angle_deg': err,
                 'ellipticity': float(torch.abs(Rc[0, 1])
                                      - torch.abs(Rc[1, 0]))})
    print(f'{label} eigen: |ev|={abs(ev[0]):.3f},{abs(ev[1]):.3f} '
          f'dphi_eig={dphi:.1f} (err {err:.1f}) axes '
          f'{rows[0]["axis_angle_deg"]:.1f}/{rows[1]["axis_angle_deg"]:.1f} '
          f'ellip {rows[0]["ellipticity"]:.2f}/{rows[1]["ellipticity"]:.2f}',
          flush=True)
    append('eigenchannels', rows)


def spectra(rho, P, H, label, order=(9, 9)):
    lams = sorted(set(list(np.arange(620.0, 645.1, 1.0))
                      + list(np.arange(630.0, 636.01, 0.5))))
    rows = []
    for lam in lams:
        e = rc.eps_asi(lam)
        sim = __import__('torcwa').rcwa(freq=1.0 / lam, order=list(order),
                                        L=[float(P), float(P)],
                                        dtype=rc.SIM_DTYPE, device=rc.DEVICE)
        sim.add_input_layer(eps=rc.EPS_GLASS)
        sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
        sim.add_layer(thickness=float(H), eps=rho * (e - 1.0) + 1.0)
        sim.solve_global_smatrix()
        with torch.no_grad():
            Rj, Tj = rc.jones(sim, 'backward')
        m = pr.scalars(pr.port_metrics(Rj, Tj))
        rows.append({'label': label, 'lam_nm': float(lam), **m})
    F = np.array([r['F'] for r in rows])
    L = np.array([r['lam_nm'] for r in rows])
    F0 = F[np.argmin(abs(L - 633.0))]
    above = L[F >= 0.5 * F0]
    print(f'{label} spectra: F(633)={F0:.3f} peak={F.max():.3f}@'
          f'{L[F.argmax()]:.1f} half-max band ~{above.min():.0f}-'
          f'{above.max():.0f} nm (within scan)', flush=True)
    append('spectra_finalists', rows)


def fab(rho, P, H, label, order=(9, 9)):
    b = rho.numpy() > 0.5
    n = b.shape[0]
    px = P / n
    rows = []
    for bias_px in (-4, -2, 0, 2, 4):
        if bias_px < 0:
            bb = ndi.binary_erosion(b, iterations=-bias_px)
        elif bias_px > 0:
            bb = ndi.binary_dilation(b, iterations=bias_px)
        else:
            bb = b
        bb = bb & (pr.design_mask(n, P).numpy() > 0.5)
        r2 = torch.tensor(bb.astype(np.float32))
        for dH in (-10.0, -5.0, 0.0, 5.0, 10.0):
            if bias_px != 0 and dH != 0.0:
                continue
            m = pr.eval_full(r2, P, H + dH, order)
            rows.append({'label': label, 'bias_nm': bias_px * px,
                         'dH_nm': dH, **{k: m[k] for k in
                                         ('F', 'T', 'co', 'A',
                                          'phase_err_deg')}})
    for r in rows:
        print(f"{label} fab bias={r['bias_nm']:+.1f} dH={r['dH_nm']:+.0f}: "
              f"F={r['F']:.3f} T={r['T']:.3f} co={r['co']:.3f}", flush=True)
    append('fab_robustness', rows)


def conv(rho, P, H, label):
    rows = []
    for o in (9, 11, 13, 15):
        with torch.no_grad():
            Rj, Tj = pr.jones_theta0(rho, P, H, (o, o))
        m = pr.scalars(pr.port_metrics(Rj, Tj))
        evR = np.linalg.eigvals(Rj.numpy().astype(complex))
        evT = np.linalg.eigvals(Tj.numpy().astype(complex))
        closure = m['Rtot'] + m['T'] + m['A']
        rows.append({'label': label, 'order': o, **m,
                     'evR0_re': evR[0].real, 'evR0_im': evR[0].imag,
                     'evR1_re': evR[1].real, 'evR1_im': evR[1].imag,
                     'evT0_abs': abs(evT[0]), 'evT1_abs': abs(evT[1]),
                     'closure': closure})
        print(f"{label} o{o}: F={m['F']:.4f} T={m['T']:.4f} co={m['co']:.4f}"
              f" evR=({evR[0]:.3f},{evR[1]:.3f})", flush=True)
    append('convergence', rows)


if __name__ == '__main__':
    torch.set_num_threads(2)
    what, path, P, H, label = sys.argv[1], sys.argv[2], float(sys.argv[3]), \
        float(sys.argv[4]), sys.argv[5]
    rho = torch.tensor(np.load(path).astype(np.float32))
    fns = {'eigen': eigen, 'spectra': spectra, 'fab': fab, 'conv': conv}
    for k in (fns if what == 'all' else [what]):
        fns[k](rho, P, H, label)
    print(f'QUALIFY_DONE {label}', flush=True)
