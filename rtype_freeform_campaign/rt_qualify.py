"""Finalist qualification: angle robustness, PB rotation, spectral window,
multipole forensics, complex-t Argand, Fourier-order convergence.

usage: python rt_qualify.py <stage_glob_tag> ...   (tags under refinement/)
Writes results/{angle_scan_finalists, pb_rotation_finalists,
multipole_finalists, convergence_finalists, spectra_finalists,
t_argand_finalists}.csv
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import torch

import rt_core as rc

R = rc.HERE / 'results'


def load_tag(tag):
    for stage in ('refinement', 'coarse', 'finalists'):
        p = rc.HERE / stage / tag
        if (p / 'final.json').exists():
            rec = json.loads((p / 'final.json').read_text())
            rho = torch.tensor(np.load(p / 'rho_binary.npy'))
            return rec, rho
    raise FileNotFoundError(tag)


def jones_at(rho, P, H, lam=rc.LAM0, order=(9, 9), theta=0.0, phi=0.0):
    e = rc.eps_asi(lam)
    import torcwa
    sim = torcwa.rcwa(freq=1.0 / float(lam), order=list(order),
                      L=[float(P), float(P)], dtype=rc.SIM_DTYPE,
                      device=rc.DEVICE)
    sim.add_input_layer(eps=rc.EPS_GLASS)
    sim.set_incident_angle(inc_ang=math.radians(theta),
                           azi_ang=math.radians(phi))
    sim.add_layer(thickness=float(H), eps=rho * (e - 1.0) + 1.0)
    sim.solve_global_smatrix()
    return rc.jones(sim, 'backward')


def angle_scan(tags):
    out = R / 'angle_scan_finalists.csv'
    done = set()
    if out.exists():
        with open(out) as f:
            done = {(r['tag'], float(r['theta']), float(r['phi']))
                    for r in csv.DictReader(f)}
    fields = None
    for tag in tags:
        rec, rho = load_tag(tag)
        for th in (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60):
            for ph in (0, 45, 90):
                if (tag, float(th), float(ph)) in done:
                    continue
                with torch.no_grad():
                    Rj, Tj = jones_at(rho, rec['P'], rec['H'],
                                      theta=th, phi=ph)
                m = rc.device_metrics(Rj, Tj)
                row = {'tag': tag, 'theta': th, 'phi': ph, **m}
                if fields is None:
                    fields = list(row.keys())
                new = not out.exists()
                with open(out, 'a', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=fields)
                    if new:
                        w.writeheader()
                    w.writerow(row)
        print(f'angle {tag} done', flush=True)


def pb_rotation(tags):
    out = R / 'pb_rotation_finalists.csv'
    rows = []
    for tag in tags:
        rec, rho = load_tag(tag)
        phis = []
        for th in range(0, 181, 15):
            rr = ndi.rotate(rho.numpy(), th, reshape=False, order=0,
                            mode='constant', cval=0.0)
            rr = torch.tensor((rr > 0.5).astype(np.float32))
            # safety: confirm still inside envelope
            n = rr.shape[0]
            ax = (np.arange(n) + 0.5) / n * rec['P'] - rec['P'] / 2
            X, Y = np.meshgrid(ax, ax, indexing='ij')
            outside = float(rr.numpy()[(X ** 2 + Y ** 2)
                                       > rc.r_design(rec['P']) ** 2].sum())
            with torch.no_grad():
                Rj, Tj = jones_at(rr, rec['P'], rec['H'])
            Rc = rc.circular(Rj)
            amp2 = float(torch.abs(Rc[0, 1]) ** 2)
            ph = math.degrees(float(torch.angle(Rc[0, 1])))
            phis.append((th, ph, amp2))
            rows.append({'tag': tag, 'theta_rot': th, 'phase_deg': ph,
                         'R_cross': amp2, 'pixels_outside_envelope':
                         outside})
        th_a = np.array([p[0] for p in phis], dtype=float)
        ph_a = np.unwrap(np.radians([p[1] for p in phis]))
        slope = np.polyfit(th_a, np.degrees(ph_a), 1)[0]
        resid = np.degrees(ph_a) - np.polyval(
            np.polyfit(th_a, np.degrees(ph_a), 1), th_a)
        amps = np.array([p[2] for p in phis])
        print(f'PB {tag}: slope={slope:+.3f} deg/deg (expect -2), '
              f'rms={np.std(resid):.1f} deg, R_cross '
              f'{amps.min():.3f}-{amps.max():.3f}', flush=True)
        rows.append({'tag': tag, 'theta_rot': -1, 'phase_deg': slope,
                     'R_cross': float(np.std(resid)),
                     'pixels_outside_envelope': float(amps.min())})
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def spectra(tags):
    out = R / 'spectra_finalists.csv'
    rows = []
    for tag in tags:
        rec, rho = load_tag(tag)
        for lam in np.arange(600.0, 670.0 + 0.1, 5.0):
            with torch.no_grad():
                Rj, Tj = jones_at(rho, rec['P'], rec['H'], lam=float(lam))
            m = rc.device_metrics(Rj, Tj)
            rows.append({'tag': tag, 'lam_nm': float(lam), **m})
        print(f'spectra {tag} done', flush=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def forensics(tags):
    out = R / 'multipole_finalists.csv'
    rows = []
    for tag in tags:
        rec, rho = load_tag(tag)
        row = {'tag': tag}
        for pol in ('x', 'y'):
            f = rc.moments_families(rho, rec['P'], rec['H'], rc.LAM0,
                                    (9, 9), pol, n_xy=48, nz=7)
            for k in ('f_ED', 'f_MD', 'f_EQ', 'f_MQ', 'px_in_ED',
                      'py_in_ED', 'mx_in_MD', 'my_in_MD'):
                row[f'{k}_{pol}'] = float(f[k])
        fams = {k: row[f'f_{k}_x'] for k in ('ED', 'MD', 'EQ', 'MQ')}
        fams_y = {k: row[f'f_{k}_y'] for k in ('ED', 'MD', 'EQ', 'MQ')}
        row['class_x'] = max(fams, key=fams.get)
        row['class_y'] = max(fams_y, key=fams_y.get)
        rows.append(row)
        print(f'forensics {tag}: x->{row["class_x"]} '
              f'({fams[row["class_x"]]:.2f}) y->{row["class_y"]} '
              f'({fams_y[row["class_y"]]:.2f})', flush=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def convergence(tags):
    out = R / 'convergence_finalists.csv'
    rows = []
    for tag in tags:
        rec, rho = load_tag(tag)
        for o in (9, 11, 13, 15):
            with torch.no_grad():
                Rj, Tj = jones_at(rho, rec['P'], rec['H'], order=(o, o))
            m = rc.device_metrics(Rj, Tj)
            rows.append({'tag': tag, 'order': o, **m})
            print(f'conv {tag} o{o}: Rc={m["R_cross"]:.3f} '
                  f'err={m["pb_phase_err_deg"]:.0f}', flush=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == '__main__':
    torch.set_num_threads(2)
    what = sys.argv[1]
    tags = sys.argv[2:]
    {'angle': angle_scan, 'pb': pb_rotation, 'spectra': spectra,
     'forensics': forensics, 'conv': convergence}[what](tags)
    print(f'QUALIFY_{what.upper()}_DONE', flush=True)
