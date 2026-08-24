"""
ed_eq_detuning.py — §19 causal detuning trajectory (Stage A observable run)
===========================================================================
Candidate: P0750_H0250_seed011 (bright-ED / dark-EQ Fano structure found by
the Q-free pilot). One physical parameter: uniform IN-PLANE scale alpha
(period P -> alpha*P, pattern scales with the cell, h fixed) — this detunes
the narrow EQ-driven feature across the broad ED background.

For each alpha: locate the resonance (coarse scan, lossless material), then
fine-sample it (0.25 nm), and record per lambda:
    exact moments (px..Qxz), C_px, C_Qxz, T/R, full scattered channel
    amplitudes (TORCWA authority), exact channel integrals (up/dn),
    even/odd channel parts and the radiation-normalized ED-EQ phase
        dphi_rad = arg(even_px_term) - arg(odd_Qxz_term)   [up channel]
    stored-energy proxy U = int eps|E|^2 dV (dense grid) at the fine points.
Then pole-fit Q_rad from t(omega), record P_rad-related channel powers at
resonance. All quantities are OBSERVABLES; nothing is optimized.
Checkpointed per (alpha, lambda); idempotent.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_HERE.parent))
import ed_eq_core as core                      # noqa: E402
from ed_eq_qualify import pole_fit             # noqa: E402

CAND = 'P0750_H0250_seed011'
PILOT = _HERE / 'results' / 'pilot' / CAND
OUT = _HERE / 'results' / 'detuning'
ALPHAS = [0.94, 0.96, 0.975, 0.985, 0.99, 0.995, 1.0, 1.005, 1.01, 1.02,
          1.035, 1.05, 1.07]
ORDER = [9, 9]
N_XY, NZ = 48, 9
Z0 = 376.730313668
EPS0, C0 = core.EPS0, core.C0


def scan_point(rho, P, h, lam, order=ORDER):
    eps_si = core.si_eps(float(lam), lossless=True)
    n = rho.shape[0]
    with torch.no_grad():
        sim = core.build_sim(rho, P, h, float(lam), order, eps_si=eps_si)
        amps = core.channel_amplitudes(sim)
        x_ax, z_ax, E, _ = core.fields_3d(sim, P, h, N_XY, NZ)
        idx = (torch.floor(x_ax / P * n).long()) % n
        eps3 = (rho[idx][:, idx] * (eps_si - 1.0) + 1.0)[:, :, None] \
            .expand(N_XY, N_XY, NZ)
        mo = core.torch_moments(E, eps3, x_ax, z_ax, float(lam))
        k = 2 * math.pi / (lam * 1e-9)
        omega = 2 * math.pi * C0 / (lam * 1e-9)
        A_cell = (P * 1e-9) ** 2
        # exact channel integrals
        chi = (eps3 - 1.0).to(torch.complex128)
        Jx = (-1j * omega * EPS0) * chi * E[0].to(torch.complex128)
        xm = ((x_ax - P / 2) * 1e-9).to(torch.float64)
        zm = ((z_ax - h / 2) * 1e-9).to(torch.float64)
        Zc = zm.reshape(1, 1, -1)

        def tz(F):
            F = torch.trapezoid(F, xm, dim=0)
            F = torch.trapezoid(F, xm, dim=0)
            return torch.trapezoid(F, zm, dim=0)
        E_up = complex(-(Z0 / (2 * A_cell)) * tz(Jx * torch.exp(-1j * k * Zc)))
        E_dn = complex(-(Z0 / (2 * A_cell)) * tz(Jx * torch.exp(+1j * k * Zc)))
        # stored-energy proxy (dimensionless: eps|E|^2 averaged over cell*h)
        U = float(torch.mean(eps3.real * sum(torch.abs(Ei) ** 2 for Ei in E)))

    px, my, Qxz = complex(mo['px']), complex(mo['my']), complex(mo['Qxz'])
    even_px = -(Z0 / (2 * A_cell)) * (-1j * omega * px)
    odd_Q = -(Z0 / (2 * A_cell)) * (-1j * k) * (-(1j * omega / 6) * Qxz)
    odd_m = -(Z0 / (2 * A_cell)) * (-1j * k) * my
    row = {'lam_nm': float(lam)}
    for key in ('txx', 'rxx'):
        v = complex(amps[key])
        row[key + '_re'], row[key + '_im'] = v.real, v.imag
    row['T'] = abs(complex(amps['txx'])) ** 2 + abs(complex(amps['tyx'])) ** 2
    row['R'] = abs(complex(amps['rxx'])) ** 2 + abs(complex(amps['ryx'])) ** 2
    row['P_out_spec'] = row['T'] + row['R']       # all open channels (specular-only)
    for t in ('px', 'py', 'pz', 'mx', 'my', 'mz', 'Qxz', 'Qyz', 'Qxy'):
        v = complex(mo[t])
        row[t + '_re'], row[t + '_im'] = v.real, v.imag
    cE = k ** 4 / (6 * math.pi * EPS0 ** 2)
    row['C_px'] = cE * abs(px) ** 2
    row['C_Qxz'] = (k ** 6 / (720 * math.pi * EPS0 ** 2)) * 2 * abs(Qxz) ** 2
    row['E_up_re'], row['E_up_im'] = E_up.real, E_up.imag
    row['E_dn_re'], row['E_dn_im'] = E_dn.real, E_dn.imag
    row['even_px_re'], row['even_px_im'] = even_px.real, even_px.imag
    row['odd_Q_re'], row['odd_Q_im'] = odd_Q.real, odd_Q.imag
    row['odd_m_re'], row['odd_m_im'] = odd_m.real, odd_m.imag
    row['dphi_rad_deg'] = math.degrees(np.angle(even_px) - np.angle(odd_Q))
    row['U_proxy'] = U
    return row


def run_alpha(alpha, rho, P0, h):
    P = alpha * P0
    out_csv = OUT / f'alpha_{alpha:.3f}.csv'
    rows = []
    done = set()
    if out_csv.exists():
        rows = list(csv.DictReader(open(out_csv)))
        done = {round(float(r['lam_nm']), 3) for r in rows}

    def save():
        rows.sort(key=lambda r: float(r['lam_nm']))
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)

    coarse = np.arange(1272.0, 1392.0 + 0.1, 2.0)
    for lam in coarse:
        if round(float(lam), 3) in done:
            continue
        rows.append(scan_point(rho, P, h, lam))
        save()
    # locate narrow feature: max |dT/dlam| on the coarse comb
    cr = sorted([r for r in rows if abs(float(r['lam_nm']) * 4 % 8) < 1e-6
                 or True], key=lambda r: float(r['lam_nm']))
    ll = np.array([float(r['lam_nm']) for r in cr])
    TT = np.array([float(r['T']) for r in cr])
    i0 = int(np.argmax(np.abs(np.gradient(TT, ll))))
    lam_c = ll[i0]
    fine = np.arange(lam_c - 6.0, lam_c + 6.0 + 0.01, 0.25)
    for lam in fine:
        if round(float(lam), 3) in done:
            continue
        rows.append(scan_point(rho, P, h, float(lam)))
        save()
    print(f'alpha={alpha}: {len(rows)} points, feature near {lam_c:.0f} nm',
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, nargs=2, default=[0, 1])
    ap.add_argument('--threads', type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((PILOT / 'config.json').read_text())
    rho = torch.tensor(np.load(PILOT / 'rho_binary.npy'), dtype=torch.float32)
    for alpha in ALPHAS[args.shard[0]::args.shard[1]]:
        run_alpha(alpha, rho, float(cfg['P']), float(cfg['h']))
    print('DETUNING_SHARD_DONE', flush=True)


if __name__ == '__main__':
    main()
