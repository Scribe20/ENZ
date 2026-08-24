"""
ed_eq_qualify.py — Stage-A qualification of pilot candidates
============================================================
For each completed pilot binary geometry:

1. Wavelength scan lam0 +/- 40 nm (1 nm): TORCWA T/R + scattered 0th-order
   channel amplitudes (authority for radiation), dense-lite fields ->
   EXACT current multipoles (C_px, C_Qxz, all Cartesian components,
   family weights, toroidal diagnostic), exact channel-integral
   reconstruction (up/down) with residual vs TORCWA.
2. Resonance identification + complex pole fit of t(omega):
       t = a + b / (omega - omega0 + i*gamma)
   on the LOSSLESS material (k -> 0)  => Q_rad = omega0 / (2*gamma);
   repeated with k = 1e-4              => Q_loaded; 1/Q_abs = 1/Ql - 1/Qr.
   Q is RECORDED, never used for selection (contract).
3. Candidate ledger row (all 8 selection criteria + recorded Q).

Checkpointed per candidate (skip-if-complete); idempotent.
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

RESULTS = _HERE / 'results'
PILOT = RESULTS / 'pilot'
QUAL = RESULTS / 'qualify'
LAM0 = 1332.5
LAMS = np.arange(LAM0 - 40.0, LAM0 + 40.0 + 0.01, 1.0)
ORDER_Q = [9, 9]
N_XY, NZ = 48, 9
Z0 = 376.730313668
EPS0, C0 = core.EPS0, core.C0


def scan_candidate(run_dir, out_csv, lossless=False, k_override=None,
                   order=ORDER_Q, lams=LAMS):
    cfg = json.loads((run_dir / 'config.json').read_text())
    P, h = float(cfg['P']), float(cfg['h'])
    rho = torch.tensor(np.load(run_dir / 'rho_binary.npy'), dtype=torch.float32)
    rows = []
    done = set()
    if out_csv.exists():
        rows = list(csv.DictReader(open(out_csv)))
        done = {round(float(r['lam_nm']), 2) for r in rows}
    n = rho.shape[0]
    for lam in lams:
        if round(float(lam), 2) in done:
            continue
        eps_si = core.si_eps(float(lam), lossless=lossless, k_override=k_override)
        with torch.no_grad():
            sim = core.build_sim(rho, P, h, float(lam), order, eps_si=eps_si)
            amps = core.channel_amplitudes(sim)
            x_ax, z_ax, E, _ = core.fields_3d(sim, P, h, N_XY, NZ)
            idx = (torch.floor(x_ax / P * n).long()) % n
            eps3 = (rho[idx][:, idx] * (eps_si - 1.0) + 1.0)[:, :, None] \
                .expand(N_XY, N_XY, NZ)
            mo = core.torch_moments(E, eps3, x_ax, z_ax, float(lam))
            # exact channel integrals (scattered up/down, center-referenced)
            k = 2 * math.pi / (lam * 1e-9)
            omega = 2 * math.pi * C0 / (lam * 1e-9)
            chi = (eps3 - 1.0).to(torch.complex128)
            Jx = (-1j * omega * EPS0) * chi * E[0].to(torch.complex128)
            xm = ((x_ax - P / 2) * 1e-9).to(torch.float64)
            zm = ((z_ax - h / 2) * 1e-9).to(torch.float64)
            Zc = zm.reshape(1, 1, -1)
            A_cell = (P * 1e-9) ** 2

            def tz(F):
                F = torch.trapezoid(F, xm, dim=0)
                F = torch.trapezoid(F, xm, dim=0)
                return torch.trapezoid(F, zm, dim=0)
            E_up = complex(-(Z0 / (2 * A_cell)) * tz(Jx * torch.exp(-1j * k * Zc)))
            E_dn = complex(-(Z0 / (2 * A_cell)) * tz(Jx * torch.exp(+1j * k * Zc)))
        row = {'lam_nm': float(lam)}
        for key in ('txx', 'rxx', 'tyx', 'ryx'):
            v = complex(amps[key])
            row[key + '_re'], row[key + '_im'] = v.real, v.imag
        row['T'] = abs(complex(amps['txx'])) ** 2 + abs(complex(amps['tyx'])) ** 2
        row['R'] = abs(complex(amps['rxx'])) ** 2 + abs(complex(amps['ryx'])) ** 2
        for t in ('px', 'py', 'pz', 'mx', 'my', 'mz',
                  'Qxx', 'Qyy', 'Qzz', 'Qxy', 'Qxz', 'Qyz', 'Tx', 'Ty', 'Tz'):
            v = complex(mo[t])
            row[t + '_re'], row[t + '_im'] = v.real, v.imag
        cE = k ** 4 / (6 * math.pi * EPS0 ** 2)
        row['C_px'] = cE * abs(complex(mo['px'])) ** 2
        row['C_Qxz'] = (k ** 6 / (720 * math.pi * EPS0 ** 2)) * 2 * abs(complex(mo['Qxz'])) ** 2
        row['E_up_re'], row['E_up_im'] = E_up.real, E_up.imag
        row['E_dn_re'], row['E_dn_im'] = E_dn.real, E_dn.imag
        rows.append(row)
        rows.sort(key=lambda r: float(r['lam_nm']))
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    return rows


def pole_fit(lams_nm, t_complex):
    """Fit t(omega) = a + b/(omega - omega0 + i gamma) around the strongest
    resonant feature. Returns (lam0_nm, Q, ok)."""
    from scipy.optimize import least_squares
    om = 2 * math.pi * C0 / (np.asarray(lams_nm) * 1e-9)
    t = np.asarray(t_complex)
    # locate feature: max curvature of |t|
    d2 = np.abs(np.gradient(np.gradient(np.abs(t))))
    i0 = int(np.argmax(d2[2:-2])) + 2
    sl = slice(max(0, i0 - 15), min(len(om), i0 + 16))
    oms, ts = om[sl], t[sl]

    def resid(p):
        a = p[0] + 1j * p[1]; b = (p[2] + 1j * p[3]) * 1e12
        om0, g = p[4], abs(p[5])
        model = a + b / (oms - om0 + 1j * g)
        return np.concatenate([(model - ts).real, (model - ts).imag])
    p0 = [t.real.mean(), t.imag.mean(), 0.1, 0.0, om[i0], (om[0] - om[-1]) / 40]
    try:
        r = least_squares(resid, p0, max_nfev=20000)
        om0, g = r.x[4], abs(r.x[5])
        lam_res = 2 * math.pi * C0 / om0 * 1e9
        Q = om0 / (2 * g)
        ok = bool(r.success and lams_nm.min() < lam_res < lams_nm.max() and Q > 0)
        return float(lam_res), float(Q), ok
    except Exception:
        return float('nan'), float('nan'), False


def analyze(run_dir, qdir):
    main_csv = qdir / 'spectra_main.csv'
    lossless_csv = qdir / 'spectra_lossless.csv'
    lossy_csv = qdir / 'spectra_lossy.csv'
    rows = list(csv.DictReader(open(main_csv)))
    lam = np.array([float(r['lam_nm']) for r in rows])
    g = lambda kk, rr=rows: np.array([float(r[kk]) for r in rr])
    cplx = lambda base, rr=rows: np.array(
        [complex(float(r[base + '_re']), float(r[base + '_im'])) for r in rr])

    Cpx, CQxz = g('C_px'), g('C_Qxz')
    led = {'run_id': run_dir.name}
    cfg = json.loads((run_dir / 'config.json').read_text())
    led.update({'P': cfg['P'], 'h': cfg['h'], 'seed': cfg['seed'],
                'fill': cfg.get('fill', float('nan'))})

    def peak_fwhm(v):
        i = int(np.argmax(v)); half = v[i] / 2
        above = lam[v >= half]
        return lam[i], (above[-1] - above[0]) if len(above) > 1 else np.nan, \
            bool(0 < i < len(v) - 1)
    led['lam_px'], led['fwhm_px'], led['px_interior'] = peak_fwhm(Cpx)
    led['lam_Qxz'], led['fwhm_Qxz'], led['Qxz_interior'] = peak_fwhm(CQxz)
    led['split_nm'] = abs(led['lam_px'] - led['lam_Qxz'])
    led['split_over_fwhm'] = led['split_nm'] / max(
        min(led['fwhm_px'], led['fwhm_Qxz']), 1e-9)

    i0 = int(np.argmin(np.abs(lam - LAM0)))
    pxv = {t: cplx(t)[i0] for t in ('px', 'py', 'pz', 'mx', 'my', 'mz',
                                    'Qxx', 'Qyy', 'Qzz', 'Qxy', 'Qxz', 'Qyz')}
    p2 = {t: abs(pxv[t]) ** 2 for t in pxv}
    led['px_frac_ED'] = p2['px'] / (p2['px'] + p2['py'] + p2['pz'] + 1e-300)
    qsum = (p2['Qxx'] + p2['Qyy'] + p2['Qzz']
            + 2 * (p2['Qxy'] + p2['Qxz'] + p2['Qyz']))
    led['Qxz_frac_EQ'] = 2 * p2['Qxz'] / (qsum + 1e-300)
    k = 2 * math.pi / (LAM0 * 1e-9)
    cE = k ** 4 / (6 * math.pi * EPS0 ** 2)
    Cp = cE * (p2['px'] + p2['py'] + p2['pz'])
    Cm = cE / C0 ** 2 * (p2['mx'] + p2['my'] + p2['mz'])
    CQe = cE / 120 * k ** 2 * qsum
    led['EDEQ_frac'] = (Cp + CQe) / (Cp + Cm + CQe + 1e-300)
    led['MD_frac'] = Cm / (Cp + Cm + CQe + 1e-300)
    led['S_px_t'] = float(Cpx[i0] / (cfg['P'] * 1e-9) ** 2)
    led['S_Qxz_t'] = float(CQxz[i0] / (cfg['P'] * 1e-9) ** 2)

    # recorded (not selecting) Q
    for tag, csvp in [('rad', lossless_csv), ('loaded', lossy_csv)]:
        if csvp.exists():
            rr = list(csv.DictReader(open(csvp)))
            ll = np.array([float(r['lam_nm']) for r in rr])
            tt = np.array([complex(float(r['txx_re']), float(r['txx_im']))
                           for r in rr])
            lr, Q, ok = pole_fit(ll, tt)
            led[f'lam_res_{tag}'], led[f'Q_{tag}'], led[f'Qfit_ok_{tag}'] = lr, Q, ok
    if led.get('Q_rad') and led.get('Q_loaded') and \
            np.isfinite(led['Q_rad']) and np.isfinite(led['Q_loaded']) and \
            led['Q_rad'] > led['Q_loaded'] > 0:
        led['Q_abs'] = 1 / (1 / led['Q_loaded'] - 1 / led['Q_rad'])
    return led


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, nargs=2, default=[0, 1])
    ap.add_argument('--threads', type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    runs = sorted([d for d in PILOT.iterdir()
                   if (d / 'config.json').exists()
                   and json.loads((d / 'config.json').read_text())
                   .get('status') == 'completed'])
    runs = runs[args.shard[0]::args.shard[1]]
    ledger = []
    for rd in runs:
        qd = QUAL / rd.name
        qd.mkdir(parents=True, exist_ok=True)
        print(f'{rd.name}: qualification scans', flush=True)
        scan_candidate(rd, qd / 'spectra_main.csv')
        scan_candidate(rd, qd / 'spectra_lossless.csv', lossless=True)
        scan_candidate(rd, qd / 'spectra_lossy.csv', k_override=1e-4)
        led = analyze(rd, qd)
        (qd / 'ledger_row.json').write_text(json.dumps(led, indent=1,
                                                       default=float))
        ledger.append(led)
        print(f"{rd.name}: split={led['split_nm']:.1f}nm "
              f"pxED={led['px_frac_ED']:.2f} QxzEQ={led['Qxz_frac_EQ']:.2f} "
              f"EDEQ={led['EDEQ_frac']:.2f} Q_rad={led.get('Q_rad', float('nan')):.0f}",
              flush=True)
    print('QUALIFY_SHARD_DONE', flush=True)


if __name__ == '__main__':
    main()
