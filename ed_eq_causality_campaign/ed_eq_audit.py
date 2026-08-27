"""
ed_eq_audit.py — Stage-A forensic audit driver
==============================================
Subcommands (all checkpointed / idempotent):

  families   complete 4-family fractions (f_ED+f_MD+f_EQ+f_MQ = 1) and
             component purities for ALL 18 pilot candidates at key
             wavelengths (lam0, C_px-peak, C_Qxz-peak, fitted pole);
             writes candidate_ledger_v2.csv + families_at_wavelengths.csv.
  qrefine    adaptive spectral refinement + JOINT complex t/r shared-pole
             fits with strict Q_RESOLVED gates and per-point energy
             residuals; writes q_validation_v2.csv.
  needle     forensic treatment of P0750_H0350_seed029 (adaptive ladder,
             then complex-frequency pole search; integrity checks).
  matrices   champion order/grid/origin matrices for Q and fractions.

Toroidal terms remain diagnostics (never a fifth family).
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
AUD = RESULTS / 'audit'
LAM0 = 1332.5
EPS0, C0 = core.EPS0, core.C0
ORDER_A = [9, 9]           # Stage-A fidelity (audited at higher orders in
                           # the matrices subcommand)

P_COMPS = ('px', 'py', 'pz')
M_COMPS = ('mx', 'my', 'mz')
QE_COMPS = ('Qxx', 'Qyy', 'Qzz', 'Qxy', 'Qxz', 'Qyz')
QM_COMPS = ('Qmxx', 'Qmyy', 'Qmzz', 'Qmxy', 'Qmxz', 'Qmyz')


def decompose_at(rho, P, h, lam, order=ORDER_A, lossless=True, n_xy=48, nz=9,
                 origin_nm=None):
    eps_si = core.si_eps(float(lam), lossless=lossless)
    n = rho.shape[0]
    with torch.no_grad():
        sim = core.build_sim(rho, P, h, float(lam), order, eps_si=eps_si)
        amps = core.channel_amplitudes(sim)
        x_ax, z_ax, E, _ = core.fields_3d(sim, P, h, n_xy, nz)
        idx = (torch.floor(x_ax / P * n).long()) % n
        eps3 = (rho[idx][:, idx] * (eps_si - 1.0) + 1.0)[:, :, None] \
            .expand(n_xy, n_xy, nz)
        mo = core.torch_moments(E, eps3, x_ax, z_ax, float(lam),
                                origin_nm=origin_nm)
    T = abs(complex(amps['txx'])) ** 2 + abs(complex(amps['tyx'])) ** 2
    R = abs(complex(amps['rxx'])) ** 2 + abs(complex(amps['ryx'])) ** 2
    return mo, {'T': T, 'R': R, 'en_res': abs(T + R - 1),
                'txx': complex(amps['txx']), 'rxx': complex(amps['rxx'])}


def family_row(mo, lam):
    k = mo['k']
    cE = k ** 4 / (6 * math.pi * EPS0 ** 2)
    C = {}
    for t in P_COMPS:
        C[t] = cE * abs(complex(mo[t])) ** 2
    for t in M_COMPS:
        C[t] = cE / C0 ** 2 * abs(complex(mo[t])) ** 2
    for t in QE_COMPS:
        w = 1 if t in ('Qxx', 'Qyy', 'Qzz') else 2
        C[t] = cE / 120 * k ** 2 * w * abs(complex(mo[t])) ** 2
    for t in QM_COMPS:
        w = 1 if t in ('Qmxx', 'Qmyy', 'Qmzz') else 2
        C[t] = cE / 120 * (k / C0) ** 2 * w * abs(complex(mo[t])) ** 2
    C_ED = sum(C[t] for t in P_COMPS)
    C_MD = sum(C[t] for t in M_COMPS)
    C_EQ = sum(C[t] for t in QE_COMPS)
    C_MQ = sum(C[t] for t in QM_COMPS)
    tot = C_ED + C_MD + C_EQ + C_MQ
    r = {'lam_nm': float(lam),
         'f_ED': C_ED / tot, 'f_MD': C_MD / tot,
         'f_EQ': C_EQ / tot, 'f_MQ': C_MQ / tot,
         'sum_check': (C_ED + C_MD + C_EQ + C_MQ) / tot,
         'C_total_exact': tot}
    for t in P_COMPS:
        r[f'{t}_given_ED'] = C[t] / (C_ED + 1e-300)
    for t in M_COMPS:
        r[f'{t}_given_MD'] = C[t] / (C_MD + 1e-300)
    for t in QE_COMPS:
        r[f'{t}_given_EQ'] = C[t] / (C_EQ + 1e-300)
    r['Cpx_total_fraction'] = C['px'] / tot
    r['Cmy_total_fraction'] = C['my'] / tot
    r['CQxz_total_fraction'] = C['Qxz'] / tot
    r['ED_EQ_balance'] = min(r['f_ED'], r['f_EQ']) / max(r['f_ED'], r['f_EQ'])
    # toroidal DIAGNOSTIC only (never added to the family sum)
    r['CT_diag_over_CED'] = (cE * abs(1j * k * complex(mo['Tx'])) ** 2) / (C_ED + 1e-300)
    return r


def classify(r):
    if r['f_ED'] + r['f_EQ'] >= 0.80 and r['f_ED'] >= 0.20 and r['f_EQ'] >= 0.20 \
            and r['px_given_ED'] >= 0.80 and r['Qxz_given_EQ'] >= 0.80:
        return 'clean_balanced_ED_EQ'
    if r['f_ED'] >= 0.6:
        return 'ED_dominated' + ('_with_EQ' if r['f_EQ'] >= 0.15 else '')
    if r['f_EQ'] >= 0.6:
        return 'EQ_dominated' + ('_with_ED' if r['f_ED'] >= 0.15 else '')
    if r['f_MD'] >= 0.4:
        return 'MD_contaminated'
    return 'mixed_higher_order'


def cmd_families(args):
    AUD.mkdir(parents=True, exist_ok=True)
    long_rows, ledger = [], []
    for d in sorted(PILOT.iterdir()):
        if not (d / 'config.json').exists():
            continue
        cfg = json.loads((d / 'config.json').read_text())
        if cfg.get('status') != 'completed':
            continue
        rho = torch.tensor(np.load(d / 'rho_binary.npy'), dtype=torch.float32)
        P, h = float(cfg['P']), float(cfg['h'])
        led = json.loads((QUAL / d.name / 'ledger_row.json').read_text())
        lams = {'lam0': LAM0, 'px_peak': led['lam_px'], 'Qxz_peak': led['lam_Qxz']}
        if led.get('Qfit_ok_rad') and np.isfinite(led.get('lam_res_rad', np.nan)):
            lams['pole'] = led['lam_res_rad']
        rows_c = {}
        for tag, lam in lams.items():
            mo, ch = decompose_at(rho, P, h, lam)
            r = family_row(mo, lam)
            r.update({'run_id': d.name, 'which': tag, 'en_res': ch['en_res']})
            r['class'] = classify(r)
            long_rows.append(r)
            rows_c[tag] = r
            print(f"{d.name} @{tag}({lam:.1f}): f_ED={r['f_ED']:.3f} "
                  f"f_MD={r['f_MD']:.3f} f_EQ={r['f_EQ']:.3f} "
                  f"f_MQ={r['f_MQ']:.3f} [{r['class']}]", flush=True)
        base = rows_c['lam0']
        base['old_EDEQ_frac'] = led.get('EDEQ_frac')
        base['old_MD_frac'] = led.get('MD_frac')
        ledger.append(base)
    for path, rows in [(RESULTS / 'families_at_wavelengths.csv', long_rows),
                       (RESULTS / 'candidate_ledger_v2.csv', ledger)]:
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print('FAMILIES_DONE', flush=True)


# ---------------------------------------------------------------------------
# Joint shared-pole fit + adaptive refinement
# ---------------------------------------------------------------------------

def joint_pole_fit(lam_nm, t, r, window=None):
    """s(om) = c0 + c1*(om-om_bar) + a/(om-omp) for s in {t, r}, shared
    complex pole omp. Returns dict with lam_pole, Q, residual, ok."""
    from scipy.optimize import least_squares
    om = 2 * math.pi * C0 / (np.asarray(lam_nm) * 1e-9)
    idx = np.argsort(om)
    om, t, r = om[idx], np.asarray(t)[idx], np.asarray(r)[idx]
    om_bar = om.mean()
    sc = om.max() - om.min()

    def unpack(p):
        c0t = p[0] + 1j * p[1]; c1t = (p[2] + 1j * p[3]) / sc
        at = (p[4] + 1j * p[5]) * sc
        c0r = p[6] + 1j * p[7]; c1r = (p[8] + 1j * p[9]) / sc
        ar = (p[10] + 1j * p[11]) * sc
        omp = om_bar + p[12] * sc + 1j * p[13] * sc
        return c0t, c1t, at, c0r, c1r, ar, omp

    def resid(p):
        c0t, c1t, at, c0r, c1r, ar, omp = unpack(p)
        mt = c0t + c1t * (om - om_bar) + at / (om - omp)
        mr = c0r + c1r * (om - om_bar) + ar / (om - omp)
        return np.concatenate([(mt - t).real, (mt - t).imag,
                               (mr - r).real, (mr - r).imag])

    best = None
    for g_om in (0.0, 0.15, -0.15):
        for g_g in (0.02, 0.005):
            p0 = [t.real.mean(), t.imag.mean(), 0, 0, 0.01, 0,
                  r.real.mean(), r.imag.mean(), 0, 0, 0.01, 0, g_om, -g_g]
            try:
                res = least_squares(resid, p0, max_nfev=40000)
                if best is None or res.cost < best.cost:
                    best = res
            except Exception:
                continue
    if best is None:
        return {'ok': False}
    c0t, c1t, at, c0r, c1r, ar, omp = unpack(best.x)
    lam_pole = 2 * math.pi * C0 / omp.real * 1e9
    Q = omp.real / (-2 * omp.imag) if omp.imag < 0 else np.inf
    rms = math.sqrt(2 * best.cost / len(om) / 4)
    scale = max(np.abs(t).max(), np.abs(r).max())
    # crude CI from JtJ
    try:
        J = best.jac
        cov = np.linalg.pinv(J.T @ J) * (2 * best.cost / max(len(J) - 14, 1))
        dg = math.sqrt(abs(cov[13, 13])) * sc
        dQ = Q * dg / abs(omp.imag) if omp.imag != 0 else np.inf
    except Exception:
        dQ = np.nan
    inside = lam_nm.min() <= lam_pole <= lam_nm.max()
    return {'ok': True, 'lam_pole': lam_pole, 'Q': Q, 'dQ': dQ,
            'rms_rel': rms / scale, 'inside': bool(inside),
            'gamma': -omp.imag}


def adaptive_qrefine(rho, P, h, lam_seed, tag, max_rounds=5, order=ORDER_A):
    """Refine sampling until >=20 samples inside the fitted FWHM; joint fit;
    stability gates. Returns (fit record, samples list)."""
    pts = {}

    def sample(lams):
        for lam in lams:
            key = round(float(lam), 4)
            if key in pts:
                continue
            mo, ch = decompose_at(rho, P, h, float(lam), order=order,
                                  n_xy=32, nz=5)
            pts[key] = {'t': ch['txx'], 'r': ch['rxx'], 'en': ch['en_res'],
                        'my': complex(mo['my']), 'Qxz': complex(mo['Qxz']),
                        'px': complex(mo['px'])}

    span, step = 12.0, 0.5
    center = lam_seed
    fit = {'ok': False}
    for rnd in range(max_rounds):
        sample(np.arange(center - span, center + span + 1e-9, step))
        keys = sorted(k for k in pts if center - span <= k <= center + span)
        lam_a = np.array(keys)
        t = np.array([pts[k]['t'] for k in keys])
        r = np.array([pts[k]['r'] for k in keys])
        fit = joint_pole_fit(lam_a, t, r)
        if not fit.get('ok') or not fit.get('inside', False):
            span *= 1.5
            step = span / 30
            continue
        fwhm_nm = (fit['lam_pole'] * 1e-9) ** 2 * 2 * fit['gamma'] / (2 * math.pi * C0) * 1e9
        n_in = int(np.sum(np.abs(lam_a - fit['lam_pole']) < fwhm_nm / 2))
        print(f'  {tag} round {rnd}: pole {fit["lam_pole"]:.3f} nm '
              f'Q={fit["Q"]:.0f} fwhm={fwhm_nm:.3f} nm samples-in-fwhm={n_in} '
              f'rms={fit["rms_rel"]:.1e}', flush=True)
        if n_in >= 20 and step <= fwhm_nm / 10:
            break
        center = fit['lam_pole']
        span = max(3 * fwhm_nm, 1.0)
        step = fwhm_nm / 12
    # stability: refit on 2 window widths
    fwhm_nm = (fit['lam_pole'] * 1e-9) ** 2 * 2 * fit['gamma'] / (2 * math.pi * C0) * 1e9 \
        if fit.get('ok') else np.nan
    stab = np.nan
    if fit.get('ok'):
        qs = []
        for wmult in (1.5, 2.5):
            keys = sorted(k for k in pts
                          if abs(k - fit['lam_pole']) <= wmult * fwhm_nm)
            if len(keys) >= 10:
                f2 = joint_pole_fit(np.array(keys),
                                    np.array([pts[k]['t'] for k in keys]),
                                    np.array([pts[k]['r'] for k in keys]))
                if f2.get('ok'):
                    qs.append(f2['Q'])
        if len(qs) == 2 and all(np.isfinite(qs)):
            stab = abs(qs[0] - qs[1]) / max(qs)
    # energy gate: worst |T+R-1| among points inside the fit-relevant
    # window (2.5x FWHM, matching the widest stability refit); fall back
    # to all sampled points when there is no resolved window.
    if fit.get('ok') and np.isfinite(fwhm_nm):
        in_keys = [k for k in pts if abs(k - fit['lam_pole']) <= 2.5 * fwhm_nm]
        en_worst = max(pts[k]['en'] for k in in_keys) if in_keys \
            else max(p['en'] for p in pts.values())
    else:
        en_worst = max(p['en'] for p in pts.values())
    keys = sorted(pts)
    n_in = int(np.sum(np.abs(np.array(keys) - fit.get('lam_pole', np.nan)) < fwhm_nm / 2)) \
        if fit.get('ok') else 0
    resolved = bool(fit.get('ok') and fit.get('inside') and n_in >= 20
                    and fit['rms_rel'] < 0.05 and (stab == stab and stab < 0.15)
                    and en_worst < 5e-3)
    rec = {'tag': tag, 'lam_pole': fit.get('lam_pole', np.nan),
           'Q_pole': fit.get('Q', np.nan), 'Q_unc': fit.get('dQ', np.nan),
           'fwhm_nm': fwhm_nm, 'samples_in_fwhm': n_in,
           'rms_rel': fit.get('rms_rel', np.nan), 'window_stability': stab,
           'energy_res_worst': en_worst, 'order': order[0],
           'Q_RESOLVED': resolved}
    return rec, pts


def cmd_qrefine(args):
    AUD.mkdir(parents=True, exist_ok=True)
    targets = []
    for d in sorted(QUAL.iterdir()):
        led_p = d / 'ledger_row.json'
        if not led_p.exists():
            continue
        led = json.loads(led_p.read_text())
        q = led.get('Q_rad', np.nan)
        if led.get('Qfit_ok_rad') and np.isfinite(q) and q > 100 and q < 1e5:
            targets.append((d.name, led['lam_res_rad']))
    targets = targets[args.shard[0]::args.shard[1]]
    out_csv = RESULTS / f'q_validation_v2_shard{args.shard[0]}.csv'
    done = set()
    rows = []
    if out_csv.exists():
        rows = list(csv.DictReader(open(out_csv)))
        done = {r['tag'] for r in rows}
    for name, lam_seed in targets:
        if name in done:
            continue
        cfg = json.loads((PILOT / name / 'config.json').read_text())
        rho = torch.tensor(np.load(PILOT / name / 'rho_binary.npy'),
                           dtype=torch.float32)
        print(f'qrefine {name} (seed {lam_seed:.1f} nm)', flush=True)
        rec, pts = adaptive_qrefine(rho, float(cfg['P']), float(cfg['h']),
                                    float(lam_seed), name)
        rows.append(rec)
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        np.savez_compressed(AUD / f'qrefine_{name}.npz',
                            lam=np.array(sorted(pts)),
                            t=np.array([pts[k]['t'] for k in sorted(pts)]),
                            r=np.array([pts[k]['r'] for k in sorted(pts)]),
                            en=np.array([pts[k]['en'] for k in sorted(pts)]),
                            my=np.array([pts[k]['my'] for k in sorted(pts)]),
                            Qxz=np.array([pts[k]['Qxz'] for k in sorted(pts)]),
                            px=np.array([pts[k]['px'] for k in sorted(pts)]))
    print('QREFINE_SHARD_DONE', flush=True)


def cmd_needle(args):
    name = 'P0750_H0350_seed029'
    cfg = json.loads((PILOT / name / 'config.json').read_text())
    rho = torch.tensor(np.load(PILOT / name / 'rho_binary.npy'),
                       dtype=torch.float32)
    P, h = float(cfg['P']), float(cfg['h'])
    AUD.mkdir(parents=True, exist_ok=True)
    rho_vals = set(np.unique(rho.numpy()).tolist())
    print(f'needle: geometry binary check: values={rho_vals}', flush=True)
    rec, pts = adaptive_qrefine(rho, P, h, 1310.0, name + '_o9', max_rounds=7)
    print('needle [9,9] result:', rec, flush=True)
    rec11, _ = adaptive_qrefine(rho, P, h, rec.get('lam_pole', 1310.0),
                                name + '_o11', max_rounds=5, order=[11, 11])
    print('needle [11,11] result:', rec11, flush=True)
    # complex-frequency pole probe at [9,9] (works even if real-axis
    # sampling cannot resolve the width)
    lam_c = rec.get('lam_pole', 1310.0)
    om0 = 2 * math.pi * C0 / (lam_c * 1e-9)

    import torcwa as _torcwa

    def t_at(om_complex):
        # TORCWA freq is in 1/nm: freq = omega/(2*pi*c) in 1/m, then *1e-9
        freq_nm = om_complex / (2 * math.pi * C0) * 1e-9
        lam_re = 2 * math.pi * C0 / om_complex.real * 1e9
        eps_si = core.si_eps(float(lam_re), lossless=True)
        eps_t = torch.tensor(eps_si, dtype=core.SIM_DTYPE)
        sim = _torcwa.rcwa(freq=complex(freq_nm), order=list(ORDER_A),
                           L=[P, P], dtype=core.SIM_DTYPE, device=core.DEVICE)
        sim.add_input_layer(eps=core.SUBSTRATE_EPS)
        sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
        sim.add_layer(thickness=h, eps=rho * eps_t + (1.0 - rho) * 1.0)
        sim.solve_global_smatrix()
        sim.source_planewave(amplitude=[1.0, 0.0], direction='forward')
        amps = core.channel_amplitudes(sim)
        return complex(amps['txx'])

    try:
        g = rec.get('fwhm_nm', 0.1)
        g_om = om0 * (g / lam_c) / 2 if np.isfinite(g) and 0 < g < 10 \
            else om0 * 1e-5

        def inv_t(om_c):
            # Newton root-finds 1/t -> 0 (pole of t). A Fano transmission
            # ZERO nearby makes t = 0 exactly; floor |t| so the probe is
            # repelled from the zero instead of crashing.
            t = t_at(om_c)
            if not (np.isfinite(t.real) and np.isfinite(t.imag)) \
                    or abs(t) < 1e-14:
                t = complex(1e-14, 0.0)
            return 1.0 / t

        om = om0 - 1j * g_om
        converged = False
        for it in range(8):
            d = g_om * 0.3
            f0, fp, fm = inv_t(om), inv_t(om + d), inv_t(om - d)
            der = (fp - fm) / (2 * d)
            if abs(der) == 0 or not np.isfinite(abs(der)):
                break
            step = f0 / der
            if abs(step) > 0.01 * om0:
                step *= 0.01 * om0 / abs(step)
            om = om - step
            if abs(step) < 1e-9 * om0:
                converged = True
                break
        tfin = t_at(om)
        Qc = om.real / (-2 * om.imag) if om.imag < 0 else np.inf
        lamp = 2 * math.pi * C0 / om.real * 1e9
        print(f'needle complex-frequency pole: lam={lamp:.4f} nm '
              f'Q_pole={Qc:.0f} |t(om_p)|={abs(tfin):.3e} '
              f'converged={converged}', flush=True)
        rec['complex_freq_Q'] = float(Qc)
        rec['complex_freq_lam'] = float(lamp)
        rec['complex_freq_converged'] = bool(converged)
        rec['complex_freq_abs_t'] = float(abs(tfin))
    except Exception as e:
        print(f'needle complex-frequency probe failed: {e}', flush=True)
    # tiny-perturbation response: single-pixel erosion
    import scipy.ndimage as ndi
    rho_e = torch.tensor(ndi.binary_erosion(rho.numpy() > 0.5).astype(np.float32))
    rec_e, _ = adaptive_qrefine(rho_e, P, h, rec.get('lam_pole', 1310.0),
                                name + '_eroded', max_rounds=4)
    print('needle eroded-geometry result:', rec_e, flush=True)
    (AUD / 'needle_forensics.json').write_text(json.dumps(
        {'o9': rec, 'o11': rec11, 'eroded': rec_e}, indent=1, default=float))
    print('NEEDLE_DONE', flush=True)


def cmd_matrices(args):
    name = 'P0750_H0250_seed011'
    cfg = json.loads((PILOT / name / 'config.json').read_text())
    rho = torch.tensor(np.load(PILOT / name / 'rho_binary.npy'),
                       dtype=torch.float32)
    P, h = float(cfg['P']), float(cfg['h'])
    AUD.mkdir(parents=True, exist_ok=True)
    out = {}
    # order matrix: refit pole + fractions at each order
    for order in [[11, 11], [13, 13], [15, 15]]:
        rec, _ = adaptive_qrefine(rho, P, h, 1331.0, f'{name}_o{order[0]}',
                                  max_rounds=4, order=order)
        mo, ch = decompose_at(rho, P, h, rec.get('lam_pole', 1331.0),
                              order=order)
        fr = family_row(mo, rec.get('lam_pole', 1331.0))
        out[f'order_{order[0]}'] = {**rec,
                                    **{k: fr[k] for k in ('f_ED', 'f_MD',
                                       'f_EQ', 'f_MQ', 'px_given_ED',
                                       'my_given_MD', 'Qxz_given_EQ')}}
        print(f"matrices order {order}: {out[f'order_{order[0]}']}", flush=True)
    # grid matrix (fractions only; solve is grid-independent)
    lam_p = out['order_13'].get('lam_pole', 1331.0)
    for (nxy, nz) in [(32, 5), (48, 9), (64, 15), (96, 21)]:
        mo, ch = decompose_at(rho, P, h, lam_p, n_xy=nxy, nz=nz)
        fr = family_row(mo, lam_p)
        out[f'grid_{nxy}x{nz}'] = {k: fr[k] for k in ('f_ED', 'f_MD', 'f_EQ',
                                   'f_MQ', 'Cpx_total_fraction',
                                   'Cmy_total_fraction', 'CQxz_total_fraction')}
    # origin matrix
    for tag, o in [('center', None), ('x+P8', (P/2+P/8, P/2, h/2)),
                   ('y+P8', (P/2, P/2+P/8, h/2)), ('z+H4', (P/2, P/2, h/2+h/4))]:
        mo, ch = decompose_at(rho, P, h, lam_p, origin_nm=o)
        fr = family_row(mo, lam_p)
        out[f'origin_{tag}'] = {k: fr[k] for k in ('f_ED', 'f_MD', 'f_EQ',
                                'f_MQ', 'Cpx_total_fraction',
                                'Cmy_total_fraction', 'CQxz_total_fraction')}
    (AUD / 'champion_matrices.json').write_text(json.dumps(out, indent=1,
                                                           default=float))
    print('MATRICES_DONE', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['families', 'qrefine', 'needle', 'matrices'])
    ap.add_argument('--shard', type=int, nargs=2, default=[0, 1])
    ap.add_argument('--threads', type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    {'families': cmd_families, 'qrefine': cmd_qrefine,
     'needle': cmd_needle, 'matrices': cmd_matrices}[args.cmd](args)


if __name__ == '__main__':
    main()
