"""Phases 2-6 analysis on the master scan (no new solves).

Derives: empirical port couplings g_up/g_dn vs TORCWA (Phase 2, coupling
verification without free-space assumptions), background split (Phase 3),
ladder-model reconstructions A-D with residuals (Phase 4), Kerker /
directionality diagnostics with port-impedance-corrected powers (Phase 5),
and broadband band-width metrics (Phase 6).

Conventions:
  t_sc = txx - tbg, r_sc = rxx - rbg  (TORCWA power-normalized amplitudes)
  E_up/E_dn: exact induced-current channel integrals (field units)
  ladder: E_up ~ even_px + odd_m + odd_Q + eps2 + ..., E_dn ~ even - odd + eps2
  eps2 = -(Z0/2A)(-k^2/2) I2   (ALL 2nd-order even content: Qm_yz +
        octupole + mean-radius corrections - labeled as such, not "MQ")
  g_up = E_up/t_sc, g_dn = E_dn/r_sc: single complex constants of the
        layered-port normalization, fitted as band medians; per-lambda
        deviation quantifies the layered-background correction.
"""
import cmath
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import stage_core as sc

R = sc.RESULTS


def merge(cand, nsh):
    parts = [pd.read_csv(R / f'scan_{cand}_shard{i}.csv') for i in range(nsh)]
    df = pd.concat(parts).sort_values('lam_nm').reset_index(drop=True)
    df.to_csv(R / f'{cand}_multipole_spectra.csv', index=False)
    return df


def cx(df, stem):
    return df[stem + '_re'].to_numpy() + 1j * df[stem + '_im'].to_numpy()


def analyze(cand='p0550'):
    nsh = 3 if cand == 'p0550' else 1
    df = merge(cand, nsh)
    lam = df['lam_nm'].to_numpy()
    k = 2 * math.pi / (lam * 1e-9)
    txx, rxx = cx(df, 'txx'), cx(df, 'rxx')
    tbg, rbg = cx(df, 'tbg'), cx(df, 'rbg')
    Eu, Ed = cx(df, 'E_up'), cx(df, 'E_dn')
    ev, oQ, om = cx(df, 'even_px'), cx(df, 'odd_Q'), cx(df, 'odd_m')
    I2 = cx(df, 'I2')
    P_nm = 550.0 if cand == 'p0550' else 750.0
    A_cell = (P_nm * 1e-9) ** 2
    e2 = -(sc.Z0 / (2 * A_cell)) * (-k ** 2 / 2) * I2

    t_sc, r_sc = txx - tbg, rxx - rbg
    g_up_l = Eu / t_sc
    g_dn_l = Ed / r_sc
    g_up = np.median(g_up_l.real) + 1j * np.median(g_up_l.imag)
    g_dn = np.median(g_dn_l.real) + 1j * np.median(g_dn_l.imag)

    # exactness of the ladder vs the exact integral (unit-consistent):
    lad_up = ev + om + oQ + e2
    lad_dn = ev - om - oQ + e2
    ladder_resid_up = np.abs(Eu - lad_up) / np.maximum(np.abs(Eu), 1e-12)
    ladder_resid_dn = np.abs(Ed - lad_dn) / np.maximum(np.abs(Ed), 1e-12)

    # channel-normalized amplitudes in TORCWA t-units:
    A_ED_top, A_EQ_top, A_MD_top = ev / g_up, oQ / g_up, om / g_up
    A_ED_bot, A_EQ_bot, A_MD_bot = ev / g_dn, -oQ / g_dn, -om / g_dn
    dphi_top = np.degrees(np.angle(A_ED_top / A_EQ_top))
    dphi_bot = np.degrees(np.angle(A_ED_bot / A_EQ_bot))

    # Phase 4 models (TORCWA units, background added):
    models = {
        'A_EDonly': (tbg + ev / g_up, rbg + ev / g_dn),
        'B_ED_EQ': (tbg + (ev + oQ) / g_up, rbg + (ev - oQ) / g_dn),
        'C_ED_EQ_MD': (tbg + (ev + oQ + om) / g_up,
                       rbg + (ev - oQ - om) / g_dn),
        'D_plus_2nd': (tbg + lad_up / g_up, rbg + lad_dn / g_dn),
        'X_exact_integral': (tbg + Eu / g_up, rbg + Ed / g_dn),
        'noED': (tbg + (oQ + om + e2) / g_up, rbg + (-oQ - om + e2) / g_dn),
        'noEQ': (tbg + (ev + om + e2) / g_up, rbg + (ev - om + e2) / g_dn),
    }
    scale_t, scale_r = np.mean(np.abs(txx)), np.mean(np.abs(rxx))
    # error_*: relative to that channel's own mean amplitude (harsh for the
    # small-|r| band); error_*_abs: per unit incident amplitude (the
    # physically comparable scale across channels).
    errs = {name: {'error_t': float(np.mean(np.abs(tm - txx)) / scale_t),
                   'error_r': float(np.mean(np.abs(rm - rxx)) / scale_r),
                   'error_t_abs': float(np.mean(np.abs(tm - txx))),
                   'error_r_abs': float(np.mean(np.abs(rm - rxx)))}
            for name, (tm, rm) in models.items()}
    mod_df = {'lam_nm': lam, 'T_full': np.abs(txx) ** 2,
              'R_full': np.abs(rxx) ** 2}
    for name, (tm, rm) in models.items():
        mod_df[f'T_{name}'] = np.abs(tm) ** 2
        mod_df[f'R_{name}'] = np.abs(rm) ** 2
    pd.DataFrame(mod_df).to_csv(R / f'{cand}_models_tr.csv', index=False)

    # Phase 5 directionality (scattered power, port-impedance corrected;
    # air index 1 up, n_sub down; field amplitudes Eu, Ed):
    P_top = 1.0 * np.abs(Eu) ** 2
    P_bot = sc.N_SUB * np.abs(Ed) ** 2
    eta_dir = (P_top - P_bot) / np.maximum(P_top + P_bot, 1e-30)
    # intra-scattered ED-EQ cancellation measure per port:
    xi_top = np.abs(A_ED_top + A_EQ_top) / np.maximum(
        np.abs(A_ED_top) + np.abs(A_EQ_top), 1e-30)
    xi_bot = np.abs(A_ED_bot + A_EQ_bot) / np.maximum(
        np.abs(A_ED_bot) + np.abs(A_EQ_bot), 1e-30)
    ratio_top = np.abs(A_EQ_top) / np.maximum(np.abs(A_ED_top), 1e-30)

    out = pd.DataFrame({
        'lam_nm': lam,
        'T': df['T'], 'R': df['R'], 'en_res': df['en_res'],
        'abs_t_sc': np.abs(t_sc), 'abs_r_sc': np.abs(r_sc),
        'abs_tbg': np.abs(tbg), 'abs_rbg': np.abs(rbg),
        'arg_tbg_deg': np.degrees(np.angle(tbg)),
        'arg_rbg_deg': np.degrees(np.angle(rbg)),
        'g_up_abs': np.abs(g_up_l), 'g_up_arg_deg': np.degrees(np.angle(g_up_l)),
        'g_dn_abs': np.abs(g_dn_l), 'g_dn_arg_deg': np.degrees(np.angle(g_dn_l)),
        'A_ED_top_abs': np.abs(A_ED_top), 'A_EQ_top_abs': np.abs(A_EQ_top),
        'A_MD_top_abs': np.abs(A_MD_top),
        'A_ED_bot_abs': np.abs(A_ED_bot), 'A_EQ_bot_abs': np.abs(A_EQ_bot),
        'A_MD_bot_abs': np.abs(A_MD_bot),
        'dphi_top_deg': dphi_top, 'dphi_bot_deg': dphi_bot,
        'sum_top_abs': np.abs(A_ED_top + A_EQ_top),
        'sum_bot_abs': np.abs(A_ED_bot + A_EQ_bot),
        'xi_top': xi_top, 'xi_bot': xi_bot, 'ratio_EQ_ED_top': ratio_top,
        'eta_dir': eta_dir,
        'ladder_resid_up': ladder_resid_up, 'ladder_resid_dn': ladder_resid_dn,
        'f_ED': df['f_ED'], 'f_EQ': df['f_EQ'], 'f_MD': df['f_MD'],
        'f_MQ': df['f_MQ'], 'ED_EQ_balance': df['ED_EQ_balance'],
        'px_given_ED': df['px_given_ED'], 'Qxz_given_EQ': df['Qxz_given_EQ'],
    })
    out.to_csv(R / f'{cand}_channel_amplitudes.csv', index=False)

    # Phase 6 band widths (p0550):
    bands = {}
    def bandwidth(mask):
        if not mask.any():
            return {'nm': 0.0, 'range': None}
        l = lam[mask]
        return {'nm': float(l.max() - l.min()),
                'range': [float(l.min()), float(l.max())],
                'contiguous': bool(np.all(np.diff(np.where(mask)[0]) == 1))}
    m1 = (df['f_ED'] + df['f_EQ'] >= 0.80).to_numpy()
    m2 = ((df['f_ED'] >= 0.20) & (df['f_EQ'] >= 0.20)
          & (df['px_given_ED'] >= 0.80) & (df['Qxz_given_EQ'] >= 0.80)).to_numpy()
    m3 = (df['ED_EQ_balance'] >= 0.8).to_numpy()
    bands['f_ED_plus_f_EQ_ge_0.80'] = bandwidth(m1)
    bands['clean_criterion'] = bandwidth(m1 & m2)
    bands['balance_ge_0.8'] = bandwidth(m3)
    bands['eta_dir_ge_0.6'] = bandwidth(np.asarray(eta_dir >= 0.6))
    bands['eta_dir_le_-0.6'] = bandwidth(np.asarray(eta_dir <= -0.6))
    bands['xi_bot_le_0.3'] = bandwidth(np.asarray(xi_bot <= 0.3))
    bands['xi_top_le_0.3'] = bandwidth(np.asarray(xi_top <= 0.3))

    iRmin, iTmin = int(np.argmin(df['R'])), int(np.argmin(df['T']))
    summ = {
        'cand': cand,
        'g_up': {'abs': abs(g_up), 'arg_deg': math.degrees(cmath.phase(g_up)),
                 'per_lam_abs_dev_max': float(np.max(np.abs(g_up_l - g_up))
                                              / abs(g_up))},
        'g_dn': {'abs': abs(g_dn), 'arg_deg': math.degrees(cmath.phase(g_dn)),
                 'per_lam_abs_dev_max': float(np.max(np.abs(g_dn_l - g_dn))
                                              / abs(g_dn))},
        'model_errors': errs,
        'ladder_resid_up_median': float(np.median(ladder_resid_up)),
        'R_min': {'lam': float(lam[iRmin]), 'R': float(df['R'][iRmin]),
                  'xi_bot': float(xi_bot[iRmin]),
                  'dphi_bot_deg': float(dphi_bot[iRmin])},
        'T_min': {'lam': float(lam[iTmin]), 'T': float(df['T'][iTmin])},
        'eta_dir': {'max': float(np.max(eta_dir)), 'min': float(np.min(eta_dir)),
                    'at_lam0': float(eta_dir[np.argmin(np.abs(lam - 1332.5))])},
        'bands': bands,
        'en_res_worst': float(df['en_res'].max()),
    }
    (R / f'{cand}_channel_summary.json').write_text(
        json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))
    return out, summ


if __name__ == '__main__':
    import sys
    analyze(sys.argv[1] if len(sys.argv) > 1 else 'p0550')
