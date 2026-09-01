"""Phase G: low-T scan across ALL existing qualified data (no new solves).

Sources:
  - 18 Stage-A pilot candidates: ed_eq_causality_campaign qualify
    spectra_main.csv (1292.5-1372.5, 1 nm; T/R + complex p/m/Qe moments;
    MQ available only via mqspec for 5 candidates - flagged).
  - P0550 thickness family (phase_sweep, at lam0 and coarse lambda map).
  - P0750 master scan (full 4-family rows).
  - Stage-A detuning trajectory of P0750 (alpha family).
  Old ED/MD campaign candidates are EXCLUDED: different (clamped)
  material model - not comparable; noted in the report.

Writes results/existing_candidate_tmin_ranking.csv, printed ranked.
"""
import glob
import math

import numpy as np
import pandas as pd

import stage_core as sc

EPS0, C0 = sc.EPS0, sc.C0
QUAL = sc.CAMP / 'results' / 'qualify'


def fam3(row, lam):
    k = 2 * math.pi / (lam * 1e-9)
    cE = k ** 4 / (6 * math.pi * EPS0 ** 2)
    def a2(stem):
        return row[f'{stem}_re'] ** 2 + row[f'{stem}_im'] ** 2
    Cp = cE * (a2('px') + a2('py') + a2('pz'))
    Cm = cE / C0 ** 2 * (a2('mx') + a2('my') + a2('mz'))
    CQ = cE / 120 * k ** 2 * (a2('Qxx') + a2('Qyy') + a2('Qzz')
                              + 2 * (a2('Qxy') + a2('Qxz') + a2('Qyz')))
    tot = Cp + Cm + CQ
    comps = {'px': cE * a2('px') / tot, 'my': cE / C0 ** 2 * a2('my') / tot,
             'Qxz': cE / 120 * k ** 2 * 2 * a2('Qxz') / tot}
    dom = sorted(comps, key=comps.get, reverse=True)[:2]
    return Cp / tot, Cm / tot, CQ / tot, '+'.join(dom)


def main():
    rows = []
    for d in sorted(glob.glob(str(QUAL / '*/spectra_main.csv'))):
        name = d.split('/')[-2]
        df = pd.read_csv(d)
        i = int(df['T'].idxmin())
        r = df.loc[i]
        lam = float(r.lam_nm)
        fED, fMD, fEQ, dom = fam3(r, lam)
        mq = sc.RESULTS / 'audit' / f'mqspec_{name}.csv'
        fMQ = np.nan
        if mq.exists():
            m = pd.read_csv(mq)
            j = (m.lam_nm - lam).abs().idxmin()
            if abs(m.lam_nm[j] - lam) <= 1.01:
                fMQ = float(m.f_MQ[j])
        A = 1 - float(r['T']) - float(r['R'])
        rows.append({'candidate': name, 'source': 'stageA_qualify_1nm',
                     'lam_Tmin': lam, 'Tmin': float(r['T']),
                     'R': float(r['R']), 'A': A,
                     'en_res': abs(A) if np.isnan(fMQ) else abs(A),
                     'f_ED_3fam': fED, 'f_MD_3fam': fMD, 'f_EQ_3fam': fEQ,
                     'f_MQ': fMQ, 'dominant': dom,
                     'confidence': 'MEDIUM (1-nm grid; lossy-main scenario; '
                                   'A includes numerical residual)'})
    # P0750 master (1-nm full rows, exact 4-family)
    d = pd.read_csv(sc.RESULTS / 'p0750_multipole_spectra.csv')
    i = int(d['T'].idxmin())
    r = d.loc[i]
    rows.append({'candidate': 'P0750_H0250_seed011', 'source':
                 'stageB_master_1nm', 'lam_Tmin': float(r.lam_nm),
                 'Tmin': float(r['T']), 'R': float(r['R']),
                 'A': 1 - float(r['T']) - float(r['R']),
                 'en_res': float(r.en_res),
                 'f_ED_3fam': np.nan, 'f_MD_3fam': np.nan,
                 'f_EQ_3fam': np.nan, 'f_MQ': float(r.f_MQ),
                 'dominant': 'my+Qxz (exact: f_MD %.2f f_EQ %.2f)' % (
                     r.f_MD, r.f_EQ),
                 'confidence': 'HIGH pending sub-nm refinement (this stage)'})
    # P0550 h-family at lam0
    sw = pd.read_csv(sc.RESULTS / 'p0550_phase_sweep.csv')
    f = sw[sw.lam_nm == 1332.5]
    i = int(f['T'].idxmin())
    r = f.loc[i]
    rows.append({'candidate': f'P0550 h-family (h={r.h_nm:g})',
                 'source': 'stageB_hsweep_lam0', 'lam_Tmin': 1332.5,
                 'Tmin': float(r['T']), 'R': float(r['R']),
                 'A': 1 - float(r['T']) - float(r['R']),
                 'en_res': float(r.en_res), 'f_ED_3fam': np.nan,
                 'f_MD_3fam': np.nan, 'f_EQ_3fam': np.nan,
                 'f_MQ': float(r.f_MQ),
                 'dominant': 'px+Qxz (exact: f_ED %.2f f_EQ %.2f)' % (
                     r.f_ED, r.f_EQ),
                 'confidence': 'HIGH ([9,9]; composition partially degraded '
                               'at this h)'})
    out = pd.DataFrame(rows).sort_values('Tmin')
    out.to_csv(sc.RESULTS / 'existing_candidate_tmin_ranking.csv',
               index=False)
    pd.set_option('display.width', 250)
    print(out[['candidate', 'source', 'lam_Tmin', 'Tmin', 'R', 'A', 'f_MQ',
               'dominant']].head(12).to_string(index=False,
               float_format=lambda v: f'{v:.4f}'))
    print('TMIN_RANKING_DONE')


if __name__ == '__main__':
    main()
