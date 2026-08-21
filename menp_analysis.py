"""
menp_analysis.py
================

Analysis of the MENP qualification campaign (menp_torcwa_campaign.py) for
the three frozen candidates: builds per-candidate spectra figures,
convergence/origin/substrate/material tables, extracts peak positions,
linewidths, relative phases, and emits the quantitative inputs for the
12 qualification questions and the final 3-candidate ranking.

Cross-section conventions: per-component dipole radiation weights
    C_px = k^4 |px|^2 / (6 pi eps0^2),   C_mz = k^4 |mz|^2 / (6 pi eps0^2 c^2)
(the same vacuum-background formal weights as MENP; per unit cell).
"""

import csv
import json
from pathlib import Path

import numpy as np

from menp_port import C0, EPS0

RESULTS = Path(__file__).resolve().parent / 'results_ed_md_coexcitation'
MENP_OUT = RESULTS / 'menp'
CANDS = ['P0870_H0100_seed029', 'P0830_H0100_seed011', 'P0900_H0100_seed011']


def load_csv(path):
    if not Path(path).exists():
        return []
    return list(csv.DictReader(open(path)))


def cplx(row, base):
    return float(row[f'{base}_re']) + 1j * float(row[f'{base}_im'])


def comp_weights(row):
    """Per-component dipole radiation weights + family cross sections."""
    lam = float(row['lam_nm']) * 1e-9
    k = 2 * np.pi / lam
    cE = k ** 4 / (6 * np.pi * EPS0 ** 2)
    cM = cE / C0 ** 2
    out = {'lam_nm': float(row['lam_nm'])}
    for c in 'xyz':
        out[f'C_p{c}'] = cE * abs(cplx(row, f'p{c}')) ** 2
        out[f'C_m{c}'] = cM * abs(cplx(row, f'm{c}')) ** 2
    for k2 in ['Cp', 'Cm', 'CQe', 'CQm', 'CT', 'CpT',
               'S_ED_3slice', 'S_MD_3slice', 'S_ED_vol', 'S_MD_vol']:
        out[k2] = float(row[k2]) if row.get(k2, '') not in ('', 'nan') else np.nan
    out['px'] = cplx(row, 'px')
    out['mz'] = cplx(row, 'mz')
    out['rel_phase_deg'] = np.degrees(np.angle(out['px']) - np.angle(out['mz']))
    return out


def peak_and_fwhm(lam, v):
    i = int(np.argmax(v))
    interior = 0 < i < len(v) - 1
    lam_pk = lam[i]
    if interior:
        y0, y1, y2 = v[i - 1], v[i], v[i + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-300:
            lam_pk = lam[i] + 0.5 * (y0 - y2) / den * (lam[1] - lam[0])
    half = v[i] / 2
    above = lam[v >= half]
    fwhm = (above[-1] - above[0]) if len(above) > 1 else np.nan
    return float(lam_pk), float(fwhm), bool(interior)


def lorentz_fit(lam, v):
    """Single-Lorentzian fit; returns (center, fwhm, ok)."""
    from scipy.optimize import curve_fit
    def L(x, a, x0, g, b):
        return a * (g / 2) ** 2 / ((x - x0) ** 2 + (g / 2) ** 2) + b
    i = int(np.argmax(v))
    try:
        popt, _ = curve_fit(L, lam, v, p0=[v[i], lam[i], 4.0, np.min(v)],
                            maxfev=20000)
        return float(popt[1]), float(abs(popt[2])), True
    except Exception:
        return np.nan, np.nan, False


def analyze_candidate(name):
    out = MENP_OUT / name
    fine = [comp_weights(r) for r in load_csv(out / 'fine_decomposition.csv')]
    if not fine:
        print(f'{name}: no fine data yet')
        return None
    lam = np.array([r['lam_nm'] for r in fine])
    g = lambda k: np.array([r[k] for r in fine])

    res = {'name': name}
    # peaks
    for key, tag in [('C_px', 'px'), ('C_mz', 'mz'),
                     ('S_ED_3slice', 'SED'), ('S_MD_3slice', 'SMD')]:
        pk, fw, interior = peak_and_fwhm(lam, g(key))
        c, w, ok = lorentz_fit(lam, g(key))
        res[f'peak_{tag}'] = pk
        res[f'fwhm_{tag}'] = fw
        res[f'lorentz_center_{tag}'] = c
        res[f'lorentz_fwhm_{tag}'] = w
        res[f'interior_{tag}'] = interior
    res['split_px_mz_nm'] = abs(res['peak_px'] - res['peak_mz'])
    res['split_proxy_nm'] = abs(res['peak_SED'] - res['peak_SMD'])

    # values at target and at joint peak
    i0 = int(np.argmin(np.abs(lam - 1332.5)))
    fam = {k: g(k) for k in ['Cp', 'Cm', 'CQe', 'CQm', 'CT']}
    famsum = fam['Cp'] + fam['Cm'] + fam['CQe'] + fam['CQm']
    px2 = g('C_px'); py2 = g('C_py'); pz2 = g('C_pz')
    mx2 = g('C_mx'); my2 = g('C_my'); mz2 = g('C_mz')
    res['at_target'] = {
        'C_px': px2[i0], 'C_mz': mz2[i0],
        'px_frac_of_ED': px2[i0] / (px2[i0] + py2[i0] + pz2[i0]),
        'mz_frac_of_MD': mz2[i0] / (mx2[i0] + my2[i0] + mz2[i0]),
        'Cp_frac': fam['Cp'][i0] / famsum[i0],
        'Cm_frac': fam['Cm'][i0] / famsum[i0],
        'CQe_frac': fam['CQe'][i0] / famsum[i0],
        'CQm_frac': fam['CQm'][i0] / famsum[i0],
        'CT_over_Cp': fam['CT'][i0] / fam['Cp'][i0],
        'balance_px_mz': min(px2[i0], mz2[i0]) / max(px2[i0], mz2[i0]),
        'rel_phase_deg': g('rel_phase_deg')[i0],
    }
    # proxy tracking: normalized-shape correlation
    def shapecorr(a, b):
        a = (a - a.mean()) / (a.std() + 1e-300)
        b = (b - b.mean()) / (b.std() + 1e-300)
        return float(np.mean(a * b))
    res['corr_Cpx_SED'] = shapecorr(px2, g('S_ED_3slice'))
    res['corr_Cmz_SMD'] = shapecorr(mz2, g('S_MD_3slice'))
    res['corr_CQe_SED'] = shapecorr(fam['CQe'], g('S_ED_3slice'))
    res['rel_phase_std_deg'] = float(np.std(np.unwrap(
        np.radians(g('rel_phase_deg'))) * 180 / np.pi))

    # convergence tables
    for stage, keycol in [('order_scan', 'order'), ('grid_scan', 'n_xy'),
                          ('origin_scan', 'origin'), ('substrate_scan', 'substrate'),
                          ('material_scan', 'n_si')]:
        rows = load_csv(out / f'{stage}.csv')
        res[stage] = []
        for r in rows:
            cw = comp_weights(r)
            entry = {keycol: r.get(keycol, ''),
                     'lam_nm': cw['lam_nm'],
                     'C_px': cw['C_px'], 'C_mz': cw['C_mz'],
                     'Cp': cw['Cp'], 'Cm': cw['Cm'], 'CQe': cw['CQe'],
                     'CQm': cw['CQm'], 'CT': cw['CT'],
                     'rel_phase_deg': cw['rel_phase_deg']}
            if stage == 'grid_scan':
                entry['nz'] = r['nz']
            res[stage].append(entry)
    res['_fine'] = fine
    return res


def plot_candidate(res):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    name = res['name']
    fine = res['_fine']
    lam = np.array([r['lam_nm'] for r in fine])
    g = lambda k: np.array([r[k] for r in fine])
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))

    ax = axs[0, 0]
    for k, style, col in [('C_px', '-', 'tab:red'), ('C_py', ':', 'salmon'),
                          ('C_pz', '--', 'darkred'), ('C_mz', '-', 'tab:blue'),
                          ('C_mx', ':', 'lightblue'), ('C_my', '--', 'navy')]:
        ax.semilogy(lam, np.maximum(g(k), 1e-24), style, color=col, label=k, lw=1.4)
    ax.axvline(1332.5, color='k', ls='--', lw=0.8)
    ax.set_title('Cartesian dipole radiation weights')
    ax.set_xlabel('wavelength (nm)'); ax.legend(fontsize=7, ncol=2)

    ax = axs[0, 1]
    famsum = g('Cp') + g('Cm') + g('CQe') + g('CQm')
    for k, col in [('Cp', 'tab:red'), ('Cm', 'tab:blue'), ('CQe', 'tab:orange'),
                   ('CQm', 'tab:purple'), ('CT', 'tab:green')]:
        ax.plot(lam, g(k) / famsum, color=col, label=k + '/sum', lw=1.5)
    ax.axvline(1332.5, color='k', ls='--', lw=0.8)
    ax.set_title('multipole family fractions (exact ME; CT diagnostic)')
    ax.set_xlabel('wavelength (nm)'); ax.set_ylim(0, 1); ax.legend(fontsize=8)

    ax = axs[1, 0]
    for k, col, lab in [('C_px', 'tab:red', 'C_px (norm)'),
                        ('S_ED_3slice', 'darkred', 'S_ED proxy (norm)'),
                        ('C_mz', 'tab:blue', 'C_mz (norm)'),
                        ('S_MD_3slice', 'navy', 'S_MD proxy (norm)')]:
        v = g(k); ax.plot(lam, v / v.max(), color=col, lw=1.4, label=lab,
                          ls='-' if k.startswith('C') else '--')
    ax.axvline(1332.5, color='k', ls='--', lw=0.8)
    ax.set_title('multipole spectra vs near-field proxies (normalized)')
    ax.set_xlabel('wavelength (nm)'); ax.legend(fontsize=8)

    ax = axs[1, 1]
    ax.plot(lam, np.unwrap(np.radians(g('rel_phase_deg'))) * 180 / np.pi,
            'k-', lw=1.5)
    ax.axvline(1332.5, color='k', ls='--', lw=0.8)
    ax.set_title('relative phase arg(px) - arg(mz)')
    ax.set_xlabel('wavelength (nm)'); ax.set_ylabel('degrees')

    fig.suptitle(f'{name} - exact multipole decomposition ([13,13], binary, '
                 f'96x96x21)', fontsize=12)
    fig.tight_layout()
    fig.savefig(MENP_OUT / name / 'decomposition_summary.png', dpi=140,
                bbox_inches='tight')
    plt.close(fig)


def main():
    summary = {}
    for name in CANDS:
        res = analyze_candidate(name)
        if res is None:
            continue
        plot_candidate(res)
        summary[name] = {k: v for k, v in res.items() if k != '_fine'}
        t = res['at_target']
        print(f"\n===== {name} =====")
        print(f"  peak(C_px) = {res['peak_px']:.2f} nm (FWHM {res['fwhm_px']:.2f}), "
              f"peak(C_mz) = {res['peak_mz']:.2f} nm (FWHM {res['fwhm_mz']:.2f})")
        print(f"  px-mz peak splitting = {res['split_px_mz_nm']:.3f} nm "
              f"(proxy splitting {res['split_proxy_nm']:.3f} nm)")
        print(f"  at 1332.5 nm: px/ED = {t['px_frac_of_ED']:.3f}, "
              f"mz/MD = {t['mz_frac_of_MD']:.3f}")
        print(f"  families: Cp {t['Cp_frac']*100:.1f}%  Cm {t['Cm_frac']*100:.1f}%  "
              f"CQe {t['CQe_frac']*100:.1f}%  CQm {t['CQm_frac']*100:.1f}%  "
              f"CT/Cp {t['CT_over_Cp']:.2f}")
        print(f"  balance C_px/C_mz = {t['balance_px_mz']:.3f}, "
              f"rel phase = {t['rel_phase_deg']:.1f} deg "
              f"(std over line {res['rel_phase_std_deg']:.1f} deg)")
        print(f"  proxy correlation: corr(C_px,S_ED) = {res['corr_Cpx_SED']:.3f}, "
              f"corr(C_mz,S_MD) = {res['corr_Cmz_SMD']:.3f}, "
              f"corr(CQe,S_ED) = {res['corr_CQe_SED']:.3f}")
    (MENP_OUT / 'qualification_summary.json').write_text(
        json.dumps(summary, indent=1, default=float))
    print(f"\nwrote {MENP_OUT / 'qualification_summary.json'}")


if __name__ == '__main__':
    main()
