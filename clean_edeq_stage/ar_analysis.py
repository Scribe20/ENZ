"""AR-audit analysis: merged tables, bandwidths (with clean-ED-EQ
intersections), order/grid/energy qualification, and the summary JSON.
Run after ar_audit rt shards and ar_keypts complete.
"""
import json

import numpy as np
import pandas as pd

import stage_core as sc

R = sc.RESULTS
HS = [225.0, 227.2, 235.0, 250.0]
LAM0 = 1332.5
R_FRESNEL = 0.18699 ** 2


def bands(mask, lams, min_nm=3.0):
    out, start = [], None
    for i, (l, m) in enumerate(zip(lams, mask)):
        if m and start is None:
            start = l
        if (not m or i == len(lams) - 1) and start is not None:
            end = l if m else lams[i - 1]
            if end - start >= min_nm:
                out.append([float(start), float(end)])
            start = None
    return out


def clean_mask_interp(h, lams):
    """Clean-ED-EQ criterion interpolated from the phase-sweep coarse map
    (5-nm lambda, 10-nm h grid; band edges accurate to ~+-3 nm)."""
    sw = pd.read_csv(R / 'p0550_phase_sweep.csv')
    sw = sw[sw.lam_nm % 5 == 0]
    hs = np.sort(sw.h_nm.unique())
    h_lo = hs[hs <= h].max() if (hs <= h).any() else hs.min()
    h_hi = hs[hs >= h].min() if (hs >= h).any() else hs.max()
    w = 0.0 if h_hi == h_lo else (h - h_lo) / (h_hi - h_lo)
    cols = ['f_ED', 'f_EQ', 'px_given_ED', 'Qxz_given_EQ']
    out = {}
    for c in cols:
        lo = sw[sw.h_nm == h_lo].sort_values('lam_nm')
        hi = sw[sw.h_nm == h_hi].sort_values('lam_nm')
        v = (1 - w) * lo[c].to_numpy() + w * hi[c].to_numpy()
        out[c] = np.interp(lams, lo.lam_nm.to_numpy(), v)
    return ((out['f_ED'] >= 0.2) & (out['f_EQ'] >= 0.2)
            & (out['f_ED'] + out['f_EQ'] >= 0.8)
            & (out['px_given_ED'] >= 0.8) & (out['Qxz_given_EQ'] >= 0.8))


def main():
    df = pd.concat([pd.read_csv(R / f'ar_rt_shard{i}.csv') for i in range(4)])
    df = df.drop_duplicates(['case', 'h_nm', 'lam_nm', 'order'])
    df.to_csv(R / 'ar_rt_all.csv', index=False)
    d9 = df[df.order == 9]
    bare = d9[d9.case == 'bare'].sort_values('lam_nm')
    Rb = bare.set_index('lam_nm').f_R

    summ = {'fresnel_check_max_dev': float((Rb - R_FRESNEL).abs().max()),
            'R_bare_analytic': R_FRESNEL, 'per_h': {}, 'orders': {},
            'baselines_at_lam0': {}}

    for case in ('uniform', 'simple'):
        for h in HS:
            p = d9[(d9.case == case) & (np.isclose(d9.h_nm, h))]
            if len(p) < 100:
                continue
            i0 = (p.lam_nm - LAM0).abs().idxmin()
            summ['baselines_at_lam0'][f'{case}_h{h:g}'] = float(p.loc[i0, 'f_R'])

    comp_rows = []
    for h in HS:
        p = d9[(d9.case == 'p0550') & (np.isclose(d9.h_nm, h))] \
            .sort_values('lam_nm')
        lam, Rp = p.lam_nm.to_numpy(), p.f_R.to_numpy()
        Rbv = Rb.reindex(lam).to_numpy()
        clean = clean_mask_interp(h, lam)
        i0, im = int(np.argmin(np.abs(lam - LAM0))), int(np.argmin(Rp))
        en = np.abs(p.f_T.to_numpy() + Rp - 1.0)
        rec = {
            'R_at_lam0': float(Rp[i0]), 'T_at_lam0': float(p.f_T.iloc[i0]),
            'R_min': float(Rp[im]), 'lam_Rmin': float(lam[im]),
            'en_res_at_Rmin': float(en[im]), 'en_res_at_lam0': float(en[i0]),
            'R_backward_at_lam0': float(p.b_R.iloc[i0]),
            'max_abs_Rf_minus_Rb': float(np.max(np.abs(Rp - p.b_R.to_numpy()))),
            'bands_A_R_le_0.05': bands(Rp <= 0.05, lam),
            'bands_B_below_bare': bands(Rp < Rbv, lam),
            'bands_C_half_bare': bands(Rp <= 0.5 * Rbv, lam),
            'bands_clean': bands(clean, lam),
            'bands_clean_and_below_bare': bands(clean & (Rp < Rbv), lam),
            'bands_clean_and_half_bare': bands(clean & (Rp <= 0.5 * Rbv), lam),
            'AR_gain_bare_at_lam0': float(Rbv[i0] - Rp[i0]),
            'suppression_vs_bare_at_lam0': float(Rp[i0] / Rbv[i0]),
        }
        u = d9[(d9.case == 'uniform') & (np.isclose(d9.h_nm, h))] \
            .sort_values('lam_nm')
        if len(u) == len(p):
            rec['suppression_vs_film_at_lam0'] = float(
                Rp[i0] / u.f_R.to_numpy()[i0])
            rec['suppression_vs_film_band_median'] = float(
                np.median(Rp / u.f_R.to_numpy()))
        summ['per_h'][f'h{h:g}'] = rec
        for lam_i, Rp_i, Rb_i, cl in zip(lam, Rp, Rbv, clean):
            comp_rows.append({'h_nm': h, 'lam_nm': lam_i, 'R_p0550': Rp_i,
                              'R_bare': Rb_i, 'clean': bool(cl),
                              'AR_gain': Rb_i - Rp_i})
    pd.DataFrame(comp_rows).to_csv(R / 'p0550_ar_comparison.csv', index=False)

    # order convergence (R/T-only grids)
    for h in HS:
        rec = {}
        for o in (9, 11, 13, 15):
            p = df[(df.case == 'p0550') & (np.isclose(df.h_nm, h))
                   & (df.order == o)].sort_values('lam_nm')
            if len(p) < 20:
                continue
            lam, Rp = p.lam_nm.to_numpy(), p.f_R.to_numpy()
            Rbv = np.full_like(Rp, R_FRESNEL)
            i0, im = int(np.argmin(np.abs(lam - LAM0))), int(np.argmin(Rp))
            rec[f'o{o}'] = {
                'R_at_lam0': float(Rp[i0]), 'R_min': float(Rp[im]),
                'lam_Rmin': float(lam[im]),
                'bw_A_nm': float(sum(b - a for a, b in
                                     bands(Rp <= 0.05, lam))),
                'bw_B_nm': float(sum(b - a for a, b in
                                     bands(Rp < Rbv, lam))),
            }
        summ['orders'][f'h{h:g}'] = rec
    spot = R / 'ar_o17_spot.csv'
    if spot.exists():
        summ['orders']['o17_spot'] = pd.read_csv(spot).to_dict('records')
    gm = R / 'ar_grid_matrix.csv'
    if gm.exists():
        summ['grid_matrix'] = pd.read_csv(gm).to_dict('records')

    (R / 'ar_summary.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ['per_h'], indent=1)[:3000])
    print('\nORDERS:', json.dumps(summ['orders'], indent=1)[:2000])
    print('AR_ANALYSIS_DONE')


if __name__ == '__main__':
    main()
