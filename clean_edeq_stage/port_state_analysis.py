"""Port-state audit analysis (Phases A-F, I-P, T-U): all machine-readable
outputs from merged scan data. Run after port_state_scan shards finish.
"""
import json
import math

import numpy as np
import pandas as pd

import stage_core as sc

R = sc.RESULTS
LAM0 = 1332.5
P550, P750 = 550.0, 750.0


def cx(d, s):
    return d[s + '_re'].to_numpy() + 1j * d[s + '_im'].to_numpy()


def e2_of(d, P_nm):
    k = 2 * math.pi / (d['lam_nm'].to_numpy() * 1e-9)
    A = (P_nm * 1e-9) ** 2
    return -(sc.Z0 / (2 * A)) * (-k ** 2 / 2) * cx(d, 'I2')


def unwrap_deg(a):
    return np.degrees(np.unwrap(np.radians(a)))


def main():
    summ = {}
    ps_full = pd.concat([pd.read_csv(R / f'ps_full_shard{i}.csv')
                         for i in range(4)]).drop_duplicates(
        ['kind', 'name', 'h_nm', 'lam_nm', 'order'])
    ps_smx = pd.concat([pd.read_csv(R / f'ps_smx_shard{i}.csv')
                        for i in range(4)]).drop_duplicates(
        ['kind', 'name', 'h_nm', 'lam_nm', 'order'])

    # ---------------- P0550 h-family at lam0 (A-D) ----------------
    sw = pd.read_csv(R / 'p0550_phase_sweep.csv')
    f0 = sw[sw.lam_nm == LAM0].copy()
    f0['kind'] = 'sweep'
    href = ps_full[ps_full.kind == 'p0550_href'].copy()
    common = [c for c in f0.columns if c in href.columns]
    fam = pd.concat([f0[common], href[common]]) \
        .drop_duplicates('h_nm').sort_values('h_nm').reset_index(drop=True)
    h = fam.h_nm.to_numpy()
    ev, oQ, om = cx(fam, 'even_px'), cx(fam, 'odd_Q'), cx(fam, 'odd_m')
    e2 = e2_of(fam, P550)
    t, r = cx(fam, 'txx'), cx(fam, 'rxx')
    tbg, rbg = cx(fam, 'tbg'), cx(fam, 'rbg')
    dphi_top_w = np.degrees(np.angle(ev / oQ))
    dphi_bot_w = np.degrees(np.angle(ev / -oQ))
    dphi_top_u = unwrap_deg(dphi_top_w)
    dphi_bot_u = unwrap_deg(dphi_bot_w)
    phi_t_u = unwrap_deg(np.degrees(np.angle(t)))
    phi_r_u = unwrap_deg(np.degrees(np.angle(r)))
    A_abs = 1 - fam['T'].to_numpy() - fam['R'].to_numpy()

    pd.DataFrame({'h_nm': h,
                  'dphi_bot_wrapped_deg': dphi_bot_w,
                  'dphi_bot_unwrapped_deg': dphi_bot_u,
                  'dphi_top_unwrapped_deg': dphi_top_u,
                  'phi_t_unwrapped_deg': phi_t_u,
                  'phi_r_unwrapped_deg': phi_r_u}) \
        .to_csv(R / 'p0550_h_phase_unwrapped.csv', index=False)
    pd.DataFrame({'h_nm': h, 't_re': t.real, 't_im': t.imag,
                  'r_re': r.real, 'r_im': r.imag,
                  'tbg_re': tbg.real, 'tbg_im': tbg.imag,
                  'rbg_re': rbg.real, 'rbg_im': rbg.imag,
                  'T': fam['T'], 'R': fam['R'], 'A': A_abs}) \
        .to_csv(R / 'p0550_h_complex_rt.csv', index=False)
    mcols = ['h_nm', 'f_ED', 'f_MD', 'f_EQ', 'f_MQ', 'ED_EQ_balance',
             'px_given_ED', 'Qxz_given_EQ', 'my_given_MD']
    fam[mcols].to_csv(R / 'p0550_h_multipoles.csv', index=False)

    # slopes and smoothness
    dpdh = np.gradient(dphi_bot_u, h)
    mask_knob = (h >= 215) & (h <= 240)
    lin = np.polyfit(h[mask_knob], dphi_bot_u[mask_knob], 1)
    resid = dphi_bot_u[mask_knob] - np.polyval(lin, h[mask_knob])
    summ['p0550_phase_knob'] = {
        'monotonic': bool(np.all(np.diff(dphi_bot_u) < 0)),
        'global_slope_deg_per_nm': float(
            (dphi_bot_u[-1] - dphi_bot_u[0]) / (h[-1] - h[0])),
        'slope_215_240_deg_per_nm': float(lin[0]),
        'linear_resid_rms_deg_215_240': float(np.std(resid)),
        'slope_range_deg_per_nm': [float(dpdh.min()), float(dpdh.max())],
        'total_dphi_span_deg': float(dphi_bot_u.max() - dphi_bot_u.min()),
    }

    # mode identity interval (C): D_comp from h*=227.5-ish reference + purities
    ref = fam.iloc[(fam.h_nm - 227.5).abs().idxmin()]
    D = np.sqrt((fam.f_ED - ref.f_ED) ** 2 + (fam.f_MD - ref.f_MD) ** 2
                + (fam.f_EQ - ref.f_EQ) ** 2 + (fam.f_MQ - ref.f_MQ) ** 2)
    fam['D_comp_ref'] = D
    ok = (D <= 0.10) & (fam.px_given_ED >= 0.95) & (fam.Qxz_given_EQ >= 0.7) \
        & (fam.f_ED + fam.f_EQ >= 0.85)
    hs_ok = h[ok.to_numpy()]
    summ['mode_identity_interval'] = {
        'criterion': 'D_comp(vs h=227.5) <= 0.10 AND px|ED >= 0.95 AND '
                     'Qxz|EQ >= 0.70 AND f_ED+f_EQ >= 0.85',
        'h_range': [float(hs_ok.min()), float(hs_ok.max())],
        'contiguous': bool(np.all(np.diff(np.where(ok)[0]) == 1)),
    }
    lo, hi = hs_ok.min(), hs_ok.max()
    mm = (h >= lo) & (h <= hi)
    summ['phase_coverage_over_identity_interval'] = {
        'dphi_bot_deg': float(dphi_bot_u[mm].max() - dphi_bot_u[mm].min()),
        'arg_t_deg': float(phi_t_u[mm].max() - phi_t_u[mm].min()),
        'arg_r_deg': float(phi_r_u[mm].max() - phi_r_u[mm].min()),
        'R_range': [float(fam['R'][mm].min()), float(fam['R'][mm].max())],
        'T_range': [float(fam['T'][mm].min()), float(fam['T'][mm].max())],
    }
    c1 = np.corrcoef(dphi_bot_u, phi_t_u)[0, 1]
    c2 = np.corrcoef(dphi_bot_u, phi_r_u)[0, 1]
    s_t = np.polyfit(dphi_bot_u[mm], phi_t_u[mm], 1)[0]
    summ['device_vs_multipole_phase'] = {
        'corr_dphi_vs_arg_t': float(c1), 'corr_dphi_vs_arg_r': float(c2),
        'd_arg_t_per_d_dphi_in_interval': float(s_t)}

    # current-pattern overlap (C)
    import glob
    ovs = sorted(glob.glob(str(R / 'audit' / 'overlap' / 'ex_mid_h*.npz')))
    if len(ovs) >= 3:
        maps = {}
        for p in ovs:
            z = np.load(p)
            maps[float(z['h'])] = z['Ex']
        hs_o = sorted(maps)
        ovl = []
        for a, b in zip(hs_o[:-1], hs_o[1:]):
            u, v = maps[a].ravel(), maps[b].ravel()
            ovl.append(abs(np.vdot(u, v)) / (np.linalg.norm(u)
                                             * np.linalg.norm(v)))
        summ['current_overlap'] = {
            'h_pairs_step_nm': 5.0,
            'min_neighbor_overlap': float(min(ovl)),
            'overlap_at_215_240': float(min(
                o for a, o in zip(hs_o[:-1], ovl) if 214 <= a <= 240)),
        }

    # ---------------- S-matrix (E) ----------------
    def smx_table(kind, key):
        d = ps_smx[ps_smx.kind == kind].sort_values(key)
        t21 = cx(d, 'f_t'); r11 = cx(d, 'f_r')
        t12 = cx(d, 'b_t'); r22 = cx(d, 'b_r')
        recip = np.abs(t12 - t21)
        uni = np.abs(np.abs(r11) ** 2 + np.abs(t21) ** 2 - 1)
        out = pd.DataFrame({key: d[key].to_numpy(),
                            'r11_re': r11.real, 'r11_im': r11.imag,
                            'r22_re': r22.real, 'r22_im': r22.imag,
                            't21_re': t21.real, 't21_im': t21.imag,
                            't12_re': t12.real, 't12_im': t12.imag,
                            'R_f': np.abs(r11) ** 2, 'R_b': np.abs(r22) ** 2,
                            'T': np.abs(t21) ** 2,
                            'reciprocity_resid': recip,
                            'unitarity_resid': uni})
        return out, {'max_reciprocity_resid': float(recip.max()),
                     'max_unitarity_resid': float(uni.max())}
    smx_h, s1 = smx_table('smx_h', 'h_nm')
    smx_h.to_csv(R / 'p0550_smatrix_vs_h.csv', index=False)
    smx_l, s2 = smx_table('smx_p0750', 'lam_nm')
    smx_l.to_csv(R / 'p0750_smatrix_vs_lam.csv', index=False)
    summ['smatrix'] = {'p0550_vs_h': s1, 'p0750_vs_lam': s2}

    # ---------------- P0750 fine resonance (I, J, L) ----------------
    d1 = pd.read_csv(R / 'p0750_multipole_spectra.csv')
    fine = ps_full[ps_full.kind == 'p0750_fine'].copy()
    common = [c for c in d1.columns if c in fine.columns]
    p7 = pd.concat([d1[common], fine[common]]).drop_duplicates('lam_nm') \
        .sort_values('lam_nm').reset_index(drop=True)
    p7.to_csv(R / 'p0750_highres_rt.csv', index=False)
    lam7 = p7.lam_nm.to_numpy()
    T7, R7 = p7['T'].to_numpy(), p7['R'].to_numpy()
    A7 = 1 - T7 - R7
    im = int(np.argmin(T7))
    # parabolic refine on the 0.2-nm core
    l3, t3 = lam7[im - 1:im + 2], T7[im - 1:im + 2]
    den = t3[0] - 2 * t3[1] + t3[2]
    lam_star = l3[1] - 0.5 * (t3[2] - t3[0]) / den * (l3[1] - l3[0])
    summ['p0750_Tmin'] = {
        'Tmin_sampled': float(T7[im]), 'lam_Tmin_sampled': float(lam7[im]),
        'lam_Tmin_parabolic': float(lam_star),
        'R_at_Tmin': float(R7[im]), 'A_at_Tmin': float(A7[im]),
        'en_res_at_Tmin': float(p7.en_res.iloc[im]),
        'step_nm_core': 0.2,
        'fractions_at_Tmin': {k: float(p7[k].iloc[im]) for k in
                              ('f_ED', 'f_MD', 'f_EQ', 'f_MQ')},
        'purities_at_Tmin': {k: float(p7[k].iloc[im]) for k in
                             ('my_given_MD', 'Qxz_given_EQ',
                              'px_given_ED')},
    }
    # t-plane ladder (per-row exact port coupling g_up = E_up / (t - tbg))
    ev7, oQ7, om7 = cx(p7, 'even_px'), cx(p7, 'odd_Q'), cx(p7, 'odd_m')
    e27 = e2_of(p7, P750)
    t7, tb7 = cx(p7, 'txx'), cx(p7, 'tbg')
    Eu7 = cx(p7, 'E_up')
    g7 = Eu7 / (t7 - tb7)
    pieces = {'ED': ev7 / g7, 'MD': om7 / g7, 'EQ': oQ7 / g7,
              '2nd': e27 / g7}
    arg_rows = []
    for i2 in range(len(p7)):
        arg_rows.append({'lam_nm': lam7[i2],
                         'tbg_re': tb7[i2].real, 'tbg_im': tb7[i2].imag,
                         **{f't_{k}_re': v[i2].real for k, v in pieces.items()},
                         **{f't_{k}_im': v[i2].imag for k, v in pieces.items()},
                         't_full_re': t7[i2].real, 't_full_im': t7[i2].imag})
    pd.DataFrame(arg_rows).to_csv(R / 'p0750_t_argand.csv', index=False)
    # removal tests at lam_Tmin (K)
    i = im
    combos = {
        'bg_only': tb7[i],
        'bg+ED': tb7[i] + pieces['ED'][i],
        'bg+MD': tb7[i] + pieces['MD'][i],
        'bg+EQ': tb7[i] + pieces['EQ'][i],
        'bg+2nd': tb7[i] + pieces['2nd'][i],
        'bg+ED+EQ': tb7[i] + pieces['ED'][i] + pieces['EQ'][i],
        'bg+MD+EQ': tb7[i] + pieces['MD'][i] + pieces['EQ'][i],
        'bg+ED+MD+EQ': tb7[i] + pieces['ED'][i] + pieces['MD'][i]
        + pieces['EQ'][i],
        'ladder_all': tb7[i] + sum(v[i] for v in pieces.values()),
        'remove_ED': tb7[i] + sum(v[i] for k, v in pieces.items()
                                  if k != 'ED'),
        'remove_MD': tb7[i] + sum(v[i] for k, v in pieces.items()
                                  if k != 'MD'),
        'remove_EQ': tb7[i] + sum(v[i] for k, v in pieces.items()
                                  if k != 'EQ'),
        'remove_2nd': tb7[i] + sum(v[i] for k, v in pieces.items()
                                   if k != '2nd'),
        'full_TORCWA': t7[i],
    }
    rem = pd.DataFrame([{'model': k, 't_re': v.real, 't_im': v.imag,
                         'abs_t': abs(v), 'T_model': abs(v) ** 2}
                        for k, v in combos.items()])
    rem.to_csv(R / 'p0750_component_removal.csv', index=False)
    summ['p0750_removal_at_Tmin'] = {k: float(abs(v) ** 2)
                                     for k, v in combos.items()}
    # internal vs external cancellation (L)
    odd_res = np.abs(om7 + oQ7) / np.maximum(np.abs(om7) + np.abs(oQ7),
                                             1e-30)
    j = int(np.argmin(odd_res))
    summ['p0750_internal_vs_external'] = {
        'lam_internal_null': float(lam7[j]),
        'internal_residual_min': float(odd_res[j]),
        'lam_Tmin': float(lam7[im]),
        'offset_nm': float(lam7[im] - lam7[j]),
    }

    # ---------------- M/N: same composition, opposite function ----------
    F = fam.reset_index(drop=True)
    pairs = []
    for i2 in range(len(F)):
        for j2 in range(i2 + 1, len(F)):
            a, b = F.iloc[i2], F.iloc[j2]
            D = math.sqrt((a.f_ED - b.f_ED) ** 2 + (a.f_MD - b.f_MD) ** 2
                          + (a.f_EQ - b.f_EQ) ** 2 + (a.f_MQ - b.f_MQ) ** 2)
            pairs.append({'h1': a.h_nm, 'h2': b.h_nm, 'D_comp': D,
                          'dR': abs(a.R - b.R), 'dT': abs(a['T'] - b['T']),
                          'R1': a.R, 'R2': b.R})
    pr = pd.DataFrame(pairs)
    pr['score'] = pr.dR / (pr.D_comp + 0.02)
    best = pr.sort_values('score', ascending=False).head(25)
    best.to_csv(R / 'same_composition_opposite_function_pairs.csv',
                index=False)
    b0 = best.iloc[0]
    summ['best_same_comp_pair'] = b0.to_dict()

    # ---------------- O/P: composition- and phase-port maps -------------
    rank = pd.read_csv(R / 'existing_candidate_tmin_ranking.csv')
    rows_map = []
    for _, rr in F.iterrows():
        rows_map.append({'set': 'P0550_hfam', 'id': f'h{rr.h_nm:g}',
                         'f_ED': rr.f_ED, 'f_MD': rr.f_MD, 'f_EQ': rr.f_EQ,
                         'f_MQ': rr.f_MQ, 'T': rr['T'], 'R': rr.R})
    for _, rr in p7.iterrows():
        rows_map.append({'set': 'P0750_lam', 'id': f'l{rr.lam_nm:g}',
                         'f_ED': rr.f_ED, 'f_MD': rr.f_MD, 'f_EQ': rr.f_EQ,
                         'f_MQ': rr.f_MQ, 'T': rr['T'], 'R': rr.R})
    led = pd.read_csv(sc.CAMP / 'results' / 'candidate_ledger_v2.csv')
    qual_t = {r.candidate: r for _, r in rank.iterrows()}
    for _, rr in led.iterrows():
        rows_map.append({'set': 'stageA_lam0', 'id': rr.run_id,
                         'f_ED': rr.f_ED, 'f_MD': rr.f_MD, 'f_EQ': rr.f_EQ,
                         'f_MQ': rr.f_MQ, 'T': np.nan, 'R': np.nan})
    pd.DataFrame(rows_map).to_csv(R / 'composition_port_map.csv',
                                  index=False)
    # phase-port predictors on the h family
    r_sc = r - rbg
    phase_sc_bg = np.degrees(np.angle(r_sc / rbg))
    pred = {
        'corr_R_vs_dphi_bot_wrapped': float(np.corrcoef(
            np.abs(((dphi_bot_w + 180) % 360) - 180), fam['R'])[0, 1]),
        'corr_R_vs_scatteredphase': float(np.corrcoef(
            np.abs(np.abs(phase_sc_bg) - 180), fam['R'])[0, 1]),
        'corr_R_vs_f_ED': float(np.corrcoef(fam.f_ED, fam['R'])[0, 1]),
        'corr_R_vs_B': float(np.corrcoef(fam.ED_EQ_balance, fam['R'])[0, 1]),
    }
    summ['phase_vs_composition_predictors'] = pred

    # ---------------- T/U: complex-plane convergence ---------------------
    ar = pd.read_csv(R / 'ar_rt_all.csv')
    conv = []
    for o in (9, 11, 13, 15):
        d = ar[(ar.case == 'p0550') & (np.isclose(ar.h_nm, 227.2))
               & (ar.order == o)]
        if len(d) == 0:
            continue
        i2 = (d.lam_nm - LAM0).abs().idxmin()
        rr = d.loc[i2]
        conv.append({'which': 'p0550_h227.2_lam0', 'order': o,
                     're_r': rr.f_r_re, 'im_r': rr.f_r_im,
                     're_t': rr.f_t_re, 'im_t': rr.f_t_im})
    for _, rr in ps_full[ps_full.kind == 'p0750_ord'].iterrows():
        conv.append({'which': 'p0750_lam1334', 'order': rr.order,
                     're_r': rr.rxx_re, 'im_r': rr.rxx_im,
                     're_t': rr.txx_re, 'im_t': rr.txx_im})
    rr = p7.iloc[(p7.lam_nm - 1334.0).abs().idxmin()]
    conv.append({'which': 'p0750_lam1334', 'order': 9,
                 're_r': rr.rxx_re, 'im_r': rr.rxx_im,
                 're_t': rr.txx_re, 'im_t': rr.txx_im})
    pd.DataFrame(conv).to_csv(R / 'complex_convergence.csv', index=False)

    (R / 'port_state_summary.json').write_text(
        json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))
    print('PORT_STATE_ANALYSIS_DONE')


if __name__ == '__main__':
    main()
