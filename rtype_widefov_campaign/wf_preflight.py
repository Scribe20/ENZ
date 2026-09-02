"""Wide-FOV preflight (spec sections 3-8, 43, 57 steps 1-4):
  1. freeze + record conventions and material
  2. exact diffraction-order thresholds per period
  3. fixed padding table
  4. paper-rectangle full fine angle map (theta 0-85 x phi 0-90, [9,9])
     + near-grazing 88-deg diagnostic + energy-closure checks
  5. runtime benchmark of the angular-minibatch objective.
"""
import csv
import json
import math
import time

import numpy as np
import torch

import wf_core as wf
import rt_core as rc

R = wf.HERE / 'results'
R.mkdir(exist_ok=True)

P_R, H_R, WX, WY = 226.0, 170.0, 160.0, 96.0


def rect_rho(n=128):
    ax = (np.arange(n) + 0.5) / n * P_R - P_R / 2
    X, Y = np.meshgrid(ax, ax, indexing='ij')
    return torch.tensor(((np.abs(X) <= WX / 2) & (np.abs(Y) <= WY / 2))
                        .astype(np.float32))


def main():
    torch.set_num_threads(2)
    eps = rc.EPS_ASI_633
    conv = {
        'lambda0_nm': 633.0,
        'material': f'a-Si Franta 2013, n+ik = {eps**0.5:.6f}',
        'glass_n': rc.N_GLASS,
        'incident_medium': 'air (TORCWA backward; input layer = glass)',
        'transmitted_medium': 'glass substrate',
        'angle_definition': ('theta_air = physical air-side incidence; '
                             'inc_ang(glass) = asin(sin(theta_air)/1.457); '
                             'k_par = k0 sin(theta_air) exact; azi = phi'),
        'circular_convention': ('C=(1/sqrt2)[[1,1],[i,-i]], sigma+=(x+iy)/'
                                'sqrt2 at theta=0; R_circ = C^H R C applied '
                                'to TORCWA transverse pol labels at all '
                                'angles, identically for every structure'),
        'R_cross': 'mean(|Rc01|^2, |Rc10|^2)', 'R_co':
        'mean(|Rc00|^2, |Rc11|^2)',
        'padding_rule': 'max(20 nm, 0.10 P), fixed per period, never swept',
    }
    json.dump(conv, open(R / 'conventions.json', 'w'), indent=1)

    # padding table
    rows = [{'P': P, 'padding_nm': rc.padding(P),
             'r_design_nm': rc.r_design(P)} for P in wf.PERIODS]
    wf.write_rows(R / 'padding_by_period.csv', rows)
    print('padding:', rows, flush=True)

    # exact diffraction thresholds
    dif = wf.diffraction_thresholds()
    wf.write_rows(R / 'diffraction_thresholds.csv', dif)
    for d in dif:
        print(f"P={d['P']:.0f}: first order opens at theta_air="
              f"{d['theta_open_air']:.1f} (air) / "
              f"{d['theta_open_glass']:.1f} (glass) [999=never]", flush=True)

    # rectangle fine angle map at [9,9]
    rho = rect_rho()
    t0 = time.time()
    rows = []
    for th in list(np.arange(0.0, 85.1, 5.0)) + [88.0]:
        for ph in np.arange(0.0, 90.1, 15.0):
            with torch.no_grad():
                Rj, Tj = wf.jones_angle(rho, P_R, H_R, float(th), float(ph),
                                        order=(9, 9))
            s = wf.angle_scores(Rj, Tj)
            m = rc.device_metrics(Rj, Tj)
            rows.append({'tag': 'rectangle', 'theta': float(th),
                         'phi': float(ph), 'R_cross': float(s['Rc']),
                         'R_co': float(s['co']), 'T_cross': float(s['Tc']),
                         'T_tot': float(s['Tt']), 'A': float(s['A']),
                         'abs_rx': m['abs_rx'], 'abs_ry': m['abs_ry'],
                         'dphi_r_deg': m['dphi_r_deg'],
                         'energy_x': m['R_total_x'] + m['T_total_x']
                         + m['A_x']})
        print(f'rect theta={th:.0f} done ({time.time()-t0:.0f}s)',
              flush=True)
    wf.write_rows(R / 'rectangle_reference_angle_map.csv', rows)
    r0 = rows[0]['R_cross']
    print(f'rectangle: R_cross(0)={r0:.3f}', flush=True)

    # energy-closure sanity: lossless counterfactual at 80 deg must give
    # R+T ~ 1 (checks power normalization of oblique S-parameters)
    import material_model  # noqa: F401  (silence linters)
    e_l = complex(eps.real, 0.0)
    for order in ((7, 7), (9, 9)):
        import torcwa
        sim = torcwa.rcwa(freq=1.0 / 633.0, order=list(order),
                          L=[P_R, P_R], dtype=rc.SIM_DTYPE, device=rc.DEVICE)
        sim.add_input_layer(eps=rc.EPS_GLASS)
        sim.set_incident_angle(inc_ang=wf.glass_angle(80.0),
                               azi_ang=math.radians(45.0))
        sim.add_layer(thickness=H_R, eps=rho * (e_l - 1.0) + 1.0)
        sim.solve_global_smatrix()
        Rj, Tj = wf.jones_dev(sim)
        Rt = float((torch.abs(Rj) ** 2).sum(dim=0)[0]
                   + (torch.abs(Tj) ** 2).sum(dim=0)[0])
        print(f'lossless closure order {order}: R+T(p,80deg,phi45) = '
              f'{Rt:.4f}', flush=True)
    # theta = 0 exact: ps-corrected must match xy jones
    with torch.no_grad():
        sim0 = wf.build_sim_angle(rho, P_R, H_R, 0.0, 0.0, order=(9, 9))
        Rd, Td = wf.jones_dev(sim0)
        Rx, Tx = rc.jones(sim0, 'backward')
        print(f'theta=0 ps-vs-xy max|dR| = '
              f'{float((Rd - Rx).abs().max()):.2e} '
              f'max|dT| = {float((Td - Tx).abs().max()):.2e}', flush=True)

    # benchmark: order-7 vs order-9 jones fwd+bwd, moments cost
    x = torch.randn(96, 96, requires_grad=True)
    mask = rc.design_mask(96, 226.0)
    kern = rc.conic_filter_kernel(96, 226.0, 15.0)
    for order in ((5, 5), (7, 7), (9, 9)):
        t0 = time.time()
        rho_b = rc.filt_project(torch.sigmoid(x), kern, 4.0, mask=mask)
        scores = []
        for th, ph in ((0, 0), (30, 45), (60, 0), (75, 90), (80, 45)):
            Rj, Tj = wf.jones_angle(rho_b, 226.0, 170.0, th, ph,
                                    order=order)
            scores.append(wf.angle_scores(Rj, Tj))
        L = wf.robust_loss(scores)
        L.backward()
        print(f'benchmark 5-angle iter, order {order}: '
              f'{time.time()-t0:.1f}s', flush=True)
        x.grad = None
    t0 = time.time()
    rho_b = rc.filt_project(torch.sigmoid(x), kern, 4.0, mask=mask)
    f = rc.moments_families(rho_b, 226.0, 170.0, 633.0, (7, 7), 'x')
    (f['f_ED']).backward()
    print(f'moments_families (32x32x5, order 7) grad: '
          f'{time.time()-t0:.1f}s', flush=True)
    print('PREFLIGHT_DONE', flush=True)


if __name__ == '__main__':
    main()
