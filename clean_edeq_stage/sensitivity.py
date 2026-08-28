"""Phases 8-9: sensitivity maps and phase-knob selection for P0550.

(1) Adjoint (autograd) maps at a chosen wavelength: gradients wrt the
    64x64 density of
      phi   = arg(px / Qxz)          (ED-EQ relative phase; channel
                                      constants drop out of the ratio)
      bal   = log(C_ED / C_EQ)       (balance log-ratio; |d bal| small
                                      <=> B_ED_EQ preserved)
      aED   = log C_ED,  aEQ = log C_EQ
    Saved to results/audit/sens_maps_<lam>.npz + png.

(2) Interpretable-knob comparison at the same wavelength by central
    finite differences of the same observables:
      h (thickness), alpha (global in-plane scale via P), n (index scale)
    plus the decoupled pixel direction M = g_phi - proj_{g_bal} g_phi
    from (1). Figure of merit: |d phi/d beta| per unit |d bal/d beta|
    at matched observable scale.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

import stage_core as sc
import ed_eq_core as core

AUD = sc.RESULTS / 'audit'
NAME = 'P0550_H0250_seed011'


def observables(rho_t, P, h, lam, eps_scale=1.0):
    eps_si = core.si_eps(float(lam)) * 1.0
    n_c = complex(eps_si) ** 0.5 * eps_scale
    eps_si = n_c ** 2
    sim = core.build_sim(rho_t, P, h, float(lam), sc.ORDER, eps_si=eps_si)
    x_ax, z_ax, E, _ = core.fields_3d(sim, P, h, sc.N_XY, sc.NZ)
    n = rho_t.shape[0]
    idx = (torch.floor(x_ax / P * n).long()) % n
    eps3 = (rho_t[idx][:, idx] * (eps_si - 1.0) + 1.0)[:, :, None] \
        .expand(sc.N_XY, sc.N_XY, sc.NZ)
    mo = core.torch_moments(E, eps3, x_ax, z_ax, float(lam))
    Cp, Cm, CQe, CQm = core.family_weights4(mo)
    q = mo['px'] / mo['Qxz']
    phi = torch.atan2(q.imag, q.real)
    bal = torch.log(Cp) - torch.log(CQe)
    return {'phi': phi, 'bal': bal,
            'aED': torch.log(Cp), 'aEQ': torch.log(CQe),
            'f_ED': (Cp / (Cp + Cm + CQe + CQm)),
            'f_EQ': (CQe / (Cp + Cm + CQe + CQm)), 'mo': mo}


def grad_maps(lam):
    rho, P, h = sc.load_ref(NAME)
    maps = {}
    for key in ('phi', 'bal'):
        rho_t = rho.clone().requires_grad_(True)
        obs = observables(rho_t, P, h, lam)
        obs[key].backward()
        maps[key] = rho_t.grad.detach().numpy().copy()
        print(f'grad map {key}: |g| mean {np.abs(maps[key]).mean():.3e} '
              f'max {np.abs(maps[key]).max():.3e}', flush=True)
    g_phi, g_bal = maps['phi'].ravel(), maps['bal'].ravel()
    proj = (g_phi @ g_bal) / max(g_bal @ g_bal, 1e-30)
    M = (g_phi - proj * g_bal).reshape(maps['phi'].shape)
    Mn = M / np.abs(M).max()
    cos = (g_phi @ g_bal) / (np.linalg.norm(g_phi) * np.linalg.norm(g_bal))
    AUD.mkdir(parents=True, exist_ok=True)
    np.savez(AUD / f'sens_maps_{lam:.1f}.npz', g_phi=maps['phi'],
             g_bal=maps['bal'], M=Mn, rho=rho.numpy(),
             cos_phi_bal=cos)
    print(f'cos(g_phi, g_bal) = {cos:.3f}', flush=True)
    return maps['phi'], maps['bal'], Mn, rho, P, h


def knob_fd(lam):
    g_phi, g_bal, Mn, rho, P, h = grad_maps(lam)

    def obs_at(rho_t, P_, h_, eps_scale=1.0):
        with torch.no_grad():
            o = observables(rho_t, P_, h_, lam, eps_scale)
        return (float(o['phi']) * 180 / math.pi, float(o['bal']),
                float(o['f_ED']), float(o['f_EQ']))

    knobs = {}
    dh = 2.0
    p_hi, p_lo = obs_at(rho, P, h + dh), obs_at(rho, P, h - dh)
    knobs['thickness_per_nm'] = [(a - b) / (2 * dh) for a, b in zip(p_hi, p_lo)]
    da = 0.005
    p_hi, p_lo = obs_at(rho, P * (1 + da), h), obs_at(rho, P * (1 - da), h)
    knobs['scale_per_pct'] = [(a - b) / (2 * da * 100) for a, b in zip(p_hi, p_lo)]
    dn = 0.005
    with torch.no_grad():
        o_hi = observables(rho, P, h, lam, eps_scale=1 + dn)
        o_lo = observables(rho, P, h, lam, eps_scale=1 - dn)
    knobs['index_per_pct'] = [
        (float(o_hi['phi']) - float(o_lo['phi'])) * 180 / math.pi / (2 * dn * 100),
        (float(o_hi['bal']) - float(o_lo['bal'])) / (2 * dn * 100),
        (float(o_hi['f_ED']) - float(o_lo['f_ED'])) / (2 * dn * 100),
        (float(o_hi['f_EQ']) - float(o_lo['f_EQ'])) / (2 * dn * 100)]
    db = 0.05
    Mt = torch.tensor(Mn, dtype=torch.float32)
    r_hi = torch.clamp(rho + db * Mt, 0, 1)
    r_lo = torch.clamp(rho - db * Mt, 0, 1)
    p_hi, p_lo = obs_at(r_hi, P, h), obs_at(r_lo, P, h)
    knobs['pixel_dir_per_0.05'] = [(a - b) / 2 for a, b in zip(p_hi, p_lo)]

    table = {}
    for kname, (dphi, dbal, dfe, dfq) in knobs.items():
        # decoupling merit: degrees of phase per 0.01 of |bal| change
        merit = abs(dphi) / max(abs(dbal), 1e-9) * 0.01
        table[kname] = {'dphi_deg': dphi, 'dbal': dbal, 'df_ED': dfe,
                        'df_EQ': dfq, 'deg_per_0.01bal': merit}
        print(f'{kname:22s} dphi={dphi:+.3f} dbal={dbal:+.4f} '
              f'merit(deg per 0.01 bal)={merit:.2f}', flush=True)
    (AUD / f'knob_table_{lam:.1f}.json').write_text(
        json.dumps(table, indent=1))
    print('SENSITIVITY_DONE', flush=True)
    return table


if __name__ == '__main__':
    import sys
    torch.set_num_threads(2)
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 1332.5
    knob_fd(lam)
