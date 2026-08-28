"""Phase 15: field-gradient (quadrupole-coupling) diagnostic for P0550.

Computes |dEx/dz| and |dEz/dx| maps at lam0 on a dense grid (64x64x41
through the slab plus evaluation right at the top/substrate interfaces),
plus the symmetrized gradient combination (dEx/dz + dEz/dx)/2 that the
Qe_xz interaction samples. Reports interface and hotspot statistics.
No claims about emitters - platform diagnostic only.
"""
import json

import numpy as np
import torch

import stage_core as sc
import ed_eq_core as core

NAME = 'P0550_H0250_seed011'
LAM = 1332.5
NXY, NZDENSE = 64, 41


def main():
    rho, P, h = sc.load_ref(NAME)
    eps_si = core.si_eps(LAM)
    with torch.no_grad():
        sim = core.build_sim(rho, P, h, LAM, sc.ORDER, eps_si=eps_si)
        x_ax, z_ax, E, H = core.fields_3d(sim, P, h, NXY, NZDENSE)
    Ex = E[0].numpy()
    Ez = E[2].numpy()
    dx = float(x_ax[1] - x_ax[0]) * 1e-9
    dz = float(z_ax[1] - z_ax[0]) * 1e-9
    dEx_dz = np.gradient(Ex, dz, axis=2)
    dEz_dx = np.gradient(Ez, dx, axis=0)
    sym = 0.5 * (dEx_dz + dEz_dx)
    stats = {}
    for nameq, q in (('abs_dEx_dz', np.abs(dEx_dz)),
                     ('abs_dEz_dx', np.abs(dEz_dx)),
                     ('abs_sym_grad', np.abs(sym)),
                     ('abs_Ex', np.abs(Ex))):
        stats[nameq] = {
            'volume_mean': float(q.mean()), 'volume_max': float(q.max()),
            'top_interface_mean': float(q[:, :, -1].mean()),
            'top_interface_max': float(q[:, :, -1].max()),
            'bottom_interface_mean': float(q[:, :, 0].mean()),
            'bottom_interface_max': float(q[:, :, 0].max()),
            'p99': float(np.quantile(q, 0.99)),
        }
    # normalized figure of merit: gradient per incident field per k
    k = 2 * np.pi / (LAM * 1e-9)
    stats['grad_enhancement_vs_planewave'] = {
        'sym_top_max_over_kE0': float(np.abs(sym)[:, :, -1].max() / k),
        'sym_volume_p99_over_kE0': float(np.quantile(np.abs(sym), 0.99) / k),
        'note': 'planewave |dEx/dz| = k E0; values >1 mean gradient '
                'exceeds the incident-planewave gradient scale',
    }
    (sc.RESULTS / 'p0550_gradient_diag.json').write_text(
        json.dumps(stats, indent=1))
    np.savez(sc.RESULTS / 'audit' / 'gradient_maps.npz',
             x_nm=x_ax.numpy(), z_nm=z_ax.numpy(),
             dEx_dz=dEx_dz, dEz_dx=dEz_dx, sym=sym, Ex=Ex, rho=rho.numpy())
    print(json.dumps(stats['grad_enhancement_vs_planewave'], indent=1))
    print('GRADIENT_DIAG_DONE')


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
