"""Locality / LPA diagnostic (spec sec 46): a 4x1 supercell of the
champion with PB rotations 0/45/90/135 deg forms a phase-gradient
metagrating (period 4P). Under the local-phase approximation the
cross-circular reflection routes into ONE first diffraction order with
efficiency ~ R_cross(single atom) = the acceptance-map value; the
measured order-resolved efficiency directly quantifies neighbor
coupling. Also: uniform-rotation supercell sanity check vs the
single-cell result.

usage: python wf_lpa.py <tag>
"""
import json
import math
import sys

import numpy as np
import scipy.ndimage as ndi
import torch

import wf_core as wf
import rt_core as rc
from wf_anglemap import load_geometry


def supercell(rho, rots):
    tiles = []
    for a in rots:
        rr = ndi.rotate(rho.numpy(), a, reshape=False, order=0,
                        mode='constant', cval=0.0)
        tiles.append((rr > 0.5).astype(np.float32))
    return torch.tensor(np.concatenate(tiles, axis=0))


def run(tag):
    import torcwa
    rho, P, H, _ = load_geometry(tag)
    e = rc.eps_asi()
    out = {}
    for name, rots in (('uniform', [0, 0, 0, 0]),
                       ('gradient', [0, 45, 90, 135])):
        sc = supercell(rho, rots)
        sim = torcwa.rcwa(freq=1.0 / wf.LAM0, order=[29, 7],
                          L=[4 * float(P), float(P)],
                          dtype=rc.SIM_DTYPE, device=rc.DEVICE)
        sim.add_input_layer(eps=rc.EPS_GLASS)
        sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
        sim.add_layer(thickness=float(H), eps=sc * (e - 1.0) + 1.0)
        sim.solve_global_smatrix()
        # circular cross amplitude per reflected order m (x along ramp):
        # sigma+ in, sigma- out = (1/2)[(rxx - ryy) - i(rxy + ryx)]
        res = {}
        for m in range(-3, 4):
            rj = {}
            for pol in ('xx', 'yx', 'xy', 'yy'):
                with torch.no_grad():
                    rj[pol] = complex(sim.S_parameters(
                        orders=[m, 0], direction='backward',
                        port='reflection', polarization=pol,
                        ref_order=[0, 0]))
            cross = 0.5 * ((rj['xx'] - rj['yy'])
                           - 1j * (rj['xy'] + rj['yx']))
            co = 0.5 * ((rj['xx'] + rj['yy'])
                        + 1j * (rj['xy'] - rj['yx']))
            res[m] = {'R_cross_m': abs(cross) ** 2,
                      'R_co_m': abs(co) ** 2}
        out[name] = res
        tot_cross = sum(v['R_cross_m'] for v in res.values())
        print(f'{tag} {name}: per-order R_cross ' +
              ' '.join(f'm={m}:{res[m]["R_cross_m"]:.3f}'
                       for m in range(-2, 3)) +
              f' | total {tot_cross:.3f}', flush=True)
    single = json.loads(
        (wf.HERE / 'refinement' / tag / 'final.json').read_text())
    out['single_cell_R_cross0'] = single['R_cross0']
    (wf.HERE / 'results' / f'lpa_{tag}.json').write_text(
        json.dumps(out, indent=1, default=float))
    print('LPA_DONE', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    run(sys.argv[1])
