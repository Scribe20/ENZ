"""P0750 true per-order transmission minimum: small R/T-only lambda scans
at [11,11], [13,13], [15,15] around the resonance (the pole shifts with
order, so a fixed-wavelength comparison understates convergence).
Writes results/p0750_order_tmin.csv.
"""
import numpy as np
import pandas as pd
import torch

import stage_core as sc
from ar_audit import rt_both
import ed_eq_core as core


def main():
    rho, P, h = sc.load_ref('P0750_H0250_seed011')
    rows = []
    for o in (11, 13, 15):
        lams = np.arange(1329.0, 1338.0 + 0.01, 0.5)
        for lam in lams:
            eps_si = core.si_eps(float(lam))
            with torch.no_grad():
                sim = core.build_sim(rho, P, h, float(lam), [o, o],
                                     eps_si=eps_si)
                rr = rt_both(sim)
            rows.append({'order': o, 'lam_nm': float(lam), **rr})
            print(f'o{o} {lam:.1f} T={rr["f_T"]:.4f}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(sc.RESULTS / 'p0750_order_tmin.csv', index=False)
    for o in (11, 13, 15):
        d = df[df.order == o]
        i = d.f_T.idxmin()
        print(f'order {o}: Tmin={d.f_T[i]:.4f} at {d.lam_nm[i]:.1f} '
              f'R={d.f_R[i]:.4f}')
    print('P0750_ORDER_MIN_DONE', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
