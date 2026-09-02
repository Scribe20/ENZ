"""Acceptance-angle metrics (spec sections 27-28, 43) from the fine
angle maps. For every mapped tag computes, on the identical grid:
  A. theta_50: largest contiguous range from 0 with
     min_phi R_cross(theta) >= 0.5 R_cross(0)
  B. theta_20: same with min_phi R_cross >= 0.20
  C. dominance range: R_cross > max(R_co, T_cross) for all phi
     (T_co folded into T via T_tot; dominance uses the listed channels)
  D. paper-matched metric: the reference paper's own angular metric is
     not recoverable in this environment (no Supplement available);
     stated explicitly rather than substituted.
Plus solid-angle-weighted <R_cross> over 0-85 and simple mean, worst
angle, R_co stats. Writes results/acceptance_metrics.csv.
"""
import numpy as np
import pandas as pd

import wf_core as wf

R = wf.HERE / 'results'


def metrics_for(df):
    df = df[df.theta <= 85.0]
    ths = sorted(df.theta.unique())
    r0 = float(df[(df.theta == 0)].R_cross.mean())
    minphi = {t: float(df[df.theta == t].R_cross.min()) for t in ths}
    meanphi = {t: float(df[df.theta == t].R_cross.mean()) for t in ths}

    def contiguous(pred):
        last = -1.0
        for t in ths:
            if pred(t):
                last = t
            else:
                break
        return last

    th50 = contiguous(lambda t: minphi[t] >= 0.5 * r0)
    th20 = contiguous(lambda t: minphi[t] >= 0.20)

    def dominant(t):
        sub = df[df.theta == t]
        return bool(((sub.R_cross > sub.R_co)
                     & (sub.R_cross > sub.T_cross)
                     & (sub.R_cross > sub.T_tot - sub.T_cross)).all())
    thdom = contiguous(dominant)

    bw = wf.band_weights(ths)
    omega = sum(bw[t] * meanphi[t] for t in ths)
    iworst = df.R_cross.idxmin()
    return {'R_cross0': r0,
            'theta_50': th50, 'theta_20': th20, 'theta_dom': thdom,
            'Rc_omega_085': float(omega),
            'Rc_mean_085': float(df.R_cross.mean()),
            'Rc_min_085': float(df.R_cross.min()),
            'theta_worst': float(df.loc[iworst, 'theta']),
            'phi_worst': float(df.loc[iworst, 'phi']),
            'co_mean': float(df.R_co.mean()),
            'co_max': float(df.R_co.max()),
            'T_mean': float(df.T_tot.mean()),
            'T_max': float(df.T_tot.max()),
            'A_mean': float(df.A.mean()),
            'A_max': float(df.A.max())}


def main():
    df = pd.read_csv(R / 'full_angle_maps.csv')
    rows = []
    for tag, sub in df.groupby('tag'):
        m = metrics_for(sub.reset_index(drop=True))
        rows.append({'tag': tag, **m})
        print(f"{tag}: Rc0={m['R_cross0']:.3f} th50={m['theta_50']:.0f} "
              f"th20={m['theta_20']:.0f} thdom={m['theta_dom']:.0f} "
              f"omega={m['Rc_omega_085']:.3f} min={m['Rc_min_085']:.3f}",
              flush=True)
    pd.DataFrame(rows).to_csv(R / 'acceptance_metrics.csv', index=False)
    print('ACCEPTANCE_DONE', flush=True)


if __name__ == '__main__':
    main()
