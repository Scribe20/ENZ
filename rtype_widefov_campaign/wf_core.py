"""Wide-FOV angle-aware R-type campaign core (633 nm PB meta-atom).

Builds on the validated rt_core machinery (materials, D2 filter chain,
Jones/circular conventions) but replaces the theta=0 objective with an
ANGULAR-RESPONSE-SURFACE objective from iteration 0.

Conventions (frozen, identical for every structure incl. baselines):
  - Stack: glass substrate (TORCWA input layer) / patterned a-Si (H) / air.
  - DEVICE illumination from the AIR side ('backward'): incident medium
    air, reflection port air, transmission port glass. This is the
    paper-relevant R-type direction (validated in rt_baseline.py).
  - ANGLES ARE AIR-SIDE physical incidence angles theta_air. TORCWA's
    inc_ang lives in the input (glass) layer, so we set
    inc_ang = asin(sin(theta_air)/n_glass); k_par = k0 sin(theta_air)
    exactly. theta_air in [0, 90) maps to glass angles [0, 43.3) - no
    TIR anywhere in the domain. azi_ang = phi (in-plane azimuth).
  - Jones r/t: 2x2 complex in TORCWA's transverse 'x'/'y' polarization
    labels at the given (theta,phi); circular combos via the same
    C = (1/sqrt2)[[1,1],[i,-i]] as the previous campaign. At theta=0
    this is exactly the CP basis (sigma+ = (x+iy)/sqrt2). R_cross :=
    mean(|Rc01|^2, |Rc10|^2), R_co := mean(|Rc00|^2, |Rc11|^2).
  - lambda0 = 633 nm; a-Si Franta 2013 (n=4.2827, k=0.0687 at 633);
    glass n=1.457. No endpoint clamping (dataset brackets 633 nm).
  - ONE fixed rotation-safe padding per period: pad = max(20, 0.10 P)
    (same rule as the previous campaign); r_design = P/2 - pad. Never
    swept, same for all H / methods / seeds.
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'rtype_freeform_campaign'))
import rt_core as rc                                        # noqa: E402
import torcwa                                               # noqa: E402

LAM0 = rc.LAM0
N_GLASS = rc.N_GLASS

PERIODS = [200.0, 213.0, 226.0, 239.0, 252.0]
HEIGHTS = [140.0, 155.0, 170.0, 185.0, 200.0]

THETA_POOL = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 80.0]
PHI_POOL = [0.0, 45.0, 90.0]


def glass_angle(theta_air):
    """Input-layer (glass) angle giving air-side k_par = k0 sin(theta)."""
    return math.asin(math.sin(math.radians(theta_air)) / N_GLASS)


def build_sim_angle(rho, P, H, theta_air, phi, lam=LAM0, order=(7, 7)):
    e = rc.eps_asi(lam)
    sim = torcwa.rcwa(freq=1.0 / float(lam), order=list(order),
                      L=[float(P), float(P)], dtype=rc.SIM_DTYPE,
                      device=rc.DEVICE)
    sim.add_input_layer(eps=rc.EPS_GLASS)
    sim.set_incident_angle(inc_ang=glass_angle(theta_air),
                           azi_ang=math.radians(phi))
    sim.add_layer(thickness=float(H), eps=rho * (e - 1.0) + 1.0)
    sim.solve_global_smatrix()
    return sim


_DIN = torch.diag(torch.tensor([-1.0 + 0j, 1.0 + 0j]))


def jones_dev(sim):
    """Device Jones matrices at arbitrary (theta, phi), extracted in the
    exact flux-normalized p/s eigenbasis and sign-corrected to reduce to
    the validated xy-basis Jones at theta -> 0:
        R_dev = R_ps @ diag(-1,1)          (incident p -> -x at theta=0)
        T_dev = diag(-1,1) @ T_ps @ diag(-1,1)
    Validated in wf_pstest.py: lossless R+T closure = 1.000 +- 3e-4 up to
    theta_air = 85 deg at any azimuth (the xy-basis power normalization
    is exact only at phi = 0/90). Helicity powers |Rc|^2 are invariant
    under the in-plane frame rotation with phi."""
    r = [[None, None], [None, None]]
    t = [[None, None], [None, None]]
    for i, po in ((0, 'p'), (1, 's')):
        for j, pi_ in ((0, 'p'), (1, 's')):
            r[i][j] = sim.S_parameters(orders=[0, 0], direction='backward',
                                       port='reflection',
                                       polarization=po + pi_,
                                       ref_order=[0, 0]).reshape(())
            t[i][j] = sim.S_parameters(orders=[0, 0], direction='backward',
                                       port='transmission',
                                       polarization=po + pi_,
                                       ref_order=[0, 0]).reshape(())
    Rps = torch.stack([torch.stack(r[0]), torch.stack(r[1])])
    Tps = torch.stack([torch.stack(t[0]), torch.stack(t[1])])
    return Rps @ _DIN, _DIN @ Tps @ _DIN


def jones_angle(rho, P, H, theta_air, phi, lam=LAM0, order=(7, 7)):
    sim = build_sim_angle(rho, P, H, theta_air, phi, lam, order)
    return jones_dev(sim)


def angle_scores(R, T):
    """Differentiable per-angle score tensors from the Jones pair."""
    Rc = rc.circular(R)
    Tc = rc.circular(T)
    r_cross = 0.5 * (torch.abs(Rc[0, 1]) ** 2 + torch.abs(Rc[1, 0]) ** 2)
    r_co = 0.5 * (torch.abs(Rc[0, 0]) ** 2 + torch.abs(Rc[1, 1]) ** 2)
    R_tot = (torch.abs(R) ** 2).sum(dim=0)
    T_tot = (torch.abs(T) ** 2).sum(dim=0)
    absorb = 0.5 * ((1 - R_tot - T_tot).clamp(min=-0.1)).sum()
    return {'Rc': r_cross, 'co': r_co, 'Tt': 0.5 * T_tot.sum(),
            'A': absorb, 'Tc': 0.5 * (torch.abs(Tc[0, 1]) ** 2
                                      + torch.abs(Tc[1, 0]) ** 2)}


def robust_loss(scores, R0=None):
    """Angular robust objective: mean + softmin + lower-tail of R_cross
    minus penalty terms; optional smooth PB phase/amplitude terms from
    the theta=0 Jones matrix R0 (never a wrapped angle)."""
    Rcs = torch.stack([s['Rc'] for s in scores])
    cos_ = torch.stack([s['co'] for s in scores])
    Tts = torch.stack([s['Tt'] for s in scores])
    As = torch.stack([s['A'] for s in scores])
    softmin = -torch.logsumexp(-8.0 * Rcs, 0) / 8.0
    k = max(1, len(scores) // 3)
    tail = torch.sort(Rcs).values[:k].mean()
    robust = 0.4 * Rcs.mean() + 0.35 * softmin + 0.25 * tail
    L = (-robust + 0.15 * cos_.mean() + 0.15 * Tts.mean()
         + 0.30 * As.mean())
    if R0 is not None:
        rx, ry = R0[0, 0], R0[1, 1]
        num = (rx * torch.conj(ry)).real
        den = torch.abs(rx) * torch.abs(ry) + 1e-9
        L = L + 0.10 * 0.5 * (1.0 + num / den) \
            + 0.05 * (torch.abs(rx) - torch.abs(ry)) ** 2
    return L


def minibatch(it, iters, seed, worst):
    """Deterministic structured angular minibatch (5 states/iter).
    Curriculum never drops oblique incidence; slot for mined worst
    angles once a full-pool eval exists."""
    rng = np.random.default_rng(seed * 7919 + it)
    frac = it / max(1, iters)
    if frac < 0.25:
        lows, mids, highs = [0.0], [30.0], [60.0]
    elif frac < 0.6:
        lows, mids, highs = [0.0, 15.0], [45.0], [60.0, 75.0]
    else:
        lows, mids, highs = [0.0, 15.0], [30.0, 45.0], [60.0, 75.0, 80.0]

    def pick(pool):
        return float(pool[rng.integers(len(pool))])

    def phi():
        return float(PHI_POOL[rng.integers(3)])

    batch = [(0.0, float(PHI_POOL[it % 3]))]
    batch.append((pick(mids), phi()))
    batch.append((pick(highs), phi()))
    if worst and rng.random() < 0.5:      # hard-angle oversampling
        th, ph = worst[rng.integers(len(worst))]
        batch.append((float(th), float(ph)))
    else:
        batch.append((pick(highs), phi()))
    if worst:
        th, ph = worst[rng.integers(len(worst))]
        batch.append((float(th), float(ph)))
    else:
        batch.append((pick(lows + mids + highs), phi()))
    return batch


def full_pool_eval(rho, P, H, order=(7, 7)):
    """No-grad evaluation over the full training pool (7 theta x 3 phi).
    Returns rows (dicts) and the worst-30% angle list by R_cross."""
    rows = []
    with torch.no_grad():
        for th in THETA_POOL:
            for ph in PHI_POOL:
                R, T = jones_angle(rho, P, H, th, ph, order=order)
                s = angle_scores(R, T)
                rows.append({'theta': th, 'phi': ph,
                             'R_cross': float(s['Rc']),
                             'R_co': float(s['co']),
                             'T_tot': float(s['Tt']),
                             'A': float(s['A'])})
    srt = sorted(rows, key=lambda r: r['R_cross'])
    worst = [(r['theta'], r['phi']) for r in srt[:6]]
    return rows, worst


def band_weights(thetas):
    """Solid-angle band weights for a sorted theta sample list: each
    sample owns [midpoint below, midpoint above]; w = cos(lo)-cos(hi)."""
    th = np.asarray(sorted(thetas), dtype=float)
    edges = np.concatenate([[0.0], (th[:-1] + th[1:]) / 2,
                            [th[-1] + (th[-1] - th[-2]) / 2]])
    edges = np.clip(edges, 0.0, 90.0)
    lo, hi = np.radians(edges[:-1]), np.radians(edges[1:])
    w = np.cos(lo) - np.cos(hi)
    return {t: w[i] / w.sum() for i, t in enumerate(th)}


def pool_summary(rows):
    """Ledger metrics from full-pool rows (spec section 26/28)."""
    import numpy as _np
    Rc = _np.array([r['R_cross'] for r in rows])
    co = _np.array([r['R_co'] for r in rows])
    Tt = _np.array([r['T_tot'] for r in rows])
    A_ = _np.array([r['A'] for r in rows])
    ths = sorted({r['theta'] for r in rows})
    bw = band_weights(ths)
    # phi-average per theta, then solid-angle weight
    omega = sum(bw[t] * _np.mean([r['R_cross'] for r in rows
                                  if r['theta'] == t]) for t in ths)
    iworst = int(Rc.argmin())
    q = int(max(1, len(rows) // 4))
    return {'Rc_mean': float(Rc.mean()), 'Rc_min': float(Rc.min()),
            'Rc_soft': float(-np.log(np.mean(np.exp(-8 * Rc))) / 8),
            'Rc_tail25': float(np.sort(Rc)[:q].mean()),
            'Rc_omega': float(omega),
            'co_mean': float(co.mean()), 'co_max': float(co.max()),
            'T_mean': float(Tt.mean()), 'T_max': float(Tt.max()),
            'A_mean': float(A_.mean()), 'A_max': float(A_.max()),
            'theta_worst': rows[iworst]['theta'],
            'phi_worst': rows[iworst]['phi']}


def diffraction_thresholds():
    """Exact specular-only limits: first nonzero order (m,n) opens in
    medium n_med when |k0 sin(th)(cos,sin)phi + G| < k0 n_med for some
    (m,n) != 0. Worst case over phi/G-direction: sin(th) > lam/P - n_med.
    Returns per-P opening theta_air (deg) in air and glass (>90 = never)."""
    out = []
    for P in PERIODS + [271.0]:
        row = {'P': P}
        for med, n_med in (('air', 1.0), ('glass', N_GLASS)):
            s = LAM0 / P - n_med
            row[f'sin_open_{med}'] = s
            row[f'theta_open_{med}'] = (math.degrees(math.asin(s))
                                        if s <= 1.0 else 999.0)
        out.append(row)
    return out


def write_rows(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
