# Angular failure forensics

## 1. Exact failure budget (spec sec 39)

For a diagonal Jones response the identity
  1 = R_cross + T_tot + A + imbalance + retardance + mixing
holds exactly with imbalance = (|rx|-|ry|)^2/4 and retardance =
|rx||ry|(1+cos dphi)/2 (both are the two halves of R_co). Closure was
verified = 1.000 on every row (results/failure_budget.csv). phi = 0;
diffraction term identically zero (specular-only domain).

Paper rectangle: its loss at theta = 0-40 is RETARDANCE (0.44 -> 0.30
lost to the 109-deg phase error; imbalance small). Its retardance loss
falls with angle (0.016 at 60 deg) - obliquity naturally walks dphi
toward pi - which is why the rectangle is angle-TOLERANT: it starts
mistuned and never had a sharp condition to lose.

Old theta0-optimized champions (oldA/oldB): the mirror image. Nearly
perfect at theta = 0 (retardance loss 0.004-0.064), then the tuned
condition detunes catastrophically: retardance loss 0.15-0.35 by
20-60 deg plus rising T leakage, with absorption 0.3-0.5 throughout.
R_cross collapses 0.51-0.53 -> 0.08-0.15. ANSWER (Q23): retardance
failure (B) is what fails first, compounded by absorption (D);
amplitude imbalance (A) is secondary (<= 0.09); mixing (F) negligible
at phi = 0; diffraction (E) zero; modal change (C) accompanies oldB's
EQ -> mixed transition.

New angle-aware designs: retardance loss <= 0.10 (newA) at ALL angles
- the optimizer found states whose half-wave condition is angularly
FLAT. Their remaining deficit is T (0.24-0.33) + A (0.23-0.35):
amplitude, not phase.

## 2. The PB-law collapse (spec sec 36) - the campaign's sharpest limit

Physical rotation of the hard-binary structures, alpha = 0..180, fit
phase_cross = phase0 + s*alpha:

| theta | rectangle s (rms) | newA s (rms) | newB s (rms) |
|---|---|---|---|
| 0  | -1.99 (1.8)  | -1.99 (1.9)  | -1.99 (2.0)  |
| 30 | -2.05 (7.5)  | -2.06 (11.9) | -1.92 (12.1) |
| 45 | -2.16 (19.1) | -2.17 (25.2) | -1.77 (24.1) |
| 60 | -2.43 (42.4) | -2.40 (41.7) | -0.34 (27.5) |
| 75 | +0.05 (8.1)  | +0.09 (12.6) | -0.05 (9.4)  |

Three very different geometries, one envelope: the -2*alpha law is
clean to 30 deg, marginal at 45, broken at 60, and GONE at 75 - where
the fitted slope ~ 0 means the cross-circular phase is pinned by the
incidence-plane physics and no longer responds to element orientation.
The large near-grazing R_cross (0.5-0.8 for every structure, and
Omega = 0.135-0.149 even for bare glass / a uniform film) is therefore
orientation-independent Fresnel-type conversion: REAL power in the
cross channel, USELESS for encoding metalens phase. Consequently:

- "acceptance angle" by R_cross alone overstates the metalens FOV;
- the PB-usable FOV of every single-layer element tested here is
  ~<= 45-55 deg incidence;
- within that range the angle-aware designs carry 1.4-1.7x the
  rectangle's cross amplitude with ~5x less co-pol leakage.

## 3. Absorption counterfactual (spec sec 42; diagnostic only)

k = 0 (Re(n) kept), frozen geometries, phi = 0: newA lossless would be
R_cross = 0.45-0.60 across 0-60 deg (real: 0.31-0.36); the rectangle
0.33-0.42 (real: 0.23-0.28). The angular SHAPE is essentially
unchanged in both - no hidden dispersion collapse - so the residual
gap to unity is ABSORPTION-dominated (answer Q26: most of the
remaining ceiling, and almost none of the angular variation, is loss).

## 4. Mode dispersion remark (spec sec 41)

No sharp resonance tracks through the angular window for the new
designs (their responses are spectrally and angularly smooth; order
convergence at the 1e-3 level, no high-Q features). The old EQ
bow-tie's identity change (EQ 0.48 at 0 deg -> mixed by 20 deg) is the
modal-dispersion signature that accompanies its retardance detuning.
A quantitative d(lambda_res)/d(theta) is not extractable for the new
designs because they do not present an isolable resonance center in
the operating window - stated rather than fabricated.

## 5. Scope notes

- Failure budgets and the PB table use phi = 0 (s/p symmetric plane);
  the full maps show the phi spread and are the basis of the
  acceptance metrics.
- The 1st-order ladder (Argand) reconstructs s-channel transmission
  well at all angles but degrades on the p channel at >= 60 deg
  (multi-scattering growth); Argand conclusions are drawn only where
  the ladder tracks the full amplitude.
