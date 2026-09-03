# Perfect-R ceiling campaign - main report (in progress; theta=0 sections frozen)

Objective: coherent global-phase-invariant fidelity F_ideal to the ideal
reflective PB half-wave operator exp(i psi)[[0,1],[1,0]] (circular basis),
with augmented constraints on transmission and co-pol and NO absorption
penalty (spec secs 1-4; validated in
PERFECT_R_CONVENTIONS_AND_VALIDATION.md). Fixed 15-nm rotation-safe
padding; D2 / C2 / FULL symmetry branches; 1-3 islands allowed at
discovery; no multipole terms in any objective.

## 1. What the workspace already contained (mining, 184 geometries)

Under corrected conventions the historical partial solutions are
genuinely partial (PERFECT_R_WORKSPACE_MINING.md): the best F_ideal was
0.527 (old A, P271/H185; T 0.08, co 0.08, A 0.31); the "mirror" pieces
reach R_tot 0.84 with T 0.002 but zero retardance (F = 0); the
phase-perfect piece (old EQ bow-tie) has co 0.005 and 10-deg error at
F 0.505 but A 0.43; the lowest-T piece (0.016) and lowest-A piece (0.15)
sit at F 0.52 and 0.31. Nine incomplete checkpoints were recovered; none
beat the recorded finals (best 0.498). Under the ideal-fidelity metric,
EVERY historical champion collapses to min F <= 0.09 somewhere in
0-50 deg (results/mining_angular_set.csv).

## 2. Stage I - normal-incidence ceiling in real a-Si (232 D2 runs,
##    6 P x 6 H, 4 base + 7 extra seed families at the 8 top basins)

| | F_ideal | T | R_co | A | phase err | islands | min Si / gap (nm) |
|---|---|---|---|---|---|---|---|
| best fab-valid: D2 P278/H230 (newA-warm) | **0.585** | 0.010 | 0.006 | 0.399 | 3 deg | 1 | 61 / 78 |
| runner-up: D2 P272/H230 | 0.580 | 0.007 | 0.008 | 0.406 | 5 deg | 1 | 65 / 77 |
| discovery-only: D2 P278/H260 (rect-warm) | 0.617 | 0.008 | 0.002 | 0.372 | 6 deg | 3 | **14 / 3** (pathological) |
| historical best (old A) | 0.527 | 0.054 | 0.065 | 0.355 | 39 deg | 1 | 110 / 24 |

Findings that are now final at theta = 0:
- Transmission and co-pol are SOLVED as constraints (T ~ 0.01, co ~
  0.006, phase error 3-6 deg) - the ideal-operator objective with hard
  T/co continuation removes exactly the failure channels the previous
  R_cross objectives left open (newA: T 0.24).
- The entire residual deficit is absorption: A = 0.37-0.43 in every
  leading state. 1 - F ~ A within a few percent.
- F rises monotonically with period across the device grid
  (226 -> 278) and peaks at H = 200-260; H = 290 is worse (no H = 320
  extension warranted).
- The 15-nm padding basin outperforms the old 10%-rule basins at every
  shared (P, H) (sec 9 / Q9: yes, the tighter envelope opened a better
  basin).
- Multi-island states appear among the discovery leaders (0.617 with 3
  islands) but the fab-valid frontier (>= 30 nm features) is
  single-island at 0.585; multi-island leaders carry 3-15 nm slots and
  are held to the fab-robust re-optimization gate (sec 13) before any
  claim (Q10: helps at discovery, unproven after fab enforcement -
  reopt results below).

## 3. Symmetry branches (Q11, Q12)

| branch | runs | best F (any) | best F fab-valid | T / co / A of best fab-valid |
|---|---|---|---|---|
| D2 | 232 | 0.617 (pathological) | **0.585** | 0.010 / 0.006 / 0.399 |
| C2 | 24 | 0.577 (3 isl, 2.8-nm gaps) | 0.561 | 0.021 / 0.018 / 0.401 |
| FULL | 16 | 0.553 (2.8-nm gaps) | 0.498 | 0.021 / 0.006 / 0.465 |

Removing the mirror symmetries did NOT open a better reflective state:
C2 and FULL converge to the same F ~ 0.55-0.58 plateau with the same
A ~ 0.40, and their fab-valid members are below D2's. The D2 (C2v)
constraint of the previous "freeform" campaigns was not what limited
them (Q11: no; Q12: no).

## 4. THE ceiling experiment - lossless optimization from scratch (Q24-27)

24 runs with Im(n_aSi) = 0 from initialization, identical envelope,
constraints, filters and seeds (lossless_ceiling/):

| | F_ideal | T | R_co | A |
|---|---|---|---|---|
| best lossless-optimized (D2 P278/H170, random seed) | **0.985** | 0.002 | 0.013 | 0 |
| best lossless fab-valid (D2 P272/H200, warm) | 0.983 | 0.013 | 0.003 | 0 |
| best real-material fab-valid | 0.585 | 0.010 | 0.006 | 0.399 |

Lossless optimization approaches UNITY (0.985) in the same single-layer,
15-nm-padded, D2 design space in which real a-Si stalls at 0.585. The
gap (0.40) equals the absorbed fraction of the real state. Therefore:
the single-layer / footprint / symmetry / port geometry is NOT the
limit - the ideal reflective PB operator is realizable by this
architecture - and MATERIAL ABSORPTION is the ceiling (verdict category
C is the operative one at theta = 0; the large-P footprint diagnostic
and the angular stages are reported in the sections that follow).

(sections 5+ appended as the campaign proceeds: large-P footprint
diagnostic, fab-robust re-optimization, Stage II/III angular
continuation with PB-rotation fidelity, forensics, qualification,
30 answers, verdict)

## 5. Basin hopping, multi-island and the fabrication gate (secs 11, 13, 21, 33)

Controlled latent mutations (3 strengths, parent-child provenance in
mutation/) of the two fab-valid Stage-I leaders showed the parents were
NOT at their basin ceiling:

| generation | best child | F | T | co | A | err | islands | min Si / gap / edge (nm) |
|---|---|---|---|---|---|---|---|---|
| parents | D2 P278/H230 | 0.585 | 0.010 | 0.006 | 0.399 | 3 | 1 | 61 / 78 / 9 |
| gen 1 (4 runs, all beat parents) | mut4 P272/H230 | 0.639 | 0.010 | 0.016 | 0.335 | 18 | 1 | 71 / 2.8 / 1.9 |
| gen 1, fully fab-valid | mut1 P278/H230 | **0.615** | 0.008 | 0.005 | 0.372 | 9 | 1 | 72 / 72 / 5.3 |
| gen 2 (6 runs) | mut16 P278/H230 | **0.650** | 0.012 | 0.010 | 0.328 | 13 | 1 | 72 / 2.9 / 0.04 |

Every gain came with absorption dropping (0.40 -> 0.33): the mutations
find states that store less energy in the a-Si. The higher-F children
carry a one-pixel (2.9 nm) air sliver and ride the rotation-safe
envelope edge; the best FULLY fab-valid real-material state is mut1 at
F = 0.615 (72/72 nm features, 5 nm edge clearance).

Multi-island (Q10): 3-island states reach the highest raw discovery F
(0.617-0.623) but every one carries 3-15 nm slivers, and under the
erosion/dilation-robust objective (joint eta = 0.35/0.5/0.65) they
collapse (0.623 -> 0.476 nominal, 0.09 dilated). Multi-island does not
survive the fabrication gate; the fab-valid frontier is single-island.

Direct fabrication-robustness test of the clean champion (spec sec 33,
frozen binary geometry, [9,9]):

| perturbation | F | T | co | phase err |
|---|---|---|---|---|
| nominal | 0.585 | 0.010 | 0.006 | 3 deg |
| height -10 / -5 / +5 / +10 nm | 0.560 / 0.579 / 0.581 / 0.569 | <= 0.022 | <= 0.047 | 17-31 deg |
| lateral bias -5.8 nm (erode) | 0.278 | 0.340 | 0.155 | 70 deg |
| lateral bias +5.8 nm (dilate) | 0.417 | 0.183 | 0.036 | 0.4 deg |
| lateral bias +-11.6 nm | 0.185 / 0.007 | 0.42 / 0.85 | - | - |

Height tolerance is excellent (+-10 nm costs < 0.03), but the state is
LATERALLY CRITICAL: a 2-pixel (5.8 nm) boundary bias costs 0.17-0.31
of F and re-opens transmission (0.18-0.34). The erosion/dilation-robust
re-optimization of the same champion converged to F ~ 0.48 - i.e. the
ROBUSTNESS-CONSTRAINED real-material ceiling at theta = 0 is ~0.5, not
0.6. This is the honest answer to "does it survive a 2-3 nm
perturbation": the ideal-operator state needs its boundary held to
~+-3 nm laterally.

## 6. Footprint diagnostic (sec 16, theta = 0 only, NOT device candidates)

P = 300/H170 (first completed trio): F = 0.590 / 0.605 / 0.541 with
A = 0.37-0.38 - about +0.06 over the device-grid value at the same
height (0.535 at P278/H170) and far below the lossless ceiling. The
remaining P = 330/400 runs are appended below when complete.

Cross-candidate fabrication sensitivity (frozen binaries, [9,9];
lateral bias in 2-pixel steps of 5.8 nm; height +-5/10 nm):

| candidate | F nominal | F at -5.8 / +5.8 nm | F at -11.6 / +11.6 nm | F at H -10 / +10 nm |
|---|---|---|---|---|
| champ585 (P278/H230) | 0.585 | 0.278 / 0.417 | 0.007 / 0.185 | 0.560 / 0.569 |
| mut1 (fab-valid, P278/H230) | 0.615 | 0.439 / 0.330 | 0.089 / 0.182 | 0.591 / 0.615 |
| mut4 (P272/H230) | 0.639 | 0.514 / 0.303 | 0.178 / 0.188 | 0.614 / 0.649 |
| mut14 (P278/H230) | 0.644 | 0.528 / 0.283 | 0.211 / 0.167 | 0.614 / 0.657 |

All four share one signature: height is a benign knob (<= 0.03 loss
over +-10 nm, the mutation children even improve slightly at +10 nm),
lateral boundary position is critical (a 5.8-nm bias costs 0.1-0.3, an
11.6-nm bias destroys the state). The mutation children are somewhat
more erosion-tolerant than the Stage-I champion (0.51-0.53 vs 0.28 at
-5.8 nm). Any claim of F > 0.6 in real a-Si therefore carries the
explicit requirement of ~+-3 nm lateral CD control.

## 7. Robustness-constrained ceiling and basin saturation

Erosion/dilation-robust re-optimization (joint eta = 0.35/0.5/0.65,
35-nm filter, warm-started from each leader):

| start | robust F (nominal) | eroded / dilated F | T | A | features (nm) |
|---|---|---|---|---|---|
| D2 P272/H230 (mut4, 0.639) | **0.539** | 0.337 / 0.342 | 0.018 | 0.425 | 71 / 60 |
| C2 P272/H200 (0.577) | 0.524 | 0.509 / 0.449 | 0.019 | 0.448 | 8.5 / 2.8 (still slivers) |
| D2 P278/H230 (champ, 0.585) | 0.482 | 0.391 / 0.234 | 0.093 | 0.424 | 78 / 72 |
| D2 P278/H260 3-island (0.623) | 0.476 | 0.445 / 0.090 | 0.107 | 0.378 | 14 / 55 |

Requiring the ideal operator to survive +-1 projection-threshold step
(~ +-6 nm) costs 0.1-0.15 of F: the robustness-constrained
real-material ceiling is F ~ 0.52-0.54 (T ~ 0.02, A ~ 0.43). Three
generations of basin hopping saturate at F = 0.648-0.651 (raw,
sliver-bearing) - the raw normal-incidence real-material ceiling in
the P272-278 / H230 basin is ~0.65 with A = 0.33, and the full
hierarchy of honest numbers is therefore:

    raw discovery      0.65   (2.9-nm slivers, edge-riding)
    fab-valid (>=30nm) 0.615  (mut1, 72/72 nm)
    robust (+-6 nm)    0.54
    lossless-optimized 0.985  (same space, k = 0)

## 8. Complex-port forensics at theta = 0 (secs 29-31; Q13-14, Q17-19)

First-order ladders (t = t_bg + t_ED + t_MD + t_EQ + residual, and the
reflection analogue; per-channel exact coupling calibration; MQ carried
in the residual and reported by its exact power fraction) for the three
real-material leaders, principal channels x (p-like) and y (s-like):

| candidate | ch. | |t|^2 full | |r|^2 full | ladder pieces |ED| / |MD| / |EQ| (t or r) | f_MQ | residual |
|---|---|---|---|---|---|---|
| champ585 | x | 0.013 | 0.480 | r: 0.32 / 0.37 / **0.77** ; t: 0.24 / 0.40 / **0.84** | 0.06 | 0.23-0.50 |
| champ585 | y | 0.008 | 0.702 | r: 0.60 / **0.93** / 0.18 ; t: 0.38 / **0.87** / 0.17 | 0.01 | 0.16-0.22 |
| mut1 (0.615) | x | - | 0.571 | r: 0.11 / 0.18 / **0.75** | 0.07 | 0.20 |
| mut1 (0.615) | y | 0.011 | 0.669 | r: 0.71 / 0.76 / 0.37 ; t: 0.41 / 0.64 / 0.31 | 0.03 | 0.26-0.37 |
| mut22 (0.651) | x | 0.000 | 0.705 | r: 0.19 / 0.05 / **0.81** ; t: 0.14 / 0.05 / **0.89** | 0.07 | 0.20-0.49 |
| mut22 (0.651) | y | 0.023 | 0.616 | r: 0.68 / 0.50 / 0.55 ; t: 0.39 / 0.41 / 0.45 | 0.06 | 0.31-0.46 |

The ideal-operator optimizer did NOT return to the paper's ED/MD
recipe. Every leader is a hybrid: the x (p-like) channel is
ELECTRIC-QUADRUPOLE-LED in both its transmission cancellation
(|t_x|^2 = 0.000-0.013, EQ piece 0.84-0.89 vs ED <= 0.24) and its
reflection (EQ 0.75-0.81), while the y channel is dipole-led
(MD 0.5-0.93 with ED 0.6-0.7). The pi reflection-phase difference is
thus set BETWEEN a quadrupolar x-response and a dipolar y-response -
a higher-order transmission-zero mechanism of exactly the causal type
the P0750 audit established (external t -> 0 by background + multipole
sum, not a dark internal mode), now realised for both principal
channels simultaneously. The ladders are first-order: residuals of
0.2-0.5 (MQ power fraction 0.01-0.07 plus higher orders / multiple
scattering) are shown, not hidden; the qualitative EQ-led assignment
is robust because the EQ piece exceeds the residual in every x row.
Absorption (A 0.33-0.40) is the energy the resonant EQ/MD currents
dissipate in k = 0.069 a-Si - the same currents that make the
cancellation work. Topology (Q17-19): all leaders are single
rounded, notched bars inside the envelope - simple in outline, but the
simplicity is NOT evidence of an ED/MD optimum: the current pattern is
quadrupolar on x.
