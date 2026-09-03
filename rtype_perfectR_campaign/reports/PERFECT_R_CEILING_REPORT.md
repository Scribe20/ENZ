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
| D2 P278/H230 (mut22, 0.651) | **0.588** | 0.366 / 0.326 | 0.003 | 0.408 | 72 / 26 |
| D2 P278/H230 (mut2, 0.631) | 0.569 | 0.373 / 0.347 | 0.011 | 0.414 | 67 / 61 |
| D2 P272/H230 (mut4, 0.639) | 0.539 | 0.337 / 0.342 | 0.018 | 0.425 | 71 / 60 |
| C2 P272/H200 (0.577) | 0.524 | 0.509 / 0.449 | 0.019 | 0.448 | 8.5 / 2.8 (still slivers) |
| D2 P278/H230 (champ, 0.585) | 0.482 | 0.391 / 0.234 | 0.093 | 0.424 | 78 / 72 |
| D2 P278/H260 3-island (0.623) | 0.476 | 0.445 / 0.090 | 0.107 | 0.378 | 14 / 55 |

Requiring the ideal operator to survive +-1 projection-threshold step
(~ +-6 nm) costs 0.1-0.15 of F: the robustness-constrained
real-material ceiling is F ~ 0.54-0.59 (T <= 0.02, A ~ 0.41-0.43). Three
generations of basin hopping saturate at F = 0.648-0.651 (raw,
sliver-bearing) - the raw normal-incidence real-material ceiling in
the P272-278 / H230 basin is ~0.65 with A = 0.33, and the full
hierarchy of honest numbers is therefore:

    raw discovery      0.65   (2.9-nm slivers, edge-riding)
    fab-valid (>=30nm) 0.615  (mut1, 72/72 nm)
    robust (+-6 nm)    0.57-0.59
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

## 9. Stage II - small-angle continuation (theta 0-30, phi 0/45/90; sec 22)

Four theta=0 leaders were continued with 4-state angular minibatches,
multi-failure hard mining (worst F, largest T, largest R_co) and the
same constraint continuation (continuation/, cont30_ledger.csv):

| parent | F0 after | pool F mean / min (0-30) | T max | co max | A mean | rotation fidelity mean / min | features |
|---|---|---|---|---|---|---|---|
| mut1 (fab-valid 0.615) | 0.598 | **0.444 / 0.211** | 0.161 | 0.108 | 0.462 | **0.329 / 0.135** | 78 / 2.9 |
| Stage-I champ (0.585) | 0.502 | 0.331 / 0.209 | 0.140 | 0.116 | 0.518 | 0.226 / 0.052 | 78 / 72 |
| P272/H230 (0.580) | 0.495 | 0.321 / 0.229 | 0.159 | 0.100 | 0.530 | 0.211 / 0.068 | 77 / 77 |
| C2 P278/H200 (0.561) | 0.517 | 0.438 / 0.208 | 0.184 | 0.133 | 0.461 | 0.186 / **0.006** | 107 / 96 |

Findings: (i) every near-ideal theta=0 state loses ~half its fidelity by
30 deg (pool means 0.32-0.44, minima 0.21-0.23 at (30, 90)) even after
angle-aware continuation - the ideal operator is phase-critical in
k_par exactly as the previous campaign's PB-law analysis implied;
(ii) absorption RISES off-axis (A mean 0.46-0.53), so the angular loss
is again dissipative; (iii) the C2 candidate matches the D2 leader on
static pool fidelity but its PHYSICAL-ROTATION fidelity collapses
(min 0.006): without the two mirrors the principal axes do not stay
locked to the motif orientation under oblique incidence, so C2 states
are not PB-addressable off-axis (Q11 answered on the device level:
C2 does not outperform D2, and fails the rotation test); (iv) the
mutation child remains the best device candidate on every axis.

Stage III (0-55 with physical-rotation states in the loss, sec 23-24)
on the mut1 continuation, in progress at this writing: pool F mean
0.34, min 0.01 at theta = 55 (phi = 0) - the challenge angle is not
held by any single-layer state, consistent with the measured PB-law
collapse between 45 and 60 deg.

## 10. Numerical and spectral qualification of the fab-valid leader (secs 32, 34)

mut1 (D2 P278/H230, 72/72 nm features), Fourier-order convergence on
complex eigenvalues (closure = 1.000 at every order):

| order | F | T | co | A | eig_R0 (re, im) | eig_R1 (re, im) | |eig_T| |
|---|---|---|---|---|---|---|---|
| 9 | 0.615 | 0.008 | 0.005 | 0.372 | 0.052 + 0.754i | -0.189 - 0.796i | 0.07 / 0.11 |
| 11 | 0.611 | 0.018 | 0.004 | 0.367 | 0.130 + 0.734i | -0.238 - 0.787i | 0.13 / 0.14 |
| 13 | 0.610 | 0.021 | 0.004 | 0.366 | 0.153 + 0.725i | -0.255 - 0.783i | 0.15 / 0.14 |
| 15 | 0.606 | 0.026 | 0.003 | 0.365 | 0.184 + 0.710i | -0.269 - 0.779i | 0.17 / 0.15 |

F is stable to 0.009 over orders 9-15; the complex reflection
eigenvalues drift by ~0.13 in the real part (a slow phase rotation of
both eigenvalues together, leaving their ratio - the half-wave relation
- intact: |eig| 0.76/0.82 -> 0.73/0.82, phase difference 189 -> 193 deg)
while transmission drifts 0.008 -> 0.026. All conclusions in this
report use margins far larger than these drifts, but the transmission
floor should be quoted as "< 0.03", not 0.008.

Eigenchannels (sec 31): eigenpolarizations are exactly linear along x
and y (axis angles 0.00/90.00 deg, ellipticity 0.000) for both mut1 and
mut22 - the D2 constraint delivers clean principal axes; eigenvalue
magnitudes 0.76/0.82 (mut1) and 0.84/0.78 (mut22) with retardance
189 and 193 deg (errors 9 and 13 deg).

Spectrum 620-645 nm (0.5-nm steps near 633): F rises monotonically
0.543 (620) -> 0.615 (633) -> 0.667 (645, the scan edge), with
T <= 0.010 and co <= 0.008 everywhere and A falling 0.44 -> 0.34. The
state is broadband (the entire scan lies above half-maximum) - it is
NOT a high-Q needle; 633 nm sits on the rising flank of a broad
resonance whose peak is slightly red of the design wavelength. Both
labels of sec 32 therefore coincide: the practical broadband champion
IS the peak-R physics champion of this basin.

## 11. Stage III - device PB range 0-55 deg with physical-rotation states in the loss (secs 23-25)

The Stage-II leader (mut1 lineage) was continued on the pool theta =
{0,15,30,40,50,55} x phi = {0,45,90} with, in the second half, the
motif physically rotated by alpha = {30,60,90} at theta = {0,30,45,50}
and scored against the rotated ideal operator U_alpha; multi-failure
mining oversampled the worst F, largest T, largest R_co and worst
rotated state. Frozen hard-binary result (continuation/, cont55):

| theta (deg) | F min / mean (static, over phi) | T max | co max |
|---|---|---|---|
| 0 | 0.518 / 0.518 | 0.100 | 0.013 |
| 15 | 0.406 / 0.473 | 0.081 | 0.019 |
| 30 | 0.193 / 0.318 | 0.171 | 0.110 |
| 40 | 0.193 / 0.285 | 0.124 | 0.208 |
| 50 | 0.115 / 0.235 | 0.118 | 0.300 |
| 55 (challenge) | 0.011 / 0.174 | 0.114 | 0.404 |

Rotated-operator fidelity F(U_alpha) of the same geometry:

| theta | alpha = 30 | 60 | 90 |
|---|---|---|---|
| 0 | 0.513 | 0.512 | 0.518 |
| 30 | 0.157 | 0.326 | 0.506 |
| 45 | 0.142 | 0.211 | 0.426 |
| 50 | 0.113 | 0.179 | 0.402 |

Reading: at normal incidence the rotated fidelity is flat in alpha
(0.51-0.52) - the PB law is exact, as it must be for a D2 motif. Off
axis the fidelity becomes strongly ORIENTATION-DEPENDENT: a motif
rotated 90 deg relative to the incidence plane keeps 0.40-0.51 up to
50 deg, but rotated 30 deg it drops to 0.11-0.16 by 30-50 deg. In a
metalens every orientation occurs, so the usable figure is the minimum
over alpha: ~0.5 at 0 deg, ~0.16 at 30 deg, ~0.11 at 50 deg, ~0 at 55.
Co-pol leakage grows to 0.30-0.40 at 50-55 deg while T stays <= 0.17.
This is the SAME phenomenology the wide-FOV campaign measured for the
paper rectangle and its dipolar champions (PB law clean to 30, marginal
at 45, gone by 60-75): the ideal-operator states, being phase-critical
resonant states, follow the same curve from a higher starting point.
Angle-aware optimization with rotation states in the loss did not move
that boundary; it only preserved F0 (0.565 vs 0.615 for the theta0
version) while flattening the 0-15 deg region.

Direct bias test of the robust-optimized state (robust588, frozen
binary): F = 0.588 nominal; 0.432 / 0.397 at -5.8 / +5.8 nm;
0.158 / 0.236 at +-11.6 nm; 0.558-0.584 over H +-10 nm. Compared with
the Stage-I champion (0.278 / 0.417 at +-5.8 nm) the robust objective
did buy real erosion tolerance (+0.15 at -5.8 nm) at the same nominal
F - the +-6 nm lateral criticality is softened, not removed.

## 12. Raw-best qualification (mut22, F 0.651) and the physical-rotation law (secs 24, 32-34)

mut22 (D2 P278/H230, 2.9-nm sliver, edge-riding): convergence orders
9/11/13/15 -> F 0.651/0.640/0.636/0.631, T 0.012/0.026/0.031/0.039,
closure 1.000; spectrum F(633) = 0.651, peak 0.690 at 645 nm, whole
620-645 scan above half-maximum (broadband, like mut1); bias
sensitivity F = 0.499 / 0.304 at -5.8 / +5.8 nm, 0.213 / 0.159 at
+-11.6 nm; height +-10 nm: 0.620-0.658. Same signature as every other
leader: height-benign, laterally critical, order-stable to ~0.02.

Physical-rotation PB law (alpha = 0..180 in 15-deg steps, phi = 0,
[9,9]), Stage-III device leader:

| theta | fitted slope (ideal -2) | rms (deg) | F(U_alpha) min / mean | R_cross range over alpha |
|---|---|---|---|---|
| 0 | -2.003 | 8.6 | 0.541 / 0.555 | 0.541-0.565 |
| 30 | -2.035 | 15.2 | 0.172 / 0.283 | 0.207-0.546 |
| 45 | -2.150 | 14.5 | 0.147 / 0.222 | 0.166-0.451 |
| 50 | -2.144 | 15.3 | 0.097 / 0.177 | 0.100-0.425 |

robust588 at theta = 0: slope -1.979, rms 8.1, F(U_alpha) 0.545-0.575
(oblique rows appended in results/pb_matrix_fidelity.csv).

Reading: the geometric-phase SLOPE stays within 7% of -2 up to 50 deg
for this state (better than the dipolar wide-FOV champions, whose
slope broke at 60 deg), so the phase law itself is not what fails first
here. What fails is the OPERATOR FIDELITY under rotation: the
cross-circular amplitude becomes strongly orientation-dependent off
axis (R_cross 0.21-0.55 at 30 deg, 0.10-0.43 at 50 deg), so the
worst-orientation fidelity falls to 0.17 (30 deg) and 0.10 (50 deg).
For a 360-deg metalens the worst orientation is the binding figure.
Answer to "does its physical rotation obey the PB law?": in phase,
yes to 50 deg; in amplitude/fidelity, only to ~15-20 deg at the
> 0.4 level.

## 13. Footprint diagnostic update (sec 16)

Large-P normal-incidence runs (ceiling_largeP/, [9,9] optimization,
NOT device candidates): P300/H170 best 0.605 (A 0.375); P330/H170 best
0.636 (5 islands, A 0.327). Relaxing the footprint from 278 to 330 nm
buys ~+0.10 at fixed H but the absorbed fraction stays 0.33-0.38 - the
large-P states hit the same wall as the device-grid states, at a level
still far below the lossless 0.985. (P400 and H230 rows are appended
to results/ceiling_largeP_ledger.csv as they complete; none can change
the category because absorption, not footprint, is what the lossless
comparison isolates.)

## 14. The 30 required answers (sec 38)

1. **Highest F_ideal at theta=0 in real a-Si:** 0.651 raw (mut22, D2
   P278/H230, 2.9-nm sliver, edge-riding); 0.615 fully fab-valid (mut1,
   72/72 nm); 0.588 erosion/dilation-robust (robust588).
2. **Highest R_cross(0) with T_tot < 0.10:** 0.651 (T = 0.012) - every
   leader satisfies T < 0.10 by construction.
3. **With T_tot < 0.05:** 0.651 (T = 0.012); T <= 0.03 holds to order
   15 (0.039 at [15,15] for mut22, 0.026 for mut1).
4. **Lowest T while R_cross > 0.5:** 0.001-0.003 (robust588: T 0.003 at
   F 0.588; several Stage-I states at T 0.002-0.007, F 0.55-0.57).
5. **> 0.6:** T = 0.008 (mut1, F 0.615) / 0.010 (mut16-22, F 0.65).
6. **> 0.7:** none - no real-material state reached F or R_cross 0.7.
7. **Does any real candidate approach R_cross > 0.8?** No. Raw maximum
   0.651 after 250+ theta0 runs, 3 mutation generations (saturating at
   0.648-0.651) and three symmetry branches; the footprint-relaxed
   diagnostic reaches 0.636 at P330. The lossless twin reaches 0.985.
8. **Limiting failure:** ABSORPTION. In every leader T <= 0.01,
   R_co <= 0.02, phase error 3-18 deg, and 1 - F ~ A = 0.33-0.40.
9. **Does the 15-nm padding open a better basin?** Yes: at every shared
   (P,H) the 15-nm-envelope runs beat the old 10%-rule basins
   (0.585 vs 0.527 best-vs-best; the leaders use 90-95% of the envelope
   radius, which the old rule forbade).
10. **Do multiple islands help?** At discovery, marginally (3-island
    states 0.617-0.623 vs 0.585 single-island) - but every multi-island
    leader carries 3-15-nm slivers and collapses under the
    erosion/dilation-robust objective (0.623 -> 0.476). No fab-valid
    multi-island state beat the fab-valid single-island frontier.
11. **Does C2 outperform D2?** No: best C2 0.577 (sliver) / 0.561
    fab-valid vs D2 0.651 / 0.615; and the C2 candidate fails the
    physical-rotation test off-axis (rotation-fidelity min 0.006 over
    0-30 deg) because its principal axes are not locked to the motif.
12. **Does unrestricted freeform outperform C2?** No: best FULL 0.553
    (sliver) / 0.498 fab-valid - the lowest of the three branches; it
    reproduces the same A ~ 0.40 plateau.
13. **Is the old ED/MD family still selected?** No. The x (p-like)
    channel of every leader is EQ-led in both its transmission
    cancellation (EQ piece 0.84-0.89) and its reflection (0.75-0.81);
    only the y channel is dipole-led.
14. **Or a higher-order transmission-zero mechanism?** Yes - an
    EQ-led external transmission zero on x (|t_x|^2 = 0.000-0.013)
    coexisting with a dipole-led zero on y, i.e. the P0750-type causal
    mechanism (background + multipole sum -> 0 at the port) realised
    for both principal channels with a pi retardance between them.
15. **What did the workspace contain that was overlooked?** Separately
    solved pieces: mirror-like states (R_tot 0.84, T 0.002, no
    retardance), a phase-perfect low-co state (10 deg, co 0.005, A
    0.43), low-T (0.016) and low-A (0.15) states; and 9 incomplete
    checkpoints. Their warm/mix seeds (newA-warm, rect-warm,
    mirror+oldB mixes) are what seeded the winning basin.
16. **Promising incomplete checkpoints beating recorded finalists?**
    No - the best recovered checkpoint reached 0.498 (< 0.527 recorded).
17. **Does the ideal matrix require a topology unlike newA?** Not in
    outline: the leaders are single notched/rounded bars descended from
    a newA warm start - but at 15-nm padding, larger P (272-278) and
    H = 230, with a quadrupolar x-current rather than newA's dipolar one.
18. **Does the optimum remain a simple rounded bar?** Yes in silhouette
    (single island, 60-78-nm features), with small notches/slivers
    that the mutations add at the envelope edge.
19. **Is the simplicity evidence of a true robust optimum?** No. The
    simple outline hides a phase-critical resonant state: +-5.8 nm of
    lateral bias costs 0.1-0.3 of F, and the multipolar content is
    quadrupolar on x. Robustness had to be bought explicitly (0.588
    robust vs 0.651 raw).
20. **Best PB-compatible 0-50 deg device candidate:** the Stage-III
    continuation of mut1 (continuation/cont55, D2 P278/H230): F0 0.565,
    static F 0.52/0.41/0.19/0.19/0.12 at 0/15/30/40/50 deg (phi-min).
21. **Its worst-angle F_ideal:** 0.115 at 50 deg (static, phi = 0-90);
    0.097 worst-orientation at 50 deg under physical rotation; 0.011
    at the 55-deg challenge point.
22. **Its T_tot and R_co:** T <= 0.17 over 0-55 (0.10 at normal
    incidence after continuation); R_co 0.013 at 0 deg rising to 0.30
    at 50 and 0.40 at 55 deg - co-pol leakage, not transmission, is
    the off-axis failure channel.
23. **Does its physical rotation obey the PB law?** In phase, yes:
    fitted slope -2.00 / -2.04 / -2.15 / -2.14 at 0/30/45/50 deg
    (rms 9-15 deg). In operator fidelity, only near normal incidence:
    F(U_alpha) is flat (0.54-0.56) at 0 deg but orientation-dependent
    off axis (min 0.17 at 30, 0.10 at 50 deg).
24. **Best lossless-optimized candidate:** D2 P278/H170 random seed,
    F = 0.985, T 0.002, co 0.013 (fab-valid runner-up 0.983 at
    P272/H200 with 105/94-nm features).
25. **Does lossless optimization approach unity?** Yes: 0.985 - the
    ideal reflective PB operator is realisable in this exact
    single-layer, 15-nm-padded, D2 design space.
26. **Is the real-material ceiling material-limited?** Yes, dominantly:
    identical space, identical constraints, identical seeds: 0.985
    (k = 0) vs 0.651 (k = 0.069); the 0.33 gap equals the absorbed
    fraction of the real state.
27. **Or architecture/footprint-limited?** Not primarily. The footprint
    diagnostic adds only ~+0.10 (P330) while A stays 0.33-0.38, and
    symmetry relaxation (C2/FULL) adds nothing. The architecture's own
    ceiling is > 0.98; the angular/PB range IS an architecture property
    (single-layer PB states lose operator fidelity beyond ~15-30 deg
    and the law beyond ~50 deg) - that is the one architecture limit,
    and it is not what caps F at normal incidence.
28. **Is "perfect R" plausible in single-layer a-Si at 633 nm?** No.
    With k = 0.069 the resonant currents that produce the double
    transmission zero dissipate ~1/3 of the power; F caps at ~0.65 raw,
    ~0.6 fab-valid, ~0.59 robust, and falls to ~0.1-0.2 by 30-50 deg.
29. **Minimal platform change to remove the dominant limit:** lower
    Im(n) at 633 nm AT SIMILAR INDEX (n ~ 3.9-4.3, k < 0.01 - e.g.
    hydrogenated/annealed a-Si:H), keeping single-layer, D2, 15-nm
    padding, P ~ 270-280, H ~ 230: the lossless twin shows this exact
    geometry class supports F > 0.98. A lower-index low-loss dielectric
    (GaP-like n = 3.31, exploratory sec 16) removes absorption
    (A 0.02-0.04) but loses the half-wave state in the same P/H window
    (F 0.25-0.46, retardance error ~90 deg) - it would need a taller,
    re-optimized layer. Kept separate from the a-Si result (sec 28).
30. **Strongest defensible verdict:** below.

## 15. VERDICT

**C. MATERIAL-LOSS-LIMITED.**

Evidence: (i) in the identical design space the lossless-optimized
ceiling is 0.985 while real a-Si saturates at 0.65 raw / 0.615
fab-valid / 0.59 robust after 250+ multi-start runs, three symmetry
branches, multi-island search, footprint relaxation and three
generations of basin hopping - and 1 - F equals the absorbed fraction;
(ii) transmission (<= 0.01), co-pol (<= 0.02) and retardance (3-18 deg)
are all solved at normal incidence, so no port channel other than
absorption is left to fix; (iii) neither symmetry relaxation nor
footprint relaxation moves the absorption plateau (0.33-0.40).

Qualifiers that the verdict carries explicitly: the campaign is a
STRONG IMPROVEMENT over every prior state (F 0.527 -> 0.651, with T and
co-pol reduced 5-10x) but NOT near-ideal (category B at the device
level); the states are height-tolerant but laterally critical
(+-6 nm); and the PB-usable angular range of the ideal operator is
~15-30 deg at the > 0.4 fidelity level, with the phase law surviving to
50 deg - an architecture property of single-layer PB reflectors that
this campaign did not move. "Perfect R" in this platform requires a
lower-loss material, not a different geometry.

Robust-optimized state (robust588) under physical rotation: slope
-1.98 (rms 8) at 0 deg, -2.15 (rms 20) at 30, -2.30 (rms 36) at 45,
+0.43 (rms 46) at 50 deg; F(U_alpha) min 0.545 / 0.150 / 0.093 / 0.066.
The +-6-nm-robust state is therefore MORE angle-fragile than the
Stage-III leader (whose slope held to 50 deg): robustness to boundary
bias and robustness to k_par are not the same property, and the
angle-aware continuation is what buys the latter.

Stage III on the +-6-nm-robust state (robust588 lineage): F0 0.453,
pool F mean 0.269, min 0.122 (at 40 deg), 0.207 at the 55-deg
challenge point; rotation-fidelity min 0.039. It is flatter but lower
than the mut1 lineage everywhere below 50 deg, and its rotated-operator
fidelity is worse; the device finalist therefore remains the mut1
Stage-III continuation. (Both rows: results/cont55_ledger.csv; per-angle
tables: results/angular_perfectR_finalists.csv.)

## 16. Exploratory (sec 28, kept separate): a low-loss high-index material in the SAME geometry class

To test the material-limited verdict directly rather than only by the
k = 0 counterfactual, four runs used a GaP-like index
n = 3.31 + 0.003i (eps = 10.96 + 0.02i) at 633 nm with the identical
envelope, constraints, seeds and H = 200/230 (lowloss_GaP/). Result:
F = 0.25-0.46 with T = 0.14-0.33, R_co = 0.36-0.40, A = 0.02-0.04 and
retardance errors of 82-104 deg. Absorption is indeed gone, but at the
lower index the same P/H window no longer supports the double
transmission zero plus pi retardance - the index-thickness budget of
the H <= 230 nm layer is insufficient. This sharpens the answer to
Q29: the minimal platform change is LOWER LOSS AT SIMILAR INDEX
(e.g. a-Si:H / annealed a-Si with k < 0.01, n ~ 3.9-4.3), not simply a
lower-index low-loss dielectric; a GaP/TiO2 platform would require a
taller re-optimized layer. This section is exploratory and is not part
of the apples-to-apples a-Si ceiling result.
