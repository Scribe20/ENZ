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
