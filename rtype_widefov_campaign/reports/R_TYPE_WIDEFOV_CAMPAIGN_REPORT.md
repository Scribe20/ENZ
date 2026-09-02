# R-type wide-FOV campaign - from-scratch angle-aware freeform design
## 633 nm reflective geometric-phase meta-atom for a 360-deg metalens

Everything here is air-side illumination, exact p/s-basis flux-normalized
Jones matrices (see R_TYPE_WIDEFOV_NUMERICAL_QUALIFICATION.md - the
xy-basis oblique normalization used implicitly at phi=45 in the previous
campaign is provably wrong and was replaced), hard-binary geometries at
order [9,9], identical grids and conventions for every structure
including the paper rectangle and the old normal-incidence champions.

## The central question and the answer

**Can a PB-rotatable, fabrication-constrained, single-layer freeform
a-Si meta-atom produce a reflection-dominant cross-circular state that
remains useful over a large fraction of a hemisphere?**

**Yes - with a caveat that binds every single-layer element equally.**
Angle-aware optimization from iteration 0 produced structures that are
cross-circular-dominant over the whole evaluated hemisphere (newA:
theta_50 = theta_20 = dominance range = 85 deg, worst-angle
R_cross = 0.204, solid-angle mean 0.401 vs rectangle 0.301), with 5x
lower co-polarized leakage than the paper rectangle. The caveat: the
GEOMETRIC-PHASE LAW ITSELF (phase = -2 alpha under element rotation)
degrades identically for every tested structure - clean to 30 deg,
marginal at 45 deg, broken at 60 deg, gone at 75 deg - so the
metalens-usable FOV of ANY single-layer R-type element is bounded by
PB-law validity (~45-55 deg), not by R_cross magnitude. Within that
PB-valid range the new designs still beat the rectangle at every angle.

## MASTER TABLE (spec section 50)

All under identical metrics: theta 0-85 x phi 0-90 fine maps, [9,9],
hard-binary. Omega = solid-angle-weighted <R_cross>; worst = min over
the map; PB slope from physical rotation alpha = 0..180 deg.

| | paper rectangle | old A (theta0-opt) P271/H200 | old B (theta0-opt) P271/H215 | NEW A P239/H200 | NEW B P252/H185 |
|---|---|---|---|---|---|
| P / H / pad (nm) | 226/170/- | 271/200/27.1 | 271/215/27.1 | 239/200/23.9 | 252/185/25.2 |
| R_cross(0) | 0.229 | 0.526 | 0.505 | 0.314 | 0.380 |
| mean R_cross (0-85) | 0.276 | 0.281 | 0.272 | 0.375 | 0.387 |
| solid-angle Omega | 0.301 | 0.256 | 0.244 | **0.401** | 0.398 |
| worst-angle R_cross | 0.098 | 0.015 | 0.016 | **0.204** | 0.156 |
| theta_50 | 55 | 10 | 10 | **85** | 50 |
| theta_20 | 40 | 15 | 15 | **85** | 50 |
| dominance range | never | 15 | 15 | **85** | 25 |
| R_co mean / max | 0.293/0.496 | 0.136/0.316 | 0.111/0.367 | **0.057/0.116** | 0.091/0.265 |
| T_tot mean / max | 0.149/0.400 | 0.186/0.546 | 0.175/0.469 | 0.243/0.339 | 0.250/0.470 |
| PB slope th=0 / 60 | -1.99 / -2.43 | - | - | -1.99 / -2.40 | -1.99 / -0.34 |
| min Si / air feature | 96 / - | 110 / 24 | 104 / 24 | 87 / 52 | 97 / 66 |
| dominant multipoles | ED(x)/MD(y) | ED/MD 3-way | EQ bow-tie | ED-MD dipolar | MD(p)/ED(s) dipolar |

(old A/B co-pol and T stats from their fine maps; their PB rotation was
not re-run - their theta_50 = 10 deg already disqualifies them. The
rectangle's dominance range is "never" because R_co > R_cross at and
near normal incidence - the paper element is co-pol-dominant.)

Null baselines under identical metrics: bare glass Omega = 0.135,
fill-matched uniform film Omega = 0.149, both with R_cross(0) = 0 and
never reaching 0.20. This is the "free" grazing-Fresnel floor that any
acceptance claim must discount; the campaign's structural numbers stand
far above it (and the PB-collapse analysis shows the grazing conversion
is orientation-independent and hence not metalens-usable anyway).

## What the previous campaign's champions are worth at angle

Under the same metrics the previous normal-incidence champions collapse:
theta_50 = 10 deg, worst-angle R_cross = 0.015-0.016, Omega BELOW the
rectangle (0.244-0.256 vs 0.301). The angular failure budget (exact
decomposition, closure = 1.000) identifies the mechanism: RETARDANCE
DETUNING - their tuned Delta_phi = pi collapses within 20 deg (phase
loss up to 0.35 of unity by 40-60 deg), while absorption stays high.
The previous campaign's "wide-angle caveat" is thus confirmed and
sharpened: it was not an architecture limit but a basin property of
theta0-first optimization. Angle-aware optimization from iteration 0
in the SAME single-layer space recovers hemisphere-scale usefulness.

## Answers to the campaign's structural questions (sections 7, 9, 29)

- **Does the robust optimum move to smaller P?** No. The robust score
  rises monotonically toward P = 239-258 nm (all specular-only to 85+
  deg, so this is not a diffraction artifact). P = 200-213 is strictly
  worse for both methods. The small-P hypothesis is refuted in this
  design space.
- **Does it move toward H = 170?** No. The robust score still rises at
  H = 200; the one-sided extension found H = 200-210 optimal (H = 210
  slightly softer). H = 170 controls (fully refined + qualified) reach
  Omega 0.362 (A) / 0.372 (B) - both beat the rectangle at the paper's
  own height, so same-height T/R fabrication remains viable, but the
  robust optimum sits at H = 185-200.
- **Trade-off visibility:** A_P258_H200 pushes Omega to 0.448 with a
  worst-angle hole (0.077 at (45,90)); A_P239_H200 gives the flattest
  hemisphere (worst 0.204). The angular Pareto front is real and now
  mapped (FIGURE wf_F12).

## VERDICT (conservative, per section 55)

**STRONG GO - FREEFORM MOVES THE ANGULAR PARETO FRONT**, with the
explicit PB-validity qualifier. Angle-aware freeform:
- improves wide-angle useful reflection over the rectangle everywhere
  (Omega +33%, worst-angle 2.1x, theta_20 40 -> 85 deg),
- improves PB/leakage metrics massively (hemispheric co-pol 0.293 ->
  0.057; the rectangle is never cross-dominant, newA always is),
- and does so with fabricable (>=52 nm features), single-island,
  envelope-safe, reproducible (3 seeds), order-converged geometries.

What it does NOT do - and, on this evidence, no single-layer element
does: extend the geometric-phase LAW past ~45-55 deg incidence. The
-2 alpha law degrades identically for the rectangle, newA and newB
(three very different geometries), which is evidence of an
architecture-level ceiling ON THE PB MECHANISM (not on R_cross). A
full "ARCHITECTURE LIMIT" verdict for the law would still require
non-D2 or multi-layer counterexamples; within this campaign it is
recorded as a consistent, structure-independent observation.

## Report set

- R_TYPE_WIDEFOV_METHOD_A.md - the ED/MD-gated branch
- R_TYPE_WIDEFOV_METHOD_B.md - the port-only branch + multipole choice
- R_TYPE_ANGULAR_FAILURE_FORENSICS.md - budgets, PB collapse, lossless
- R_TYPE_ANGULAR_PARETO.md - the peak-vs-FOV front
- R_TYPE_WIDEFOV_NUMERICAL_QUALIFICATION.md - conventions, ps-basis
  finding, diffraction, convergence, seeding provenance
- The 30 required answers: section at the end of this file.
