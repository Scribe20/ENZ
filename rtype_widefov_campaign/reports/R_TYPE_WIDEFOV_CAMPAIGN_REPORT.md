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

## The 30 required answers (spec section 54)

1. **Does angle-aware optimization from iteration 0 beat the previous
   normal-incidence-first approach on angular robustness?** Decisively.
   Same design space, same solver: theta_50 10 deg (old) -> 50-85 deg
   (new); worst-angle R_cross 0.015 -> 0.156-0.204; Omega 0.244-0.256
   -> 0.398-0.448. The old failure was a basin property, not an
   architecture limit.
2. **Does the robust optimum move to smaller P?** No - it moves the
   other way: robust score rises monotonically to P = 239-258; P =
   200-213 is strictly worse for both methods (all specular-only, so
   not a diffraction artifact).
3. **Toward H ~ 170?** No. Optimum H = 185-210 (score still rising at
   H = 200; one-sided extension found H = 210 slightly softer than
   H = 200). H = 170 is viable but sub-optimal.
4. **Best P/H for Method A:** P = 233-258, H = 190-210 (champion
   P239/H200; peak-Omega member P258/H200).
5. **Best P/H for Method B:** P = 226-252, H = 185-200 (champion
   P252/H185).
6. **Paper rectangle's exact acceptance under OUR metrics:**
   theta_50 = 55, theta_20 = 40, dominance range = NEVER (R_co >
   R_cross at and near theta = 0), Omega = 0.301, worst 0.098 at
   (60,90). PB slope -1.99 at 0 deg, collapsed by 60-75 deg.
7. **theta_50 new A:** 85 deg (full evaluated hemisphere).
8. **theta_50 new B:** 50 deg.
9. **theta_20:** newA 85 deg; newB 50 deg (rectangle 40; old champions
   15).
10. **Dominance range:** newA 85 deg (cross beats co, T_cross, T_co at
    every mapped point); newB 25 deg (T_cross exceeds it at mid
    angles); rectangle never; oldA/oldB 15 deg.
11. **Solid-angle hemispheric <R_cross> (0-85):** newA 0.401, newB
    0.398, A-alt P258 0.448, rectangle 0.301, oldA 0.256, oldB 0.244,
    bare 0.135, film 0.149.
12. **Highest R_cross(0):** among wide-FOV finalists, newB 0.380
    (A-alt 0.375, newA 0.314). The old theta0 champions' 0.51-0.53
    remain the peak-only records - at theta_50 = 10 deg.
13. **Best wide-angle average:** A_P258_H200 (Omega 0.448, mean
    0.420).
14. **Best worst-angle performance:** newA A_P239_H200: 0.204 minimum
    over the whole 0-85 x 0-90 map (2.1x the rectangle's 0.098).
15. **Clear peak/FOV Pareto trade-off?** Within the angle-aware family
    yes (flat-floor vs peak-Omega members, mapped in wf_F12); but the
    old champions are NOT on the front - theta0-first optimization is
    strictly dominated.
16. **Does Method A retain ED/MD over angle?** Yes: p ED-dominant
    (0.66-0.94) from 20 deg up (MD-leaning right at 0), s MD-dominant
    (0.65-0.93) throughout; no EQ takeover anywhere.
17. **Does Method B remain EQ-led?** Method B never became EQ-led: the
    angle-aware port-only optimizer chose a dipolar state outright.
    (The OLD theta0 EQ bow-tie holds EQ only at theta = 0 and loses it
    by 20 deg.)
18. **Does wide-angle Method B return to a lower-order dipolar state?**
    Yes - MD(p)/ED(s), the mirrored role assignment of the paper's
    recipe, with f_EQ <= 0.30 at every tested angle.
19. **Which multipolar basis gives the flattest angular response?**
    The measured answer: low-order ED/MD mixtures (both new champions);
    dipole-led cancellation survives obliquity, and the p-channel
    naturally purifies toward ED (growing pz) without breaking the
    response.
20. **Is higher-order character correlated with poorer angular
    robustness?** In this design space, yes - as measurement, not
    dogma: the only EQ-dominant state (oldB) is also the most
    angle-fragile (theta_50 = 10 deg), and its EQ fraction itself
    decays within 20 deg. No angle-robust EQ state was found by
    either method; absence of evidence for one is noted, not claimed
    as impossibility.
21. **PB slope ~ -2 at theta = 60?** No. -2.4 with 42 deg rms
    (rectangle and newA alike); newB -0.34. The law is broken at 60.
22. **At 75?** Gone: fitted slopes +0.05/+0.09/-0.05 - the cross phase
    no longer responds to element rotation at all.
23. **What fails first with angle?** For the theta0-optimized states:
    retardance (the tuned Delta_phi = pi detunes within 20 deg; exact
    budget, closure 1.000), compounded by absorption; imbalance
    secondary; mixing negligible; diffraction zero. For the
    angle-aware states nothing "fails" until the PB LAW itself goes
    (45-60 deg) - their phase stays flat (retardance loss <= 0.10) and
    the residual is T+A amplitude.
24. **Is P271 itself a major cause of the previous collapse?** Not
    primarily. The P-dependent robust sweep peaks at P239-258, so P271
    is past the optimum but close to it; the dominant cause is the
    theta0-first BASIN (retardance detuning), plus P271's glass-side
    order opening at 61.5 deg which the new grid avoids entirely.
    Stated per the sweep, not assumed.
25. **Is H = 200-215 itself a major cause?** No - the opposite: the
    angle-aware optimum sits AT H = 200 (newA) and H = 185 (newB).
    Height was not the problem; the optimization target was.
26. **How much collapse remains with absorption removed?** The angular
    SHAPE is nearly unchanged at k = 0 (newA 0.45-0.60 across 0-60 vs
    0.31-0.36 real): absorption sets the level (~35-40% relative),
    dispersion sets almost none of the angular variation. The old
    champions' collapse is NOT rescued by k = 0 in shape - their
    failure is phase detuning.
27. **Does any candidate beat the rectangle over a meaningful angular
    interval?** newA beats it at ALL 126 mapped (theta, phi) points in
    R_cross (pointwise ratio: min 1.03x, median 1.39x) and at every
    angle on co-pol purity. newB beats it at 120/126 points.
28. **Does any candidate dominate on BOTH angular-average R_cross and
    co-pol leakage?** Yes - newA: Omega +33% (0.401 vs 0.301) with
    hemispheric co-pol 0.057 vs 0.293 (5.1x lower); newB similarly
    (0.398, 0.091).
29. **Is there an H = 170 candidate that improves the paper while
    preserving same-height compatibility?** Yes, qualified in full:
    A_P239_H170 (Omega 0.362, theta_50 45, worst 0.138, co 0.113) and
    B_P239_H170 (Omega 0.372, theta_50 35). Both beat the rectangle at
    its own height; both are below the H = 185-210 optimum.
30. **Is a single-layer wide-FOV freeform R-type actually feasible?**
    For the SCATTERING state: yes - demonstrated, reproducible,
    fabricable, order-converged (newA: hemisphere-wide dominance,
    worst 0.204). For the full 360-metalens FUNCTION: only up to the
    PB-law validity boundary (~45-55 deg incidence), which this
    campaign measured to be structure-independent across three very
    different geometries. Extending the LAW beyond that in a single
    layer found no counterexample here and likely needs a different
    architecture (multilayer / non-local engineering) - recorded as a
    measured, consistent ceiling, not a proven impossibility.
