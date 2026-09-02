# Method B - angle-aware port-only freeform (discovery branch)

No multipolar quantity anywhere in the objective; exact decomposition
run only AFTER freezing. Seeds: smooth D2 random freeform (blob +
isotropic bump + weak random-sign quadrupolar term) - no rectangle
encoding, never warm-started from Method A or the old EQ bow-tie.

## The central discovery question: what does Maxwell choose when
## ANGULAR ROBUSTNESS is the target?

**A low-order dipolar state - the wide-angle optimizer does NOT
rediscover the EQ bow-tie.** The champion's exact fractions (phi = 0):
p-channel MD-dominant (0.63-0.68) at theta = 0-40 crossing to ED at
60-75; s-channel ED-dominant (0.58-0.70) throughout; f_EQ never
dominant at any tested angle (peak 0.30 in a canceling role). Note the
role SWAP relative to the paper's assignment (paper: x->ED, y->MD;
newB: p->MD, s->ED) - the port-only optimizer found the mirrored
dipolar solution. The forward-transmission ladder is likewise
dipole-led at all angles, with deep s-channel suppression
(|t_s|^2 = 0.004-0.041 through 60 deg).

Meanwhile the previous theta0-optimized EQ bow-tie (oldB), decomposed
with the same machinery, holds EQ dominance ONLY at theta = 0
(f_EQ = 0.48) and degrades to a mixed dipolar state by 20 deg - while
its device performance collapses (theta_50 = 10 deg). In this design
space, higher-order character and angular fragility go together as a
MEASURED pair (spec Q17: EQ does not survive; Q18: yes, wide-angle
optimization returns to dipoles; Q20: correlated - by measurement,
not assumption).

## Champion: B_P252_H185_s47 (refined, [9,9])

- R_cross(0) = 0.380 (the strongest theta=0 among finalists);
  Omega = 0.398; worst mapped 0.156 (at 55,0); theta_50 = theta_20 =
  50 deg; dominance range 25 deg (limited by T_cross at mid angles).
- Hemispheric co-pol 0.091 mean / 0.265 max.
- Fab: single island, 97 nm Si / 66 nm air minimum features -
  the cleanest geometry of the campaign.
- PB rotation: -1.99 (rms 2.0) at 0; -1.92/12.1 at 30; -1.77/24.1 at
  45; collapsed (-0.34) at 60. Same validity envelope as newA and the
  rectangle.
- Provenance: seed 47 under the documented rescue rule (randomized
  anisotropy amplitude); reproducibility of the B family confirmed at
  P226/H200 (seeds 11/23/47 all land in one topology family).

## Two honest Method-B pathologies, documented

1. **rx = ry saddle trapping** (R_cross(0) ~ 0) hit several seed-11
   runs, systematically at small P. Rescue seeds with stronger random
   anisotropy recovered most points, EXCEPT:
2. **A competing co-pol-mirror attractor at P239/H200 and P252/H200**:
   BOTH rescue seeds re-trapped there (R_cross(0) = 0.003-0.019 with
   otherwise reasonable pool means). Without mechanism gates, the
   port-only objective has a genuine second basin at these lattice
   points that sacrifices normal incidence entirely. Method A at the
   same points never traps - the theta=0 multipole gates act as an
   effective anti-saddle regularizer. This is a real methodological
   finding, not noise.

## H = 170 control

B_P239_H170 (refined + mapped): Omega = 0.372, theta_50 = 35, worst
0.076, T leakage high at mid angles (mean 0.394). Beats the rectangle
on Omega at the paper height but with weaker floors than the A control.
