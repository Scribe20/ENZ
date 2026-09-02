# The angular Pareto front (spec secs 29-30)

Axes: R_cross(0) (peak) vs solid-angle-weighted <R_cross> 0-85 and vs
acceptance angles (figures wf_F3, wf_F12). Points: all 50 coarse + all
refined/seed/rescue runs + baselines, identical hard-binary [9,9]
metrics.

## The front

- **Flat-hemisphere extreme**: A_P239_H200 (Omega 0.401, worst 0.204,
  theta_20 = 85). Nothing beats its floor.
- **Peak-Omega extreme**: A_P258_H200 (Omega 0.448, R_cross(0) 0.375)
  with a (45,90) hole (0.077) -> theta_50 = 45.
- **Balanced high-peak**: B_P252_H185 (R_cross(0) 0.380, Omega 0.398,
  worst 0.156).
- The old theta0 champions sit OFF the front: their peak (0.51-0.53)
  buys Omega 0.244-0.256 - BELOW the rectangle - and theta_50 = 10.
- The rectangle sits at (0.229, 0.301): dominated by every finalist on
  both axes simultaneously (e.g. newA: +37% peak AND +33% Omega AND
  2.1x worst-angle AND 5x lower co-pol).

**Is there a peak-FOV trade-off (Q15)?** Within the angle-aware family,
yes and it is now mapped: pushing Omega/peak (P252-258) opens mid-angle
azimuthal holes; pushing the floor (P233-239) costs ~0.05 of Omega and
~0.06 of peak. Between families the "trade-off" seen previously
(theta0 champions vs rectangle) was not a front at all - theta0-first
optimization lands strictly inside the true front.

## Structural trends (Q2-Q5)

- Period: robust score rises monotonically P200 -> P239-258 for both
  methods (both Omega and floors). The small-P hypothesis (sec 7) is
  refuted: k_par P phase progression does not limit this design space
  before P ~ 260; lateral design freedom wins.
- Height: best at H = 185-210; H = 140-155 is strictly worse; H = 170
  controls beat the rectangle but not the H = 200 family.
- Best regions: Method A P = 233-258 / H = 190-210; Method B
  P = 226-252 / H = 185-200.

## Reading the front honestly

Omega integrates near-grazing R_cross that the PB-collapse analysis
shows is orientation-independent (not phase-addressable). A
PB-usability-weighted front (restricting to theta <= 45-55) compresses
all Omega values but preserves every ranking stated above: newA/newB
keep 1.4-1.7x the rectangle's cross amplitude and the theta0 champions
still collapse. No conclusion in this report depends on the grazing
region.
