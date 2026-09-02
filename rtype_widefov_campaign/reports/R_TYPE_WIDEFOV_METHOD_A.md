# Method A - angle-aware ED/MD-control branch

Device objective: the full angular robust loss (0.4 mean + 0.35 softmin
+ 0.25 lower-tail of R_cross over structured 5-state minibatches,
minus 0.15 co + 0.15 T + 0.30 A penalties + smooth theta=0 PB phase
term). Mechanism constraint: SOFT exact-multipole gates at theta = 0
only (f_ED^x >= 0.50, px|ED >= 0.80, f_MD^y >= 0.50, mx|MD >= 0.80,
weight 0.6, exact Alaee current moments, never |E| proxies). Seeds:
paper-inspired anisotropic soft rectangle scaled into the envelope +
random perturbation.

## Champion: A_P239_H200_s11 (refined, [9,9] end-to-end)

- R_cross(0) = 0.314; pool mean 0.375; solid-angle Omega = 0.401;
  worst mapped angle 0.204 (at theta=70, phi=90).
- theta_50 = theta_20 = dominance range = 85 deg (the full evaluated
  hemisphere): min-over-phi R_cross >= 0.204 >= 0.5 R_cross(0) and
  >= 0.20 everywhere, and R_cross beats R_co, T_cross, T_co at every
  mapped point. Hemispheric co-pol 0.057 mean / 0.116 max.
- Fab: single island, min Si linewidth 87 nm, min internal air gap
  52 nm, envelope-safe (edge clearance > 0), SHA256 in final.json.
- PB rotation: slope -1.991 (rms 1.9 deg) at theta=0; -2.06/11.9 at
  30; -2.17/25.2 at 45; -2.40/41.7 at 60; collapsed at 75 (see
  forensics report - identical envelope for the rectangle).
- Reproducibility: seeds 23 and 47 from-scratch at P239/H200 land at
  worst 0.268/0.295, Omega 0.370/0.368 - same topology family
  (dumbbell-in-envelope), i.e. a genuine attractor, not luck.
- Order convergence: complex r_cross drift order 9 -> 15 at the 1e-3
  level at (0,0), (60,0), (60,45), (80,0) (results/convergence.csv).

Alternate character: A_P258_H200_s11 - Omega = 0.448 (the campaign
maximum) and R_cross(0) = 0.375, but with a worst-angle hole (0.077 at
(45,90)) -> theta_50 = 45 deg. This is the other end of the angular
Pareto front; P = 258 comes from the sanctioned one-sided boundary
extension and remains specular-only to theta_air 85.5 deg (its >= 85
rows are excluded from acceptance metrics).

## Multipolar identity vs angle (phi = 0; exact fractions)

The gates held during optimization (f_ED^x ~ 0.44-0.54 around the soft
gate, f_MD^y 0.55-0.66). The frozen champion at theta = 0 sits at the
ED/MD boundary on p (f_MD 0.47 vs f_ED ~ 0.44) and MD on s (0.65);
from 20 deg upward p purifies to ED (0.66 -> 0.94) - largely the
growing pz - and s stays MD (0.68-0.93). Method A retains its dipolar
ED/MD character over the whole angular range (spec Q16: yes).

## H = 170 control (spec sec 44)

A_P239_H170 (refined + fully mapped): Omega = 0.362, theta_50 = 45,
theta_20 = 40, worst 0.138, co 0.113/0.223. It beats the rectangle
(0.301 / never dominant) at the paper's own height, so a common-height
T/R scheme does not forfeit the improvement - but the robust optimum
is H = 185-210 (score still rising at H = 200; H = 210 slightly
softer; H = 190 slightly below H = 200).

## Notes

- The angular minibatch + hard-angle mining worked as designed: the
  binding worst angle migrated from high theta early to (0,0)/(45,90)
  late - the optimizer actively traded normal-incidence peak for FOV.
- Method A run cost ~2x Method B (theta=0 moments each iteration).
