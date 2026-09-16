# REPORT — from the Q-Dz parent campaign to an activation-relevant design workflow

Package `enz_jz_inverse_design/qdz_activation/` (this branch), built on the unchanged
`qdz_parent/` baseline (`claude/jz-1302-qdz-parent`, f7b0110) and the unchanged native hot-electron ITO model
of `nonlinear_activation_best/` (`claude/jz-1302-nonlinear-activation`, f8b5896, merged).  Method and audit:
`METHOD.md`.  Machine-readable results: `outputs/candidates/<tag>/stage{A,B}.json`, `stageB_scan.npz`,
`outputs/tables/*.{csv,json,md}`, `outputs/tests/sanity_tests.json`.  Figures: `outputs/figures/`.

> Evidence classes used below.  **[code]** tested code path (`outputs/tests/sanity_tests.json`, 10/10);
> **[computed]** production Stage A/B numbers of this branch at RCWA order [9,9] with order checks to [11,11];
> **[repo]** existing repository evidence (certifications of `qdz_parent`, the nonlinear lane report);
> **[provisional]** depends on the literature-derived hot-electron band model (Lane B of the nonlinear lane).


## 0. Summary

1. **What the old campaign optimized.**  `qdz_parent` maximized the parent (no-ITO) interface participation `eta_Dz`
   under a finite-Q proxy constraint and a wavelength constraint, and certified exact poles post hoc.  It never
   inserted the real ITO and never looked at the transmission background.  Its certified designs split into a
   warm-started REFLECTOR family (parent T at lambda_E 0.04-0.47, Q_r 77-546) and a transmissive fresh4242 family
   (T 0.87-0.96, Q_r 228-809)  [repo].
2. **New workflow (this branch).**  Stage A adds a systematic parent background `T_bg` and resonance contrast
   `C_res`; Stage B inserts the real 23-nm ITO and evaluates the small-signal transmission change `dT` between the
   physical cold ITO and the hot-electron state Te = 1000 K of the repository's own (provisional) model, together
   with `R`, `A`, `F_x,y,z`, `F_tot`, `eta_z`, loaded poles, order convergence, linearity, mechanism decomposition
   and Rayleigh safety, at order [9,9], for 22 frozen geometries  [code, computed].
3. **Result.**  No campaign design combines a transmissive background with a resonant, positive, interior
   transmission activation of useful size.  The largest interior positive dT among the 20 campaign/baseline
   designs are +0.0142 (pilot_h525_Q50_fresh8080, 1271 nm, a broadband ITO absorber: cold T 0.21, A 0.70) and
   +0.0133 (fr_Q50_final2_h680, 1314 nm, on a loaded Q = 36 line: cold T 0.17, A 0.78); every design with a
   transmissive parent (T_bg >= 0.75) except that pilot gives dT <= +0.0063.  The SNU deck cylinder, never
   optimized here, gives dT = +0.0189 at 1279 nm from a cold T of 0.55 (S_T = 0.54, loaded Q = 52, exact
   loss-scaling gamma_rad/gamma_nr = 1.46), i.e. the largest activation, the largest normalized sensitivity and
   by far the most usable cold transmission  [computed].
4. **Why.**  The real ITO over-damps every high-`eta_Dz` parent: with `Q_nr ~ |eps_ITO|^2/(eps'' eta_Dz_P) =
   0.43/eta_Dz_P` the campaign's parents (eta_Dz_P 0.03-0.13) sit 6-120 x beyond the critical-coupling value
   `eta_Dz_P ~ 0.43/Q_r`; their loaded lines are erased (no AAA pole for the whole fresh4242 family) or reduced to
   few-percent features, and only the broadband background responds.  The cylinder's eta_Dz_P = 0.013 keeps its
   loaded line visible.  `eta_Dz`, `Q_r` and `F_z` are therefore individually insufficient; the loaded
   radiative/non-radiative ratio and the transmissive background decide  [computed, mechanism robust].
5. **Robust vs provisional.**  Signs of dT, the ranking by S_T, the loss of the fresh4242 lines and the rejection
   of the reflector family depend only on the supplied cold data and the fixed Drude perturbation direction
   (S_T varies < 5 % between Te = 600 and 2000 K for the cylinder and the fresh designs, 20 % for
   fr_Q50_final2_h680; 84-100 % of dT comes from Re(delta_eps)).  Absolute dT per kelvin and any intensity axis
   are provisional (Lane B)  [computed / provisional].
6. **Numerics.**  Small-signal dT on narrow loaded lines needs order [9,9]: [5,5]/[7,7] give the wrong sign
   at 1341 nm for fr_Q100_final3_h640.  Several unguarded maxima sit 10-15 nm above the glass Rayleigh
   wavelength (1251.2 nm) or at the 1400-nm end of the a-Si:H data and are excluded from the operating-point
   selection by a 20-nm / 5-nm guard  [computed].
7. **Frozen for independent FDTDX validation (section 15):** the deck cylinder (positive control), the two campaign
   finalists above, and fr_Q200_fresh4242_h700 (the campaign's own transmissive high-Q / high-eta_Dz parent whose
   line vanishes under loading).

## 1. What the existing `qdz_parent` campaign actually optimized  [repo]

`optimize_parent.py` maximizes `eta_Dz` of the no-ITO parent under two soft constraints, `Q_proxy ~ Q_target`
(symmetric log penalty, never rewarding a larger Q) and `lambda_r ~ lambda_E` (one target linewidth), with
structural geometry constraints (128 x 128 grid, hard 12 % air pad, 40-nm filter, tanh projection).  It never
looked at the far field beyond R and T at lambda_E, never inserted the ITO (the `hybrid_certify.py` and
`detuning.py` stages exist but were never run), and its 10 exact certifications (`outputs/certified/`) show
the trade space it produced: warm-started designs (final0/1/3 lineages, Q_r 77-500) that are broadband
REFLECTORS at lambda_E (parent R 0.53-0.98, T 0.04-0.47) with `eta_Dz` 0.043-0.14, and the fresh4242 lineage
(Q_r 228 and 452, h = 700 nm) that is transmissive (parent T 0.87 / 0.92 at lambda_E) with `eta_Dz` 0.076 /
0.082.  Two more certified designs have no usable resonance at all (pilot_h525_Q50_fresh8080: no pole;
pilot_h525_Q100_fresh4242: pole 78 nm away).

## 2. Why `eta_Dz`, `Q_r` and `F_z` are individually insufficient as activation metrics

* `eta_Dz` measures how much longitudinal displacement the PARENT mode puts on the future interface per unit
  stored energy.  It contains no ITO loss and no far-field information: a mode can have a large `eta_Dz` and
  be a perfect mirror (parent T = 0.04), or be so strongly damped by the real ITO that its line disappears.
  Quantitatively (Stage B, `critical_coupling_estimate`): the ITO-induced non-radiative Q of a parent mode is
  `Q_nr ~ |eps_ITO|^2 / (eps'' eta_Dz_P) = 0.432 / eta_Dz_P` at lambda_E, so `eta_Dz_P = 0.03-0.1` gives
  `Q_nr ~ 4-14`, far below every certified `Q_r` (77-809): all high-`eta_Dz` parents are over-damped by the
  ITO by factors 6-100 and their loaded lines are 5-15 x broader and 5-15 x shallower than the parent lines.
* `Q_r` alone rewards internal build-up; at fixed `eta_Dz` a larger `Q_r` only deepens the over-damping ratio
  `gamma_nr/gamma_r = Q_r/Q_nr`.  The useful quantity is the ratio, not either factor.
* `F_z` (the Jz/Fz objective) is the cold-state longitudinal ITO absorption.  Maximizing it produced unity
  absorbers whose transmission channel is shut (T = 0.003) in both the cold and the hot state; the nonlinear
  lane showed that heating then opens REFLECTION, not transmission [repo].  Heating needs `F_tot`, not `F_z`,
  and a transmission activation needs a transmissive background, which `F_z` actively destroys.

## 3. Definitions introduced here (all computed, all with sanity tests)

| quantity | definition | where |
|---|---|---|
| `T_bg` | median total parent T over `L_bg` = ENZ window [lambda_E +- 50 nm] ∩ single-order window [lambda_R + 10 nm, 1400 nm] minus +-min(3 FWHM, 25 nm) around EVERY significant parent pole; wavelengths stored; undefined if < 5 samples survive | `parent_background.py` |
| `C_res` | `|T_bg - T(lambda_r)|` at the certified pole (exact solve) and the extremum form over +-3 FWHM (dense rescan), with the local depth `T_max - T_min` | idem |
| `dT`, `S_T` | `T(eps1) - T(eps0)` at the same wavelength, geometry, order, incidence, polarization and z-quadrature; `S_T = dT / |eps1 - eps0|` | `loaded_sensitivity.py` |
| heating | cold `F_tot = F_x + F_y + F_z` (= A, per unit area) at the operating wavelength; `eta_z = F_z / F_tot` reported separately as the longitudinal-channel diagnostic | idem |
| `lambda_op` | post hoc: argmax dT (positive) in the safe window, refined at 0.5 nm; argmin dT and argmax |S_T| reported; lambda_E and lambda_r are reference points only | idem |
| safety | dispersive Rayleigh set, margin of lambda_op in nm / relative / loaded linewidths; order checks [5,5]-[11,11] with explicit tolerances -> `numerically_unresolved` | `rayleigh.py`, idem |

## 4. How the physical epsilon perturbation was chosen  [repo, provisional]

The repository's only nonlinear ITO model is the hot-electron model of `nonlinear_activation_best/ito_nonlinear.py`
(Kane-band Drude weight, N calibrated to the Drude fit of the supplied cold data, gamma constant; exact
reduction to `ITO_nk.csv` at 300 K).  Its native variable is the electron temperature `Te`; the default
perturbation is `Te = 1000 K`: `eps_ITO(lambda_E)` moves from `0 + 0.4320 i` to `+0.0363 + 0.4273 i`
(`|delta_eps| = 0.0366`, the ENZ crossing 1302.3 -> 1309.4 nm; Drude-weight loss 1.06 %).  No arbitrary complex
`delta_eps` was invented.  What is robust and what is not: the complex DIRECTION of `delta_eps`
(`arg = -atan(gamma/omega) = -7.3 deg`) follows from the Drude fit of the supplied cold data only, so the sign
of `dT` and the normalized `S_T` are robust to first order; the magnitude per kelvin (and every intensity
axis) is provisional.  The linear-response check (Te = 600 / 1000 / 2000 K) and the real-only / imaginary-only
decomposition quantify how far each result is from that first-order regime.  The perturbation layer is
configurable (`--kind drude_weight --delta`, `real_only`, `imag_only`; `custom` is flagged non-physical).

## 5. Numerical validation  [code]

`tests/sanity_tests.py`, 10/10 passed (`outputs/tests/sanity_tests.json`): material layer = CSV at 300 K
(0.0) and = native model at 1000 K (1.6e-16); dT = dR = dA = dFz = 0 exactly for eps1 = eps0; R+T+A = 1 (0.0)
and |F_tot - A| = 2.9e-5 (< 1e-3); planar-film sign convention vs an independent transfer-matrix code (T to
1e-10, dT > 0 = more transmission, same sign at 1280 / 1302 / 1340 nm); the `qdz_parent` baseline reproduced
to 1.4e-11 with the unchanged code at [5,5], [7,7], [9,9]; exact-pole Q used and the ill-posed proxy of
`fr_Q100_final3_h640` (7.9e12) refused; total = specular T above the glass Rayleigh wavelength (1e-14) and
not below it (0.012 at 1246 nm); identical sims except eps_ITO; angle hook (theta = 3 deg: closure 1e-12,
|F_tot - A| = 2.9e-5); dispersive vs fixed-index Rayleigh 0.09 nm; background-set logic (cap, undefined case).


## 6. Which candidates were re-evaluated  [computed]

All 22 frozen geometries of `candidates.registry()` were pushed through Stage A and Stage B at RCWA order
[9,9] (2-nm grid plus dense refinement windows, order checks [5,5]/[7,7]/[9,9]/[11,11] at the selected points,
linearity / direction checks): the 10 exact-pole-certified `qdz_parent` designs (Q_target 20-500), the 8
frontier runs that were never certified (their exact poles and parent metrics were computed here at [9,9] and
are labelled as such), the two Jz/Fz finalists that had been parent-certified (final3 = the JZ best design,
final1), and two reference structures that no campaign optimized: the SNU deck cylinder (d = 405.7 nm,
h = 970.5 nm, P = 588 nm; positive control) and the bare air/ITO/glass film (analytic control).  The complete
table is `outputs/tables/candidates.{csv,json,md}` (reproduced in section 12); per-candidate details are in
`outputs/candidates/<tag>/stage{A,B}.json` and the scans in `stageB_scan.npz`.

Cross-check against the existing nonlinear lane [repo]: for the deck cylinder this pipeline gives a cold loaded
T minimum of 0.215 at 1292.5 nm, an absorption maximum of 0.465 at 1291.3 nm and an exact loaded pole
Q = 52.4 at 1294.8 nm; the nonlinear lane ([7,7], its own scripts) reported 0.224 at 1294 nm, 0.466 at
1292 nm and Q_loaded ~ 55.  For the JZ best design it gives dT = -0.0003 at lambda_E and a positive dT only at
1266 nm (15 nm from the glass Rayleigh anomaly); the nonlinear lane found the same two facts (|dT| <= 0.0025 at
the resonance, largest positive swing at 1268 nm, order-sensitive).  The two independent implementations agree.

## 7. Do high Q, high eta_Dz and high transmission sensitivity correlate?  [computed]

No.  The decisive quantity is what the real ITO does to the parent line, and that is set by the ratio of the
ITO-induced loss rate to the radiative rate, not by either factor alone:

* The transmissive fresh4242 parents (Q_r = 228 and 452 at h = 700 nm, T_bg = 0.97, far-field contrast 0.30,
  eta_Dz_P = 0.052 / 0.055) LOSE their resonance completely when the 23-nm ITO is inserted: no loaded pole is
  found by AAA, the cold loaded T is flat at 0.87-0.92, <|E_z|^2>_ITO falls to 0.7-0.9, and the activation is a
  featureless dT = +0.0024 / +0.0026 (S_T = 0.063 / 0.067) with an "absorption -> transmission" signature.  The
  parent-only estimate predicted exactly this (Q_loaded,est = 8, gamma_nr/gamma_r = 28 and 57).
* The reflector family (warm-started Q_target = 100 / 200 designs, Q_r = 77-107, eta_Dz_P = 0.03) keeps a
  loaded line (Q 64-90) but on a T_bg = 0.10-0.20 mirror background: dT_max,pos <= +0.0001 and the dominant
  channel is "absorption -> reflection, no transmission opening" (dR up to +0.017, dT at the line -0.008 to
  -0.011).  These are the designs the note's hypothesis wanted rejected, and T_bg rejects them.
* The two largest interior campaign responses come from designs WITHOUT a useful parent resonance in the
  usual sense: fr_Q50_final2_h680 (an uncertified frontier run whose exact [9,9] parent pole has Q = 32 and whose
  parent poles fill the whole ENZ window, so T_bg is undefined) gives dT = +0.0133 at 1314 nm on a loaded Q = 36
  line (T 0.166 -> 0.179, S_T = 0.36, A = 0.78, eta_z = 0.92, 63 nm from the anomaly, converged, linear within
  20 %), and pilot_h525_Q50_fresh8080 (below) gives +0.0142 at 1271 nm.  Both are ITO absorbers with a cold
  transmission of 0.17-0.21.
* The highest certified Q_r among aligned designs (fr_Q100_final3_h640, Q_r = 499, eta_Dz_P = 0.042, T_bg = 0.25)
  gives the largest positive campaign dT that sits on a real loaded line: +0.0052 at 1341 nm (T 0.262 -> 0.267,
  loaded Q = 170, S_T = 0.13, linear, converged), while at lambda_E it does the opposite (dT = -0.013, T -> R).
* The design with the highest parent eta_Dz of all (0.194, pilot_h525_Q50_fresh8080) has NO certified parent
  resonance; loaded, it is a broadband ITO absorber (A = 0.70, eta_z = 0.97, <|E_z|^2>_ITO = 14) whose
  transmission opens as the ITO heats: dT = +0.0096 at lambda_E (T 0.157 -> 0.167, S_T = 0.26) and
  +0.0142 at its interior optimum 1271 nm (T 0.205 -> 0.219, S_T = 0.41, converged to 1e-4 between [9,9] and
  [11,11], linear within 6 %), i.e. the largest campaign response, but from a low cold transmission and
  without a resonance to narrow it (exact loss scaling of its only loaded pole: Q_rad = 13, Q_nr = 7.9).
* The deck cylinder, with eta_Dz_P = 0.013 (3-8 x smaller than any campaign design) and Q_r = 172 on a
  T_bg = 0.97 background with a 0.95 far-field contrast, keeps a visible loaded line (Q = 52, T_min = 0.22,
  A_max = 0.47) and yields dT = +0.019 at 1279 nm (T 0.548 -> 0.567, S_T = 0.54, "R,A -> T", cold A = 0.31,
  eta_z = 0.90, 387 nm from any Rayleigh anomaly, converged to 1e-4 across [5,5]-[11,11], linear to 0.9 %
  between Te = 600 and 2000 K, 100 % from Re(delta_eps)) and dT = -0.017 at 1296 nm on the other shoulder of
  the same line.  It beats every campaign design by 2-10 x in S_T at a usable cold transmission.

The mechanism is the one quantified in section 2: eta_Dz_P = 0.03-0.07 puts the campaign's parents 6-60 x
beyond the ITO-critical-coupling value eta_Dz_P ~ 0.43/Q_r, the loaded line is either erased (fresh4242) or
reduced to a few-percent far-field feature, and the small-signal transmission change is then only the
broadband background response.  The cylinder's smaller interface participation is what keeps its loaded Q
at 52 and its transmission sensitivity large.  The first-order estimate is a screening indicator only: it
is right in sign and order of magnitude for every candidate (the exact loaded Q is 52 where it predicts 28
for the cylinder, and no loaded line where it predicts Q ~ 8) but it cannot replace the exact loaded pole,
and it mis-assigns modes when several parent poles lie in the window.

At the parent level (certified set, `outputs/tables/certified_Q_vs_buildup.json`) the hypothesis "larger Q
buys internal build-up at the expense of interface participation" is NOT supported: on the seven aligned
certified parents log Q_r correlates positively both with U_mid at lambda_r (Spearman +0.86) and with
eta_Dz at lambda_r (+0.75).  The trade-off appears only after loading.

## 8. Are broadband-reflector solutions rejected?  [computed]

Yes, by T_bg together with the cold loaded transmission: every design with T_bg < 0.3 (the Q_target 100/200
warm family, the JZ finalists, the final3-lineage frontier runs, fr_Q100_final3_h640) has an interior positive
dT <= +0.0093 and a cold loaded T <= 0.26 at its operating point (median loaded T over the window 0.03-0.15);
their largest unguarded dT values (+0.0124 for jzfz_final1, +0.0142 for fr_Q50_final0_h680, +0.0107 for
fr_Q500_final0_h380) sit exactly on the safe-window edges, 10 nm above the glass Rayleigh anomaly or at the
1400-nm end of the material data, and change by 10-35 % between orders [5,5] and [11,11] there.  In the
reflector family proper the dominant loaded response is "absorption -> reflection" or "T -> R"; the only
positive dT of the JZ best design (+0.0073, S_T = 0.21) occurs 15 nm from the glass Rayleigh anomaly at
T = 0.08 and is exactly the order-sensitive near-threshold state the nonlinear lane had flagged.  The threshold sweep (`outputs/tables/threshold_sweep.json`) shows
which combinations of (T_bg_min, C_res_min, F_tot,min, eta_z,min) leave which survivors; no single
threshold is hard-coded anywhere.

## 9. Robust versus provisional conclusions

Robust (independent of the hot-electron band model; they use only the supplied cold data, the fixed
perturbation direction and the loaded cold spectra):
1. the far-field visibility and Q of the ITO-loaded line, hence the ranking of the candidates by S_T
   (linear-response checks: S_T changes by < 5 % between Te = 600 and 2000 K for every finalist);
2. the sign of dT at each wavelength (fixed by arg(delta_eps) = -7.3 deg, itself fixed by the Drude fit of
   the supplied cold data; > 84 % of dT comes from Re(delta_eps) in every case);
3. the rejection of the reflector family and the loss of the fresh4242 resonance under loading;
4. the cold-state heating metric F_tot and eta_z (pure cold-state quantities).
Provisional (Lane B of the nonlinear lane): the absolute size of dT per kelvin, every intensity or pulse-energy
axis, and any statement about Te beyond ~6000 K.  Nothing here needed those.

## 10. Numerical failure modes found

* Fourier-order sign errors at narrow loaded lines: for fr_Q100_final3_h640 at 1341 nm the [5,5] and [7,7]
  values of dT are -0.0049 and +0.0019 against +0.0052 at [9,9] and +0.0046 at [11,11]; the map order of the
  Jz/Fz campaign is not sufficient for small-signal differences on a Q ~ 170 loaded line.  All selections
  use [9,9] and are flagged if [9,9] -> [11,11] moves dT by more than max(0.02, 20 %).
* Rayleigh proximity: several extremal dT values (fresh4242 dT_min, pilot fresh8080 dT_max, the JZ best
  dT_max) sit 10-15 nm above the glass Rayleigh wavelength 1251.2 nm; the safety margin is reported per point
  and these points are not used as operating points.
* The background definition needs a cap on the pole-exclusion half-width: a Q ~ 20 parent pole (FWHM ~ 60 nm)
  would otherwise empty the ENZ window; with the cap the number of surviving samples is reported and the
  definition sweep (`definition_sweep.json`) shows the sensitivity of T_bg to k_excl and D_op.
* AAA finds no loaded pole for the fresh4242 family and for the pilots without a parent pole; the absence is
  reported as such, not replaced by a proxy.
* The Q proxy of `qdz_parent` is ill-posed for three certified designs (Q_proxy 7.9e12, 1.4e3 vs 452, 4.8e7);
  only exact poles are used.

## 11. What a topology-level sensitivity optimization would need (not run)

The frozen-candidate study shows dT is numerically stable (order-converged) and linear in the perturbation
for the finalists, and every quantity entering it is already a differentiable torch scalar in
`forward.evaluate` (the two-state solve is two forward passes).  A second-phase optimizer is therefore
feasible, but its formulation must change: maximize the positive loaded dT (or S_T) at a post-hoc
lambda_op, subject to a finite parent Q_target, a TRANSMISSIVE background (T_bg), a visible loaded line, a
cold F_tot above a swept minimum, and an interface participation held near the critical-coupling value
eta_Dz_P ~ 0.43/Q_r rather than maximized.  Term scales: dT is O(1e-2) per 0.04 of |delta_eps|, so a
log-loss on S_T (unity at a factor e) and the existing unity-scaled Q and lambda constraints are
commensurate; a T_bg constraint as a hinge on log(T_bg/T_bg,min).  This is proposed, not implemented.

<!-- TABLES START -->
## 12. Tables (generated by `select_pareto.py`; sorted by the interior positive dT)

20 campaign / baseline designs + 2 references.  `T_bg`, `C_res_ext`: parent (no ITO); `T0_loaded_median`:
median cold loaded T over the safe window; `lam_op_int`, `dT_int`, `S_T_int`, `T0_int`, `Ftot0_int`, `eta_z_int`: the interior
operating point (argmax dT with lambda >= lambda_R + 20 nm and <= 1395 nm); `lam_op_pos`, `dT_max_pos`: the unguarded argmax
(flagged `at_window_edge_pos` when it sits on the safe-window edge); `gamma_nr_over_gamma_r_est`: parent-only first-order
estimate; `loaded_Q_pos`: exact loaded AAA pole nearest lambda_op; `unresolved_pos`: [9,9] -> [11,11] change of dT above
max(0.02, 20 %); `dominated_activation_axes`: dominated on (dT_int, T_bg, C_res_ext, F_tot, eta_z, T0_loaded_median, Q_r, eta_Dz).

| tag | kind | Q_target | Q_r | eta_Dz_lamE | eta_Dz_P_lamE | T_bg | C_res_ext | T0_loaded_median | lam_op_int | dT_int | S_T_int | T0_int | Ftot0_int | eta_z_int | lam_op_pos | dT_max_pos | at_window_edge_pos | channel_pos | dT_max_neg | gamma_nr_over_gamma_r_est | loaded_Q_pos | unresolved_pos | dominated_activation_axes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deck_cylinder_P588_h970 | reference | - | 172 | 0.014 | 0.013 | 0.97 | 0.95 | 0.85 | 1279.2 | +0.0189 | +0.536 | 0.548 | 0.315 | 0.90 | 1279.2 | +0.0189 | False | R,A->T (useful) | -0.0172 | 5.2 | 5 | False | - |
| pilot_h525_Q50_fresh8080 | qdz_certified | 50.0 | - | 0.194 | 0.128 | 0.75 | - | 0.14 | 1271.2 | +0.0142 | +0.406 | 0.205 | 0.698 | 0.96 | 1261.2 | +0.0147 | True | A->T (absorption to transmission, useful) | -0.0101 | - | - | False | False |
| fr_Q50_final2_h680 | qdz_uncertified | 50.0 | 32 | 0.058 | 0.039 | - | - | 0.11 | 1314.1 | +0.0133 | +0.356 | 0.166 | 0.781 | 0.92 | 1314.1 | +0.0133 | False | A->T (absorption to transmission, useful) | -0.0046 | 2.9 | 36 | False | False |
| fr_Q500_final0_h380 | qdz_uncertified | 500.0 | 8 | 0.163 | 0.120 | 0.23 | 0.69 | 0.11 | 1393.2 | +0.0093 | +0.221 | 0.503 | 0.493 | 0.91 | 1399.7 | +0.0107 | True | A->T (absorption to transmission, useful) | -0.0106 | 2.1 | 10 | False | False |
| fr_Q50_fresh8080_h480 | qdz_uncertified | 50.0 | 44 | 0.118 | 0.094 | 0.86 | 0.62 | 0.77 | 1377.2 | +0.0063 | +0.155 | 0.779 | 0.192 | 0.80 | 1378.2 | +0.0063 | False | A->T (absorption to transmission, useful) | -0.0148 | 9.6 | - | False | False |
| jzfz_final1 | jzfz_baseline | - | - | 0.101 | 0.063 | 0.09 | - | 0.03 | 1271.2 | +0.0062 | +0.179 | 0.066 | 0.917 | 0.91 | 1261.2 | +0.0124 | True | A->T,R | -0.0024 | - | 10 | False | False |
| jzfz_final3 | jzfz_baseline | - | 220 | 0.111 | 0.071 | 0.03 | 0.92 | 0.03 | 1271.2 | +0.0054 | +0.154 | 0.063 | 0.913 | 0.91 | 1266.0 | +0.0073 | False | A->T,R | -0.0020 | 36.1 | 184 | False | False |
| fr_Q100_final3_h640 | qdz_certified | 100.0 | 499 | 0.060 | 0.042 | 0.25 | 0.38 | 0.15 | 1341.2 | +0.0052 | +0.133 | 0.262 | 0.689 | 0.86 | 1341.2 | +0.0052 | False | A->T,R | -0.0134 | 48.7 | 170 | False | False |
| fr_Q50_final0_h680 | qdz_uncertified | 50.0 | 6 | 0.083 | 0.052 | 0.43 | 0.44 | 0.11 | 1313.2 | +0.0044 | +0.119 | 0.107 | 0.861 | 0.96 | 1261.2 | +0.0142 | True | R,A->T (useful) | -0.0059 | 0.8 | 12 | False | False |
| fr_Q20_final2_h440 | qdz_uncertified | 20.0 | 17 | 0.158 | 0.105 | 0.31 | 0.60 | 0.06 | 1301.2 | +0.0042 | +0.114 | 0.058 | 0.845 | 0.95 | 1300.7 | +0.0042 | False | A->T (absorption to transmission, useful) | -0.0071 | 4.2 | 98 | False | False |
| fr_Q20_final1_h420 | qdz_certified | 20.0 | 546 | 0.138 | 0.098 | 0.30 | 0.18 | 0.06 | 1297.4 | +0.0036 | +0.100 | 0.057 | 0.837 | 0.97 | 1297.4 | +0.0036 | False | A->T,R | -0.0075 | 124.0 | - | False | False |
| pilot_h525_Q100_fresh4242 | qdz_certified | 100.0 | 809 | 0.030 | 0.011 | 0.96 | 0.45 | 0.84 | 1329.2 | +0.0035 | +0.092 | 0.856 | 0.096 | 0.65 | 1329.2 | +0.0035 | False | A->T (absorption to transmission, useful) | +0.0014 | 19.8 | - | False | False |
| fr_Q200_fresh4242_h700 | qdz_certified | 200.0 | 228 | 0.076 | 0.052 | 0.97 | 0.30 | 0.90 | 1333.2 | +0.0026 | +0.067 | 0.904 | 0.072 | 0.53 | 1333.2 | +0.0026 | False | A->T (absorption to transmission, useful) | -0.0037 | 27.7 | - | False | False |
| fr_Q500_fresh4242_h700 | qdz_certified | 500.0 | 452 | 0.082 | 0.055 | 0.97 | 0.30 | 0.91 | 1329.2 | +0.0024 | +0.063 | 0.907 | 0.067 | 0.50 | 1328.7 | +0.0024 | False | A->T (absorption to transmission, useful) | -0.0013 | 57.0 | - | False | False |
| fr_Q20_final3_h380 | qdz_uncertified | 20.0 | 21 | 0.139 | 0.099 | 0.19 | 0.71 | 0.13 | 1293.2 | +0.0008 | +0.021 | 0.022 | 0.895 | 0.97 | 1261.2 | +0.0025 | True | R->T,A | -0.0080 | 4.7 | 8 | False | False |
| bare_ito_film_P825 | reference | - | - | - | - | 0.96 | - | 0.91 | 1393.2 | +0.0007 | +0.016 | 0.909 | 0.035 | 0.00 | 1399.7 | +0.0007 | True | R,A->T (useful) | +0.0005 | - | - | False | - |
| fr_Q200_final1_h700 | qdz_certified | 200.0 | 107 | 0.043 | 0.030 | 0.20 | 0.31 | 0.14 | 1273.6 | -0.0003 | -0.007 | 0.155 | 0.554 | 0.81 | 1273.6 | -0.0003 | False | T->R | -0.0108 | 7.4 | 90 | False | True |
| fr_Q500_final3_h600 | qdz_uncertified | 500.0 | 6 | 0.166 | 0.118 | 0.19 | 0.69 | 0.05 | 1363.2 | -0.0018 | -0.044 | 0.047 | 0.911 | 0.97 | 1363.2 | -0.0018 | False | T->R,A | -0.0070 | 1.7 | 8 | False | False |
| fr_Q200_final3_h600 | qdz_uncertified | 200.0 | 11 | 0.161 | 0.113 | 0.17 | 0.82 | 0.05 | 1369.2 | -0.0020 | -0.049 | 0.052 | 0.910 | 0.96 | 1262.2 | +0.0028 | True | A->R (absorption to reflection; NO transmission opening) | -0.0059 | 2.9 | 265 | False | False |
| pilot_h675_Q100_warm_final1 | qdz_certified | 100.0 | 79 | 0.044 | 0.029 | 0.10 | 0.57 | 0.09 | 1393.9 | -0.0031 | -0.074 | 0.079 | 0.915 | 0.95 | 1266.2 | -0.0030 | False | A->R (absorption to reflection; NO transmission opening) | -0.0083 | 5.3 | 69 | False | False |
| fr_Q100_final1_h680 | qdz_certified | 100.0 | 77 | 0.044 | 0.030 | 0.10 | 0.55 | 0.09 | 1387.2 | -0.0034 | -0.083 | 0.088 | 0.911 | 0.95 | 1265.6 | -0.0032 | False | T,A->R | -0.0086 | 5.3 | 68 | False | False |
| fr_Q100_final0_h700 | qdz_certified | 100.0 | 81 | 0.047 | 0.032 | 0.15 | 0.59 | 0.09 | 1271.2 | -0.0036 | -0.104 | 0.040 | 0.677 | 0.79 | 1265.2 | +0.0001 | False | A->R (absorption to reflection; NO transmission opening) | -0.0077 | 5.9 | 64 | False | False |

Non-dominated set: fr_Q100_final0_h700, fr_Q100_final1_h680, fr_Q100_final3_h640, fr_Q200_final3_h600, fr_Q200_fresh4242_h700, fr_Q20_final1_h420, fr_Q20_final2_h440, fr_Q20_final3_h380, fr_Q500_final0_h380, fr_Q500_final3_h600, fr_Q500_fresh4242_h700, fr_Q50_final0_h680, fr_Q50_final2_h680, fr_Q50_fresh8080_h480, jzfz_final1, jzfz_final3, pilot_h525_Q100_fresh4242, pilot_h525_Q50_fresh8080, pilot_h675_Q100_warm_final1.

### 12.1 Threshold sweep (`threshold_sweep.json`, 384 combinations; survivors need dT_int > 0)

| T_bg_min | C_res_min | F_tot,min | eta_z,min | survivors (designs only) | best dT_int |
|---|---|---|---|---|---|
| 0.3 | 0.05 | 0.02 | 0.0 | 6: fr_Q200_fresh4242_h700, fr_Q20_final2_h440, fr_Q500_fresh4242_h700, fr_Q50_final0_h680, fr_Q50_fresh8080_h480, pilot_h525_Q100_fresh4242 | fr_Q50_fresh8080_h480 |
| 0.5 | 0.05 | 0.02 | 0.0 | 4: fr_Q200_fresh4242_h700, fr_Q500_fresh4242_h700, fr_Q50_fresh8080_h480, pilot_h525_Q100_fresh4242 | fr_Q50_fresh8080_h480 |
| 0.5 | 0.2 | 0.1 | 0.5 | 1: fr_Q50_fresh8080_h480 | fr_Q50_fresh8080_h480 |
| 0.7 | 0.2 | 0.1 | 0.7 | 1: fr_Q50_fresh8080_h480 | fr_Q50_fresh8080_h480 |
| 0.7 | 0.3 | 0.2 | 0.8 | 0:  | - |
| 0.9 | 0.3 | 0.2 | 0.8 | 0:  | - |

### 12.2 Correlations across the 20 designs (`correlations.json`)

| x | y | n | Spearman rho | p |
|---|---|---|---|---|
| Q_r | dT_max_pos | 18 | -0.13 | 0.598 |
| eta_Dz_lamE | dT_max_pos | 20 | 0.38 | 0.098 |
| eta_Dz_P_lamE | dT_max_pos | 20 | 0.37 | 0.107 |
| T_bg | dT_max_pos | 19 | 0.23 | 0.344 |
| gamma_nr_over_gamma_r_est | dT_max_pos | 18 | -0.1 | 0.699 |
| Ftot0_pos | dT_max_pos | 20 | 0.29 | 0.214 |
| eta_z_pos | dT_max_pos | 20 | 0.43 | 0.056 |
| Q_r | S_T_pos | 18 | -0.15 | 0.553 |
| eta_Dz_lamE | S_T_pos | 20 | 0.4 | 0.082 |
| Q_r | loaded_Q_pos | 13 | 0.54 | 0.055 |

Certified parents only (`certified_Q_vs_buildup.json`): Spearman of log Q_r with U_mid(lambda_r) = +0.83,
with eta_Dz(lambda_r) = +0.76 (n = 10); aligned subset (|detuning| < 1.5 linewidths, n = 7):
U_mid +0.86, eta_Dz +0.75.

### 12.3 Sensitivity of T_bg to its definition (`definition_sweep.json`; k = linewidths excluded per pole, D = ENZ half-window in nm)

| tag | T_bg (k=1, D=30) | (k=3, D=50, default) | (k=5, D=80) | undefined for |
|---|---|---|---|---|
| bare_ito_film_P825 | 0.96 | 0.96 | 0.96 | 0/12 |
| deck_cylinder_P588_h970 | 0.94 | 0.97 | 0.95 | 0/12 |
| fr_Q100_final0_h700 | 0.15 | 0.15 | 0.17 | 0/12 |
| fr_Q100_final1_h680 | 0.11 | 0.10 | 0.11 | 0/12 |
| fr_Q100_final3_h640 | 0.31 | 0.25 | 0.22 | 0/12 |
| fr_Q200_final1_h700 | 0.26 | 0.20 | - | 2/12 |
| fr_Q200_final3_h600 | 0.17 | 0.17 | 0.20 | 0/12 |
| fr_Q200_fresh4242_h700 | 0.97 | 0.97 | 0.97 | 0/12 |
| fr_Q20_final1_h420 | 0.26 | 0.30 | 0.28 | 0/12 |
| fr_Q20_final2_h440 | 0.20 | 0.31 | 0.31 | 0/12 |
| fr_Q20_final3_h380 | 0.13 | 0.19 | 0.18 | 0/12 |
| fr_Q500_final0_h380 | 0.27 | 0.23 | 0.23 | 0/12 |
| fr_Q500_final3_h600 | 0.18 | 0.19 | 0.19 | 0/12 |
| fr_Q500_fresh4242_h700 | 0.97 | 0.97 | 0.97 | 0/12 |
| fr_Q50_final0_h680 | 0.34 | 0.43 | - | 4/12 |
| fr_Q50_final2_h680 | 0.80 | - | - | 7/12 |
| fr_Q50_fresh8080_h480 | 0.75 | 0.86 | 0.92 | 0/12 |
| jzfz_final1 | 0.04 | 0.09 | 0.16 | 0/12 |
| jzfz_final3 | 0.03 | 0.03 | - | 4/12 |
| pilot_h525_Q100_fresh4242 | 0.96 | 0.96 | 0.96 | 0/12 |
| pilot_h525_Q50_fresh8080 | 0.76 | 0.75 | 0.73 | 0/12 |
| pilot_h675_Q100_warm_final1 | 0.11 | 0.10 | 0.11 | 0/12 |
<!-- TABLES END -->

## 13. Selection: reading the table without a scalar objective

The decision surface is (interior positive dT, S_T, cold loaded T at lambda_op, parent T_bg, contrast, cold
F_tot, eta_z, Rayleigh margin, order convergence).  Read across it:

| role | candidate | why |
|---|---|---|
| positive control (not a campaign design) | deck_cylinder_P588_h970 | dT_int +0.0189 at 1279 nm, S_T 0.54, cold T 0.55, T_bg 0.97, contrast 0.95, F_tot 0.31, eta_z 0.90, margin 387 nm, loaded Q 52 with gamma_rad/gamma_nr = 1.46 (exact); non-dominated on every axis but F_tot |
| campaign finalist A (resonant) | fr_Q50_final2_h680 | dT_int +0.0133 at 1314 nm on a loaded Q = 36 line, S_T 0.36, F_tot 0.78, eta_z 0.92, margin 63 nm; cold T only 0.17 and parent T_bg undefined (window full of poles); uncertified frontier run (exact pole computed here) |
| campaign finalist B (broadband) | pilot_h525_Q50_fresh8080 | dT_int +0.0142 at 1271 nm, S_T 0.41, F_tot 0.70, eta_z 0.96, T_bg 0.75; no parent pole, no loaded line (Q_rad 13 / Q_nr 7.9), cold T 0.21, margin 20 nm (guard limit) |
| parent-Q / eta_Dz control | fr_Q200_fresh4242_h700 | the campaign's transmissive Q_r = 228 parent (T_bg 0.97, contrast 0.30, eta_Dz 0.076): loaded line erased, dT_int +0.0026, cold T 0.90 |

The threshold sweep (section 12.1) makes the same statement quantitatively: requiring a transmissive parent
(T_bg >= 0.5) and any visible parent contrast leaves only the fresh lineages and the pilot fresh8080 among the
campaign designs, none of which exceeds dT_int = +0.0142, and adding a cold-heating floor F_tot >= 0.2 removes
the fresh4242 family (F_tot 0.07-0.10).  The campaign did not produce a design that is simultaneously
transmissive after loading (cold T >= 0.3), resonant (visible loaded line) and sensitive (dT_int >= 0.01); the
cylinder is.  This is the scientific outcome, and the FDTDX stage validates it on exactly these four frozen
geometries.

## 14. Exact commands

Environment: Python 3.11, torch (CPU), scipy, numpy, matplotlib in the system interpreter; `/opt/venv-fdtdx`
(fdtdx main-branch snapshot 98aef1c, jax 0.10.2) for the FDTDX stage.  All commands from
`enz_jz_inverse_design/qdz_activation/`.

    # sanity tests (17 s)
    python tests/sanity_tests.py --threads 4
    # the full frozen-candidate campaign as run here (two workers, ~6.5 h on 4 cores)
    python run_stageAB.py --worker 0 --nworkers 2 --threads 2 --order 9 9 --order-A 9 9 --orders-check 5 5 7 7 9 9 11 11 --step 2 --Te 1000
    python run_stageAB.py --worker 1 --nworkers 2 --threads 2 --order 9 9 --order-A 9 9 --orders-check 5 5 7 7 9 9 11 11 --step 2 --Te 1000
    # one candidate, another perturbation kind / size, or an oblique-incidence run through the same code path
    python run_stageAB.py --tags deck_cylinder_P588_h970 --threads 4 --kind drude_weight --delta 0.0106
    python run_stageAB.py --tags fr_Q50_final2_h680 --threads 4 --theta 4 --phi 0 --out outputs/candidates_theta4
    # selection tables, sweeps, correlations, figures; report tables
    python select_pareto.py && python finalize_report.py
    # exact loaded certification (loss-scaling continuation) of selected designs
    python loaded_certify_finalists.py --tags deck_cylinder_P588_h970 fr_Q50_final2_h680 --order 7 7 --threads 4
    # cheap angle look (not an ensemble) at a candidate's selected wavelengths
    python angle_check.py --tag deck_cylinder_P588_h970 --thetas 0 2 4 6 8
    # independent FDTDX validation of the frozen finalists (reference + 600-fs production + plots, ~1 h each)
    cd fdtdx_validation && ./run_validation.sh deck_cylinder_P588_h970 fr_Q50_final2_h680 pilot_h525_Q50_fresh8080 fr_Q200_fresh4242_h700

A topology-level sensitivity optimizer (section 11) is proposed, not implemented; no command exists for it.

## 15. Independent FDTDX validation

FDTDX_PLACEHOLDER


