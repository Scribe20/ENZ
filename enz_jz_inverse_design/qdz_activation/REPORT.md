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
* The highest certified Q_r among aligned designs (fr_Q100_final3_h640, Q_r = 499, eta_Dz_P = 0.042, T_bg = 0.25)
  gives the largest positive campaign dT that sits on a real loaded line: +0.0052 at 1341 nm (T 0.262 -> 0.267,
  loaded Q = 170, S_T = 0.13, linear, converged), while at lambda_E it does the opposite (dT = -0.013, T -> R).
* The design with the highest parent eta_Dz of all (0.194, pilot_h525_Q50_fresh8080) has NO certified parent
  resonance; loaded, it is a broadband ITO absorber (A = 0.70, eta_z = 0.97, <|E_z|^2>_ITO = 14) whose
  transmission opens as the ITO heats: dT = +0.0096 at lambda_E (T 0.157 -> 0.167, S_T = 0.26), i.e. the
  largest campaign response at lambda_E itself, but from a low cold transmission and without a resonance to
  narrow it.
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

Yes, by T_bg alone: every design with T_bg < 0.3 (the Q_target 100/200 warm family, the JZ finalists,
fr_Q100_final3_h640) has dT_max,pos <= +0.007, and in all of them the dominant loaded response is
"absorption -> reflection" or "T -> R"; the only positive dT of the JZ best design (+0.0073, S_T = 0.21)
occurs 15 nm from the glass Rayleigh anomaly at T = 0.08 and is exactly the order-sensitive near-threshold
state the nonlinear lane had flagged.  The threshold sweep (`outputs/tables/threshold_sweep.json`) shows
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

RESULTS_TABLE_PLACEHOLDER

