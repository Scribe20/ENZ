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

RESULTS_PLACEHOLDER
