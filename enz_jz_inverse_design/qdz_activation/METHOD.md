# METHOD — Q-Dz ACTIVATION extension (`enz_jz_inverse_design/qdz_activation/`)

Branch `claude/gifted-shannon-0aogad`, re-pointed onto the Q-Dz parent baseline `claude/jz-1302-qdz-parent`
(commit f7b0110) and merged with `claude/jz-1302-nonlinear-activation` (commit f8b5896, adds only
`nonlinear_activation_best/`).  Nothing in `qdz_parent/`, `forward.py`, `materials.py`, `analysis.py`,
`config.py`, `optimizer.py` or `nonlinear_activation_best/` is modified; no historical output is
rewritten.  All new code lives in this directory and every physical routine is imported from those
packages.

## 0. Audit of the baseline (what the existing campaign actually did)

| item | finding (with source) |
|---|---|
| Objective of `qdz_parent` | maximize `eta_Dz` of the NO-ITO parent subject to `Q_proxy ~ Q_target` and `lambda_r ~ lambda_E` (`optimize_parent.py`: `L = -log eta_Dz + L_Q + L_lam`, both constraint terms unity at a factor-e / one-linewidth error); exact post-hoc pole certification (`certify_parent.py`, AAA on r/t at [9,9]) |
| `eta_Dz` | `d_ITO * INT_A rho |Dz/eps0|^2 dA / INT_V rho eps_Si |E|^2 dV` at the prospective Si/ITO plane (glass side, Parseval-exact `Dz_mn = Ky Hx - Kx Hy` before the inverse convolution); `eta_Dz_P` uses the full-layer Parseval energy `W_layer` as denominator (`parent_fwd.py`).  Field-scale invariant (audit check 1), contact-region selective (check 2) |
| `Q_proxy` | closed-form quadratic fit of `1/W_Si` over 5 probes spanning +-0.25 kappa_target; calibrated median error 4.7 %; **ill-posed** whenever there is no peak at lambda_E (`well_posed = False`: Q_proxy 7.9e12 for `fr_Q100_final3_h640`, 1.4e3 vs exact 452 for `fr_Q500_fresh4242_h700`).  Never used here; only the exact AAA pole `Q_r` is |
| exact `Q_r` | `analysis.significant_poles` (r/t-agreeing, damped, significant AAA poles, Rayleigh branch points excluded, dense local rescan).  Parent materials lossless (a-Si:H k = 0, glass k ~ 1e-263 over 1150-1400 nm) so `Q_pole = Q_r` |
| `R`, `T`, `A`, `F_x,y,z`, `F_tot` | `forward.rt_all_orders`: power over ALL propagating orders and both output polarizations; `A = 1 - R - T`; `F_c = (w/2) Im eps_ITO INT_ITO |E_c|^2 dV / P_inc`, `P_inc = 0.5 P^2`; `F_tot = F_x+F_y+F_z = A` (ITO is the only lossy layer; residual = Fourier truncation, `config.TOL identity_abs = 1e-3`) |
| total vs specular T | for P = 825 nm the only diffraction threshold above 900 nm is the glass (+-1,0) Rayleigh wavelength `lambda_R = n_glass(lambda) P = 1251.23 nm` (dispersive; 1251.14 with the fixed n_glass(1300) of `analysis.rayleigh_wavelengths`).  Above it exactly one order propagates in both media and `T_total == T_00` to 1e-14 (verified per candidate in Stage A and in test T6; below it they differ by 0.012 at 1246 nm).  Every T quoted in this package is the total; the specular value is stored alongside |
| `lambda_E` | `materials.ito_zero_crossing()` on the supplied `ITO_nk.csv` at every call: **1302.281986 nm**, `eps_ITO = -4.7e-16 + 0.43195 i`; never hard-coded (audit `check3_lambda_E`) |
| state of the campaign | 15 frontier + 6 pilot optimizations; 10 exact certifications (7 frontier + 3 pilot; `outputs/certified/`); 8 frontier runs never certified; `hybrid_certify.py` and `detuning.py` were written but never run (no outputs); `collect.py` never run.  Baselines: Jz/Fz `final3` (h525) and `final1` (h575) parent-certified |
| nonlinear ITO model | branch `claude/jz-1302-nonlinear-activation`, `nonlinear_activation_best/ito_nonlinear.py`: `eps(lambda, Te) = eps_csv + [eps_D(Te) - eps_D(300 K)]`, Kane-band Drude weight (m0* = 0.3964 m_e, C = 0.4191 eV^-1, literature), N calibrated to the Drude fit of the supplied data (eps_inf 3.417, wp 2.6948 rad/fs, gamma 0.1847 rad/fs = 121.6 meV), gamma constant.  Exact reduction to the CSV at 300 K.  **Provisional (Lane B)** per its own `NONLINEAR_MODEL_AUDIT.md`.  Its REPORT already documents, for the Jz/Fz best design (thick slab, unity absorption), that heating closes absorption and opens REFLECTION while T stays < 0.01: the "absorption -> reflection" failure mode |

Consequences drawn for this extension: (i) the certified `Q_r`, `eta_Dz`, `eta_Dz_P` are reused as they are;
(ii) the only nonlinear material model in the repository is used as the perturbation source, in its native
parameter (electron temperature), and its provisional status is carried into every conclusion;
(iii) `F_z` is not optimized; (iv) the Q-target family 20 / 50 / 100 / 200 (+ the 500 extension already
present) is kept.

## 1. Perturbation layer (`ito_perturbation.py`)

`eps0(lambda) = materials.eps_ito(lambda)`; `eps1(lambda) = eps0 + delta_eps`, with (default)

    delta_eps(lambda) = -[wp^2(Te) - wp0^2] / (w^2 + i gamma w),   Te = 1000 K   (delta = 1 - wp^2/wp0^2 = 0.0106)

identical to `ITOHot.eps(lambda, Te) - eps_ito(lambda)` to 1e-16 (test T1).  At lambda_E:
`eps0 = 0 + 0.43195i`, `eps1 = +0.03626 + 0.42732i`, `|delta_eps| = 0.0366` (8.5 % of |eps0|), the ENZ crossing
moves 1302.3 -> 1309.4 nm.  **Robust vs provisional:** the complex DIRECTION of `delta_eps`,
`arg = -atan(gamma/w) = -7.3 deg` at lambda_E, is fixed by the Drude fit of the SUPPLIED cold data alone;
the MAGNITUDE per kelvin comes from the literature band model.  Hence `S_T = dT/|delta_eps|` is robust to
first order, `dT` per kelvin (or per pJ) is provisional.  Kinds `drude_weight` (scalar delta, the Stage-18
precedent of `enz_highq_driven_ez_audit`), `real_only`, `imag_only` (mechanism decomposition) and `custom`
(flagged NON-PHYSICAL, diagnostics only) are available; every output records the kind, the native
parameters (Te, wp^2 ratio, mu, N, eps_inf, wp0, gamma) and the resulting complex eps values.

## 2. Stage A — useful parent state (`parent_background.py`)

For a frozen hard-binary parent (no ITO), from the certification spectrum at [9,9] (or a pass computed
here for uncertified candidates, labelled as such):

* `T_bg` = median (also mean/min/max/std) of the TOTAL parent transmission over the systematic set
  `L_bg = {lambda in [lambda_E - D_op, lambda_E + D_op] ∩ [lambda_R + m_R, 1400]} \ U_poles [lambda_p +- min(k_excl FWHM_p, hw_max)]`
  with D_op = 50 nm, m_R = 10 nm, k_excl = 3, hw_max = D_op/2 = 25 nm (defaults; all recorded; the
  selection stage re-derives T_bg for k_excl in {1,2,3,5} x D_op in {30,50,80} nm from the stored spectra).
  Every significant parent pole is excluded, not only the one nearest lambda_E; a broad low-Q pole whose
  exclusion would empty the set hits the cap (recorded `cap_active`); fewer than 5 surviving samples ->
  `T_bg` undefined (reported, not silently filled).  The wavelengths used are stored.
* `C_res_center = |T_bg - T(lambda_r)|` (exact solve at the certified pole wavelength) and
  `C_res_ext = max |T_bg - T(lambda)|` over +-3 FWHM (25-point dense rescan at the spectrum order), plus the
  local depth `T_max - T_min`.  The extremum form is the far-field visibility of a Fano feature whose
  extremum is offset from the pole centre.
* per-order check (total vs (0,0)) at lambda_R + m_R, lambda_E and 1398 nm; dispersive Rayleigh set.
* `parent_metrics` recomputed at lambda_E and lambda_r with the unchanged `qdz_parent` code: for certified
  candidates this is a regression check against `certification.json` (rel. diff. ~1e-12); for uncertified
  ones the only value.

## 3. Stage B — real-ITO small-signal transmission sensitivity (`loaded_sensitivity.py`)

Frozen geometry; `forward.evaluate(..., eps_ito=eps0/eps1)` at order [9,9], n_z = 7, x-pol, normal
incidence; identical settings for the two states (test T7).  Grid: 2 nm from lambda_R - 10 nm to 1400 nm,
plus dense refinement windows +-3 FWHM (step FWHM/5, >= 0.25 nm) around every parent pole with Q > 30 in
the safe window and around the coarse loaded T minimum, plus a 0.5-nm rescan +-4 nm around the coarse
optimum.  Per wavelength: T0, T1, dT, R0, R1, dR, A0, A1, dA, F_x,y,z, F_tot (both states), eta_z = F_z/F_tot,
<|E_z|^2>_ITO, T_00 (specular), r, t (specular amplitudes), `S_T = dT/|delta_eps|`, identity residual.

* `lambda_op` is selected POST HOC: `lambda_op+` = argmax dT in the safe window [lambda_R + m_R, 1400],
  `lambda_op-` = argmin dT, `lambda_op|S|` = argmax |S_T|; lambda_E and the parent lambda_r are reported as
  reference points, never assumed to be the operating point.
* Response channel at each point from the signs of (dT, dR, dA) with a RELATIVE threshold (20 % of the
  largest change): e.g. `A->R (absorption to reflection; NO transmission opening)` — the failure mode
  stays visible; `A_to_R_dominant` flags a scan where max|dR| > 2 max|dT|.
* Heating: cold `F_tot` (= A; per unit area, the pixel-invariant heating quantity of the nonlinear lane)
  and, separately, `eta_z = F_z/F_tot` (longitudinal channel).  Neither is optimized.
* Loaded poles: AAA on the cold and on the perturbed specular r, t (Rayleigh excluded) -> loaded Q, pole
  shift and width change under the perturbation.
* `critical_coupling_estimate`: from PARENT quantities only, `Q_nr,est = |eps_ITO|^2 / (eps'' eta_Dz_P)`,
  `gamma_nr/gamma_r = Q_r/Q_nr,est`, `Q_loaded,est = (1/Q_r + 1/Q_nr,est)^-1`, compared with the exact loaded
  pole Q.  First order (parent field unchanged by loading, tangential E in the ITO neglected).
* Numerics: dT, T0, T1 at lambda_op+, lambda_op-, lambda_E for orders [5,5], [7,7], [9,9], [11,11];
  `numerically_unresolved` if |dT(11) - dT(9)| > max(0.02, 0.2 |dT(11)|) (both tolerances recorded);
  linear-response check S_T at Te = 600 / 1000 / 2000 K (spread < 25 % -> `linear_response`); real-only /
  imaginary-only decomposition and its additivity residual; Rayleigh safety margin of lambda_op in nm,
  relative and in loaded linewidths.
* Angle hook: theta, phi, pol are passed through everywhere (test T8: theta = 3 deg runs, closure 1e-12);
  an angle ensemble is a loop outside the module and rayleigh.py handles oblique thresholds.

## 4. Selection (`select_pareto.py`)

One row per candidate (failed and dominated ones kept), the non-dominated set on
(dT_max_pos, T_bg, C_res_ext, F_tot, eta_z, Rayleigh margin, Q_r, eta_Dz), survivors vs sweepable
thresholds (T_bg_min, C_res_min, F_tot_min, eta_z_min), the T_bg definition sweep, and Spearman/Pearson
correlations of Q_r, eta_Dz, eta_Dz_P, T_bg, C_res, gamma_nr/gamma_r, F_tot, eta_z with dT_max_pos and S_T.
No weighted scalar objective is formed.

## 5. Validation (`tests/sanity_tests.py`, `outputs/tests/sanity_tests.json`)

T1 material layer = CSV at 300 K and = native model at 1000 K (1e-16); T2 dT = 0 for eps1 = eps0 (exactly 0);
T3 closure and |F_tot - A| < 1e-3 (2.9e-5), parent R+T-1 = -2.7e-13; T4 planar-film sign convention vs an
independent transfer-matrix calculation (T to 1e-10, dT same sign; dT > 0 = more transmission in the
perturbed state); T5 qdz_parent baseline reproduced from the unchanged code (1.4e-11) and exact-pole Q used,
ill-posed proxies refused; T6 total = specular above lambda_R (1e-14), differs below; T7 identical sims except
eps_ITO; T8 angle hook; T9 dispersive vs fixed-index Rayleigh (0.09 nm); T10 background-set logic.

## 6. Runtime (4-core CPU, no GPU)

Loaded solve 0.8 s at [7,7], 2.7 s at [9,9], ~8 s at [11,11]; Stage A ~80 s, Stage B ~15 min per candidate at
[9,9] with 4 threads.  Campaign: `run_stageAB.py --worker w --nworkers 2 --threads 2` (two processes).
