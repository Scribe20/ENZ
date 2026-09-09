# REPORT — nonlinear (hot-electron) activation response of the frozen best JZ design

Design under test (frozen, not re-optimized): `../outputs/best/rho_hard_binary.npy`, P = 825 nm, h = 525 nm, air pad 12 % of P,
a-Si:H / 23-nm ITO / soda-lime glass, normal incidence, x-polarization, 150-fs pulses.  Method: `METHOD.md`; model audit:
`NONLINEAR_MODEL_AUDIT.md`.  Every number below carries a lane label:

* **A** — directly calculated from supplied data (geometry, supplied materials, RCWA at a given ε, pulse/energy normalization).
* **B** — depends on the literature-derived provisional hot-electron ITO model (Kane band, Drude weight, TTM with a calibrated G).
  It is the *type* of model the SNU deck cites (Alam 2016), anchored to our cold ITO, but **it is not the measured nonlinear
  response of our ITO**.
* **C** — not determinable from the supplied sources.

## 0. Summary

1. **Cold state reproduced exactly (A).**  All Phase-1 reference values (F_z = 0.954252, A = 0.990748, R = 0.006561,
   T = 0.002690 at [11,11]; η_z = 0.963; ⟨|E_z/E_inc|²⟩ = 19.9) are recovered before any nonlinearity is added
   (`outputs/nonlinear_best/cold_state.log`).  The Lane-B permittivity reduces to the supplied `ITO_nk.csv` at 300 K with zero deviation.
2. **The SNU model is not recoverable; the slide is reproducible only qualitatively (audit).**  The deck gives the model type, the
   T_e set, the pulse and the energy axis, but no equations and no C_e / G / C_l / γ(T_e); no nonlinear measurement of our ITO exists.
   The calculation therefore stops short of a *final physical* activation curve: all intensity-axis results are Lane B.
3. **Temperature-parametric RCWA response (B, 18 T_e × 91 λ, [7,7], selected states certified at [9,9]/[11,11]).**  Heating the
   ITO electrons *closes the absorption channel and opens reflection*, while transmission stays shut: at λ_ZE = 1302 nm,
   T stays between 0.0001 and 0.003 for every T_e ≤ 8000 K, A falls 0.994 → 0.589 (6000 K) and R rises 0.004 → 0.41 (6000 K).
   The T-minimum moves only 1302 → 1304 nm, the absorption maximum moves 1298 → 1344 nm and drops to 0.705, the absorption band
   narrows (118 → 88 nm), and the ITO ENZ crossing moves 1302 → 1484 nm.
4. **Activation curves (B).**  With the calibrated G (electron cooling time 150 fs at 2000 K), the trusted range (peak T_e ≤ 6000 K)
   at 1302 nm ends at I_peak = 5.0×10⁹ W/cm² = 5.4 pJ per 825-nm cell = 0.80 nJ per 10-µm pixel (deck normalization).
   Within it: **transmission-mode modulation is negligible in absolute terms** (|ΔT| ≤ 0.0025 at the resonance; the largest T swing
   in the single-order window is 0.079 → 0.204 at 1268 nm, an order-sensitive near-Rayleigh state), whereas **reflection-mode
   modulation is large**: R rises from 2×10⁻⁵ to 0.26 at 1294 nm (0.43 at the 8000-K model limit), from 0.004 to 0.24 at 1302 nm,
   with a 50 %-swing intensity of ≈ 2.8×10⁹ W/cm² (3.0 pJ per cell, 0.45 nJ per 10-µm pixel).  The reflected input–output
   characteristic is smooth, superlinear and non-saturating within the trusted range (softplus RMSE 0.004 of full scale; leaky-ReLU 0.02).
5. **Mechanism (B, decomposed with RCWA).**  ≥ 80 % of the reflection turn-on comes from the *real-part* change of ε_ITO (Drude-weight
   loss → ENZ red-shift → ⟨|E_z|²⟩ in the ITO drops 20 → 12 → the ITO loss rate falls below the radiative rate, critical coupling is
   lost); the ε″ change alone produces R < 0.03.  The fitted (constrained) TCMT shows the same: γ_nr 104 → 43 meV while γ_r stays
   ≈ 100–120 meV (300 → 6000 K).  This is a *loss-rate mismatch* mechanism, not a resonance-detuning mechanism: transmission stays
   closed because the a-Si slab background is mirror-like (background T ≈ 0.14 in the TCMT fit).
6. **TCMT is not a validated substitute for RCWA here.**  The single-mode two-port fit has RMSE(T) = 0.02–0.06 and predicts a cold
   T-minimum of 0.03 instead of 0.0026, so every activation curve is computed from the RCWA lookup; TCMT is used only as a descriptor of
   γ_r(T_e), γ_nr(T_e).  For the deck's cylinder the same fitting code reproduces the deck's parameters (γ_r = 10.6, γ_nr = 10.4 meV,
   λ₀ = 1292.1 nm versus the deck's 8.54 / 8.54 meV / 1291.6 nm).
7. **Comparison with the deck's cylinder (B, same model, same normalization)** — see section 7.
8. **Lane C.**  Without a pump–probe or heated-ellipsometry measurement of this ITO (G, Δε(T_e), γ(T_e), damage threshold) the
   intensity axis is provisional; no experimentally predictive fJ/pJ threshold is claimed.

## 1. Cold-state reproduction (Lane A)

| quantity | request reference | this work [11,11] | [7,7] (map order) |
|---|---|---|---|
| F_z | 0.954252486 | 0.954252486 | 0.955906 |
| F_x, F_y | 0.028233109, 0.008144136 | 0.028315299, 0.008169121 | 0.029195, 0.008099 |
| F_tot | 0.990736906 | 0.990736906 | 0.993200 |
| A, R, T | 0.990748324, 0.006561415, 0.002690261 | 0.990748324, 0.006561415, 0.002690261 | 0.993210, 0.004203, 0.002587 |
| η_z = F_z/F_tot | 0.96317446 | 0.963174 | 0.9625 |
| ⟨\|E_z/E_inc\|²⟩, max | 19.9056, 103.524 | 19.9078, 104.887 | 19.94, 101.0 |

(The F_x/F_y and max\|E_z\|² differences versus the request's quoted values are at the 10⁻⁴–10⁻² level and come from the request quoting
Phase-1 numbers at a different z-sampling; the [11,11] energy quantities match to 10⁻⁹.)  Cold spectrum 1200–1380 nm (fig1, 300-K curve):
T-minimum 0.00257 at 1302 nm, absorption maximum 0.996 at 1298 nm, absorption FWHM 118 nm (Q_loaded ≈ 11), Rayleigh anomaly of the glass
side at n_glass·P = 1251 nm.

## 2. What model is used and what it is anchored to

See `NONLINEAR_MODEL_AUDIT.md` (sections 1–5) and `METHOD.md` (§1–3).  In short: ε(λ, T_e) = ε_csv(λ) + [ε_D(λ, T_e) − ε_D(λ, 300 K)]
with the Kane-band Drude weight (m₀* = 0.3964 mₑ, C = 0.4191 eV⁻¹, N calibrated to the supplied Drude fit ω_p = 2.6948 rad/fs,
γ = 121.6 meV — the deck's 122 meV), γ constant; TTM with C_e from the band, G = 1.35×10¹⁷ W m⁻³ K⁻¹ (calibrated to the 150-fs
transmission fall time of the supplied Karimi paper; bracketed ×½ … ×2), C_l = 2.4×10⁶ J m⁻³ K⁻¹; heating by the *total* ITO absorption
A(λ_op, T_e) = F_x + F_y + F_z = 1 − R − T at the instantaneous T_e (fig5 shows A falling from 0.99 to ≈ 0.6 during the pulse at the
highest trusted intensity).  Validity: trusted T_e ≤ 6000 K; model limit 8000 K (the deck's own set); above 8000 K everything is masked.

## 3. Temperature-parametric RCWA response (Lane B) — fig1, `spectra_vs_Te.csv/npz`

| T_e [K] | ε_ITO(1302) | λ_ENZ [nm] | λ(T-min) | T-min | λ(A-max) | A-max | FWHM_A [nm] | T, R, A at 1302 nm | T, A at 1292 nm |
|---|---|---|---|---|---|---|---|---|---|
| 300 | +0.001 + 0.432i | 1302 | 1302 | 0.00257 | 1298 | 0.996 | 118 | 0.0026, 0.004, 0.994 | 0.0071, 0.993 |
| 1000 | +0.038 + 0.427i | 1309 | 1302 | 0.00222 | 1300 | 0.997 | 116 | 0.0022, 0.002, 0.996 | 0.0075, 0.990 |
| 2000 | +0.146 + 0.413i | 1331 | 1302 | 0.00134 | 1306 | 0.989 | 114 | 0.0013, 0.011, 0.987 | 0.0089, 0.963 |
| 3000 | +0.292 + 0.395i | 1363 | 1302 | 0.00057 | 1316 | 0.955 | 106 | 0.0006, 0.065, 0.935 | 0.0112, 0.887 |
| 4000 | +0.453 + 0.374i | 1400 | 1304 | 0.00014 | 1326 | 0.888 | 98 | 0.0003, 0.167, 0.833 | 0.0143, 0.768 |
| 5000 | +0.614 + 0.353i | 1441 | 1304 | 0.00004 | 1336 | 0.800 | 92 | 0.0004, 0.292, 0.708 | 0.0178, 0.640 |
| 6000 | +0.769 + 0.334i | 1484 | 1304 | 0.00012 | 1344 | 0.705 | 88 | 0.0007, 0.410, 0.589 | 0.0215, 0.527 |
| 8000 (beyond trusted) | +1.052 + 0.298i | 1575 | 1306 | 0.00040 | 1358 | 0.536 | — | 0.0020, 0.589, 0.410 | 0.0295, 0.366 |

Checks: R + T + A = 1 to 10⁻¹⁶ at every grid point; |F_tot − A| ≤ 3×10⁻⁴ (Fourier truncation; the substrate and a-Si are lossless in the
supplied files).  **Order certification** (`certification_best.csv`, 20 states, [7,7] → [11,11]): at 1292–1306 nm |ΔA| ≤ 0.012, |ΔF_z| ≤ 0.017,
|ΔT| ≤ 0.014 (largest relative change: T at 1292 nm, 8000 K, 0.029 → 0.016); at 1268 nm (17 nm above the Rayleigh anomaly) the [7,7] table
is only indicative (|ΔT| up to 0.055, |ΔA| up to 0.063).  All "resonance" conclusions are drawn from the 1290–1310-nm states.

## 4. Pulse coupling, normalization and validity (Lane A for the axis mapping, B for the temperatures)

* E_cell = I_peak · P² · τ · √(π/(4 ln 2)); the Gaussian time integral is 1.0645 τ = 159.7 fs (verified).  1 nJ on a 10 × 10 µm² pixel
  (the deck's "≥ 10-µm pixel") = 10⁻³ J/cm² = I_peak 6.26×10⁹ W/cm² = 6.8 pJ per 825-nm cell = 3.5 pJ per 588-nm cell.
* Trusted / model limits at λ_op = 1302 nm (base G): I_peak = 5.0×10⁹ / 1.0×10¹⁰ W/cm² = 5.4 / 10.9 pJ per cell = 0.80 / 1.6 nJ per 10-µm
  pixel.  With G × ½: 4.0×10⁹ / 6.3×10⁹; with G × 2: 6.3×10⁹ / 1.26×10¹⁰.  The deck's energy axis (0.15–1.55 nJ) therefore spans
  T_e,peak ≈ 2300 K (0.15 nJ) … 8000 K (1.55 nJ) for this design — its upper half is outside the trusted range of the band model.
* Energy balance: |stored − absorbed| / absorbed ≤ 10⁻⁶ in every run; time-step convergence 1 fs → 0.5 fs → 0.25 fs changes ⟨T⟩ by
  < 2×10⁻¹¹ and T_e,peak by < 0.05 K (`dt_convergence.json`).  At the trusted limit 4.1 pJ of the 5.4 pJ incident on a cell is absorbed
  in the ITO during the pulse (pulse-averaged A = 0.76, F_z = 0.70): the absorbed power does *not* stay at 0.99 — the feedback matters.
* Caveat (C): the TTM uses one T_e for the whole ITO layer of the cell.  The cold field map has max/mean |E_z|² = 5.3, so local hot-spot
  temperatures can exceed the cell average; a spatially resolved TTM was not built (the deck's model is also cell-averaged).

## 5. Activation curves (Lane B) — fig2 (heatmaps, log and linear energy axes), fig3, fig4, `activation_curves*.csv`, `threshold_summary*.csv`

Selected λ_op (rule in METHOD §4, single-order window only): 1302 nm (λ_ZE = cold T-min), 1306 nm (largest negative T swing),
1268 nm (largest positive T swing; order-sensitive), 1292 nm (the deck's pump wavelength region); reflection mode adds 1294 nm
(R_cold = 2×10⁻⁵) and 1262 nm.

**Transmission mode (trusted range, T_e,peak ≤ 6000 K):**

| λ_op | T_cold → T_high | ΔT | ratio | I at 10 / 50 / 90 % swing [W/cm²] | E_cell at 50 % | shape (I_out = ⟨T⟩·I_in, normalized fits: RMSE) | G × ½ / × 2 |
|---|---|---|---|---|---|---|---|
| 1302 | 0.00257 → 0.00063 | −0.0019 | 0.25 (−6.1 dB) | 2.4×10⁸ / 9.8×10⁸ / 2.2×10⁹ | 1.07 pJ | decreasing, saturating (slope at limit 17 % of max); linear 0.16, leaky-ReLU 0.041, sigmoid 0.070, softplus 0.073 | T_high 0.00077 / 0.00057 |
| 1306 | 0.00338 → 0.00086 | −0.0025 | 0.26 | 2.7×10⁸ / 1.1×10⁹ / 3.2×10⁹ | 1.24 pJ | decreasing; leaky-ReLU 0.017, sigmoid 0.033 | 0.00079 / 0.00102 |
| 1292 | 0.0071 → 0.0157 | +0.0086 | 2.2 (+3.4 dB) | 7.1×10⁸ / 2.5×10⁹ / 4.4×10⁹ | 2.7 pJ | increasing, not saturating; softplus 0.0006, sigmoid 0.0013, leaky-ReLU 0.012, linear 0.063 | 0.0174 / 0.0133 |
| 1268 (indicative) | 0.079 → 0.204 | +0.125 | 2.6 | 6.5×10⁸ / 2.7×10⁹ / 5.5×10⁹ | 3.0 pJ | increasing, not saturating; softplus 0.0011 | 0.222 / 0.173 |

Across the whole single-order window the pulse-averaged T never exceeds 0.26 (that value only at the 1252-nm Rayleigh edge) and stays
below 0.1 for λ ≥ 1280 nm at every trusted intensity: **the design does not produce a usable transmission-mode activation** — the thick
a-Si layer with the ITO-loaded resonance keeps the transmission channel shut in both the cold and the hot state.

**Reflection mode (trusted range):**

| λ_op | R_cold → R_high (model limit) | ratio | I at 10 / 50 / 90 % [W/cm²] | E_cell at 10 / 50 % | shape of I_refl = ⟨R⟩·I_in |
|---|---|---|---|---|---|
| 1294 | 2.2×10⁻⁵ → 0.262 (0.43) | 1.2×10⁴ (41 dB) | 1.2×10⁹ / 2.8×10⁹ / 4.5×10⁹ | 1.3 / 3.0 pJ | increasing, superlinear, not saturating; softplus 0.004, sigmoid 0.008, leaky-ReLU 0.022 |
| 1292 | 2.2×10⁻⁴ → 0.266 (0.43) | 1.2×10³ (31 dB) | 1.1×10⁹ / 2.7×10⁹ / 4.5×10⁹ | 1.2 / 3.0 pJ | same |
| 1302 | 0.0039 → 0.241 (0.42) | 61 (18 dB) | 1.4×10⁹ / 3.0×10⁹ / 4.6×10⁹ | 1.6 / 3.2 pJ | non-monotonic at low I (R dips to 0.0017 near 1000 K), then superlinear |
| 1262 | 0.144 → 0.485 (0.55) | 3.4 | 6.2×10⁸ / 2.8×10⁹ / 6.4×10⁹ | 0.7 / 3.0 pJ | increasing; softplus 0.002 |

The reflected I_out(I_in) is not a ReLU (there is no linear regime above threshold within the trusted range: dI_out/dI_in still rises at
the limit) and not a sigmoid (no saturation is reached before T_e = 6000 K); on the trusted range it is best described as a smooth
softplus-like superlinear onset.  Whether it saturates at higher energy is outside the model's validity (the 8000-K state gives R ≈ 0.43,
still rising).  All threshold intensities scale with the (unsourced) G: ×½ / ×2 in G moves them by roughly −20 % / +25 %.

## 6. Mechanism (Lane B; RCWA decomposition `mechanism_decomposition.json`, resonance descriptors §3, TCMT fig7)

| λ_op = 1302 nm, T_e | ε′ hot only (ε″ cold) → R, A, ⟨\|E_z\|²⟩ | ε″ hot only → R, A, ⟨\|E_z\|²⟩ | both → R, A, ⟨\|E_z\|²⟩ |
|---|---|---|---|
| 2000 K | 0.009, 0.990, 19.7 | 0.006, 0.992, 20.8 | 0.011, 0.987, 20.6 |
| 4000 K | 0.141, 0.859, 16.6 | 0.013, 0.984, 22.9 | 0.167, 0.833, 18.7 |
| 6000 K | 0.337, 0.662, 12.1 | 0.025, 0.971, 25.5 | 0.410, 0.589, 14.0 |

* The response is driven by Δε′ (the Drude-weight loss that moves ε′(1302 nm) from 0 to +0.77 and the ENZ crossing from 1302 to 1484 nm).
  It acts by *reducing the longitudinal field enhancement in the ITO* (⟨|E_z|²⟩ 20 → 12), i.e. by lowering the ITO loss rate γ_nr below the
  radiative rate: the resonance leaves critical coupling and the incident power is reflected instead of absorbed.  Δε″ alone (0.43 → 0.33)
  slightly *increases* |E_z|² and changes R by < 0.03.
* It is not a spectral-detuning mechanism at λ_ZE: the transmission minimum moves only 2–4 nm while the absorption maximum moves 46 nm
  (6000 K) and loses 30 % of its height; the absorption band *narrows* (118 → 88 nm), consistent with γ_nr falling.
* Constrained TCMT (fig7; `tcmt_fit_order7.csv`): γ_r = 120 → 98 meV (γ₁ air 116 → 81, γ₂ glass 4 → 17), γ_nr = 104 → 43 meV,
  ρ = γ_r/γ_nr = 1.16 → 2.3, λ₀ = 1314 → 1348 nm (300 → 6000 K); background T = 0.14, R = 0.86 (mirror-like).  Fit quality: RMSE(T) 0.02–0.05,
  RMSE(R) 0.03–0.06, RMSE(A) 0.02–0.04, T-min 0.031 vs 0.0026 (RCWA), T-min shift error ≤ +12 nm, intensity-domain RMSE(⟨T⟩) 0.03 — the
  single-mode TCMT reproduces the trends (loss of critical coupling, red-shift) but not the transmission floor, so it is **not** used for any
  activation-curve number.

## 7. Comparison with the deck's TK-BIC cylinder (Lane B, identical model, materials, order, TTM parameters and intensity grid)

CYLINDER_SECTION_PLACEHOLDER

## 8. Answers to the six questions (from the calculations; lane labels in brackets)

1. **Strong transmission modulation?**  No [B].  Within the trusted range the absolute transmission swing at the resonance is |ΔT| ≤ 0.0025
   (0.0026 → 0.0006 at 1302 nm, a −6-dB *decrease*), the largest T swing in the single-order window is 0.08 → 0.20 (1268 nm, order-sensitive),
   and ⟨T⟩ < 0.1 for λ ≥ 1280 nm at every trusted intensity.  The strong modulation is in *reflection*: 2×10⁻⁵ → 0.26 (1294 nm), 0.004 → 0.24
   (1302 nm), and in absorption (0.99 → 0.76 pulse-averaged).  The transmission channel is shut by the a-Si slab itself [A: cold T = 0.0026,
   background T ≈ 0.14 from the TCMT fit].
2. **Required intensity / energy per 825-nm cell?**  For the reflection-mode swing at 1294 nm: 10 % of the trusted swing at 1.2×10⁹ W/cm²
   (1.3 pJ per cell, 0.19 nJ per 10-µm pixel), 50 % at 2.8×10⁹ W/cm² (3.0 pJ, 0.45 nJ), the trusted limit at 5.0×10⁹ W/cm² (5.4 pJ, 0.80 nJ)
   [B, ±25 % with G × ½ … × 2].  These are model-conditional numbers (the absolute scale follows the unsourced G and the band C_e); they are
   *not* an experimentally predictive threshold [C].  The transmission-mode 50 %-swing point at 1302 nm is 9.8×10⁸ W/cm² (1.1 pJ) but the
   swing itself is only −0.002.
3. **Mechanism?**  Loss of critical coupling by a *loss-rate mismatch* [B]: the Drude-weight loss (ε′ 0 → +0.77 at 6000 K, ENZ 1302 → 1484 nm)
   reduces |E_z|² in the ITO and hence γ_nr (104 → 43 meV in the TCMT descriptor) below γ_r (≈ 100–120 meV); ≥ 80 % of the reflection turn-on
   comes from ε′, < 20 % from ε″; the T-minimum detunes by only 2–4 nm and the absorption band narrows rather than broadens.
4. **Multiple activation shapes vs λ?**  Yes, but only two families [B]: (i) blue of / at the cold resonance (≈ 1290–1310 nm) — reflection
   rises superlinearly from ~0 (softplus-like onset, non-saturating up to 6000 K), transmission barely moves (decreasing at 1300–1310 nm,
   increasing at 1290–1296 nm, both at the 10⁻³ level); (ii) far red side (≥ 1340 nm) — transmission rises modestly (0.03 → 0.05–0.09) and
   reflection rises slightly.  No wavelength gives a sigmoid-like saturating curve within the trusted range; the deck's "saturable" (transmission
   decreasing) family does exist at 1300–1310 nm but with ΔT ≈ −0.002.
5. **Threshold vs the previous TK-BIC/cylinder design (same normalization, same model)?**  CYLINDER_Q5_PLACEHOLDER
6. **Does Q_loaded ≈ 9–11 with unity absorption improve or worsen the threshold vs the higher-Q design?**  CYLINDER_Q6_PLACEHOLDER

## 9. What cannot be determined (Lane C) — and what would resolve it

* The true Δε(T_e) of *this* ITO film (band non-parabolicity and effective mass are literature values for other films), its electron–phonon
  coupling G (only the supplied paper's 130–205-fs decay times constrain it indirectly), γ(T_e), the damage threshold, and the hot-spot
  (spatially non-uniform) electron temperature.  Consequently no experimentally predictive fJ/pJ threshold is claimed; the deck's
  0.15–1.55-nJ axis can be mapped to intensity exactly [A] but to T_e only through the Lane-B model.
* Required measurements: (i) degenerate pump–probe ΔT(t, F) on the bare 23-nm film at 1300 nm (gives G and the Δε scale per absorbed
  fluence), (ii) ellipsometry of the film under known heating or a T_e-resolved Δε(λ) (gives the Drude-weight and γ dependence), (iii) the
  SNU group's own parameter file if the deck's curves are to be reproduced quantitatively.  With (i)–(ii) the same pipeline (`build_lookup.py`,
  `ttm_activation.py`) re-runs unchanged.

## 10. Files (all under `outputs/nonlinear_best/`)

`cold_state.log`, `cold_state_reference.json`, `cold_spectrum_order7.npz` — cold check · `nonlinear_parameters.json`, `epsilon_vs_Te.csv` — Lane-B model ·
`lookup_order7.npz/.log` (+ `lookup_order9`, `lookup_order11`, `certification_best.csv/json`) — RCWA tables and certification ·
`spectra_vs_Te.csv/npz` — fig1 data · `heatmap_{T,R,A,Fz}.npz`, `heatmap_*_{base,G05,G2,dt1,dt025}.npz`, `heatmap_deps_*.npz`, `ttm_meta_*.json`,
`dt_convergence.json` — TTM scans · `activation_curves.csv`, `threshold_summary.csv`, `activation_summary.json`, `swing_vs_lambda_op.csv`
(+ `*_Rmode.*` for reflection) · `TTM_traces.csv/npz` — fig5 · `tcmt_fit_order7.csv/npz`, `tcmt_summary_order7.json` — fig7 ·
`mechanism_decomposition.json` · cylinder lane: `lookup_cyl_order7.npz`, `*_cyl.*`, `tcmt_*cyl*` · figures in `figures/`
(fig1_T_lambda_Te, fig2_T_heatmap, fig2_heatmaps_all, fig2_linear_axis, fig3_T_vs_I, fig4_Iout_vs_Iin, fig5_TTM_traces, fig6_eps_vs_input,
fig7_tcmt_vs_rcwa_order7, fig_composite_slide17_analogue, and the `_Rmode` / `_cyl` variants).
