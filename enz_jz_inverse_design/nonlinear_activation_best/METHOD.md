# METHOD — nonlinear activation calculation for the frozen best JZ design

Everything here is computed for the **frozen** certified best geometry of the JZ campaign
(`../outputs/best/rho_hard_binary.npy`, P = 825 nm, h = 525 nm, hard air pad 12 % of P, a-Si:H / 23-nm ITO /
soda-lime glass, normal incidence, x-polarization).  Nothing is re-optimized.  The material lanes A / B / C of
`NONLINEAR_MODEL_AUDIT.md` are used throughout: **A** = directly calculated from supplied data, **B** = depends on
the literature-derived provisional hot-electron model, **C** = not determinable without measurements.

## 0. Cold-state reproduction (Lane A) — `outputs/nonlinear_best/cold_state*.{log,json}`, `cold_spectrum_order7.npz`

`forward.evaluate` (TORCWA, exp(−jωt), c = ε₀ = μ₀ = 1; adapter of the JZ campaign) is re-run at the Phase-1
reference wavelength with the supplied material files.  The eight reference numbers of the request (F_z, F_x, F_y,
F_tot, A, R, T, η_z, mean/max |E_z/E_inc|²) are reproduced to ≤ 1×10⁻⁶ at [11,11] before any nonlinearity is
introduced (`cold_state.log`).  The cold spectrum 1200–1380 nm at [7,7] is stored.

## 1. Hot-electron permittivity ε_ITO(λ, T_e) (Lane B) — `ito_nonlinear.py`

* Drude fit of the supplied `ITO_nk.csv` over 1150–1450 nm → ε∞ = 3.4171, ω_p = 2.6948 rad/fs,
  γ = 0.18468 rad/fs (121.6 meV; the deck's slide 10 quotes 122 meV for the same film).
* Kane non-parabolic conduction band (m₀* = 0.3964 mₑ, C = 0.4191 eV⁻¹; Liu et al. 2014 / Alam et al. 2016),
  electron density N calibrated so that the band's Drude weight at 300 K equals the fitted ω_p²
  (N = 1.280×10²⁷ m⁻³, E_F = 0.808 eV); at every T_e the chemical potential μ(T_e) is solved from fixed N and
  ω_p²(T_e) = (e²/ε₀)∫ g f / m*(E) dE.  γ is held constant (assumption; no source gives γ(T_e)).
* **Delta model**: ε(λ, T_e) = ε_csv(λ) + [ε_D(λ, T_e) − ε_D(λ, 300 K)].  Pre-declared tolerance at 300 K:
  |Δε| ≤ 1×10⁻⁶ — the reduction is exact (0.0) by construction, so the cold JZ results are unchanged.
* U(T_e) = ∫ E g f dE and C_e = dU/dT_e follow from the same band (no extra parameter).
* Validity: the deck's T_e set reaches 8000 K, so the tables are built to 8000 K; the degenerate-band description
  is an extrapolation above ≈ 6000 K (μ crosses 0 near 7000 K).  Results with peak T_e > 6000 K are shown dashed
  or masked, results with peak T_e > 8000 K are masked everywhere and excluded from every metric.

## 2. RCWA lookup table T, R, A, F_x, F_y, F_z, F_tot (λ, T_e) — `build_lookup.py`

Every material other than ITO stays at its supplied cold dispersion.  For each T_e of
{300, 400, 500, 600, 800, 1000, 1250, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000} K and each
λ of 1200–1380 nm (step 2 nm) the frozen design is solved at order [7,7] (the JZ campaign's map order) with the
`eps_ito` override of `forward.build_sim`; R and T are all-order energy fluxes, A = 1 − R − T, F_x,y,z the
Fourier–Parseval ITO loss components, F_tot = F_x + F_y + F_z.  The check |F_tot − A| ≤ 3×10⁻⁴ holds at every
(λ, T_e) (F_tot is the direct ITO dissipation; A additionally contains the glass/a-Si extinction, which is zero for
the supplied files, so the residual is the Fourier-truncation error).  Selected (λ_op, T_e) states are
re-evaluated at [9,9] and [11,11] (`certify_compare.py` → `certification_best.{csv,json}`).

## 3. Two-temperature model with self-consistent optical feedback — `ttm_activation.py`

    dU_e/dt = A(λ_op, T_e(t)) · I(t) / d_ITO − G (T_e − T_l),      C_l dT_l/dt = G (T_e − T_l)
    T_e = T_e(U_e) (inverse of the band U(T_e)),  I(t) = I_peak exp(−4 ln2 t²/τ²),  τ = 150 fs

* The heating term uses the **total** ITO absorption A = F_x + F_y + F_z = 1 − R − T at the instantaneous T_e
  (quasi-static optics, cubic-spline interpolation of the [7,7] table in T_e at fixed λ_op), so the absorbed
  power falls or rises self-consistently as the resonance detunes; nothing assumes A = 0.99.
* Integration: RK4 in (U_e, T_l), Δt = 0.5 fs over −450 fs … +1200 fs, vectorized over 61 log-spaced peak
  intensities 10⁵–10¹¹ W/cm² for every λ_op of the table (91 wavelengths).  Convergence: repeated with Δt = 1 fs and
  0.25 fs (`ttm_dt1.log`, `ttm_dt025.log`).  Energy check: the stored energy (U_e − U_300 + C_l ΔT_l)·d_ITO must equal
  ∫A I dt to machine precision (`energy_resid` in every heatmap file; ≤ 10⁻⁷).
* Pulse-averaged observables ⟨Q⟩ = ∫ Q(T_e(t)) I(t) dt / ∫ I(t) dt for Q ∈ {T, R, A, F_z, F_x, F_y, F_tot}
  (these are the energy transmittance / reflectance / absorptance of the whole pulse — what a slow detector sees),
  peak and final T_e, final T_l.  I_out(I_in) = ⟨T⟩ · I_peak.
* Parameters: G (not sourced) is calibrated so that C_e(2000 K)/G = 150 fs, the transmission fall time measured
  by Karimi et al. on a-Si/23-nm-ITO metasurfaces (supplied paper: 130–205 fs), G = 1.35×10¹⁷ W m⁻³ K⁻¹, bracketed
  by runs with G × ½ (`G05`) and G × 2 (`G2`).  C_l = 2.4×10⁶ J m⁻³ K⁻¹ (unattributed; the lattice rises < 100 K
  within the window, so it barely matters).
* Energy per cell: E_cell = I_peak · P² · τ · √(π/(4 ln 2)) (verified: the Gaussian's time integral is
  τ√(π/(4 ln2)) = 1.0645 τ); the deck's E_in axis (nJ per ≥ 10-µm pixel) maps to I_peak ≈ 6.26×10⁹ W/cm² per nJ
  on a 10 × 10 µm² pixel, i.e. 4.5 pJ per 825-nm cell per nJ.
* The solver never extrapolates the optical table above 8000 K (values are clipped and the state is flagged
  `valid_model = False`); the thermodynamic inverse U → T_e is tabulated to 12 000 K only so that such flagged
  runs terminate gracefully.  Every plotted or tabulated metric uses the trusted range (T_e,peak ≤ 6000 K) and
  states the model limit (≤ 8000 K) separately.

## 4. Figures, quantification — `make_figures.py`

fig1 T, A, R, F_z(λ; T_e); fig2 ⟨T⟩(λ_op, I_peak) with the E_cell axis and the 6000 / 8000-K contours (plus R, A,
F_z, peak T_e, Δε′, Δε″, ⟨T⟩ − T_cold panels); fig3 ⟨T⟩(I_peak) at the selected λ_op with the G-bracket band;
fig4 I_out(I_in) with shape-descriptor fits (linear, leaky-ReLU, softplus, sigmoid on the trusted range, normalized
axes; the fits describe, they do not label); fig5 TTM traces; fig6 Δε(I) and ε(T_e), λ_ENZ(T_e); composite
slide-17 analogue.  λ_op selection rule: the cold T-minimum / λ_ZE, the wavelength of the largest positive and the
largest negative trusted-range swing ΔT, plus any wavelengths given on the command line.  Metrics per λ_op
(`threshold_summary.csv`): T_low, T_high, ΔT, contrast, I and E_cell at 10 / 50 / 90 % of the trusted-range swing,
dynamic range, maximum |d⟨T⟩/dlog₁₀I|, dI_out/dI_in range, monotonicity, saturation flag, fit RMSEs.

## 5. TCMT fit and validation — `tcmt_fit.py`

Single-mode, two-port (air / glass) TCMT with a lossless non-resonant background (Fan–Suh–Joannopoulos 2003; the
deck's slide-14 formula is the special case r_d = 0, γ₁ = γ₂): seven real parameters per T_e
(E₀, γ₁, γ₂, γ_nr, r_d, two relative phases) fitted to T(λ) **and** R(λ) on the single-order window
λ > n_glass·P = 1251 nm (below it the (±1, 0) orders propagate in the glass and a two-port model is not defined).
A_TCMT = 1 − T − R is not fitted and serves as a check.  Reported: RMSE / max error of T, R, A per T_e, the
T-min shift RCWA vs TCMT (resonance-shift error), γ_r(T_e), γ_nr(T_e), Q's, and — the actual replacement test —
⟨T⟩(I) from the same TTM driven by the TCMT tables versus the RCWA tables at the selected λ_op.  The same fit is
applied to the deck cylinder's cold spectrum as a check of the fitting code against the deck's γ_r = γ_nr = 8.54 meV.

## 6. Comparison lane: the deck's TK-BIC cylinder — `build_lookup.py --geom cyl`

The deck's fabricated design (a-Si:H cylinder d = 405.7 nm, h = 970.5 nm, P = 588 nm on the same 23-nm ITO / glass)
is rasterized on the same 128 × 128 grid, solved with the same materials, order, ε_ITO(λ, T_e) table, TTM
parameters and intensity grid, and quantified with the same script, so that thresholds are compared at identical
normalization (per-cell energy at the same P²-scaling AND per-area intensity, which is the pixel-invariant quantity).
Its cold spectrum (T-min 0.224 at 1294 nm, A-max 0.466 at 1292 nm) reproduces the deck's slide-13/14 numbers
(1291.6 nm, T-min ≈ 0.25, A_pk = 0.47) with the supplied materials.
