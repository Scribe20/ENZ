# NONLINEAR_MODEL_AUDIT — what nonlinear ITO model generated the SNU activation slide, and what is actually available

Scope: every supplied source (`ITO_activation_08_19.pdf` — text and all 190 embedded slide images, the material
files, the two never-executed ENZ notebooks, `Example6.ipynb`, the Karimi et al. Nano Lett. 2023 text), the three
branches of the `Scribe20/ENZ` repository (`git grep` over all `*.py|*.md|*.ipynb|*.json|*.txt`), and the
web (search snippets only: `science.org`, `nature.com`, `arxiv.org`, `pmc.ncbi.nlm.nih.gov` are blocked by the
session's egress proxy, so no supplementary material could be downloaded).  No `.pptx` and no `ITO_first.pdf`
were supplied (searched the whole filesystem).

## 1. What nonlinear model was found

| item | status | where |
|---|---|---|
| Model TYPE | **found**: "Two-temperature model with fs speed — electron cloud response changes refractive index", citing Alam, De Leon, Boyd, *Science* 352, 795 (2016) and Karimi et al. 2023 | deck p. 9 (text), p. 17 (figures: T(λ;T_e) for T_e = 300, 600, 1000, 1500, 2000, 3000, 4500, 6000, 8000 K; transfer-function scan T(E;λ_p) for E_in = 0–2.4 nJ; RCWA vs "stabilized TCMT" curves vs peak intensity 10⁶–2×10¹⁰ W/cm², 150 fs) |
| Cold ITO permittivity | **found**: `ITO_nk.csv` (Drude–Lorentz SE fit, λ_ZE = 1302.282 nm, ε'' = 0.43195); deck p. 10(c–d): "ITO 23 nm + Drude fit", **γ_ENZ = 122 meV** | supplied file; deck image p10_Image115 |
| Drude parameters of OUR film | **derived here** from the supplied cold data: ε∞ = 3.4171, ω_p = 2.6948 rad/fs, γ = 0.18468 rad/fs = **121.6 meV** (least squares over 1150–1450 nm, rms 0.027) — the deck's 122 meV is reproduced | `ito_nonlinear.drude_fit` |
| TTM equations | **NOT found** in any supplied source or branch (the deck shows only the phrase; the repository's `enz_highq_driven_ez_audit/stage18_qt100_enz_shift_map.py` states explicitly: "no validated nonlinear ITO model (Δω_p per nJ, ε(T_e)) exists in the repository, and none is invented here") | — |
| Electron heat capacity C_e(T_e), band non-parabolicity, effective mass, electron density | **NOT found** in the sources.  Literature (web snippets): the Alam 2016 model uses the Kane non-parabolic band of Liu et al., *APL* 105, 181117 (2014) with **m₀* = 0.3964 mₑ, C = 0.4191 eV⁻¹** (their film: N = 1.5×10²⁷ m⁻³) | web snippets (values confirmed verbatim in two independent hits) |
| Electron–phonon coupling G, lattice heat capacity C_l | **NOT found**; web snippets give only ranges (metals 5.5×10¹⁶–2.6×10¹⁷ W m⁻³ K⁻¹, "ITO notably higher", recovery ≈ 300 fs; one unattributed ITO value C_l = 2.4×10⁶ J m⁻³ K⁻¹) | — |
| γ(T_e) | **NOT found**; the repository precedent (Stage 18) and Alam's model hold γ constant | — |
| Pulse / normalization | **found**: 150-fs pulses, λ_p = 1288–1300 nm, **E_in = 0.15–1.55 nJ per pixel with pixel ≥ 10 µm**, NA stop ≤ 0.14; the deck's E_in axis (0–2.4 nJ) is therefore per ≥10-µm pixel, i.e. ≈ 3.5 pJ per 588-nm unit cell per nJ, and its intensity axis (10⁶–2×10¹⁰ W/cm²) corresponds to 1 nJ ≈ 6×10⁹ W/cm² for a 10-µm pixel | deck p. 20 (image p20_Image163), p. 17 |
| Deck reference design | **found**: a-Si:H cylinder d = 405.7 nm, h = 970.5 nm, P = 588 nm on 23-nm ITO / glass; λ_dip = 1291.6 nm, γ_r = γ_nr = 8.54 meV, A_pk = 0.47, T_min ≈ 0.25; measured-T_e set on slide 17 | deck p. 13, 14, 15 |
| Measured nonlinear data of OUR ITO | **NOT supplied** (no pump–probe, no ΔT(F), no Δε(T_e)) | — |

## 2. Equations (Lane B — literature model, implemented in `ito_nonlinear.py`)

Because the exact SNU model is not recoverable, the calculation uses the published Alam-2016-type hot-electron
model, **anchored to the supplied cold data** so that it reduces exactly to `ITO_nk.csv` at 300 K:

    ε(λ, T_e) = ε_csv(λ) + [ε_D(λ, T_e) − ε_D(λ, 300 K)]                     (delta model, exact cold limit)
    ε_D(λ, T_e) = ε∞ − ω_p²(T_e) / (ω² + i γ ω)
    ω_p²(T_e) = (e²/ε₀) ∫ g(E) f(E; μ(T_e), T_e) / m*(E) dE                   (Drude weight of the Kane band)
    E(1 + C E) = ħ²k²/(2 m₀*),   m*(E) = m₀*(1 + 2 C E),   g(E) = (1/2π²)(2m₀*/ħ²)^{3/2} √(E(1+CE)) (1+2CE)
    N = ∫ g f dE  (fixed; μ(T_e) solved),   U(T_e) = ∫ E g f dE,   C_e = dU/dT_e
    TTM:  dU_e/dt = P_abs(t)/V_ITO − G (T_e − T_l),   C_l dT_l/dt = G (T_e − T_l)
    P_abs(t)/V_ITO = A(λ_op, T_e(t)) · I(t) / d_ITO   with A = F_x + F_y + F_z = 1 − R − T (TOTAL ITO absorption, RCWA)
    I(t) = I_peak exp[−4 ln2 t²/τ²],  τ = 150 fs

## 3. Numerical parameters and provenance

| parameter | value | provenance | lane |
|---|---|---|---|
| ε_csv(λ) | supplied `ITO_nk.csv` | supplied | A |
| ε∞, ω_p(300 K), γ | 3.4171, 2.6948 rad/fs, 0.18468 rad/fs (121.6 meV) | fit of the supplied data (deck: 122 meV) | A |
| m₀*, C | 0.3964 mₑ, 0.4191 eV⁻¹ | Liu et al. 2014 / Alam et al. 2016 (literature; not measured on our film) | B |
| N | 1.2802×10²⁷ m⁻³ | calibrated so that ω_p(300 K) of the Kane-band Drude weight equals the fitted ω_p | B (anchored to A) |
| E_F(300 K) | 0.808 eV | derived | B |
| C_e(T_e) | 4.06×10³ (300 K) … 2.0×10⁴ (2000 K) … 4.1×10⁴ J m⁻³ K⁻¹ (8000 K) | derived from the band model (no free parameter) | B |
| γ(T_e) | constant | assumption (repository precedent, Alam) | B |
| G | **not sourced**; used: G calibrated so that C_e(2000 K)/G = 150 fs (= the 130–205-fs transmission fall times measured by Karimi et al. on a-Si/23-nm-ITO metasurfaces, supplied paper), i.e. G ≈ 1.3×10¹⁷ W m⁻³ K⁻¹, with a ×½ / ×2 sensitivity bracket | calibrated to the supplied paper's decay times (indirect) | B |
| C_l | 2.4×10⁶ J m⁻³ K⁻¹ | unattributed web snippet; only enters through a < 100-K lattice rise within 2 ps | B |
| T_e set | 300, 600, 1000, 1500, 2000, 3000, 4500, 6000, 8000 K | deck slide 17 | A (set) |
| pulse | 150 fs FWHM Gaussian, normal incidence, x-pol | deck p. 20, task | A |

## 4. Allowed T_e / intensity range

* The band model is a degenerate-electron description; μ(T_e) crosses zero near 7000 K and the model above
  ≈ 6000 K (kT_e ≳ 0.5 eV ≈ 0.6 E_F) is an extrapolation.  The deck itself plots up to 8000 K, so the lookup table
  is built to 8000 K but every result above 6000 K is flagged.
* Intensity: explored 10⁵–10¹¹ W/cm²; results are reported only where peak T_e ≤ 8000 K, and the physically
  meaningful window is where T_e ≤ 6000 K.  ITO damage thresholds are not in the sources (not modelled).

## 5. Can the SNU slide be reproduced?

**Only qualitatively.**  The type of model (TTM + Drude with a T_e-dependent plasma frequency), the T_e set, the
pulse duration and the intensity/energy normalization are recoverable; the equations, C_e/G/C_l and the
T_e-dependence of γ are not, and no nonlinear measurement of our ITO exists in the sources.  Consequently:

* Lane A (directly calculated, source-grounded): cold spectra of the frozen design; the mapping between the
  deck's E_in per pixel and per-cell energy / peak intensity; all RCWA quantities at a GIVEN ε(λ).
* Lane B (literature-model exploratory): ε(λ, T_e), T/R/A/F_z(λ, T_e), and everything that depends on the
  T_e ↔ absorbed-energy map (C_e) or on the cooling rate (G): the intensity axis of the activation curves.
* Lane C (cannot be determined without new data): the true nonlinear response of THIS ITO film (its Δε(T_e), G,
  γ(T_e)), the damage threshold, and hence any experimentally predictive fJ/pJ threshold.

What would remove Lane C: a pump–probe ΔT(t, F) measurement on the bare 23-nm film (gives G and the Δε scale),
or ellipsometry of the film under known heating (gives Δε(T_e)), or simply the SNU group's own parameter file.
