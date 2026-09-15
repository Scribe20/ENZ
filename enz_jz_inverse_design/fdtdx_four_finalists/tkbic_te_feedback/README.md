# TKBIC electron-temperature feedback for `final3`, recomputed with FDTDX

Reproduction of the three TKBIC "Te-feedback" plots (T(λ;Te) family, T(λ,Te) map, T_eff vs intensity) for the frozen
`final3` finalist (a-Si:H pattern P 825 / h 525 nm / pad 12 % on 23 nm ITO on glass) over **1230–1280 nm**, with the
electromagnetics done in **FDTDX** and the thermal feedback done exactly as in the TKBIC share
(`TKBIC_Te_feedback_share_20260911.zip`).  Causal chain implemented:

    I_in(t) → A_ITO(λ, Te) → Te(t) → ε_ITO(λ, Te) → {R, T, A}_final3(λ, Te) → T_eff(I, λ)

Nothing in the validated campaign (`../fx_sim.py`, `../fx_post.py`, `../materials/`, `../final3/`, `../RUN_SUMMARY.md`)
is modified; everything here lives in this folder.  No optimization, geometry sweep, multipole or TCMT work was done.

## 1. What was done, step by step

| step | file | what |
|---|---|---|
| ZIP model, vendored unmodified | `tkbic_ref/ttm/_ito_eps_Te.py`, `tkbic_ref/ttm/_te_fluence_map.py`, `tkbic_ref/cache/tkbic_meep_matfits.json`, `tkbic_ref/Materials_data/ITO_nk.csv` | ε_ITO(λ,Te) = ε_meas(λ) − [ħω_p²(Te) − ħω_p0²]/(E(E+iħγ₀)), ħω_p(Te) = ħω_p0 √(m*(300)/m*(Te)), Kane m*(Te)/m₀ = 0.35(1 + 2·0.4191(1 + (π²/4)(k_B Te)²)), ħω_p0 = 1.775882 eV, ħγ₀ = 0.126846 eV; the TTM integrator.  `ITO_nk.csv` is byte-identical (SHA-256) to the campaign's `materials_data/ITO_nk.csv`. |
| per-Te causal ITO models | `ito_te_models.py` → `outputs/ito_te_models.json`, `ito_te_fit_check.png/.csv` | For every Te the whole target ε_meas + Δε_Drude(Te) is re-fitted with the same causal FDTDX form the cold final3 model uses (`ε_∞ − ω_p²/(ω²+iγ_dω) + Δε ω₀²/(ω₀²−ω²−iγ_Lω)`, same bounds and multi-start as `../materials_fit.py`, fit weight 1.0 in 1230–1280 nm, 0.5 in 1200–1400, 0.2 in 1150–1450).  The Δε_Drude term alone has *negative* oscillator strength, so it cannot be handed to FDTDX as an extra pole, and simply changing the existing Drude plasma frequency would not reproduce the ZIP target (the cold FDTDX model carries its loss in a broad Lorentz pole, not in the Drude damping). |
| FDTDX runs | `fx_te_sim.py`, `queue_te.sh`, `runs/<tag>/` | `../fx_sim.build_scene` is called unchanged (same 64×64×154 rectilinear grid, 5 ITO cells of 4.6 nm, dt 13.01 as, TFSF x-polarised Gaussian pulse from air, PML/periodic boundaries, sub-pixel a-Si:H pattern, glass / a-Si:H Lorentz models); only `fx_sim.MODELS["ITO"]` is swapped for the Te model and the recorded wavelengths are set to 1200–1400 nm / 2 nm ∪ 1230–1280 nm / 1 nm.  Two extra flux planes bracket the ITO film (one lossless cell away from each interface) so that A_ITO is measured directly on the full grid; the ITO volume fields are recorded at 1255 and 1275 nm for the cross-check.  Own empty-domain reference run (`runs/ref`, 250 fs).  Production window **600 fs** for the nine Te; extra runs: 8000 K / 900 fs, 300 K / 300 fs, and the first-launch 400-fs runs of 300 K and 8000 K (kept). |
| spectra + EM checks | `fx_te_post.py` → `outputs/final3_Te_family_fdtdx.npz/.csv`, `em_checks.json`, `fig1_*.png`, `fig2_*.png`, `diag_*.png` | R, T from the R/T planes with the campaign formulas (`plane_flux`, dt-scaled pulse-mode phasors, reference P_inc and TFSF-leakage correction); A_rt = 1 − R − T; **A_ITO = [S_z(below ITO) − S_z(above ITO)]/P_inc**; ITO volume-loss integral (ω/c)Im ε ∫|E/E_inc|²dV/P² at 1255/1275 nm; probe decay; time-window convergence; 300 K vs the existing `../final3/prod` spectrum. |
| thermal feedback | `te_feedback.py` → `outputs/final3_Teff_vs_I_fdtdx.npz/.csv`, `fig3_*.png`, `fig3_feedback_summary.json` | The ZIP's `integrate_ttm` (batched RK4, dt = 1 fs, −1…+3 ps) with the ZIP section-8 conventions: Gaussian intensity envelope FWHM 150 fs normalised to the fluence, C_e = γ_e Te (Sommerfeld, γ_e = 5.233 J m⁻³ K⁻²), C_L = 2.6·10⁶ J m⁻³ K⁻¹, g = C_e/τ_ep with τ_ep = 450 fs as the primary curve (1000 fs and g = 0 drawn as the model band), d_ITO = 23 nm, A and T interpolated from the FDTDX table linearly in λ then linearly in Te and clipped to [300, 8000] K (the ZIP `_at`), **A re-evaluated from Te(t) at every RK4 stage**, T_eff = ∫I T[λ,Te(t)]dt / ∫I dt (the ZIP `avg["T"]`).  Peak intensity → fluence with the ZIP rule F = I_peak·τ_eff, τ_eff = 150 fs·√(π/2); intensity grid logspace(10⁶, 10^10.2, 60) W/cm² (ZIP Fig 12).  Wavelength selection with the ZIP section-6 metrics (logistic fit on E_in = 0…2.4 nJ per (8 µm)² pixel, ΔT, saturation at 1.6 nJ, T₀, E₀) and the ZIP `_sel` rules; a ZIP label is attached only when its criteria are met. |

Regression of the vendored feedback path against the ZIP's own numbers (`figures/demo_summary.npz`, λ_p = 1289.6 nm,
E_in = 1 nJ): Te_pk = 8390 K (g = 0), 7537 K (τ_ep = 1000 fs), 6929 K (τ_ep = 450 fs), A_eff = 0.271 — identical to
README §2.5 of the share.

Environment: FDTDX from upstream `main` (commit 9ac6395, 2026-08-24 — the same code base as the campaign's supplied
`fdtdx-main.zip` snapshot; the PyPI 0.6.2 wheel lacks `RectilinearGrid`), JAX 0.10.2 on the 4-core CPU, four runs in
parallel (≈12 steps/s each).

## 2. Results — see §3 for the numbers; figures

* `outputs/fig1_T_lambda_Te_final3.png` — T(λ;Te), 1230–1280 nm, one curve per Te (+ the existing 300-fs 300 K spectrum dotted).
* `outputs/fig2_T_map_lambda_Te_final3.png` — T(λ,Te) map, linear interpolation between the nine computed rows (the lookup the feedback uses).
* `outputs/fig3_Teff_vs_I_final3.png` — T_eff(I) at the representative wavelengths, τ_ep = 450 fs solid, 1000 fs dashed, g = 0 dotted; lower panel peak Te.
* Diagnostics: `diag_RTA_full_band.png` (R, T, A_ITO, 1−R−T on 1200–1400 nm), `diag_time_convergence.png` (400 / 600 / 900 fs windows, 300 fs vs existing), `fig3_diag_scan_Teff_E_lambda.png` (T_eff(E_in; λ_p) scan), `ito_te_fit_check.png`.

## 3. Sanity checks and numerical caveats

(filled in from `outputs/em_checks.json`, `outputs/ito_te_models.json` and `outputs/fig3_feedback_summary.json` — see the
section "Numbers" below.)
