# TARGET AUDIT — resonant ITO power-transfer campaign (no optimization run)

Governing sentence: *inverse-design a resonant metasurface that maximizes
free-space optical power transfer into the ultrathin ITO layer at the ENZ
wavelength, while preserving a genuine photonic resonance; do not prescribe
the momentum channel, multipole composition, or polariton branch in advance.*

## A. Exact objective

    maximize  A_ITO(λ_E) = P_abs,ITO / P_inc,cell

    P_abs,ITO = (ω/2) ∫_ITO Im[ε_ITO(ω)] |E(r)|² dV     (Lorentz–Heaviside, ε0 = 1)
    P_inc,cell = ½ |E_inc|² A_cell,  |E_inc| = 1, normal incidence from air

## B. ITO-only absorption — method and cross-validation

Two independent routes, both implemented in `target_audit.py`:
(a) the volume integral above with the 7-slice midpoint quadrature over the
full ITO cell volume (all harmonics, interior points only);
(b) the closure identity A_ITO = 1 − R − T, exact here because a-Si is
lossless in this band (measured k = 0) and glass is lossless — ITO is the
only dissipative layer.
Agreement on every candidate: |A_vol − A_rt| = 1.7×10⁻⁶ (bare ITO) to
5×10⁻⁵ (metasurfaces) — the quadrature error. In-loop the differentiable
(b) is used (S-parameter path, no field reconstruction); (a) is re-computed
at final validation so the ITO attribution stays explicit.

## C. λ_E = 1433.488 nm (validated bare-film QNM at K = G₁₀(850 nm)); ε_ITO(λ_E) = −0.0742 + 0.7014i (measured CSV, real axis). Unchanged.

## D–E. λ_res and Q extraction

Authoritative: the **complex pole of the driven response** (AAA rational
approximation of a complex driven amplitude sampled on the real axis,
residue-ranked with a minimum-damping cut; recovers the exact bare-slab
pole to ~10⁻⁶ and gave hybrid poles stable to 1.5×10⁻⁵ under resampling):
λ_res = 2πc/Re ω̃, Q = Re ω̃ / (2|Im ω̃|).
Sanity: window-restricted A(λ) peak/FWHM (Q_spec). **A global-max Q_spec
is NOT valid for the padded class** — it latches onto an off-target Si Mie
resonance near 1300 nm (Q_spec ≈ 80–110, A up to 0.38), outside the ENZ band.
In-loop (differentiable) surrogate — see G.

## F. Q_min = 5.0 — derivation

Trusted resonant references: bare ENZ QNM at G₁₀(850) Q = 5.80 (|D| =
1.6×10⁻¹⁵); loaded hybrid ENZ pole Q = 5.04 (770-nm design); padded QNM
winner pole Q = 5.69; padded F_ENZ winner pole Q = 5.16; EDR-like cuboid
Mie resonance Q ≈ 39 (bare) / 52 (loaded, detuned at 1230–1240 nm).
An ENZ-coupled resonance is material-loss-limited to Q ≈ 5–6 (the bare
ENZ Q), so Q_min = 5.0 demands ≥ 86% of the maximum available coherence
without excluding ENZ loading; a Mie-class Q_min (≈40) would forbid every
ENZ-coupled state. The admissible band is therefore narrow, [5, ~6]; the
constraint's role is to reject non-resonant broadband states (bare ITO,
Q undefined, contrast ≈ 0).

## G. Resonance surrogate and Δλ_allowed = 80 nm — calibrated, not invented

First draft (half-maximum at λ_E ± λ_E/2Q_min = ±143 nm) was tested on the
references and **rejected**: it fails every ENZ-resonant reference (red-side
ratios 0.74–0.80) because ENZ absorption sits on a ≈0.15 non-resonant
baseline, and its blue point is contaminated by the 1300-nm Si resonance.
Adopted surrogate (baseline-aware contrast):

    C(Δ) = [A(λ_E) − ½(A(λ_E−Δ) + A(λ_E+Δ))] / A(λ_E)  ≥  C_min
    A(λ_E) ≥ A(λ_E ± Δ)                    (peak within ±Δ of λ_E)

Calibration at Δ = 80 nm (points 1353/1513 nm):

| reference | C(80) | pole Q |
|---|---|---|
| bare ITO (non-resonant) | −0.002 | — |
| EDR-like cuboid | +0.108 | (ENZ-band feature) |
| unpadded QNM winner | +0.121 | — |
| padded QNM winner | +0.078 | 5.69 |
| padded F_ENZ winner | +0.075 | 5.16 |

C_min = 0.075 = the contrast of the validated ENZ-coupled reference with the
lowest pole Q that still satisfies Q_min. Δ = 80 nm is the widest window clear
of the off-target Si resonance (C turns negative for the padded designs at
Δ ≥ 120 nm) and ≈ half the ENZ HWHM; hence Δλ_allowed = 80 nm.
Penalty: loss = −A(λ_E) + μ[relu(C_min − C)² + Σ± relu((A± − A_E)/A_E)²],
μ = 100 (all terms dimensionless; a 0.02 contrast violation costs 0.04 ≈ 20%
of A). Sensitivity μ ∈ {30, 100, 300} to be run on the final candidate only,
never to shape the geometry. Cost: +2 S-matrix solves per iteration.
Final acceptance uses the pole Q (E), not the surrogate.

## H–L. Confirmations

H. F_Ez and η_z are computed and logged but are **not** in the loss.
I. No ±G / harmonic term in the loss (the loss reads only R, T at three
wavelengths). J. No multipole term. K. No UP/LP wavelength targeted — the
objective is evaluated at λ_E only. L. Design class unchanged: padded-85 nm
(86.33 nm/side realized), y-mirror only, x free, seed 333, [7,7], 128², same
filter/projection/optimizer/binarization conventions.

## §14 failure test (saved candidates, λ_E)

| candidate | A_ITO | F_Ez | η_z | ENZ-band A peak | pole Q |
|---|---|---|---|---|---|
| bare ITO | 0.045 | 0 | 0 | none (monotonic) | — |
| EDR-like cuboid | 0.199 | 2.148 | 0.765 | 0.202 @ 1460 | — |
| unpadded QNM winner | 0.205 | 2.321 | 0.801 | 0.215 @ 1472 | — |
| padded QNM winner | 0.201 | 2.214 | 0.779 | 0.205 @ 1460 | 5.69 @ 1500 |
| padded F_ENZ winner | 0.208 | 2.285 | 0.778 | 0.212 @ 1460 | 5.16 @ 1457 |

Direction confirmed in mild form: the Ez-objective raised F_Ez +3.2% and
A +3.5% while lowering pole Q by 9% (5.69 → 5.16). Not a collapse in these
candidates, but the new formulation rejects continuation of that direction:
C(80) of the F_ENZ winner (0.075) sits exactly at C_min, so any further
Q loss is penalized while A gains remain rewarded.

## Status

Objective mode `ito_absorption` implemented (dormant) in
`enz_inverse_design/{config.py, optimize_enz_overlap.py}`. Awaiting sign-off
on: Q_min = 5.0, Δ = 80 nm, C_min = 0.075, μ = 100, design class = padded-85.
