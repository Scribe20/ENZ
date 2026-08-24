# SCIENTIFIC_CONTRACT — ED–EQ causality campaign (frozen before discovery)

Date frozen: 2026-08-24 (before any pilot optimization run).

## Primary question
Can freeform inverse design deliberately create a genuine electric-dipole /
electric-quadrupole (p_x + Qe_xz) multipolar state at λ₀ = 1332.5 nm, and
does high radiative Q emerge as a PHYSICAL CONSEQUENCE of the resulting
complex ED–EQ radiative interference — measured, never optimized?

## Causality rules (non-negotiable)
1. The PRIMARY discovery objective is
   `F_ED_EQ = ½[log(S_px + 1e-12) + log(S_Qxz + 1e-12)]`
   with S_px, S_Qxz the validated dimensionless per-cell radiation-
   efficiency scores of the exact current moments (ed_eq_core.py).
2. Q, linewidth, radiative leakage, R/T, group delay, BIC/EIT conditions,
   and any desired ED–EQ phase are EXCLUDED from the primary objective and
   from candidate pre-selection. They are measured afterwards as
   independent observables.
3. Stage-B functional engineering (if ever run) is a separate, labeled
   stage; its results cannot be used to claim Stage-A emergence.
4. Acceptable outcomes include the null result (co-excitation without Q
   enhancement); negative results are preserved and reported.
5. Any change of target or claim after seeing data is recorded here as a
   dated amendment.

## Frozen prior evidence (ED/MD campaign — not to be reinterpreted)
Large |Ex|² does not imply large p_x (P0870_H0100_seed029 was an
EQ/toroidal false positive of the field proxy); near-field proxies do not
identify multipolar states; these resonances can be substrate-bound.
Hence this campaign optimizes ACTUAL current multipoles, never |E|-proxies.

## Frozen methodological choices
* Material: a-Si (Franta 2013) primary, brackets Pierce/Karaman; a-Si:H
  identity caveat and loss scenarios per METHOD_VALIDATION.md; no
  extrapolation/clamping.
* Target EQ component: **Qe_xz** (derived, ED_EQ_CHANNEL_DERIVATION.md).
* Regime: **specular-only** (n_sub·P/λ₀ ≤ 0.90 at the target) so the
  radiation bookkeeping has exactly two open channels; this cleanly
  separates ED–EQ interference from near-cutoff lattice physics (the
  confound identified in the previous campaign). Lattice-assisted regimes
  are a possible follow-up, not this campaign.
* P_rad authority: TORCWA propagating channel amplitudes; MENP weights
  identify multipole content only.
* Multipole origin: cell center (canonical); origin sensitivity reported.
* No imposed geometric symmetry (none is required to isolate the
  x-polarized specular channel — derivation §1).
* Binarization discipline: filtering + β-continuation; claims only on
  hard-binary geometries qualified at multiple Fourier orders.

## Controls (mandatory before causal claims)
ED-dominated, EQ-dominated, co-excited, detuned, and (if naturally
occurring) different-relative-phase designs; plus the one-parameter
detuning trajectory of the strongest genuine candidate as the primary
causal test (λ_ED/λ_EQ alignment vs Δφ_rad vs P_rad vs Q).
