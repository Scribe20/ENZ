# Phase 2: freeform a-Si metasurface optimized for ENZ-mode overlap

Differentiable TORCWA inverse design of the lateral a-Si pattern rho(x,y)
in the stack **air / a-Si(140 nm, freeform) / ITO(23 nm) / glass**, maximizing
the *target ENZ excitation FoM*

```
F_ENZ(rho) = | ∫_ITO conj(Ez_ENZ) · Ez_scat(rho) dV |² / P_inc,cell ,
Ez_scat    = Ez_full(rho) − Ez_reference        (same illumination)
```

with the frozen Phase-1 target mode from `../enz_target/target_enz_mode.npz`
(λ_E = 1527 nm, K/k₀ = 5.946 − 12.860i).  This FoM is a **coherent complex
overlap** (conjugate on the target, no premature absolute values).  It is
*not* a coupling rate g and does not by itself prove strong coupling.

## Scaffold

The supplied pixel-based TORCWA topology-optimization example
(`original_pixel_inverse_design.ipynb`, preserved unmodified) provides the
architecture, kept as-is:
FFT Gaussian-blur filtering → tanh projection with exponential β ramp
(1→1000) → linear ρ↔ε mixing → hand-rolled Adam (cosine-decayed step,
[0,1] clamping, y-mirror symmetrization).  Only the objective changed:
1st-order diffraction efficiency → ENZ-mode overlap.

The supplied torcwa package is vendored verbatim in `third_party/torcwa`.

## Key implementation facts (from the code audit)

- **Ez sampling (approach B):** `rcwa.field_xy(layer_num=1, x, y, z_prop)`
  reconstructs the complex field inside the ITO at chosen z; the
  internal-layer path is pure torch (matmul/diag/exp/sum, no detach), so it
  stays in the autograd graph. The overlap integrates 7 midpoint z-slices.
- **Gradient validated:** in complex128 autograd matches central finite
  differences to **~2×10⁻⁸ relative** (validate_gradient.py); in complex64
  the same check shows ~20% FD noise — hence the production dtype is
  complex128 (~4.6 s/iteration on 4 CPU threads at order [7,7]).
- **Momentum matching:** the period is chosen as px = py = 770 nm so the
  (3,0) reciprocal-lattice harmonic gives |G|/k₀ = 5.9494 vs target
  Re K/k₀ = 5.9463 → mismatch 5×10⁻⁴ (diagnosed and printed at startup,
  threshold configurable). Only the (0,0) order propagates in air/glass at
  1527 nm — the targeted harmonic is evanescent (dark), as it must be.
- **Complex-K approximation (documented):** the periodic target uses only
  the phase exp(+i·Re K·x). Im K (1/|Im K| ≈ 19 nm, overdamped mode) is a
  propagation-loss diagnostic and is NOT built into the periodic target;
  the overlap tests generation of the modal field *pattern*.
- **Reference field:** identical stack with the a-Si layer at ε = 1,
  computed once, detached. At normal incidence Ez_ref ≈ 0 (planar stack has
  no Ez), so Ez_scat ≈ Ez_full — still computed and subtracted for
  generality/validation.
- **Target normalization:** ∫_ITO |Ez_target|² dV = 1 on the exact overlap
  grid (residual printed). Target tensors are `.detach()`-ed.
- **z-mapping:** TORCWA measures z_prop from the a-Si-side face of the ITO;
  the Phase-1 profile is stored glass-up, so it is evaluated at
  z_p1 = d − z_prop (sign flip of Ez is a global phase; |a| unaffected).
- **±K degeneracy:** a₊ (target e^{+iK'x}) is the objective; a₋ is logged
  every iteration as a diagnostic. `TARGET_DIRECTION = "bidir"` switches to
  (|a₊|²+|a₋|²)/P_inc explicitly — never silently.
- **ε_aSi honesty note:** the supplied `Materials_data/aSiH.txt` covers
  192–999 nm only; the supplied Materials class would silently clamp at
  1527 nm. ε_aSi = (3.48)² is therefore an explicit config parameter
  (typical PECVD a-Si:H at 1.5 µm), not a supplied-data value.
- **ITO loss kept:** ε_ITO = −0.590 + 0.845i from the Phase-1 npz; spectra
  use the Phase-1 CSV dispersion.

## Files

```
config.py                    every parameter + provenance ledger
target_mode.py               npz loader, momentum diagnostic, target builder
torcwa_forward.py            stack builder, differentiable Ez-in-ITO, R/T
objective.py                 complex overlap a_ENZ and F_ENZ
optimize_enz_overlap.py      main loop (Example6 architecture)
validate_gradient.py         autograd stats + finite-difference check
analyze_result.py            Figures 1-8 + Fourier/z convergence checks
tests/smoke_test.py          9-point smoke test (small config)
original_pixel_inverse_design.ipynb   supplied Example6, unmodified
third_party/torcwa/          supplied TORCWA, vendored verbatim
outputs/                     histories/ geometries/ fields/ figures/
```

Run order: `python tests/smoke_test.py` → `python validate_gradient.py` →
`python optimize_enz_overlap.py` → `python analyze_result.py`.

## Limitations (stated, not hidden)

This first-stage optimization uses a **frozen bare-structure ENZ target**;
self-consistent ENZ-mode drift and hybrid-mode tracking are not included.
The dense a-Si coverage perturbs the true eigenmode (Karimi et al. note the
superstrate shifts the ENZ-mode central frequency). Finite RCWA truncation,
the fixed 770 nm period, and the frozen-λ_E target make the overlap spectrum
a diagnostic, not a mode-tracking calculation. F_ENZ is an overlap FoM —
claims about g, Rabi splitting, or strong coupling require complex
eigenfrequency analysis (later phase).
