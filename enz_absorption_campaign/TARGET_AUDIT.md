# TARGET AUDIT — resonant ITO power-transfer campaign (generated; no optimization run)

Governing sentence: *inverse-design a resonant metasurface that maximizes free-space optical power transfer into the ultrathin ITO layer at the ENZ wavelength, while preserving a genuine photonic resonance; do not prescribe the momentum channel, multipole composition, or polariton branch in advance.*

## Exact target

    PRIMARY   maximize A_ITO(λ_E) = 1 − R − T,  λ_E = 1433.488 nm
    GATE      C = [A(λ_E) − ½(A(λ_E−d)+A(λ_E+d))]/A(λ_E) ≥ C_MIN,  d = 80 nm
              A(λ_E) ≥ A(λ_E ± d)            (three-point center-dominance test)
    PENALTY   loss = −A(λ_E) + μ[relu(C_MIN − C)² + Σ± relu((A± − A_E)/A_E)²]
    RECOMMENDED  C_MIN = -0.2817,  μ = 30

The gate is an empirical ENZ-band spectral-selectivity surrogate (a *resonance-contrast gate*). It is **not** a Q constraint and the center-dominance test does **not** prove the peak lies within ±d. Resonance wavelength and Q are established post hoc by the channel-agnostic pole analysis. λ_E is a frozen operating wavelength inherited from the validated finite-K bare-film ENZ branch (K = G₁₀, 850-nm lattice); it is not a momentum term of the loss and was not discovered by the new objective.

## A_ITO cross-validation and failure test (references)

| reference | A_vol | A_(1−R−T) | F_Ez | η_z | C(80) | pole λ / Q (r/t, in-window) | off-window r/t poles |
|---|---|---|---|---|---|---|---|
| bare ITO | 0.0449 | 0.0449 | 0.000 | 0.000 | -0.0016 | none | 1295nm(Q=76.0), 1604nm(Q=39.7) |
| EDR cuboid | 0.1987 | 0.1987 | 2.148 | 0.765 | +0.1085 | 1464 nm / 5.25 | 1201nm(Q=19.6), 1217nm(Q=100.4), 1613nm(Q=233.1) |
| unpadded QNM winner | 0.2050 | 0.2050 | 2.321 | 0.801 | +0.1211 | 1499 nm / 5.75 | 1187nm(Q=31.6), 1219nm(Q=207.3) |
| padded QNM winner | 0.2009 | 0.2009 | 2.214 | 0.779 | +0.0783 | 1460 nm / 5.02 | 1212nm(Q=89.1), 1289nm(Q=48.6), 1617nm(Q=100.7) |
| padded F_ENZ winner | 0.2078 | 0.2079 | 2.285 | 0.778 | +0.0752 | 1497 nm / 5.64 | 1210nm(Q=106.6), 1283nm(Q=37.4) |

A_ITO = (ω/2)∫_ITO Im ε|E|²dV / (½A_cell) (LH units) agrees with 1−R−T to the quadrature error on every reference; a-Si (k=0) and glass are lossless, so 1−R−T is exactly the ITO absorption and is used in-loop (differentiable S-parameter path); the volume integral is recomputed at final validation.

## Channel-agnostic pole certification

Complex r_xx(ω), t_xx(ω) (zeroth order, amplitude normalization) sampled on 46 real frequencies 1250–1700 nm; AAA poles accepted only if: inside the ENZ window 1310–1557 nm (λ_E ± HWHM of the bare ENZ QNM, Q = 5.80), found in both r and t within 2%, residue ≥ 5% of the in-window maximum of each observable, and stable to <2% under 2× coarser resampling. No QNM overlap, harmonic, or multipole enters selection. Off-window poles (the padded class's ~1300 nm Si Mie pole) are listed but cannot certify the ENZ resonance.

### No-ITO photonic pole (same geometries, a-Si/glass)

- EDR cuboid: in-window pole none; all r/t-agreeing poles: 1213nm(Q=93.8), 1221nm(Q=23.6)
- padded QNM winner: in-window pole none; all r/t-agreeing poles: 1306nm(Q=72.1)
- padded F_ENZ winner: in-window pole none; all r/t-agreeing poles: 1217nm(Q=378.9), 1300nm(Q=47.0)
- unpadded QNM winner: in-window pole none; all r/t-agreeing poles: 1194nm(Q=103.4), 1222nm(Q=214.6)

## C(80) calibration set (perturbations of saved geometries; no optimization)

n = 35 geometries: references, x/y scalings, dilation/erosion, SDF morphs between the padded winners, unpadded width changes, EDR scalings, and non-resonant controls (uniform slabs, tiny/eroded patches). Padded-class perturbations are clipped to the 85-nm mask.

| geometry | fill | A_E | C(80) | center-dom. | pole λ (nm) | Q_pole | certified |
|---|---|---|---|---|---|---|---|
| ref: bare ITO | 0.000 | 0.0449 | -0.0016 | False | — | — | False |
| ref: EDR cuboid | 0.390 | 0.1987 | +0.1085 | True | 1464 | 5.25 | True |
| ref: unpadded QNM winner | 0.594 | 0.2050 | +0.1211 | True | 1499 | 5.75 | True |
| ref: padded QNM winner | 0.495 | 0.2009 | +0.0783 | True | 1460 | 5.02 | True |
| ref: padded F_ENZ winner | 0.488 | 0.2079 | +0.0752 | True | 1497 | 5.64 | True |
| padQ x-scale 0.8 | 0.392 | 0.1843 | +0.0999 | True | 1456 | 4.98 | True |
| padQ y-scale 0.8 | 0.397 | 0.2001 | +0.1031 | True | 1463 | 5.23 | True |
| padQ x-scale 0.9 | 0.445 | 0.1990 | +0.0941 | True | 1494 | 5.46 | True |
| padQ y-scale 0.9 | 0.441 | 0.2005 | +0.0982 | True | 1461 | 5.20 | True |
| padQ x-scale 1.1 | 0.544 | 0.1994 | +0.0399 | True | 1463 | 5.47 | True |
| padQ y-scale 1.1 | 0.501 | 0.2005 | +0.0740 | True | 1466 | 5.04 | True |
| padQ x-scale 1.2 | 0.597 | 0.1850 | -0.0886 | False | 1476 | 5.23 | True |
| padQ y-scale 1.2 | 0.502 | 0.2003 | +0.0738 | True | 1463 | 5.15 | True |
| padQ dilate -3px | 0.432 | 0.1996 | +0.0958 | True | 1468 | 5.08 | True |
| padQ dilate -2px | 0.452 | 0.2008 | +0.0922 | True | 1462 | 5.03 | True |
| padQ dilate -1px | 0.473 | 0.2011 | +0.0871 | True | 1460 | 5.16 | True |
| padQ dilate +1px | 0.512 | 0.2006 | +0.0673 | True | 1459 | 5.28 | True |
| padQ dilate +2px | 0.524 | 0.2003 | +0.0576 | True | 1460 | 5.20 | True |
| padQ dilate +3px | 0.536 | 0.2001 | +0.0441 | True | 1459 | 5.23 | True |
| morph padQ->padE 0.25 | 0.493 | 0.2028 | +0.0767 | True | 1463 | 5.17 | True |
| morph padQ->padE 0.5 | 0.487 | 0.2052 | +0.0787 | True | 1463 | 5.03 | True |
| morph padQ->padE 0.75 | 0.486 | 0.2071 | +0.0774 | True | 1469 | 5.49 | True |
| unpadded dilate -3px | 0.521 | 0.2001 | +0.0630 | True | 1511 | 5.25 | True |
| unpadded dilate -1px | 0.569 | 0.2043 | +0.1224 | True | 1481 | 4.91 | True |
| unpadded dilate +1px | 0.609 | 0.2031 | +0.1231 | True | 1463 | 5.23 | True |
| unpadded dilate +3px | 0.641 | 0.1981 | +0.1304 | True | 1499 | 5.80 | True |
| EDR scale 0.7 | 0.191 | 0.1389 | +0.0840 | True | 1505 | 4.63 | False |
| EDR scale 0.85 | 0.282 | 0.1824 | +0.0990 | True | 1499 | 5.35 | True |
| EDR scale 1.15 | 0.515 | 0.1854 | +0.0919 | True | 1473 | 5.12 | True |
| EDR scale 1.3 | 0.658 | 0.1519 | -0.2817 | False | 1328 | 34.54 | True |
| control: uniform slab 0.3 | 0.300 | 0.0422 | -0.0033 | False | — | — | False |
| control: uniform slab 0.6 | 0.600 | 0.0313 | -0.0065 | False | — | — | False |
| control: uniform slab 1.0 | 1.000 | 0.0233 | -0.0085 | False | — | — | False |
| control: EDR scale 0.4 | 0.058 | 0.0624 | +0.0329 | False | — | — | False |
| control: padQ erode -6px | 0.372 | 0.1902 | +0.1018 | True | 1503 | 5.49 | True |

Certified in-window resonant states: n = 29, C ∈ [-0.2817, +0.1304]; uncertified: n = 6, max C = +0.0840. C(80) **does NOT cleanly separate** the classes. Pearson corr(C, Q_pole) over certified states = -0.860 — reported, not assumed monotonic. **Recommended C_MIN = -0.2817** (midpoint of the gap when the classes separate; otherwise the minimum certified value, flagged). Figure: figures/calibration_scatter.png.

## Gradient-scale audit for μ

| geometry | A_E | C | ‖∇A_E‖ | ‖∇C‖ | ratio |
|---|---|---|---|---|---|
| init random (unmasked) | 0.0346 | -0.0063 | 2.874e-04 | 2.975e-03 | 10.35 |
| init random (padded mask) | 0.0950 | +0.1134 | 8.056e-03 | 9.719e-03 | 1.21 |
| ref: EDR cuboid | 0.1987 | +0.1085 | 9.761e-03 | 3.404e-03 | 0.35 |
| ref: unpadded QNM winner | 0.2050 | +0.1211 | 9.333e-03 | 3.469e-03 | 0.37 |
| ref: padded QNM winner | 0.2009 | +0.0783 | 9.219e-03 | 9.526e-03 | 1.03 |
| ref: padded F_ENZ winner | 0.2079 | +0.0752 | 1.085e-02 | 9.342e-03 | 0.86 |

Median ‖∇C‖/‖∇A‖ = 0.95. The penalty gradient is 2μδ‖∇C‖ at a contrast violation depth δ; μ is chosen so that it matches the physical gradient at δ = 0.02 (≈ a quarter of C_MIN) and stays weak (< 0.3×) at δ = 0.005: **μ = 30**. No optimization was run for this.

## Q reference scale

Q ≈ 5 (Q_REF_LOADED = 5.0) is an empirical trusted loaded-resonance reference scale — bare ENZ QNM 5.80, certified loaded poles 5.0–5.7 — used only as a post-hoc low-Q sanity reference, not a fundamental ceiling and not a normalized coherence fraction. Post-hoc acceptance requires an in-window certified pole (|λ_pole − λ_E| ≤ 124 nm, the bare-ENZ HWHM) and the no-ITO photonic-pole control.

## Design class for the next run (separate from the target)

A. **padded 85 nm** — controlled comparison with the two previous padded campaigns (same seed/schedule; isolates the objective change).
B. **unpadded** — least-prior, unrestricted search matching the campaign question; comparable to the unpadded QNM winner (highest saved A_ITO 0.205 and highest C(80) 0.121).
Recommendation: **B (unpadded)** for the single next run — the scientific question is end-to-end ENZ excitation, not isolated meta-atoms, and the calibration set spans both classes so C_MIN is not class-specific; A remains the controlled follow-up.

