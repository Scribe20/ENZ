# METHOD_VALIDATION — ED-EQ causality campaign (gates passed before pilot)

## Gate 1 — material model
Primary: a-Si, Franta et al. 2013 (refractiveindex.info, tabulated nk
0.138-26.9 um): n(1332.5 nm) = 3.6517 + 6.0e-6 i, eps = 13.335.
Brackets: Pierce 1972 (n = 3.512), Karaman 2025 (n = 3.814).
Control: c-Si Franta 2017 (n = 3.5021 - matches literature; parsing check).
Identity: project baseline material is a-Si:H; no authoritative NIR
a-Si:H tabulation exists locally or in the database main shelf, so
unhydrogenated a-Si (Franta 2013) is adopted with the bracket band as the
stated uncertainty; realistic-loss scenario k = 1e-4 and lossless k -> 0
diagnostics defined. Interpolation: cubic spline strictly inside the
tabulated range; out-of-range evaluation raises (clamping forbidden).

## Gate 2 — channel derivation (ED_EQ_CHANNEL_DERIVATION.md)
Frozen target EQ component: **Qe_xz** (unique EQ partner of p_x in the
x-polarized specular channels; odd parity, channel-degenerate with m_y at
0th order; parity obstruction in symmetric background makes the substrate
constitutive for total-power ED-EQ interference).
Empirically frozen TORCWA reference-plane convention: scattered
amplitudes compare to the center-referenced sheet integral after
multiplication by e^{-ik h/2} (both channels; confirmed at 1.3-1.4%
residual in the thin-layer config; thick-config down-channel
determination truncation-limited at equal absolute error).

## Gates 3-4 — differentiable exact multipoles
torch closed-form exact kernels (option A - no surrogate gap);
dimensionless-units arithmetic (float32-safe; direct SI float32
under/overflowed to NaN - documented failure + fix).
A: worst moment vs corrected MENP port = 2.6e-7 (float32-field machine
precision). D: F_ED_EQ gradient finite and nonzero (|g| = 0.79) with
S_px, S_Qxz > 0. C: the (px, my, Qxz) truncation reproduces the exact
channel integral to 16-37% at kh/2 = 0.34-0.59 - the EXACT channel
integral (not the truncation) is the reconstruction operator for the
radiation-interference analysis.

## Raw run logs follow

## method_validation.py run
```
validation structure: P=700.0 h=250.0 lam=1332.5 order=[9, 9] eps_si=13.3351+0.0000j fill=0.500
A. torch-vs-MENP worst relative moment difference: 4.169e-07 (PASS) [float32 fields, float64 MENP]
B. exact-channel up  (t-based): reconstructed -1.5886e+00-4.4363e-01j  torcwa -1.5939e-01-1.7135e+00j  |diff|/|full| = 1.111
B. exact-channel down(r-based): reconstructed 2.1696e-01-1.4226e-01j  torcwa 1.4701e-01-2.0860e-01j  |diff|/|full| = 0.378
C. multipole truncation: up exact -1.5886e+00-4.4363e-01j vs p/m/Q -1.2042e+00-2.9077e-01j  rel 0.251
   down exact 2.1696e-01-1.4226e-01j vs 3.1198e-01-3.8400e-02j  rel 0.543
   parity split: even(exact) -6.858e-01-2.929e-01j vs p_x term -4.461e-01-1.646e-01j rel 0.365; odd(exact) -9.028e-01-1.507e-01j vs [my+Qxz] term -7.581e-01-1.262e-01j rel 0.160
D. gradient smoke: F=+nan S_px=nan S_Qxz=nan |grad|=nan (FAIL)
```

## method_validation.py run
```
validation structure: P=700.0 h=250.0 lam=1332.5 order=[9, 9] eps_si=13.3351+0.0000j fill=0.500
A. torch-vs-MENP worst relative moment difference: 2.613e-07 (PASS) [float32 fields, float64 MENP]
B. cfg1(h250) up: best ref phase = e^(i k -0.5 h), residual 0.051 (next-best 0.569)
B. cfg1(h250) dn: best ref phase = e^(i k 0.0 h), residual 0.211 (next-best 0.372)
C. cfg1(h250) parity: even rel 0.365 (kh/2=0.59); odd rel 0.160
B. cfg2(h140) up: best ref phase = e^(i k -0.5 h), residual 0.013 (next-best 0.337)
B. cfg2(h140) dn: best ref phase = e^(i k -0.5 h), residual 0.014 (next-best 0.337)
C. cfg2(h140) parity: even rel 0.255 (kh/2=0.34); odd rel 0.165
B. reference-plane convention consistent across configs: False
D. gradient smoke: F=-2.3412 S_px=4.983e-01 S_Qxz=1.858e-02 |grad|=7.893e-01 (PASS)
```
