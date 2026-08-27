# Q_EXTRACTION_AUDIT — forensic audit of Stage-A Q values

## 1. What the ORIGINAL Stage-A extraction actually did (code audit of
`ed_eq_qualify.pole_fit`, used for every ledger Q; and the trajectory
analysis reused the same function)

| item | Stage-A implementation |
|---|---|
| fit model | single complex pole + CONSTANT complex background: t(ω) = a + b/(ω − ω₀ + iγ) |
| variable | angular frequency ω = 2πc/λ (fit in ω; λ reported back) |
| background | constant complex a (no slope — inadequate for sloped Fano backgrounds) |
| data fitted | complex t_xx(λ) ONLY (r never used; no joint/shared-pole constraint) |
| fit window | ±15 samples around the max of the second derivative of |t| (max-curvature heuristic) |
| initial guesses | ONE fixed p0 (a = mean(t), b = 0.1×10¹², ω₀ at the curvature peak, γ = span/40); no multi-start |
| optimizer | scipy least_squares (default trf), max_nfev = 20000, no bounds beyond γ ≥ 0 via abs() |
| constraints/gates | "ok" = fit success AND pole inside window AND Q > 0 — nothing else |
| residual metric | none recorded |
| Q_rad definition | ω₀/(2γ) fitted on the LOSSLESS-material spectra (k → 0) |
| Q_loaded definition | same fit on k = 1e-4 spectra |
| Q_abs | **algebraically inferred**: 1/Q_abs = 1/Q_loaded − 1/Q_rad. This is NOT an independent measurement and was never validated against P_abs = (ωε₀/2)∫Im(ε)|E|²dV. |
| sampling | qualify: 1.0 nm grid (⇒ 2–9 samples per linewidth for the fitted 2–9 nm FWHM values — far below any precision standard); trajectory: 0.25 nm fine grid (9–35 samples/FWHM) |
| energy gating | none — several fitted windows contain points with |T+R−1| up to 0.197 |

Classification of the method: complex-amplitude single-pole Lorentzian fit
with constant background — NOT a power-FWHM estimate, NOT a Fano fit, NOT
an S-matrix pole computation, and NOT multi-port.

## 2. Faults established by Audit A (from saved data, before refits)

1. **Energy-conservation violations concentrate on the claimed-Q
   candidates** (lossless max|T+R−1|): P0550_H0350_s11 = 0.197,
   P0550_H0250_s29 = 0.041, P0550_H0150_s11 = 0.012 (ensemble median of
   maxima 9.3e-4). These are RCWA convergence failures at sharp features
   at order [9,9]; any Q fitted through such points is untrusted. 7/18
   lossy scans show A = 1−T−R < −1e-4 somewhere (same pathology).
2. **Resolution**: all qualify-ledger Q values (905, 597, 560, 292, 297,
   …) were fitted on the 1-nm grid — under-resolved by the ≥20–30
   samples/FWHM standard now adopted.
3. **The needle** (P0750_H0350_s29, fitted 1.4e7): fitted on a feature
   narrower than the sampling — meaningless number, correctly flagged in
   Stage A as unresolved, now being re-established with adaptive
   refinement + a complex-frequency pole probe.

## 3. Replacement method (this audit; `ed_eq_audit.py`)

* JOINT complex fit of t(ω) AND r(ω) with a SHARED complex pole ω_p and
  per-port linear complex backgrounds:
  s(ω) = c0_s + c1_s(ω−ω̄) + a_s/(ω−ω_p); Q_pole = Re(ω_p)/(−2 Im(ω_p)).
* Multi-start (6 initial guesses), scaled parameters.
* Adaptive spectral refinement until ≥20 samples inside the fitted FWHM
  and step ≤ FWHM/10.
* Per-point energy residual recorded; worst-in-window gate (< 5e-3).
* Stability gates: window-width refits (1.5×, 2.5× FWHM) must agree
  within 15%; pole inside window; joint rms residual < 5% of amplitude
  scale; parameter-covariance Q uncertainty reported.
* Q_RESOLVED = all gates pass. Anything else is reported as UNRESOLVED
  regardless of how plausible the number looks.

## 4. Stored-energy proxy audit (§19)

The Stage-A `U_peak` is exactly `mean over the sampling volume of
Re(eps)·|E|²` in TORCWA units — a dimensionless near-field energy-LIKE
proxy. It was never inserted into Q = ωU/P (Q came only from linewidths /
poles), and it carries no dispersive-material energy correction
(∂(ωε)/∂ω). It remains labeled a proxy; trend use only.
