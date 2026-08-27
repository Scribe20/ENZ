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

## 5. REFIT OUTCOMES (fixed gates; adaptive joint t/r shared-pole fits)

Machine-readable: results/q_validation_v2.csv (refit targets),
results/q_validation_v2_full.csv (all 18 candidates),
results/audit/needle_forensics.json, figures/qfit_*.png.

Targets = every v1 ledger candidate with Qfit_ok and 100 < Q < 1e5
(5 candidates) + the needle (dedicated forensics). The remaining 12
candidates have no qualifying resonance (broad response or failed v1
fit, Q < 100); P0750_H0150_seed029's v1 "Q = 7.4e6" was already
invalid in v1 (fit failed, pole at 1634.7 nm — far outside the scanned
window) and stays excluded.

| candidate | v1 Q (1-nm fit) | audit Q_pole | change | FWHM (nm) | samples/FWHM | worst in-window energy resid | Q_RESOLVED |
|---|---|---|---|---|---|---|---|
| P0550_H0150_seed011 | 905 | 881.0 +- 1.7 | -2.7% | 1.510 | 26 | 0.041 | **NO** (energy gate) |
| P0550_H0350_seed011 | 292 | 280.0 +- 0.1 | -4.1% | 4.757 | 22 | 0.0024 | YES |
| P0650_H0350_seed011 | 297 | 267.4 +- 0.1 | -10.1% | 4.982 | 22 | 0.0024 | YES |
| P0750_H0250_seed011 (champion) | 559 | 357.6 +- 1.2 | **-36%** | 3.721 | 20 | 0.0026 | YES |
| P0750_H0250_seed029 | 597 | 542.7 +- 0.2 | -9.1% | 2.418 | 28 | 0.0075 | **NO** (energy gate, marginal) |
| P0750_H0350_seed029 (needle) | 1.45e7 | 276.2 +- 0.6 | -100% (artifact) | 4.743 | 21 | 0.0009 | YES |

Findings:

1. **The champion's Stage-A qualify Q was overestimated by 56%**
   (559.5 -> 357.6): the constant-background t-only 1-nm fit is
   systematically unreliable on this sloped Fano background. (The
   trajectory's 0.25-nm fit of the same structure gave ~370 — within 3%
   of the audited value; the error was resolution + background model,
   dominated by the qualify grid.) All other surviving values shifted
   -3% to -10%.
2. **The needle is an artifact, definitively**: adaptive refinement down
   to 5e-4-nm local steps finds an ordinary resonance, Q = 276.2 (order
   [9,9]) / 261.7 (order [11,11]), pole drifting 1309.80 -> 1303.90 nm
   with order. No sub-nm feature exists anywhere in the refined windows.
   A complex-frequency Newton probe was attempted and honestly failed to
   converge (recorded converged=false); it is moot given the resolved
   real-axis fits. Single-pixel erosion moves the resonance to
   1271.7 nm, Q ~ 171 (indicative, 5 samples/FWHM) — a smooth
   perturbation response, not needle fragility. The Stage-A flag
   "unresolved, not claimed" was correct; the number 1.4e7 is hereby
   voided.
3. **Energy gating retires two Q values at order [9,9]**:
   P0550_H0150_seed011 (Q = 881; worst in-window |T+R-1| = 0.041) and
   P0750_H0250_seed029 (Q = 543; 0.0075 vs the 5e-3 gate). Both fits
   are internally excellent (rms 0.2-0.6%, stability < 0.2%) — the data,
   not the fit, is untrusted at [9,9]. An order-[11,11] recheck is
   running; its outcome updates this section (see §6).
4. **Method agreement**: for every RESOLVED candidate, the independent
   Stage-A single-pole t-only fit and the audit joint t+r shared-pole
   fit agree within 3-10% (and the champion's two fine-grid methods
   within 3%); the two Fourier orders agree within 5% where both ran
   (needle 276 vs 262; champion 358 vs 347). Q values are therefore
   established by at least two independent extractions wherever
   Q_RESOLVED = true.
5. **Q_abs remains an algebraic inference** (1/Q_abs = 1/Q_loaded -
   1/Q_rad). No independent absorption-integral extraction was performed
   in Stage A or in this audit; every Q_abs statement must carry this
   label. Q_rad and Q_loaded were, and remain, independently fitted.
6. **Detuning-trajectory Q values** (620 -> 165 vs alpha) were fitted on
   the 0.25-nm fine grid with the v1 method: adequate for the monotone
   factor-4 TREND that supports the causal conclusion (the champion
   cross-check above bounds the method error at ~3% on that grid), but
   NOT precision values; the alpha = 1.0 point is superseded by
   Q = 357.6 +- 1.2.

## 6. Order-[11,11] recheck of the two energy-gate failures

Outcome (results/q_validation_v2_o11.csv): higher Fourier order does NOT
heal either candidate — the in-window energy violation GROWS with order:

| candidate | Q at [9,9] | Q at [11,11] | energy resid [9,9] → [11,11] | verdict |
|---|---|---|---|---|
| P0550_H0150_seed011 | 881.0 ± 1.7 @ 1330.67 | 903.4 ± 6.0 @ 1330.23 | 0.041 → **0.236** | UNRESOLVED at all tested orders |
| P0750_H0250_seed029 | 542.7 ± 0.2 @ 1312.04 | 545.4 ± 0.5 @ 1308.89 | 0.0075 → **0.0212** | UNRESOLVED at all tested orders |

Interpretation: the fitted Q values are *reproducible* (2.5% and 0.5%
across orders) and the features are certainly real spectral resonances,
but the RCWA solution underneath them is not converged at the resonance
(|T+R−1| up to 24% in-window at [11,11] for the first), so no
precision-Q claim is certified for either. Stage-A statements that used
these numbers (notably "ED-only control Q ≈ 905") must be phrased as
"a sharp feature with apparent Q ~ 900, numerically uncertified at the
tested orders". Certification would require substantially higher orders
or an independent solver — out of scope for this audit.

## 7. Champion order matrix (from results/audit/champion_matrices.json)

| order | Q_pole | lam_pole (nm) | FWHM (nm) | in-window energy resid | Q_RESOLVED |
|---|---|---|---|---|---|
| [9,9] | 357.6 ± 1.2 | 1330.38 | 3.721 | 0.0026 | YES |
| [11,11] | 346.6 ± 1.3 | 1329.26 | 3.835 | 0.0032 | YES |
| [13,13] | 336.6 ± 1.3 | 1328.39 | 3.947 | 0.0052 | NO (gate 5e-3) |
| [15,15] | 328.4 ± 1.5 | 1327.90 | 4.043 | 0.0105 | NO |

The fitted Q converges smoothly (-3.1%, -2.9%, -2.4% per step; pole
drift 1.1 -> 0.9 -> 0.4 nm, converging toward ~1327.5-1328 nm), but the
energy closure at resonance DEGRADES with order for this geometry
(0.0026 -> 0.0105), crossing the gate at [13,13]. The certified value is
therefore **Q = 347 ± 1 (stat) at [11,11]**, with an order-trend
systematic of order -3%/step still visible; naive extrapolation suggests
a converged Q near ~310-320, which is NOT claimed (uncertified orders).
Any future precision claim needs an energy-conserving higher-order
solve. The same order-worsening of |T+R-1| was found on both
energy-gate-failed candidates (§6) — it is a property of these sharp
resonances in staircased TORCWA geometry at the tested orders, not of
one candidate.
