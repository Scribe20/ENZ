# P0550_PHASE_KNOB_AUDIT — thickness as a radiative-phase knob (Phases A-F)

Data: p0550_h_phase_unwrapped.csv, p0550_h_complex_rt.csv,
p0550_h_multipoles.csv, p0550_smatrix_vs_h.csv, ar_keypoints.csv
(r-Argand rows), audit/overlap/*.npz; Figures ps_fig1, ps_fig2.
Grid: h = 200-300 nm at 2.5 nm, refined to 1.25 nm in 215-240;
lam0 = 1332.5 nm; order [9,9] (order behavior certified in the AR audit).

## A. Full observable family
Every h row carries the complete exact 4-family fractions and purities,
complex r/t (both ports via the S-matrix scan), backgrounds, exact
channel integrals and ladder terms, T/R/A. A = 1-T-R stays < 1e-3 in
magnitude across the family except the known large-h drift (max
unitarity residual 0.020 at the family edge, dominated by cross-pol +
numerics; cross-pol total <= 2%).

## B. Unwrapping
The -180/+180 jump at h ~ 221 nm is wrapping. Unwrapped
Delta_phi_ED-EQ(bottom): strictly monotonic; -1.34 deg/nm over 215-240
(rms linear residual 0.38 deg), -0.85 deg/nm global, slope range
-0.28...-1.43 deg/nm; total span 85 deg over 100 nm. Top phase = bottom
+ 180 exactly (parity). Verdict: smooth, mildly nonlinear, no mode
interruption.

## C. Mode identity
Criterion: D_comp(vs h=227.5) <= 0.10 AND px|ED >= 0.95 AND
Qxz|EQ >= 0.70 AND f_ED+f_EQ >= 0.85 -> contiguous h = 215-262.5 nm.
Independent field-level check: normalized mid-slab induced-current
overlap between 5-nm neighbors >= 0.9978 over the whole 200-300 range,
>= 0.9995 within 215-240 - the SAME state, continuously deformed.

## D. Multipole phase vs device phase
corr(Dphi, arg t) = -0.977; corr(Dphi, arg r) = -0.995;
d arg(t)/d Dphi = -1.45 in the identity interval. Coverage over the
identity interval: arg(t) 76 deg, arg(r) 219 deg, while R spans
0.020-0.283 and T 0.980-0.717. A flat-optics 2pi library is NOT claimed
(76 deg only); what is claimed: the multipolar relative phase is a
faithful, nearly linear proxy of the device phases in this family.

## E. S-matrix
Power-normalized 2x2 co-pol S-matrix vs h: reciprocity
max|t12-t21| = 3.9e-5; unitarity residual (co-pol only) <= 0.020 - the
deficit is cross-pol power + weak absorption, closing to <= 1e-3 with
all four specular channels. R_f = R_b to <= 1.6e-3.

## F. Argand set (one normalization)
h = 221: |ED| = 1.56, |EQ| = 1.62 internally cancel; full r lands ON the
background (|r| = 0.182 ~ |r_bg| = 0.187). h = 227.2: detuned pair
leaves a residual opposing r_bg -> |r| = 0.141, R = 0.020 (transparent
reference). h = 235 and 250: progressively larger residual mis-phased
with bg -> R 0.040 / 0.160. Figures ps_fig2 (and ar_argand_mechanism
for the lam-Rmin panel).
