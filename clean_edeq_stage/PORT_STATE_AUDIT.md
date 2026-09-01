# PORT_STATE_AUDIT — composition, radiative phase, and the two outgoing ports

Master report of the port-state stage. Companions:
P0550_PHASE_KNOB_AUDIT.md, P0750_TRANSMISSION_ZERO_AUDIT.md,
MULTIPOLAR_PORT_STATE_MAP.md. Machine-readable: results/
{p0550_h_phase_unwrapped, p0550_h_complex_rt, p0550_h_multipoles,
p0550_smatrix_vs_h, p0750_smatrix_vs_lam, p0750_highres_rt,
p0750_t_argand, p0750_component_removal, p0750_order_tmin,
existing_candidate_tmin_ranking, same_composition_opposite_function_pairs,
composition_port_map, complex_convergence}.csv, port_state_summary.json;
figures ps_fig1…7. No topology optimization was launched (Phase Z not
triggered — the reflective state already exists in the data).

## VERDICT (Phase Y)

**VERDICT A — STRONG GO: SAME-COMPOSITION PORT-STATE PROGRAMMING**, with
its extent quantified honestly, plus the full transparent/reflective
duality across two compositions.

* Within the strictly defined mode-identity interval h = 215–262.5 nm
  (D_comp ≤ 0.10 from the h = 227.5 reference, px|ED ≥ 0.95,
  Qxz|EQ ≥ 0.70, f_ED+f_EQ ≥ 0.85, neighbor current-pattern overlap
  ≥ 0.9978, ≥ 0.9995 in the knob core), the SAME clean ED–EQ state is
  continuously reprogrammed from **R = 0.020 to R = 0.283 (14×)** and
  T = 0.980 → 0.717 purely by radiative phase (unwrapped, strictly
  monotonic). Best decoupling pair: h = 235 vs 260 — D_comp = 0.069,
  ΔR = 0.219.
* The FULL mirror state (T ≈ 0.01) is NOT reached inside the fixed
  composition — that is stated plainly. It exists in the same freeform
  platform via the composition-changing route: P0750's my/Qxz resonant
  state is a certified **T_min = 0.011 / R = 0.988 / A = 9×10⁻⁴ resonant
  mirror** whose co-polarized amplitude truly approaches the origin
  (|t_xx|² = 1.8×10⁻⁴), order-robust (T_min 0.007–0.012 at
  [9,9]/[11,11]/[13,13] tracking the shifting pole).
* Both port states are the SAME physics in complementary channels:
  transparent = background-assisted cancellation in the r-plane
  (P0550, r_bg + r_mult ≈ 0), reflective = background-assisted
  cancellation in the t-plane (P0750, t_bg + t_mult ≈ 0) — Figure 5.

## Q1 answered (P0550 phase knob)

The ±180° jump at h ≈ 221 nm is **pure phase wrapping**. Unwrapped,
Δφ_ED–EQ(bottom) is strictly monotonic over h = 200–300 nm: global slope
−0.85°/nm, knob-region (215–240) slope −1.34°/nm with rms deviation from
linearity of only 0.38°, slope range −0.28…−1.43°/nm — smooth, mildly
nonlinear, uninterrupted by any mode change (fractions, purities, and
the measured current-pattern overlap all continuous). The multipole
phase is an excellent proxy for the device phases: corr(Δφ, arg t) =
−0.977, corr(Δφ, arg r) = −0.995; d arg(t)/d Δφ ≈ −1.45. Over the
identity interval the ACTUAL device-phase coverage is **arg(t): 76°,
arg(r): 219°** (so: real but far from a 360° flat-optics library —
not claimed).

## Q2 answered (T≈0 in the same platform)

Yes — and not only in P0750. The existing-data scan
(existing_candidate_tmin_ranking.csv) finds FIVE candidates with
T ≤ 0.011 at their resonances, all of the my+Qxz odd-channel type (best:
P0550_H0350_seed029, T = 0.0034, R = 0.9965, A = 2×10⁻⁴ at 1343.5 nm,
1-nm sampling — flagged for future sub-nm confirmation; P0750 is the
certified one at 0.2-nm). The 1372.5-nm entries are scan-edge minima
(flagged); P0550_H0150_seed011's row is energy-violating (excluded).
Within the P0550 thickness family alone, T reaches only 0.365 (h = 300,
composition already drifting) — the clean ED–EQ composition does not
produce a mirror in this family.

## The unified two-port picture (Phase Q)

Sub-diffractive, quasi-lossless (k = 6×10⁻⁶ ⇒ A ≤ 10⁻³ everywhere
tested): |r|² + |t|² + (cross-pol) ≈ 1. S-matrix verified: reciprocity
|t12 − t21| ≤ 3.9×10⁻⁵ (P0550) / 3.4×10⁻⁴ (P0750); energy closes to
≤ 9×10⁻⁴ when ALL four specular channels (2 pol × 2 ports) are counted.
One caveat discovered: the freeform anisotropy converts ~1% of power to
the cross polarization at P0750's resonance (T_yx ≈ 0.0106 flat,
R_yx up to 0.06 resonant) — this cross-pol floor, not absorption and not
co-pol leakage, sets P0750's total-T minimum. P0550 stays ≤ 2%
cross-pol. **Multipole dominance alone determines neither state**: the
decisive quantity is the channel-normalized complex sum including the
background (integrity rules held throughout — see companions for the
removal tests).

## The 25 final questions (Phase X)

1. **Smooth after unwrapping?** Yes — strictly monotonic, no
   discontinuity; the jump was wrapping.
2. **d(Δφ)/dh?** −1.34°/nm in 215–240 (rms linear residual 0.38°);
   −0.85°/nm globally; range −0.28…−1.43.
3. **Mode-identity interval?** h = 215–262.5 nm (47.5 nm), contiguous,
   by D_comp ≤ 0.10 + purity gates + current-overlap ≥ 0.998.
4. **arg(t) coverage over it?** 76° (arg(r): 219°).
5. **Is Δφ_ED–EQ a useful predictor?** Yes, of both: corr −0.98 with
   arg(t), −0.99 with arg(r); and −0.93 with R (via distance from 180°).
6. **Transparent → reflective at fixed composition?** Partially:
   R 0.020 → 0.283 (14×), T 0.980 → 0.717 within the identity interval;
   the full mirror is NOT reached at fixed composition.
7. **Smallest T in the P0550 h-family?** 0.365 at h = 300 nm (λ0),
   with composition already degraded (f_ED = 0.26).
8. **Smallest trustworthy T across all existing candidates?**
   P0750_H0250_seed011: T = 0.0108 (0.2-nm sampling, energy-clean,
   order-robust). P0550_H0350_seed029 shows T = 0.0034 at 1-nm sampling
   — likely lower-T but not yet sub-nm-certified; flagged as the next
   candidate to qualify.
9. **Is P0750 genuinely a T≈0/R≈1 mirror?** Yes: T = 0.0108, R = 0.988,
   A = 8.7×10⁻⁴ — a resonant mirror, not absorptive (Phase R test),
   with ~1% polarization conversion.
10. **Accurately resolved T_min?** 0.01077 at 1333.98 nm (0.2-nm core,
    parabolic; FWHM ≈ 4 nm ⇒ 20 samples across the line); per-order:
    0.0072–0.0119 with the pole shifting ≈ −1 nm/order.
11. **At λ_Tmin:** R = 0.9884, T = 0.0108, A = 0.00087; f_ED = 0.068,
    f_MD = 0.564, f_EQ = 0.351, f_MQ = 0.017 (my|MD = 0.95,
    Qxz|EQ = 0.97).
12. **Does complex t approach the origin?** The co-polarized t_xx does:
    |t_xx| = 0.0136 (|t_xx|² = 1.8×10⁻⁴), down from |t_bg| = 0.98. The
    total-T floor is the cross-pol channel, not t_xx.
13. **What cancels t_bg?** In the ladder picture: t_ED pulls the
    background down (bg+ED: T 0.97 → 0.23) and the my/Qxz pair
    contributes its net residual plus the resonance (bg+ED+MD+EQ:
    0.163); the last stretch to 1.8×10⁻⁴ sits in higher-order terms
    (ladder truncation ≈ 0.16 at this strongly resonant point — the
    exact current integral reproduces t by construction). Attribution
    at the family level is unambiguous (next answer); at the
    single-term level it is ladder-limited and stated as such.
14. **Are m_y and Qe_xz essential?** Absolutely: remove MD → T_model
    = 5.5; remove EQ → 10.4 (each member alone wildly overshoots:
    bg+MD = 11.3, bg+EQ = 6.1; together they nearly self-cancel:
    bg+MD+EQ = 0.91). The pair is the resonant engine; ED sets the
    background-cancelling scale.
15. **Internal null = λ_Tmin?** No: internal my/Qxz null at 1333.6 nm,
    T_min at 1334.0 nm — offset +0.4 nm (resolved at 0.2 nm).
16. **What shifts it?** The direct transmission background (plus the ED
    term riding on it): T_min requires the TOTAL t to vanish, i.e. the
    net scattered field must equal −t_bg, not zero — exactly analogous
    to P0550's 221-vs-227.5 offset in the r-plane.
17. **Same composition, very different R/T?** Yes — answer 6 and
    Figure 6.
18. **Best numerical example?** h = 235 vs h = 260: D_comp = 0.069,
    ΔR = ΔT = 0.219 (R 0.040 vs 0.259) — same clean family, 6.4× R
    change.
19. **Does composition predict function well?** No: corr(R, balance) =
    0.17 across the family; and near-identical compositions span
    R 0.02–0.28. Composition sets what CAN interfere; it does not set
    the port state.
20. **Is channel-normalized phase the stronger predictor?** Yes:
    corr(R, |Δφ−180°|) = −0.93 (Figure 7B).
21. **Transparent state = background-assisted reflection cancellation?**
    Yes — confirmed with one consistent normalization (Figure 2:
    h = 221 leaves the background; h = 227.2 opposes it).
22. **Reflective state label?** Background-assisted transmission
    cancellation ≡ a resonant mirror: the my/Qxz-driven scattered field
    (with the ED term) cancels the transmitted background; A ≈ 10⁻³
    excludes absorption; ~1% goes to cross-pol.
23. **Stable under Fourier-order refinement?** Yes: P0550 knob values
    converged (AR audit); P0750 T_min 0.007–0.012 across [9,9]–[13,13]
    ([15,15] running-final at commit time; pole shift ≈ 1 nm/step is the
    known systematic).
24. **Stable in the complex plane?** complex_convergence.csv: at
    (h\*, λ0) Re/Im of r and t drift smoothly and by only a few 10⁻²
    across orders (consistent with the ±3% power statements); no hidden
    phase instability found.
25. **A defensible transparent/reflective duality?** Yes: one freeform
    multipolar platform, one convention, two complementary
    background-assisted cancellations — r-plane (transparent, clean
    ED–EQ, broadband) and t-plane (reflective, my/Qxz resonant,
    narrow) — with the composition-independent phase knob demonstrated
    quantitatively inside the first and the composition-changing route
    required for the second. That two-sentence statement is the
    publishable core.

## Integrity compliance

Background terms carried in every cancellation statement; internal ≠
external cancellation maintained (answers 15–16); T never inferred from
1−R (A and cross-pol computed everywhere); no zero claimed without the
complex amplitude approaching the origin (only t_xx qualifies); no raw
multipole phases compared (channel normalization throughout); 0.2-nm
sampling for the narrow T-zero; high-Q never used as proof of T≈0;
composition percentages never used to infer function.
