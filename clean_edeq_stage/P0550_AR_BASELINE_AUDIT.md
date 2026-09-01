# P0550_AR_BASELINE_AUDIT — is the clean ED–EQ structure actually antireflective?

Focused audit of the antireflection claim for the P0550 thickness family
against real baselines, with exported geometry. No optimization was run.
Data: results/ar_rt_all.csv (2582 solves: bare / uniform film /
fill-matched disk / P0550 at 4 thicknesses × orders [9,9]–[15,15], both
incidence directions), ar_keypoints.csv, ar_grid_matrix.csv,
ar_o17_spot.csv, ar_summary.json, p0550_ar_comparison.csv; geometry:
results/geometry/* (exact binary + SHA256), figures/p0550_geometry_*,
figures/ar_*.

## Bottom line

* The **frozen h = 250 nm reference is NOT an antireflection structure**
  (R = 0.160 at λ0 vs bare-silica 0.035): it is the directional-scattering
  /composition reference, and must not be marketed as AR.
* The **thickness-tuned h\* = 227.2 nm variant IS genuinely
  antireflective**: R(λ0) = 0.020–0.022 (order-converged) vs 0.035 bare —
  an absolute gain of 1.3–1.5 points (≈40% suppression) — and stays below
  bare silica over a contiguous **94–98 nm** band (order-robust), with the
  clean-ED–EQ ∧ below-bare intersection at **1296–1358 nm (62 nm)**.
* Against the nontrivial baselines the freeform is dramatically better:
  the laterally uniform a-Si film of the same thickness is a mirror
  (R = 0.45–0.61 at λ0; median in-band suppression ratio film→freeform
  ≈ 16×) and the fill-matched centered disk is a violently resonant
  scatterer (R swings 0.002–0.998 in-band; R = 0.23 at λ0).
* The deepest minimum is a **deep reflection minimum, not a zero**:
  R_min at 1280–1285 nm rises 0.0012 → 0.0042 → 0.0083 at
  [9,9] → [15,15] → [17,17]-spot; report as R_min ≈ 4–8×10⁻³
  (4–8× below bare). The λ0 value and all bandwidths ARE
  order-converged; the extreme depth is not.
* Mechanism (Argand-verified): **background-assisted multipolar
  cancellation**. At the naive ED–EQ 180° crossing (h ≈ 221 nm) the giant
  ED and EQ vectors (|each| ≈ 1.6 in r-units) cancel internally and the
  total reflection collapses to exactly the bare-substrate background
  (|r| = 0.182 ≈ |r_bg| = 0.187 → R ≈ R_bare). The R minimum at
  h ≈ 227.5 nm instead *detunes* the internal cancellation so that the
  surviving scattered residual ≈ −r_bg. The true condition is
  r_bg + [A_ED + A_EQ + A_MD + …]_bot ≈ 0.

## 1–3. Geometry (exported, verified)

figures/p0550_geometry_{topview,contour,3d,xz,yz,thickness_family,
current_overlay}.{png,pdf}. The meta-atom is a **single connected
freeform a-Si mesa** occupying 61.77% of the 550-nm cell, with smooth
curved interior cutouts; minimum material linewidth ≈ 245 nm (very
fab-tolerant), but the narrowest air slot is ≈ 15 nm (3 px) — the one
fab-critical feature. Hard-binary verification: min = 0.0, max = 1.0,
binary fraction = 1.0 exactly (110×110 array; 5.0 nm/px; threshold used
only for contour tracing). SHA256 of source and exported copy in
results/geometry/p0550_geometry_verification.json; the plotted array is
bit-identical to the array behind every Stage-B/AR spectrum.

## 4. Bare-substrate baseline (mandatory check)

TORCWA bare stack (empty patterned layer) vs analytic Fresnel for
silica (n = 1.46) → air: R_bare = 0.034965 both ways;
**max |R_TORCWA − R_Fresnel| = 6.9×10⁻⁷** over the full band; forward =
backward to 2×10⁻⁸. Port normalization and reference planes identical to
all other calculations.

## 5–7. Baselines and comparison (figures/ar_comparison_R + ar_single_answer)

At λ0 = 1332.5 nm:

| system | R at λ0 |
|---|---|
| P0550 freeform, h\* = 227.2 | **0.020** ([9,9]) / 0.022 ([17,17]) |
| P0550 freeform, h = 225 | 0.022 |
| P0550 freeform, h = 235 | 0.040–0.045 |
| P0550 freeform, h = 250 (frozen) | 0.160–0.167 |
| bare silica | 0.0350 |
| uniform a-Si film (227.2 nm) | 0.474 |
| fill-matched disk (227.2 nm) | 0.229 |

AR_gain_bare(λ0) at h\*: +0.0149 absolute; suppression_ratio_bare = 0.57
([9,9]) / 0.63 ([17,17]). suppression_ratio_film = 0.042 at λ0 (24×),
band-median 0.063 (16×).

## 8–9. The honest AR verdict and bandwidths (h\* = 227.2, [9,9] grid 1 nm)

* A. R ≤ 0.05: **1272–1388 nm (116 nm)** — order-stable (116 nm at every
  order to [15,15]).
* B. R < R_bare: **1273–1367 nm (94 nm)** — order-stable (94–98 nm).
* C. R ≤ 0.5·R_bare: **1276–1322 nm (46 nm)**.
* Clean balanced ED–EQ (interpolated band edges, ±3 nm): 1296–1358 nm.
* **Clean ∧ below-bare: 1296–1358 nm (62 nm)**;
  **clean ∧ half-bare: 1296–1322 nm (26 nm)**.
* h = 235: below-bare band shrinks 66 → 36 nm with order — quote 36 nm.
* h = 250: below-bare only 1414–1419 nm (5 nm) — not AR.

The earlier Stage-B "R ≤ 0.05 over 95–130 nm" statement remains true but
is NOT an antireflection claim; the AR-relevant numbers are the ones
above.

## 10. Both incidence directions

max |R_forward − R_backward| = 1.6×10⁻³ across the whole band at h\*
(6×10⁻⁶ at λ0, [17,17]); T identical to 2×10⁻⁸. As expected for a
reciprocal, quasi-lossless (k = 6×10⁻⁶) two-port with only specular
channels open — demonstrated, not assumed. The AR function is
direction-independent.

## 11–12. Mechanism (figures/ar_argand_mechanism, ar_single_answer panel D)

Complex-r construction r = r_bg + r_ED + r_EQ + r_MD + r_2nd + residual
(per-row exact port coupling; "2nd" = complete 2nd-order integral
containing Qm_yz + octupole corrections):

| point | |ED| | |EQ| | |MD| | |bg| | full |r| | reading |
|---|---|---|---|---|---|---|
| h = 221, λ0 | 1.56 | 1.62 | 0.18 | 0.187 | 0.182 | perfect internal ED–EQ cancellation ⇒ only background remains ⇒ R ≈ R_bare |
| h\* = 227.2, λ0 | 0.89 | 0.98 | 0.13 | 0.187 | 0.141 | detuned internal cancellation; residual partially opposes bg ⇒ R = 0.020 < R_bare |
| h\* = 227.2, 1285 | 1.20 | 1.26 | 0.34 | 0.187 | 0.031 | scattered residual ≈ −r_bg ⇒ deep minimum |

This is why h ≈ 227.5 beats h ≈ 221: the naive 180° crossing removes the
scattered wave but cannot remove the substrate background; the minimum
of |r_total| requires the scattered residual to cancel the background,
which happens ~6 nm of thickness (≈9° of ED–EQ phase) away. Label:
**background-assisted ED–EQ(+m_y) cancellation**, with generalized-
Kerker directionality as the underlying parity structure and Huygens-like
transmission (T = 0.98) as the consequence. R_min ≈ 0.004–0.008 is a
reflection minimum — "R-zero" is NOT claimed.

## 13–15. Numerical qualification

* Orders (R/T-only spectra): h\* R(λ0): 0.0201/0.0208/0.0215/0.0217/
  0.0221 at [9,9]/[11,11]/[13,13]/[15,15]/[17,17] — converged ≈ 0.022.
  Bandwidths A and B: stable (116; 94–98 nm). R_min depth: NOT converged
  (0.0012→0.0083); quote 4–8×10⁻³. h = 235 degrades with order (above);
  h = 250 stable and non-AR.
* Grid (at h\*, 1285 nm): R and T bit-identical across moment grids
  (solver-level quantities); fractions drift ≤ 2.8 pp from 32×32×5 to
  96×96×21 (f_ED 0.44–0.47, f_EQ 0.41–0.44, f_MD 0.12); px|ED ≥ 0.998;
  Qxz|EQ = 0.75; channel phase arg(px/Qxz) = −104.4° stable to 0.1°.
  Note: the deep-minimum wavelength sits just OUTSIDE the strict clean
  criterion (Qxz|EQ < 0.8 there); the strict-clean AR band is the 62-nm
  intersection above.
* Energy: at the claimed points |T+R−1| = 3×10⁻⁵ (λ0) and 2.3×10⁻⁴
  (R_min) — far inside the 5×10⁻³ gate. The historically flagged
  1267–1269 nm points remain excluded (outside all claimed bands).

## 17. The fifteen answers

1. **What does it look like?** A single connected freeform a-Si mesa
   with curved interior cutouts (figures/p0550_geometry_*): roughly a
   ring-like block filling most of the cell, pierced by two large
   rounded openings and thin slots.
2. **Exact parameters:** P = 550 nm; h = 250 nm frozen / 227.2 nm AR
   working point; a-Si (Franta 2013, n = 3.652 + 6×10⁻⁶i at λ0) on
   silica (n = 1.46); fill = 0.6177; min material linewidth ≈ 245 nm;
   min air slot ≈ 15 nm.
3. **R at λ0:** P0550(h\*) 0.020–0.022; bare 0.0350; uniform film 0.474;
   disk 0.229. (Frozen h = 250: 0.160.)
4. **Lower than bare?** YES at h\* (and h = 225); NO at h = 250; marginal
   at h = 235 ([15,15]: 0.045 > 0.035 at λ0 — no).
5. **By how much?** Absolute −0.013 to −0.015; relative −37…−43% at λ0;
   up to ~4–8× below bare at the 1280–1285 minimum.
6. **Band with R < R_bare:** 1273–1367 nm, 94 nm contiguous
   (order-stable 94–98 nm) at h\*.
7. **Band with R ≤ 0.5 R_bare:** 1276–1322 nm (46 nm).
8. **Low-R ∧ clean ED–EQ:** below-bare ∧ clean = 1296–1358 nm (62 nm);
   half-bare ∧ clean = 26 nm.
9. **Air-side incidence?** Identical to 1.6×10⁻³ worst-case (reciprocal
   quasi-lossless two-port) — yes, both directions.
10. **How much is specifically ED and EQ?** They are the two dominant
    vectors (|A| ≈ 0.9–1.6 each in r-units vs |bg| = 0.19, |MD| ≈
    0.13–0.34): removing either from the reconstruction raises the
    band-mean error to ≥ 0.85 per unit incident (Stage-B removal test)
    and destroys the cancellation. The suppression is impossible without
    both.
11. **How important is m_y?** Secondary but material: |MD| ≈ 0.13–0.34
    in r-units; including it cuts the reconstruction error ~2.6×; at
    the deep minimum it carries part of the background-opposing
    residual.
12. **Is the background essential?** YES — this is the audit's sharpest
    mechanistic finding: with the scattered wave internally cancelled
    (h = 221) the reflection returns to the bare value; the minimum
    exists only because the residual scattered field cancels r_bg.
13. **Correct label:** a combination — *background-assisted multipolar
    (ED+EQ+m_y) cancellation* built on *generalized-Kerker parity
    directionality*, functioning as *Huygens-like transmission*
    (T = 0.98); "antireflection" is justified only for the tuned
    h ≈ 225–228 family within the 94-nm below-bare band.
14. **Survives higher order and denser grids?** Yes for everything
    claimed (λ0 values, bandwidths, phases, fractions); the only
    non-converged quantity is the extreme depth of the minimum, quoted
    accordingly.
15. **Better than the simple baseline?** Yes, qualitatively and
    quantitatively: the fill-matched disk reaches R = 0.9983 (a mirror)
    inside the band and 0.229 at λ0 — 10× worse at λ0 — and offers no
    broadband below-bare interval; the uniform film is 16–24× worse.
    The freeform's flat, broadband, below-bare response is not
    reproduced by either trivial geometry.

## Corrections applied to prior claims

* Stage-B "R ≤ 0.05 over 95–130 nm with η ≥ 0.91" stands as a
  directional/transmission statement but is now explicitly separated
  from the AR claim (which requires the bare-substrate comparison and
  holds over 94 nm at h\* only).
* Any impression that the frozen h = 250 structure suppresses reflection
  is corrected: it is 4.6× MORE reflective than bare silica at λ0.
* The Stage-B "R = 0.0003–0.0012 near-zeros" are downgraded to
  order-unconverged deep minima (≈ 4–8×10⁻³).
