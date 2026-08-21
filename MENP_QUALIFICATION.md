# MENP multipolar qualification of three frozen ED/MD co-excitation candidates

Scope fixed by instruction: **exactly three** frozen binary candidates —
`P0870_H0100_seed029`, `P0830_H0100_seed011`, `P0900_H0100_seed011`
(run directories `results_ed_md_coexcitation/refine/<id>_lam1332p5_refined/`,
geometries = `rho_binary.npy`, resolved from the filesystem). No new topology
optimization was run; no other candidate was added.

## 1. Method qualification (before any physics claims)

* **Source audit** (`menp_port.py` docstring): MENP implements the exact
  Alaee-2018 Cartesian multipoles from J = −iωε₀(n²−1)E in SI units,
  exp(−iωt) (identical time convention to TORCWA). Audit findings:
  a `dQmxz` symmetrization bug present in all three MATLAB ME files
  (28.7% CQm shift on asymmetric test currents), an r→0 kernel NaN hazard,
  and vacuum-background/|E₀|=1 cross-section constants.
* **Validation** (`menp_validation.py`, all PASS):
  1. Python port ≡ **original MENP under GNU Octave** on the TRUE demo
     datasets (fetched from the MENP repository): rel. difference
     10⁻¹⁶–10⁻¹⁵ for every output of `exactME`, `toroidalME`,
     `toroidalME_phase`.
  2. **Reference-demo reproduction**: shipped `demo_exact.csv` /
     `demo_toroidal.csv` reproduced to ≤3×10⁻⁵ (their 5-digit rounding).
  3. **Analytic ground truth**: prescribed-p Gaussian current blob and
     prescribed-m current loop recovered to 0.05–0.3% with exact
     cross-section prefactors.
  4. Bug demonstrations faithful-vs-corrected. Production mode =
     `corrected` (symmetrized Qm, regularized kernels).
* **Fields**: dense 3D complex E on 96×96×21 (unit cell × layer, endpoints
  included, trapz-consistent), reconstructed by torcwa `field_xy` on the
  frozen binary masks; E_inc = 1 matches MENP's |E₀| = 1; positions → m,
  f → Hz. Proxy check: the extracted fields reproduce the campaign's
  S_ED/S_MD verification values exactly.
* **Background-medium assumption, investigated explicitly**: MENP's C's
  assume an isolated scatterer in vacuum. Here the moments are
  **per-unit-cell moments** of the induced current computed from the true
  substrate-aware periodic fields; C's are formal vacuum radiation weights
  used for *relative* comparison. The substrate stage (below) shows the
  silica half-space is *constitutive*: removing it collapses the resonances
  (S_ED/S_MD peaks 50–74 → 1–4, |mz| ÷4–5, peaks shift ~10 nm), consistent
  with guided-mode-type resonances tied to the near-cutoff substrate
  diffraction order (n_sub·P/λ = 0.95–0.99 for these periods).

## 2. Qualification data (cell-center origin, [13,13], 96×96×21 unless noted)

### Numerical robustness

| check | P0870_H0100_s29 | P0830_H0100_s11 | P0900_H0100_s11 |
|---|---|---|---|
| grid 64³→128×128×41: Δ\|px\|, Δ\|mz\| | 3.2%, 0.2% | 1.1%, 1.1% | 0.7%, 0.3% |
| order [13]→[17]: Δ\|px\|, Δ\|mz\| | −10%, −1.7% (CT/Cp 8.5→10.6) | **+2.4%, +1.3%** | −25%, −4.3% (Cp% 7.5→4.7) |
| order [9]→[17] family drift | ≤0.6 pt (stable) | ≤3 pt (stable) | Cp 26.9→4.7% (**fragile**) |
| origin z±H/4, x±P/8 | ≤3.3% | ≤2.9% | ≤3.7% |
| origin y+P/8 (lattice-direction ambiguity) | \|px\| +194% | \|mz\| −47% | ≤6.2% |
| rel-phase stability across orders | 11° drift | **0.4°/order (~1°)** | 37° swing |

### Multipolar content at λ₀ = 1332.5 nm (binary, exact ME)

| quantity | P0870_H0100_s29 | P0830_H0100_s11 | P0900_H0100_s11 |
|---|---|---|---|
| px within ED family | 0.999 | **1.000** | 0.993 |
| mz within MD family | 1.000 | **1.000** | 1.000 |
| Cp : Cm : CQe : CQm (%) | 1.9 : 57.0 : 38.5 : 2.6 | **47.1 : 27.0 : 16.1 : 9.8** | 7.5 : 49.4 : 33.0 : 10.1 |
| ED+MD share | 58.9% | **74.1%** | 56.9% |
| CT/Cp (toroidal vs net p) | 8.5 (anapole-like) | **1.0** | 6.5 |
| balance C_px/C_mz | 0.034 | **0.574** | 0.151 |
| arg(px)−arg(mz) | −48° (unstable ±65° over line) | **−84° (plateau, ±25°)** | 149° (swinging) |

### Wavelength-resolved peaks (fine 0.5 nm, [13,13])

| quantity | P0870 | P0830 | P0900 |
|---|---|---|---|
| peak C_px | 1327.4 nm (FWHM 11.0) | **1334.0 nm (FWHM 14.0)** | 1328.6 nm (FWHM 10.5) |
| peak C_mz | 1332.0 nm (FWHM 7.5) | **1334.7 nm (FWHM 13.5)** | 1332.0 nm (FWHM 6.5) |
| **px–mz peak splitting** | 4.58 nm | **0.71 nm** | 3.37 nm |
| proxy (S_ED–S_MD) splitting | 0.02 nm | 0.03 nm | 0.01 nm |
| corr(C_px, S_ED) / corr(CQe, S_ED) | −0.05 / **0.99** | 0.45 / 0.30 | 0.11 / 0.69 |
| corr(C_mz, S_MD) | 1.00 | 0.22* | 0.69 |

*P0830's C_mz is much broader than the proxy resonance, lowering the shape
correlation while the peaks coincide; its proxies peak 1331.5, its dipoles
1334–1335 (≈3 nm offset between near-field maximum and radiative-moment
maximum).

### Parametric realistic-NIR material check (n_Si = 3.30 / 3.45; no
trustworthy tabulated local NIR a-Si:H dataset exists — documented
parametric scan, [11,11])

| quantity | P0870 | P0830 | P0900 |
|---|---|---|---|
| resonance relocates to | 1406–1451 nm | 1422–1468 nm | 1388–1426 nm |
| px–mz splitting | 3.0 nm / 3.0 nm | **1.0 nm / 0.0 nm** | 3.0 nm / 3.0 nm |
| families at joint peak | Cp 4–5%, Cm 56% | **Cp 52–54%, Cm 21–22%** | Cp 29–32%, Cm 34–36% |
| balance C_px/C_mz | 0.07–0.09 | 0.39–0.43 | **0.81–0.95** |

## 3. The twelve questions, answered per candidate

**P0870_H0100_seed029**
1. *ED family dominated by px?* Yes formally (99.9% of |p|²) — but the net
   ED weight itself is tiny (Cp ≈ 2%).
2. *MD family dominated by mz?* Yes, 100.0%.
3. *px and mz simultaneously strong?* **No.** mz is very strong at
   1332 nm, but px is anapole-suppressed exactly there (CT/Cp ≈ 8.5, |px|
   dips at the resonance); the S_ED proxy tracks CQe (corr 0.99), not px
   (corr −0.05).
4. *Peak splitting:* 4.6 nm (px peak is a blue-side shoulder, not a
   resonance of its own).
5. *ED+MD vs EQ/MQ/TD:* MD dominant; **EQ contamination large (38.5%)**;
   TD exceeds net p.
6. *Balance:* 0.034 — heavily mz-sided.
7. *Order sensitivity:* fractions/mz stable; suppressed-px conclusion
   strengthens with order.
8. *Grid:* ≤3.2% — insensitive.
9. *Origin:* z/x insensitive; y+P/8 inflates |px| ×3 (quadrupole mixing)
   without changing the conclusion.
10. *Realistic material:* preserved (still mz-dominated, px weak).
11. *Do px/mz track proxies?* mz–S_MD perfectly; px–S_ED **not at all**.
12. *One vs two resonances:* one MD/EQ-mixed resonance with toroidal-
    suppressed electric-dipole channel.

**P0830_H0100_seed011**
1. *px dominates ED?* **Yes — 100.0%**, and ED is the largest family
   (Cp 47%).
2. *mz dominates MD?* **Yes — 100.0%** (Cm 27%).
3. *Simultaneously strong?* **Yes.** Both radiative dipoles rise together
   into a common resonance; ED+MD = 74% of the family sum at λ₀.
4. *Splitting:* **0.71 nm** ([13,13] fine grid; → 0.0–1.0 nm with
   realistic material).
5. *ED+MD vs EQ/MQ/TD:* dominant (EQ 16%, MQ 10%, CT/Cp ≈ 1.0 — toroidal
   present but net p survives; no anapole).
6. *Balance:* 0.57 at λ₀ (C_mz slightly stronger just above λ₀).
7. *Order sensitivity:* **best of the three** — |px| +2.4% and phase
   drift ~1° from [13,13]→[17,17].
8. *Grid:* ≤1.1%.
9. *Origin:* z/x insensitive; the y+P/8 lattice ambiguity reduces |mz|
   −47% (largest caveat for this candidate) but both dipoles remain
   leading at every tested origin.
10. *Realistic material:* **improves** — splitting → 0.0 nm, Cp → 54%.
11. *Track proxies?* Peaks coincide with proxies within ~3 nm; the
    radiative-dipole line is broader than the near-field line (shape
    correlations moderate).
12. *One vs two:* one mixed px/mz resonance — single co-peaked line,
    common width (14.0 vs 13.5 nm), and a flat relative-phase plateau
    (−84°) across the resonance.

**P0900_H0100_seed011**
1. *px dominates ED?* Yes (99.3%), but Cp is small at λ₀ (7.5%).
2. *mz dominates MD?* Yes (100.0%).
3. *Simultaneously strong at the same λ?* **Partially / not at λ₀.**
   px peaks at 1328.6 nm (where Cp ≈ 75%!), mz at 1332.0 nm; at λ₀ px is
   sliding into an anapole-like dip (CT/Cp 6.5).
4. *Splitting:* 3.37 nm — two distinct features.
5. *ED+MD vs EQ/MQ/TD:* MD-led with EQ 33% at λ₀; ED strong only at its
   own blue feature.
6. *Balance at λ₀:* 0.151 (but 0.81–0.95 at the joint peak under realistic
   material — its most intriguing property).
7. *Order sensitivity:* **fragile at λ₀** (Cp% 27→5 across orders; phase
   swings 37°) — λ₀ sits on a steep interference slope.
8. *Grid:* ≤0.7% — insensitive.
9. *Origin:* mildest y-sensitivity of the three (≤6%).
10. *Realistic material:* two features persist, still ~3 nm apart, but
    with excellent px/mz amplitude balance.
11. *Track proxies?* Partially (corr 0.11/0.69) — proxies merge the two
    features into one apparent line.
12. *One vs two:* **two nearly degenerate, coupled resonances** — an
    ED(px)-dominated feature ~1328.6 nm and an MD(mz)/EQ feature
    ~1332.0 nm, with interference (anapole dip) between them.

## 4. Final ranking (only these three candidates)

| rank | candidate | multipolar correctness | numerical convergence | simultaneous px/mz strength | px/mz balance | same-λ peak coincidence | overall physical credibility |
|---|---|---|---|---|---|---|---|
| **1** | **P0830_H0100_seed011** | **best** (px=100% of ED, mz=100% of MD, ED+MD=74%, CT/Cp=1) | **best** (Δ2.4% @[17,17], phase ±1°, grid ≤1.1%) | **yes — both dipoles strong at λ₀** | 0.57 (0.39–0.43 realistic) | **0.7 nm → 0.0 nm** | **high: one mixed px/mz resonance; survives every check; caveat: y-origin lattice ambiguity (−47% mz at +P/8)** |
| 2 | P0900_H0100_seed011 | good within families; Cp small at λ₀, EQ 33% | grid/origin fine but order-fragile at λ₀ | split: strong px and strong mz at λ's 3.4 nm apart | 0.15 at λ₀; **0.81–0.95 at joint peak (realistic n)** | 3.4 nm (persists ~3 nm) | moderate: genuine two-resonance px/mz system; a small parameter nudge could degenerate them — a lead, not a confirmed same-λ design |
| 3 | P0870_H0100_seed029 | mz superb, but "ED" content is EQ+toroidal; net px anapole-suppressed | fractions/mz robust; px magnitude drifts | **no** — mz only | 0.03 | 4.6 nm | high as an *mz + anapole-Ex* structure — but it does not realize the px∧mz target; S_ED proxy was tracking CQe, not px |

**Bottom line.** After exact-multipole qualification, only
**P0830_H0100_seed011** genuinely realizes the boxed target
"λ ≈ 1332.5 nm: p_x strong AND m_z strong" as net radiative Cartesian
dipoles at the same wavelength, and it does so with the best numerical
convergence of the three; the conclusion survives solver-order, spatial-
integration, origin (with the documented lattice-direction caveat),
substrate-treatment, and parametric realistic-material checks.
P0900_H0100_seed011 is a genuine but *split* two-resonance px/mz system;
P0870_H0100_seed029 is an mz-dominated resonance whose electric near-field
is toroidal/quadrupolar — a false positive of the |Ex|² proxy that only the
decomposition could expose.

Global caveats: per-unit-cell moments with vacuum-background radiation
weights (substrate shown to be constitutive); y-origin lattice ambiguity
inherent to periodic decomposition; a-Si:H model clamped at its 999-nm
endpoint for the λ₀ runs (parametric NIR scan provided); MENP `corrected`
mode used throughout (faithful-mode differences documented and validated).
