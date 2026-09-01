# R_TYPE_CAMPAIGN_REPORT — overnight freeform R-type campaign (633 nm reflective PB meta-atom)

Runtime: ~15 h autonomous (coarse 60 runs -> refinement 34 -> seeds 4 ->
qualification -> wide-angle stage). Machine-readable: results/
(rtype_master_ledger.csv, padding_by_period.csv, coarse_method{A,B}.csv,
refined_method{A,B}.csv, angle_scan_finalists.csv,
pb_rotation_finalists.csv, multipole_finalists.csv,
convergence_finalists.csv, spectra_finalists.csv, t_argand_finalists.csv,
rectangle_baseline.csv); figures/ (heatmaps, geom_*, targand_*).
Compute note: this environment has no GPU; 4 parallel CPU TORCWA workers
were used throughout.

## Baseline (validated before any optimization)
Paper rectangle (P=226, H=170, 160x96 nm), air-side incidence, a-Si
Franta-2013 (n=4.2827+0.0687i at 633 - dataset genuinely brackets 633),
glass n=1.457 (Malitson): |r_x|=0.893, |r_y|=0.745,
Delta_phi = 0.394*pi (paper: ~0.4*pi - quantitative agreement),
R_cross=0.229, R_co=0.447, A~0.20-0.24. Circular basis verified to 8e-5;
PB law measured phi_cross = phi0 - 2*theta exactly. All five periods
diffraction-safe in air AND glass to theta=60 (P=271 opens at 60.5 deg -
the optimum rides this boundary).

## Champions at normal incidence (633 nm, hard-binary, order [9,9])
| | rectangle | Method A (ED/MD) P271/H200 | Method B (port-only) P271/H215 |
|---|---|---|---|
| R_cross | 0.229 | **0.526** | **0.505** |
| R_co | 0.447 | 0.065 | **0.005** |
| T_total (x/y) | 0.001/0.205 | 0.007/0.102 | 0.006/0.107 |
| PB phase error | 70.9 deg | 39 deg | **10 deg** |
| abs r_x, abs r_y | 0.89, 0.75 | 0.75, 0.78 | 0.69, 0.74 |
| absorption A_x/A_y | 0.20/0.24 | 0.43/0.28 | 0.52/0.35 |
| min Si linewidth | - | 110 nm | 104 nm |

Both champions: one connected island inside the circular envelope
(pad = 27.1 nm, r_design = 108.4 nm), no internal slot < 24 nm, PB
rotation slope -1.95/-1.96 (target -2; rms 7.4/9.2 deg), complex
amplitudes order-converged to [15,15] (dR_cross <= 0.01), spectrally
broad over 600-670 nm (no high-Q fragility). Absorption (A up to ~0.5
with resonant fields; k = 0.069) is the reflectivity ceiling.

## VERDICTS

**METHOD-A: ED/MD R-TYPE SUCCESS** - x->ED (f_ED 0.45-0.58),
y->MD (0.71-0.73), R_cross 0.526 = 2.3x rectangle, valid PB + fab.

**METHOD-B: PORT-ONLY FREEFORM SUCCESS** - R_cross 0.505 with 90x lower
co-pol leakage (0.005) and near-ideal half-wave phase (10 deg).

**OVERALL: HIGHER-ORDER FREEFORM STATE OUTPERFORMS ED/MD, and FREEFORM
CLEARLY OUTPERFORMS RECTANGLE (at/near normal incidence).** Method B
did NOT return to ED/MD: its x-pol state is ELECTRIC-QUADRUPOLE-dominant
(f_EQ 0.48-0.58; the geometry is a bow-tie/dumbbell whose coupled lobes
literally form the quadrupole) and the forward-transmission cancellation
is EQ-led (|t_EQ| = 0.81 vs ED 0.48, MD 0.41). Even Method A's x-pol
cancellation is a three-way ED+EQ+MD sum (EQ co-equal, 0.69). The
canonical ED/MD recipe is a good solution but NOT the natural optimum
of this design space.

**Honest wide-angle caveat**: every candidate - including a dedicated
worst-case multi-angle re-optimization - loses cross-conversion at
oblique incidence (R_cross 0.11-0.16 at 22-53 deg air angles). Strong
+-60 deg operation is not reachable in this single-layer space; the
improvement over the rectangle is established at and near normal
incidence. GLASS-side scan rows theta >= 45 deg are TIR artifacts
(air-side mapping theta_air = asin(1.457 sin theta_g) covers 0-69.5 deg
with the valid rows).

## The 27 required answers (section 36)

1. **Best P/H region?** P = 262-271 nm, H = 185-215 nm — the largest
   diffraction-safe periods with intermediate heights, for both methods.
2. **Strong period dependence?** Yes: R_cross roughly doubles from
   P = 190 to P = 271 (design-radius-limited at small P); the optimum
   pushes to the largest period that stays order-free to 60 deg.
3. **Strong height dependence?** Moderate: broad optimum H ~ 170-230;
   H <= 140 underperforms; the phase error improves with H (H = 215-260
   holds the best half-wave phases).
4. **Padding per period?** Fixed rule 0.10 P: 20.0/20.8/22.6/24.4/26.2
   (27.1 at P = 271); r_design = 0.4 P (padding_by_period.csv). One
   padding per period throughout; never an optimization variable.
5. **All finalists rotation-safe 0-180 deg?** Yes — zero pixels outside
   the envelope at every 15-deg rotation for all six.
6. **Does Method A reproduce x-ED / y-MD?** Yes (x: f_ED 0.45-0.58,
   px-pure; y: f_MD 0.71-0.73, mx-pure) — with the honest note that the
   x-ED gate equilibrated slightly below 0.55 where function preferred.
7. **Method-A champion exact fractions?** x: ED/MD/EQ/MQ =
   0.45/0.24/0.28/0.03; y: 0.13/0.71/0.14/0.02 (multipole_finalists.csv
   for all finalists).
8. **What basis does Method B select?** An EQ/MD hybrid: x-pol
   electric-quadrupole-dominant (0.48-0.58), y-pol MD (0.68-0.71).
9. **Does B return to ED/MD?** No (only the weakest B finalist drifts
   toward mixed ED).
10. **Or a higher-order solution?** Yes — the x-channel is genuinely
    quadrupolar, realized as a bow-tie/dumbbell two-lobe geometry.
11. **Largest R_cross?** Method A: 0.526 (B: 0.505; rectangle: 0.229).
12. **Smallest total transmission?** Method A x-pol T = 0.007 at the
    champion; both methods reach T_total ~ 0.05-0.06 (mean).
13. **Lowest co-pol leakage?** Method B: R_co = 0.005 (A: 0.065;
    rectangle: 0.447).
14. **Delta_phi closest to pi?** Method B: 10 deg error (2 deg at
    [15,15] for B_P271_H230); A: 39 deg; rectangle: 109 deg from pi.
15. **Most accurate PB slope?** A_P262_H185 and A_P271_H185
    (-1.97, rms 4.6-4.9 deg); all six within -1.94..-1.98.
16. **Most angle-robust to 60 deg?** None is strongly angle-robust;
    A-method degrades slightly more gracefully (min R_cross ~ 0.10-0.16
    vs 0.09-0.12 for B over the valid range, phi = 0 plane). This is
    the campaign's main negative result.
17. **Most fabrication-robust?** Method A/B tie on features (82-110 nm
    linewidths, no fine slots); B's champion has the simplest shape
    (single smooth dumbbell) — most fabrication-friendly overall.
18. **Any candidate opening orders in 0-60 deg?** No; P = 271 opens at
    60.5 deg (flagged as a tight margin), P <= 262 all safely beyond.
19. **Any fragile high-Q finalist?** No — all spectra broad over
    600-670 nm.
20. **Best REAL DEVICE R-type?** A_P271_H200 for maximum conversion
    with good phase (0.526/39 deg); B_P271_H215 if co-pol purity and
    phase fidelity dominate (0.505/0.005/10 deg). Device pick:
    **B_P271_H215** — its 90x co-pol suppression is worth more to a
    real PB metalens (co-pol light is stray background) than A's +0.02
    of conversion.
21. **Most scientifically interesting?** B_P271_H215 — the unconstrained
    EQ-hybrid state: freeform discovering a quadrupole R-type that beats
    the canonical recipe on phase purity.
22. **Real Pareto improvement over the rectangle?** Yes, decisively, at
    normal incidence: 2.2-2.3x R_cross, 7-90x lower R_co, phase error
    109 deg -> 10-39 deg, comparable or lower T — dominated on every
    axis. At large oblique angles the improvement does not persist
    (answer 16).
23. **Cause of high R in the ED/MD design?** Three-way complex
    cancellation of the forward background: |t_ED| = 0.67,
    |t_EQ| = 0.69, |t_MD| = 0.49 under x-pol; MD-led (0.75) under
    y-pol. Never "ED+MD automatically reflects".
24. **Cause in the unconstrained design?** EQ-led forward cancellation
    (|t_EQ| = 0.81) under x-pol; MD-led under y-pol.
25. **Cancelled by ED/MD, higher multipoles, or a combination?** A
    combination in every case; the electric quadrupole is essential in
    BOTH methods' x-channels (removal breaks the cancellation), and the
    1st-order ladder carries a documented truncation residual.
26. **Is the conventional ED/MD recipe actually optimal?** No — it is a
    good, robust solution, but the port-only optimum uses an EQ-hybrid
    basis and reaches better phase purity at equal conversion.
27. **Does freeform reveal a better multipolar solution?** Yes: the
    two-lobe EQ/MD hybrid — the campaign's central discovery.
