# R-type wide-FOV campaign - numerical qualification

## 1. Oblique-incidence S-parameter basis (the load-bearing finding)

The campaign's first preflight check caught a real hazard: a lossless
energy-closure test of the xy-labelled TORCWA S-parameters at
theta_air = 80 deg, phi = 45 deg gave R + T = 2.48 for the paper
rectangle. Inspection of `torcwa/rcwa.py` shows the xy-basis
power-normalization factor assumes the output wave's E_z is implied by
the SAME transverse component being extracted (`1 + (K_pol/Kz)^2`),
which is exact only when Kx or Ky vanishes - i.e. at phi = 0/90 deg or
at normal incidence. At intermediate azimuths the Ex/Ey/Ez cross terms
of a single plane wave are double-counted.

**Resolution**: all oblique work in this campaign extracts the full 2x2
Jones matrices in TORCWA's p/s eigenbasis (`polarization='pp'/'ps'/
'sp'/'ss'`), whose flux normalization is exact for every propagating
order. Sign conventions are fixed once by matching the validated
normal-incidence xy Jones:

    R_dev = R_ps @ diag(-1, 1)            (incident p -> -x at theta=0)
    T_dev = diag(-1, 1) @ T_ps @ diag(-1, 1)

Validation (wf_pstest.py + preflight):
  - lossless closure R+T = 1.000 +- 3e-4 at (80,45), (80,17), (60,45),
    (85,30) deg;
  - at theta = 0 exactly, the corrected p/s Jones equals the validated
    xy Jones to 0.0 (machine identity);
  - helicity powers R_cross/R_co are invariant under the in-plane
    frame rotation with phi (checked over phi = 0..90 at theta = 50).

**Retroactive caveat for the PREVIOUS (normal-incidence) campaign**:
its `angle_scan_finalists.csv` rows at phi = 45 deg used the xy-basis
normalization and are therefore quantitatively unreliable; its phi = 0
and phi = 90 rows (including the headline wide-angle numbers from
`finalists/*/angle_table.csv`, which were phi = 0) are unaffected by
this issue.

## 2. Angle convention

theta_air is the physical air-side incidence angle (illumination from
air, the paper-relevant R-type direction). TORCWA's `inc_ang` lives in
the input (glass) layer, so inc_ang = asin(sin(theta_air)/1.457);
k_par = k0 sin(theta_air) exactly. theta_air in [0, 90) maps to glass
angles [0, 43.3): the whole air hemisphere is reachable and no TIR
occurs anywhere in the domain. The previous campaign's glass-side
angle scan rows theta_glass >= 45 deg (evanescent air side) do not
recur here.

theta = 90 deg exactly is never solved (spec sec. 35): 85 deg is the
standard near-grazing endpoint, 88 deg a stability diagnostic. The
88-deg rows behaved smoothly for the rectangle (R_cross rising
monotonically toward grazing, closure intact), so no numerical
pathology was observed, but no headline claim uses theta > 85.

## 3. Diffraction preflight (exact Bloch condition)

First nonzero order (worst case over azimuth and G-direction) opens
when sin(theta_air) > lambda/P - n_medium. At 633 nm:

| P (nm) | opens in air | opens in glass |
|---|---|---|
| 200 | never (sin>2.165) | never (sin>1.708) |
| 213 | never | never |
| 226 | never | never |
| 239 | never | never |
| 252 | never | never (sin>1.055) |
| 271 (old campaign) | never | theta_air = 61.5 deg |

The ENTIRE new period grid is specular-only over the full hemisphere in
both media - every power number in this campaign is a complete power
balance with exactly one reflected and one transmitted propagating
order. (The old P=271 champions cross a glass-side order opening at
61.5 deg; their maps here account for all open channels via the p/s
closure, and this is flagged where they are compared.)

Refinement extensions reached P = 258 nm: still specular-only
(sin > 0.996 required in glass -> opens only above theta_air = 85.5
deg; the two P258 refinement runs are marked and their >= 85 deg rows
excluded from acceptance metrics).

## 4. Numerical hierarchy

- Coarse topology optimization: order [7,7], 96x96 density grid,
  5-angle minibatches, 80 iters; mining pool sweeps at [5,5]
  (ranking only, never reported).
- Refinement: order [9,9] end-to-end, warm-started, 60 iters.
- Every reported pool/ledger metric: hard-binary geometry, full
  21-point pool, order [9,9].
- Fine angle maps (18 theta x 7 phi): order [9,9].
- Finalist convergence: orders 9/11/13/15 at (0,0), (60,0), (60,45),
  (80,0) on Re/Im of the circular cross amplitude (see
  results/convergence.csv) - power-only convergence is never used to
  certify a cancellation claim.

## 5. Material and stack

a-Si Franta 2013 (dataset genuinely brackets 633 nm; no endpoint
clamping): n + ik = 4.2827 + 0.0687i at 633 nm. Glass n = 1.457
(Malitson), eps = 2.122849. Stack: glass substrate / patterned a-Si
(H) / air; air-side illumination; reflection port air, transmission
port glass. The k = 0 counterfactual (spec sec. 42) is diagnostic-only
and never used in optimization.

## 6. Fixed envelope rule

padding(P) = max(20 nm, 0.10 P) - identical rule to the previous
campaign, one value per period, never swept, shared by both methods and
all seeds:

| P | padding | r_design |
|---|---|---|
| 200 | 20.0 | 80.0 |
| 213 | 21.3 | 85.2 |
| 226 | 22.6 | 90.4 |
| 239 | 23.9 | 95.6 |
| 252 | 25.2 | 100.8 |

The paper rectangle (160x96 at P=226) has half-diagonal 93.3 nm >
r_design(226) = 90.4 nm: the paper's own element does NOT satisfy the
rotation-safe envelope rule. It is evaluated as-is as the baseline;
all campaign geometries satisfy the envelope.

## 7. Seeding provenance

- Method A: paper-inspired anisotropic soft rectangle (scaled into the
  envelope) + smooth random perturbation.
- Method B primary grid (seed 11): smooth random blob + isotropic
  centering bump + weak fixed-amplitude (0.6) random-sign quadrupolar
  term (X^2 - Y^2)/r^2 - no rectangle encoding.
- OBSERVED PATHOLOGY: with the weak fixed anisotropy, several seed-11
  Method-B runs finished trapped at the rx = ry saddle (R_cross(0) ~ 0)
  - systematically at small P (tight envelopes) and at two large-P/H
  points. Method A at identical P/H never trapped, so this is
  initialization, not physics.
- RESCUE RULE (documented change, committed before any rescue ran):
  seeds >= 20 draw the quadrupolar amplitude from U[0.9, 2.2]
  (rgen seeded by (seed, P, H); still generic anisotropy, still no
  rectangle). The primary seed-11 grid is untouched and homogeneous.
  Rescue runs are extra seeds at the affected P/H points and are
  labelled by their seed in the ledger.

## 8. Final convergence numbers (orders 9 -> 15, four angle points
## incl. (60,45) and (80,0))

| tag | max complex r_cross drift | max R_cross drift |
|---|---|---|
| newA A_P239_H200 | 0.024 | 0.003 |
| newB B_P252_H185 | 0.068 | 0.016 |
| rectangle | 0.025 | 0.007 |

All campaign conclusions are stable against these drifts (the smallest
quoted margin, newA vs rectangle worst-angle 0.204 vs 0.098, is 30x
the largest relevant drift). Complex amplitudes, not power alone, were
checked per spec sec 48.
