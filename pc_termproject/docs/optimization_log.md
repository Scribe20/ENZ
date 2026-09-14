# Optimization log — Nanophotonics term project (square-lattice Si/air PC, complete TE+TM gap at 1550 nm)

All numbers in this log are produced by the independent Python PWEM solver (`pwem.py`) unless stated otherwise.
Provenance labels used throughout:
* **OFFICIAL** – output of `pc_evaluate.p` in MATLAB (none available in this environment: MATLAB is not installed;
  Octave cannot execute `.p` files).
* **SURROGATE-OFFICIAL** – Python (or MATLAB/Octave mirror) PWEM at the exact grading settings
  (Mmax = 9, 361 plane waves, 21 x 11 half-BZ grid, TM with eps-matrix, TE with 1/eps coefficients). This formulation
  was identified by digitising Figure 1 of the project sheet (rod example): all TM bands at X, M, Γ match the
  eps-matrix form and all TE bands match the 1/eps form to ±0.001 in a/λ, whereas the alternatives differ by
  0.01–0.05. It is therefore expected to agree with the official evaluator to ~1e-3, but this is an inference, not an
  execution of the evaluator.
* **REDUCED** – reduced basis / reduced k-set screening numbers (Stage A/B exploration only).

## 0. Environment
* MATLAB: not installed (searched PATH, /opt, /usr/local). `pc_evaluate.p` has pcode/compatibility = R2022a
  (visible in the file header) and cannot be executed here.
* Octave 8.4 installed via apt (used only to execute the MATLAB-syntax mirror `matlab/pc_pwem_official.m`).
* Python 3.11, numpy 2.4, scipy 1.17; 4 cores. One official-setting evaluation: 3 s (C4v wedge, 66 k) / 10 s (231 k).

## 1. Solver validation (Task 3)
| test | result |
|---|---|
| homogeneous eps = 12.1104 | ω = |k+G|/(2π√ε) reproduced to 1e-16 (both formulations) |
| 1-D stripes (d_Si = 0.25a) vs analytic Bloch/transfer-matrix dispersion | eps-form: error < 2e-4 at Mmax = 9; 1/eps form: 3e-3…2e-2 (slow, Li's factorisation rules) |
| alumina rods ε = 8.9, r = 0.2a (lecture PC-II p.10, PC-III p.15) | TM gap [0.3223, 0.4424] (textbook 0.32–0.44); no TE gap |
| project rod example (r = 0.2a, Si) vs digitised Figure 1 | TM at X: 0.2406 0.4161 0.5542 0.7108 … (fig: 0.240 0.416 0.553 0.711); TE at X: 0.4125 0.4526 0.6543 0.8162 (fig: 0.412 0.452 0.653 0.815) |
| wedge (66 k) vs full official grid (231 k) for C4v structures | identical band extrema (difference 0) |
| MATLAB mirror executed in Octave | identical to Python (rod, random examples) |
| basis convergence, rod example | TM (eps form) converged at Mmax = 5 (0.4161); TE (1/eps form) drifts: band-1 max 0.519 (M9) → 0.509 (M15) → ~0.50 (M25) |

Consequence: the official score at Mmax = 9 contains a systematic TE error of order +0.01 in a/λ; robustness of a design
must be judged by re-evaluating at larger Mmax and by checking that the gap is not created by that drift.

## 2. Hypotheses (from lecture PC-III p.27–29) and the tests designed for them
H1. Isolated high-ε islands open TM gaps (Ez continuous, Dz concentrated in islands) but not TE gaps. → Stage A rod families.
H2. Connected high-ε networks open TE gaps (Ex,y of the lowest band concentrated in the veins) and close/lower TM gaps
    (air-band Ez penetrates the veins). → Stage A hole/vein families.
H3. A hybrid (islands + thin connecting network) can open both gaps at the same frequency; the vein width controls the
    TE gap and the island size controls the TM gap; fill fraction sets the absolute frequency. → rod+vein, frame+rod families.
H4. The gap pair need not be the lowest gap of each polarisation; at a/λ = 0.409 and moderate fill, higher TM gaps are needed.
H5. C4v symmetry is not required by the problem; symmetric designs are a good starting subspace but an unconstrained
    optimisation should be compared. → Stage B with/without symmetry.

## 3. Stage A — parametric families (SURROGATE-OFFICIAL, wedge)
(filled in as the sweep completes; full record in logs/stageA.jsonl)
* rods (circle/square/diamond, r or s over the full range): TM gaps only, no TE gap at the target → H1 supported.
* holes (circle/square/diamond incl. overlapping holes): TE gaps but no TM gap at the target → H2 supported.
* rod + veins (r, w): first complete gaps at the target. Two regimes:
  - thin-node grid (r ≈ 0.17, w ≈ 0.18, fill 0.345): TM 3-4 [0.408, 0.424] ∩ TE 1-2 [0.316, 0.413] → 0.38 %.
  - large rods + thin veins (r = 0.32, w = 0.08, fill 0.383): TM 3-4 [0.381, 0.429] ∩ TE 2-3 [0.390, 0.469]
    → complete gap [0.3896, 0.4285], score **9.50 %**, margins lo 0.0194 / hi 0.0195 (already centred).

## 4. Stage B — gradient max-min optimisation + exact discrete refinement (SURROGATE-OFFICIAL, wedge)
Method: density ρ per C4v orbit (1176 variables), ε = 1 + ρ(ε_Si − 1), periodic conic filter (R = 2–3 px) and tanh
projection with β continuation; sequential LP (max t s.t. linearised margins ≥ t, trust region) using Hellmann–Feynman
band-edge sensitivities (verified against finite differences to 4 digits). After thresholding, an exact discrete
refinement: LP over boundary pixel-orbit flips (first-order predictions of all 264 margins, accurate to 1e-4),
flips accepted only when the exactly re-evaluated min margin improves.

| run | start | gap pair | continuous score | thresholded | after discrete refinement |
|---|---|---|---|---|---|
| test_rv | rods r=0.18 + veins 0.06 | TM1-2/TE1-2 | −18.8 % (infeasible) | 0 | – |
| pairscan (gray starts, Mmax 7) | uniform ρ=0.5 | TM1-2/any, TM2-3/TE2-3, TM2-3/TE3-4, TM3-4/TE3-4 | all negative | 0 | – |
| pairscan | uniform | TM2-3/TE1-2 | +6.8 % | 0 (gray exploited) | – |
| pairscan | uniform | TM3-4/TE1-2 | +5.0 % | 3.79 % | – |
| rv_31 | rods 0.17 + veins 0.18 | TM3-4/TE1-2 | 4.74 % | 4.74 % | 4.74 % |
| ref_rv32 (discrete only) | rods 0.32 + veins 0.08 (9.50 %) | TM3-4/TE2-3 | – | – | **10.00 %** (2 rounds) |
| rv32_32 | rods 0.32 + veins 0.08 | TM3-4/TE2-3 | 17.4 % (9.5 % gray pixels) | 0 % | **10.62 %** (11 rounds) |
| ref_d3 (depth-3 moves) | rv32_32 refined | TM3-4/TE2-3 | – | – | 10.62 % (local optimum confirmed) |

Lessons:
* Gray (intermediate-ε) pixels are strongly favoured by the continuous relaxation because the evaluator averages ε for TM
  and 1/ε for TE; a graded boundary is not realisable with binary pixels, so continuous scores are NOT design scores.
  Only thresholded + refined binary scores are reported as results.
* The TE flip prediction must use the exact change of 1/ε (linearising −Δε/ε² is off by a factor 12 for a full flip).
* Balanced max-min optima need combined flips (raise one edge, lower the other); single-flip greedy stalls, the LP does not.
* Best design so far (C4v, 10.62 %): rounded square Si block ≈ 0.6a with chamfered corners joined by veins ≈ 0.08a;
  fill 0.414; complete gap [0.38706, 0.43074]; TM 3-4 gap [0.3715, 0.4307], TE 2-3 gap [0.3871, 0.4500];
  lower edge set by TE band 2 at Γ, upper edge by TM band 4 at Γ; margins 0.0220 / 0.0217.
  Band-edge modes: TM3 max at X (Ez dipole of the block, 98 % of electric energy in Si), TM4 min at Γ (quadrupole, 92 %),
  TE2 max at Γ (ring-shaped D-energy inside the block, 92 % in Si), TE3 min at M (energy pushed into the veins/air, 44 %).
