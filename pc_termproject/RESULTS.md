# RESULTS — square-lattice Si/air 2D photonic crystal with a complete TE+TM gap centred at 1550 nm

**Provenance.** MATLAB is not available in this environment, so **no number below is an output of `pc_evaluate.p`**.
All scores are SURROGATE-OFFICIAL: the independent Python PWEM at the exact grading settings (Mmax = 9, 361 plane waves,
21 × 11 half-BZ grid, TM with the ε-matrix, TE with the 1/ε coefficients — the formulation that reproduces the project's
Figure 1 to ±0.001). The MATLAB mirror `matlab/pc_pwem_official.m`, executed in GNU Octave, gives identical numbers.
The student must run `deliverable/Nano_opt_code.m` and `pc_evaluate.p` in MATLAB to obtain the OFFICIAL score
(expected 10.6 % ± 0.3 %).

## 1. Final design (`runs/ref_rv33/mask.npy`, `deliverable/Nano_P_INITIAL_STUDENTID.mat`)
* 96 × 96 binary map, values exactly 1.0 / 12.1104, C4v-symmetric, Si fill fraction **0.4123** (3800 px).
* Geometry: a Si block ≈ 0.61a wide with chamfered/rounded corners centred in the cell, connected to its four neighbours by
  veins 8 px = 53 nm wide (0.083a). Minimum feature: the 8-px vein.
* Derivation: parametric seed (circular rod r = 0.33a + veins w = 0.08a, 5.97 %) → exact LP-guided pixel-orbit refinement
  (10 rounds) → 10.634 %. Gradient (continuous) runs and basin hopping from the r = 0.32a seed converge to the same optimum
  (10.61–10.63 %), so this is the optimum of the family, not a lucky seed.

| quantity | value |
|---|---|
| **score** | **10.634 %** (2/ω_t · min(margins)) |
| complete gap (a/λ) | [0.38719, 0.43078], width 0.04359, gap/mid-gap 10.66 % |
| in wavelength | 1471.8 nm … 1637.4 nm (target 1550 nm) |
| TM gap (bands 3-4) | [0.37195, 0.43078] — band 3 max at X, band 4 min at Γ |
| TE gap (bands 2-3) | [0.38719, 0.45154] — band 2 max at Γ, band 3 min at M |
| lower margin ω_t − ω_low | 0.02184 (TE-limited) |
| upper margin ω_high − ω_t | 0.02175 (TM-limited) |
| electric energy in Si (band-edge modes) | TM3 98 %, TM4 92 %, TE2 92 %, TE3 44 % |
| runtime of one grading-setting evaluation | 3.3 s (wedge) / 11 s (231 k) in Python; ≈ 60 s in Octave |

## 2. Verification (Stage C)
| check | result |
|---|---|
| exact grading grid (231 k) vs C4v wedge (66 k) | identical band extrema (4.6e-14) |
| fine k-grid 41 × 41 | 10.634 % (extrema at Γ, X, M) |
| Mmax 9 / 11 / 13 / 15 / 17 (official formulation) | 10.634 / 10.622 / 10.619 / 10.615 / 10.609 % |
| Mmax 9 / 11 / 13 / 15 / 17 (all-ε formulation) | 10.634 / 10.622 / 10.619 / 10.615 / 10.609 % |
| Mmax 9 / 11 / 17 (all-1/ε formulation) | 10.678 / 11.691 / 13.085 % (TM edges move up, TE-limited) |
| Mmax 5, 7 (official) | gap closed — TE low-basis error (TE2 max 0.458 / 0.419); irrelevant for Mmax ≥ 9 |
| upper edge TM4 min at Γ vs Mmax | 0.43078 → 0.43073 (converged) |
| lower edge TE2 max at Γ vs Mmax | 0.38719 → 0.38227 (moves away from target) |
| random flips of 1 % of the 664 boundary pixels (8 trials) | 10.60, 10.58, 10.57, 10.32, 10.67, 10.59, 10.58, 10.58 % |
| uniform 1-px erosion / dilation of all boundaries | 0 % / 0 % (fill 0.376 / 0.448; vein width ±26 %) |
| unconstrained (no symmetry) refinement | 10.754 % (4 pixels differ) — not adopted |

## 3. Candidate comparison (all re-evaluated on the full 231-point grid)
| design | score | fill | gap pair | note |
|---|---|---|---|---|
| project rod example r = 0.2a | 0.00 % | 0.126 | – | TM gap only (Figure 1) |
| project "solution example" (Figure 2) | 4.03 % | – | TM3-4/TE3-4 | digitised from the sheet, structure unknown |
| best rods (any r, 3 shapes) / holes / crosses / rings / frame+rod / hole+rod / two-sublattice rods | 0.00 % | – | – | Stage A, 1027 structures |
| thin-node vein grid (r 0.17 + w 0.18) | 0.38 % | 0.345 | TM3-4/TE1-2 | Stage A |
| … + gradient optimisation (rv_31) | 4.74 % | 0.316 | TM3-4/TE1-2 | |
| rods + diagonal veins (0.32, 0.16) | 3.31 % | 0.541 | TM4-5/TE3-4 | Stage A |
| … + gradient + refinement (diag_43) | 6.64 % | 0.569 | TM4-5/TE3-4 | best alternative basin |
| rods r 0.32 + veins 0.08 (seed) | 9.50 % | 0.383 | TM3-4/TE2-3 | Stage A / fine sweep optimum |
| seed + discrete refinement (ref_rv32) | 10.00 % | 0.389 | TM3-4/TE2-3 | |
| gradient + refinement (rv32_32b/c, ref_rv32_32pre) | 10.61–10.62 % | 0.415–0.417 | TM3-4/TE2-3 | |
| basin hopping (hop1) | 10.633 % | 0.411 | TM3-4/TE2-3 | |
| **r 0.33 seed + discrete refinement (ref_rv33) — FINAL** | **10.634 %** | **0.412** | TM3-4/TE2-3 | |
| unconstrained refinement (ref_nosym) | 10.754 % | 0.415 | TM3-4/TE2-3 | 4-px asymmetry, not adopted |

## 4. Files
* `deliverable/Nano_opt_code.m`, `deliverable/Nano_P_INITIAL_STUDENTID.mat` — submission (rename with initial / student id).
* `matlab/pc_pwem_official.m` — standalone MATLAB/Octave PWEM check (same formulation as the grader).
* `pwem.py`, `topopt.py`, `stageA.py`, `run_opt.py`, `run_refine.py`, `basin_hop.py`, `verify.py`, `analyze_design.py`, `plots.py`
* `figs/` — final_analysis.png, comparison_rod_seed_final.png, final_convergence.png, ablation_rod_veins.png,
  stageA_families.png, stageA_maps.png, alt_diag_analysis.png, analysis_ref_nosym.png, rv_31.png
* `docs/` — optimization_log.md, physics_explanation.md, presentation_plan.md, QA_prep.md, AI_disclosure.md,
  submission_checklist.md, remaining_manual_steps.md
* `logs/` — stageA.jsonl (1027 records), fine_sweep_rod_veins.json, candidate_table.json; `runs/*/` per-run masks and logs.
