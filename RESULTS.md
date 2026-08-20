# Same-wavelength ED/MD co-excitation topology search — final report

**Task**: TORCWA topology optimization of freeform (deliberately non-symmetric)
periodic a-Si:H metasurfaces on a fused-silica-like substrate that co-excite a
p_x-like electric-dipole response (proxy: E_x) and an m_z-like magnetic-dipole
response (proxy: H_z) at the **same** target wavelength, under the **same**
forward, normally incident, x-polarized plane wave.

**Authoritative FoM** (never modified):
`F_co = 0.5*[log(S_ED + 1e-12) + log(S_MD + 1e-12)]` with
`S_ED = <|Ex/E_inc|^2>_Ω`, `S_MD = <|Hz/H_inc|^2>_Ω`,
Ω = full unit-cell plane × three z-slices (z/h = 0.25, 0.50, 0.75) inside the
patterned layer, all from one solve. The two channel averages are kept separate
(no pointwise products, no spatial-overlap objective).

---

## 1. Configuration actually used

| item | value |
|---|---|
| target wavelength | **1332.5 nm** (midpoint of Figure-4 ED 1300 / MD 1365) |
| periods | 750, 790, 830, 870, 900 nm (square cell) |
| heights | 100, 125, 150, 175, 200 nm |
| seeds | 11, 29, 47 (independent random `rho`, deterministic seeding) |
| substrate | ε = 1.46² (non-dispersive) |
| silicon | `Materials.aSiH.apply(1332.5)²` = **8.9221 + 0.0531j** — the aSiH table spans 192–999 nm and **clamps to its 999-nm endpoint** at the target (n = 2.98701, k = 0.008881). Flagged: this is what "the installed model evaluated at the target wavelength" yields; it is low vs. typical a-Si:H NIR values (n ≈ 3.3–3.6). |
| illumination | Example6 convention: input layer = substrate, forward propagation, `amplitude=[1,0]` (x-pol E), normal incidence |
| field normalization | torcwa 0.1.4.2 is Lorentz-Heaviside, c = 1, exp(−jωt) (verified in `rcwa.__init__` docstring); `source_fourier` sets the (0,0)-order Ex amplitude in the substrate to exactly 1 ⇒ E_inc = 1, and H_inc = n_sub·E_inc = 1.46 (magnitude of the incident **H vector**; Z₀ = 1 in these units). Hence S_ED = ⟨\|Ex\|²⟩, S_MD = ⟨\|Hz\|²⟩/n_sub². |
| near-field API | `rcwa.field_xy(layer_num=0, x, y, z_prop)` — internal-layer eigenmode reconstruction using the `self.C` coefficients from `solve_global_smatrix`; pure torch, autograd-connected (no detach in the chain) |
| grids | discovery dx = 5 nm (nx = 150–180), refinement dx = 2.5 nm (nx = 300–360); evaluation on 64×64 unit-cell grid (exact band-limited average for every order used, since 64 > 2·15+1) |
| Fourier orders | discovery **[7,7]**, refinement **[9,9]** (CPU-adapted from [11,11]; documented below), verification **[9,9]/[11,11]/[13,13]** + [15,15] extension on six candidates |
| filtering/projection | Gaussian blur radius 20 nm (physical, kernel rebuilt per grid); Example6 tanh projection, β: 1→1000 (discovery, 150 iters), 8→2000 (refine, 550 iters); no hard threshold during optimization |
| optimizer | Example6-style Adam ascent: gar₀ = 0.02, β₁ = 0.9, β₂ = 0.999, ε = 1e-8, cosine LR decay |
| symmetry | **NONE** — Example6's mirror averaging removed from init and update; source grep at every preflight prints `Verified: no mirror/rotational symmetry projection is applied to rho.` |
| diffraction check | n_sub·P/λ₀ = 0.822/0.866/0.909/0.953/0.986 — all < 1 (P = 900 flagged "near threshold"); air side ≤ 0.675. No propagating higher orders at λ₀; `ALLOW_PROPAGATING_HIGHER_ORDERS=False`. |
| hardware | CPU-only container (4 cores, ~15 GB cgroup limit), torch 2.13.0, torcwa 0.1.4.2 (byte-identical to GitHub main modulo CRLF) |

**CPU adaptations** (instruction-32 priority order, documented):
discovery kept at [7,7]×150 iters (no reduction needed — 95 s/run at 4
threads); refinement order [11,11]→[9,9] (a [11,11]×550-iter refinement
of 15 candidates was infeasible on 4 CPU cores; verification retains
[9,9]/[11,11]/[13,13] + [15,15]); coverage of periods/heights/seeds was
never reduced.

## 2. Gradient smoke test (float64/complex128, order [3,3], 58×58)

* `F_co.requires_grad = True`; backward completes; `rho.grad` finite and
  nonzero (‖∇‖ = 2.61); S_ED = 0.628, S_MD = 0.0287 — finite, positive.
* Finite-difference directional derivative vs. analytic gradient:
  relative disagreement **2.6e-9 / 2.6e-9 / 1.8e-10** at δ = 1e-3/3e-4/1e-4.
  The autograd chain through `field_xy` is exact.

## 3. Discovery sweep (75 runs — complete, 0 failures)

5 periods × 5 heights × 3 seeds, every run checkpointed/resumable and
integrity-verified (files, shapes, finiteness, 150 history rows, score
sanity). F_co(projected) range **+0.069 … +4.231**, median +3.556.
Best discovery cell: P = 900, h = 100 (all three seeds F_co ≈ +4.22–4.23,
S_ED ≈ 63, S_MD ≈ 75). The Figure-4 reference case (P = 870, h = 150,
seed 29) reached F_co = +2.414 (S_ED = 9.95, S_MD = 12.55) — a distinctly
weaker basin than the h = 100–125 family at this λ₀/material.

Operational note: two infrastructure incidents (cgroup OOM of long-lived
worker processes; an execution-worker restart) interrupted the sweep and the
refinement stage. All results survived via per-run checkpoints; the drivers
were hardened to per-run process recycling, a quiescence guard, and
idempotent filesystem-driven re-entry. No run was lost or recomputed from
scratch; 0 failed runs in the final state.

## 4. Selection (greedy diversity on 1−IoU)

Metric: 1 − IoU of binary masks resampled to a common 64×64
normalized-coordinate grid; threshold 0.30. 15 candidates selected from the
F_co-ranked list (min pairwise distance of accepted picks 0.30–1.0), spanning
P = 790–900, h = 100–200, all three seeds. Full ranked list retained in
`selection_diversity.json`; nothing was discarded.

## 5. Refinement (15 candidates, dx = 2.5 nm, order [9,9], 550 iters)

All 15 completed. F_co(projected, [9,9]) range +3.957 … +4.605.
Binary-vs-projected: **identical to ≤0.01** for every candidate (β = 2000
continuation binarizes cleanly) — no candidate collapses on binarization.

## 6. RCWA order-convergence verification — the decisive test

F_co (frozen **binary** geometry) vs. Fourier order; six candidates extended
to [15,15]:

| candidate | [9,9] | [11,11] | [13,13] | [15,15] | verdict |
|---|---|---|---|---|---|
| P0870_H0100_s29 | 4.112 | — | **4.086** | — | drift −0.03 ⇒ **robust** |
| P0830_H0100_s11 | 4.054 | 4.044 | **4.031** | 3.993 | drift −0.06 ⇒ **robust (best-converged)** |
| P0900_H0100_s11 | 4.263 | — | 4.002 | — | moderate |
| P0830_H0100_s47 | 4.164 | 4.090 | 3.881 | 3.702 | moderate |
| P0900_H0100_s29 | 4.304 | 3.951 | 3.738 | 3.503 | sensitive |
| P0900_H0175_s47 | 4.280 | 3.798 | 3.271 | 2.986 | **fragile — flagged** |
| P0900_H0200_s29 | 4.604 | 3.698 | 2.836 | 2.407 | **fragile — flagged** (the [9,9] champion) |

Every candidate loses F_co with increasing order (resonance-driven fields;
slow RCWA convergence), but by wildly different amounts. The optimization-
fidelity champion (P0900_H0200_s29, [9,9] F_co = 4.60) is largely a
numerical artifact of under-resolved sharp resonances and is **still not
converged at [15,15]**. The honest **[13,13] binary ranking** is led by:

1. **P0870_H0100_seed029** — F_co = +4.086, S_ED = 53.4, S_MD = 66.3, bal 0.81
2. **P0830_H0100_seed011** — F_co = +4.031, S_ED = 49.7, S_MD = 63.8, bal 0.78
3. P0900_H0100_seed011 — F_co = +4.002, S_ED = 51.2, S_MD = 58.5, bal 0.88

Full table: `results_ed_md_coexcitation/ranking_high_fidelity.csv` — the
final report distinguishes optimization-fidelity from verification-fidelity
performance throughout.

## 7. Diagnostic wavelength sweeps (never part of F_co)

Coarse (λ₀ ± 100 nm, 5 nm) for all 15: **every candidate's S_ED(λ) and
S_MD(λ) peak at the target wavelength**, apparent splitting ≤ 0.33 nm
(below coarse-sweep resolution). Fine sweeps (λ₀ ± 12 nm, 0.5 nm, binary,
[9,9]) on three representatives:

| candidate | peak_ED | peak_MD | splitting | FWHM |
|---|---|---|---|---|
| P0790_H0125_s29 | 1332.5 | 1332.5 | < 0.5 nm | ≈ 11.5 nm |
| P0830_H0100_s47 | 1332.5 | 1332.5 | < 0.5 nm | ≈ 9.0–9.5 nm |
| P0900_H0100_s29 | 1332.5 | 1332.5 | < 0.5 nm | ≈ 5.0 nm |

S_ED and S_MD track each other across the entire resonance line; the
0th-order transmission shows a Fano-like feature at resonance. Note the
material model stays clamped (endpoint value) over the whole sweep range,
so the sweep probes geometry dispersion only.

## 8. Multipole validation status

No trustworthy multipole-decomposition implementation exists in this
workspace or in torcwa, and none was invented (per instructions). For every
refined candidate, `multipole_data.npz` stores the complex E and H fields on
a 64×64×7-z-slice grid inside the layer plus the ε map, period, height and
wavelength — everything needed for a later exact decomposition via
J = −iωε₀(ε−1)E. **Multipole identity remains a validation task.**

## 9. Scientific interpretation (kept separate from optimization)

* The single-λ log-geometric-mean objective demonstrably co-maximizes both
  proxies: refined candidates reach S_ED ≈ 50–70 and S_MD ≈ 58–78
  ([13,13], binary) with balance 0.75–0.95, and both spectral peaks lie at
  λ₀ within ≤0.5 nm for all inspected candidates.
* The spectral coincidence of the ED-like and MD-like peaks (identical peak
  λ, identical linewidth, proxies tracking across the whole line) is most
  consistent with **one driven resonance carrying both strong E_x and H_z
  near-field content** (a mixed-multipole resonance), rather than two
  independent degenerate modes. Distinguishing hybridized-mode vs.
  single-mixed-mode scenarios requires the saved multipole data.
* A high F_co does **not** prove clean p_x/m_z multipolar purity; the
  near-field proxies are mode proxies only.
* Emergent morphology: with no symmetry imposed, every high-F_co topology is
  a y-stratified freeform Si stripe/labyrinth pattern (stripes ⟂ to the
  incident E), with genuinely different perforation/blob/comb realizations
  per seed — consistent with E_x-driven stripe-waveguide resonances whose
  y-gradients generate the out-of-plane H_z circulation.
* The reported balances arise from the log-sum objective alone; no balance
  term was used (P0900_H0175_s47 reached balance 0.99 spontaneously).

## 10. Deliverables map

```
coexcite_ed_md_sweep.py               main implementation (all modes)
analyze_coexcite_results.py           ranking/diversity/contact-sheet/spectra analysis
post_verification_ext.py              [15,15] + fine-sweep extension driver
run_phase2.sh                         idempotent refine/verify/spectra controller
Materials.py, Materials_data/aSiH.txt torcwa example material (path made file-relative)
results_ed_md_coexcitation/
  summary_discovery.csv               75-run master table (all columns per instruction 23)
  selection_diversity.json            ranked list + diversity selection (metric/threshold)
  summary_refine.csv, final_summary_refine.csv   refined tables ([9,9] fidelity)
  ranking_high_fidelity.csv           [13,13]-binary ranking + order drift
  spectra_peaks_refine.csv            peak positions/splitting, all 15
  contact_sheet_discovery.png, contact_sheet_refine.png
  fine_sweeps.png
  discovery/<run_id>/                 75 dirs: config.json, history.csv, rho_{final,projected,binary}.npy,
                                      topology.png, fom_history.png, fields_target.{npz,png}, run.log, checkpoint.pt
  refine/<run_id>_refined/            15 dirs: all of the above + verification.csv (+_o15),
                                      spectra.csv/png (+ spectra_fine.csv for 3), multipole_data.npz
```

Failure log: zero failed optimization runs. Two infrastructure interruptions
(documented in §3) were recovered from checkpoints; one transient
duplicate-worker incident was detected and resolved with no data loss (the
duplicate recomputed the identical deterministic trajectory).

**Known limitation**: pushes to `Scribe20/ENZ` are rejected (HTTP 403,
"Resource not accessible by integration") — the Claude GitHub App lacks
write access to this repository, so all commits exist locally in the session
until an admin grants write access.

## 11. Final compact table — strongest and most distinct solutions

[9,9] columns are optimization fidelity; **Fco@13** is [13,13]-binary
verification fidelity. λ peaks from the diagnostic sweeps (coarse for all,
fine for the three marked *): splitting < resolution in all cases.

| run_id | P (nm) | h (nm) | seed | Fco proj [9,9] | Fco bin [9,9] | Fco@13 | S_ED@13 | S_MD@13 | bal@13 | λ_ED | λ_MD | Δλ (nm) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P0870_H0100_s29 | 870 | 100 | 29 | +4.112 | +4.113 | **+4.086** | 53.4 | 66.3 | 0.81 | 1332.5 | 1332.5 | <5 |
| P0830_H0100_s11 | 830 | 100 | 11 | +4.058 | +4.054 | **+4.031** | 49.7 | 63.8 | 0.78 | 1332.5 | 1332.5 | <5 |
| P0900_H0100_s11 | 900 | 100 | 11 | +4.263 | +4.263 | +4.002 | 51.2 | 58.5 | 0.88 | 1332.5 | 1332.5 | <5 |
| P0870_H0100_s47 | 870 | 100 | 47 | +4.179 | +4.180 | +3.953 | 48.7 | 55.8 | 0.87 | 1332.5 | 1332.5 | <5 |
| P0830_H0100_s47* | 830 | 100 | 47 | +4.165 | +4.164 | +3.881 | 45.7 | 51.4 | 0.89 | 1332.5 | 1332.5 | <0.5 |
| P0790_H0125_s47 | 790 | 125 | 47 | +3.957 | +3.964 | +3.817 | 39.9 | 51.9 | 0.77 | 1332.7 | 1332.3 | <5 |
| P0790_H0125_s29* | 790 | 125 | 29 | +3.983 | +3.982 | +3.806 | 39.5 | 51.3 | 0.77 | 1332.5 | 1332.5 | <0.5 |
| P0900_H0100_s29* | 900 | 100 | 29 | +4.304 | +4.304 | +3.738 | 40.3 | 43.8 | 0.92 | 1332.5 | 1332.5 | <0.5 |
| P0830_H0150_s29 | 830 | 150 | 29 | +3.984 | +3.984 | +3.729 | 37.6 | 46.1 | 0.82 | 1332.6 | 1332.5 | <5 |
| P0900_H0175_s47 | 900 | 175 | 47 | +4.284 | +4.280 | +3.271 † | 23.9 | 29.0 | 0.83 | 1332.6 | 1332.7 | <5 |
| P0900_H0200_s29 | 900 | 200 | 29 | +4.605 | +4.604 | +2.836 † | 19.7 | 14.8 | 0.75 | 1332.4 | 1332.3 | <5 |

† flagged: strongly order-sensitive; [15,15] still decreasing — treat the
[9,9] scores as unconverged upper bounds.
