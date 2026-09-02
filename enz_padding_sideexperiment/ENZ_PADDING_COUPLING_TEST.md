# ENZ padding coupling test — result: C (TRADE-OFF)

**Question.** Does an 85-nm hard lateral air ring around the freeform a-Si
meta-atom improve free-space → ENZ-QNM coupling (target: bare-film QNM at
1433.488 nm, ±G₁₀, Λ=850 nm, h=140 nm, d_ITO=23 nm)?

**Answer.** No — it costs ~11.4% of F_QNM (order-converged), but it fully
isolates the meta-atom (0% boundary contact) and centralizes the ITO
response (central-680 |Ez|² fraction 0.58 → 0.94) with only ~5% loss of
ITO longitudinal-field buildup. Locality improved, coupling mildly reduced.

## Headline table (final hard-binary geometries, matched everything)

| | UNPADDED | PADDED 85 nm |
|---|---|---|
| P / h_Si / d_ITO | 850 / 140 / 23 nm | 850 / 140 / 23 nm |
| F_QNM ([7,7] / [9,9] / [11,11]) | 97.34 / 96.96 / 97.07 | 85.94 / 86.19 / 85.96 |
| \|a₊\|² , \|a₋\|² | 48.67 , 48.67 (·P_inc) | 42.97 , 42.97 (·P_inc) |
| normalized mode overlap η± | 0.912 | 0.844 |
| ∫_ITO\|Ez\|²/P_inc (nm) | 106.8 | 101.8 |
| η_z (longitudinal fraction) | 0.801 | 0.779 |
| T / R / A(=ITO abs.) | 0.672 / 0.123 / 0.205 | 0.660 / 0.139 / 0.201 |
| fill fraction (cell) | 0.594 | 0.495 |
| boundary contact | 29.7% of edge pixels | **0%** |
| connected components (periodic) | 1 (y-wrapped stripe) | 1 (isolated patch) |
| min Si feature / min air gap | ≥133 / ≥133 nm | ≥133 / ≥133 nm |
| ITO \|Ez\|² participation ratio | 0.479 | 0.413 |
| ITO \|Ez\|² in central 680 window | 0.583 | **0.937** |
| spectral F_QNM peak | ~1440 nm | ~1430 nm |
| hybrid splitting | none resolved | none resolved |

Mask realization: dx = 6.640625 nm; active pixels i∈[13,114] (677.34 nm);
realized ring 86.328 nm per side; **zero** material density in the ring
after filtering, projection, and binarization (verified each stage).
Hard binarization changed the padded F by <0.2% (soft 86.06 → hard 85.94).
Energy closure: max|A| for the lossless no-ITO case 1.2e-12.
Symmetry audit: only the y-mirror is imposed (target has no y dependence),
retained identically in both arms; x is unconstrained.

## What the geometries are

Unpadded winner: a single a-Si stripe (~505 nm wide in x) spanning the
full cell in y (periodically connected) — a 1D grating bar. Padded
winner: the same motif truncated by the ring into an isolated rounded
patch ≈530(x)×670(y) nm — effectively a rectangular antenna, remarkably
close in scale to the Karimi EDR cuboid (560×500). The y-periodic
connectivity of the stripe is worth ≈12% of F_QNM; everything else about
the two solutions (spectral lineshape, absorption peak ~0.20-0.21,
overlap-integrand sign structure) is nearly identical, i.e. the padded
result is a slightly weaker version of the same ±G₁₀ conversion physics,
not a different resonance mechanism.

## With/without-ITO control (padded winner, frozen geometry)

Same qualitative behavior as the unpadded baseline: no sharp bare
resonance in-band without ITO; inserting ITO produces the broad
ENZ-related absorption feature near ~1460-1490 nm and the F_QNM(λ)/
∫|Ez|²(λ) peaks around the target. The optimized state is ITO-mediated,
not a bare Si Mie resonance that happens to sit at 1433 nm.

## Outcome classification

**C. TRADE-OFF** — padding improves localization/fabricability (full
boundary isolation, centralized ITO field, lower fill) but reduces F_QNM
by ~11.4%. It is not a NULL (the geometry class genuinely changed) and
not a NEGATIVE in the strong sense (88% of the coupling survives
isolation — boundary connectivity is helpful, not essential).

Per the protocol (§28), no extra seeds / padding sweep / anticrossing
test were run, since the padded branch did not improve the coupling
metric.

## Files

run_padded.py, compare_padded.py, outputs/geometries/ (mask + mask_info +
initial/final/hard-binary rho), outputs/histories/ (history.json,
convergence.json, fqnm_spectra.npz, padded_with_without_ito.npz,
headline_comparison.csv, compare.log), outputs/figures/ (geometry
comparison, F_QNM/Iz/A spectra, with/without-ITO overlay, field+current
maps for both arms, overlap-integrand profiles).
