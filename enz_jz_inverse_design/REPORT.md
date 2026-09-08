# REPORT — freeform a-Si:H meta-atom maximizing the longitudinal ITO dissipation F_z at the measured ENZ crossing (λ_ZE = 1302.28 nm)

Campaign `enz_jz_inverse_design`, branch `claude/jz-1302-freeform-l2o3ew` (scientific source: read-only branch
`claude/enz-eigenmode-target-u95j8m`).  Every number below is reproducible from the scripts and the json/csv files
named in the text; all figures are under `outputs/`.

> **Status of this document:** Sections 1–6 are final.  Sections 7–10 (certification, physics, final statements)
> are filled in as Stages 3–5 complete (see the end of the file for the eight required statements).

## 1. Task, objective and the mathematical audit

Stack: air / freeform a-Si:H (height h, density ρ(x,y) on a 128×128 cell of period P) / 23-nm ITO / soda-lime
glass.  Source: plane wave from air, normal incidence, lab-frame x polarization, |E_inc| = 1 (TORCWA
Lorentz–Heaviside units, c = ε0 = μ0 = 1, exp(−jωt)).  Primary quantity

    F_z = (ω/2) Im[ε_ITO(λ_ZE)] ∫_ITO |E_z|² dV / P_inc,cell,   P_inc,cell = ½ n_air cosθ |E_inc|² P²,

computed from the TOTAL driven field in the ITO (all harmonics), midpoint z quadrature strictly inside the film
(7 slices in the search, 15/31 at certification); the in-plane integral is evaluated exactly by Parseval's
identity on the Fourier coefficients of the ITO field (verified against the upstream real-space `field_xy`
route to 2×10⁻¹⁰, gate G6).  F_x, F_y use the same normalization; F_tot = F_x+F_y+F_z; η_z,abs = F_z/F_tot; and
the hard identity F_tot = 1 − R_tot − T_tot (all propagating orders) holds because ITO is the only lossy layer.

**Mathematical audit.** For a uniform ITO film at fixed λ, Im ε_ITO is a constant, so
F_z = [ω Im ε_ITO d_ITO / (2 n_air cosθ)] · ⟨|E_z/E_inc|²⟩_ITO: maximizing F_z is *proportional* to maximizing the
historical volume-averaged |E_z|² objective (`enz_direct_enz_excitation`, `OBJECTIVE = "ito_ez_volume"`).  The
constant is 0.02397 at λ_ZE with the supplied ITO (0.0707 for the old Karimi-digitized ITO at 1433.5 nm).  The
scientific distinction of this campaign is therefore not the objective's prefactor but (i) the newly supplied
measured/fitted materials, (ii) the newly determined 1302.28-nm zero crossing (old campaigns: 1433.5 nm on an ITO
whose crossing was 1419.6 nm, glass 1.4446 instead of 1.5165), (iii) unrestricted topology (both historical
`fliplr` y-mirror projections absent, Stage-0 gate G9), (iv) an outer search over P, h, padding with several
from-scratch seeds, (v) the incident-power normalization and component-resolved F_x, F_y, F_z, and (vi) the
post-hoc Q / critical-coupling / spectral certification.  (`SOURCE_AUDIT.md` §3.)

## 2. Authoritative materials (Stage 0, `materials.py`, `outputs/stage0/materials_audit.json`)

| material | file (sha256 in `materials_data/PROVENANCE.md`) | range | value at λ_ZE |
|---|---|---|---|
| ITO | `ITO_nk.csv` (Drude–Lorentz SE fit, 1-nm step) | 310–1675 nm | n = k = 0.46473, ε = 0 + 0.43195 i |
| a-Si:H | `aSi_H_measured_Postech.txt` | 246–1400 nm (rows ≥ 1000 nm: workbook Sellmeier extrapolation, k = 0) | n = 2.97540, k = 0 |
| glass | `SiO2_substrate_measured.txt` (= `soda_lime_…_comsol.txt`, soda-lime float glass) | 191–1689 nm | n = 1.51653, k = 0 |

* (n+ik)² agrees with the supplied ε₁, ε₂ columns to 3.1×10⁻⁶ (6-decimal data) over all 1366 rows (gate G2).
* **λ_ZE = 1302.282 nm** by cubic-spline interpolation of Re ε (the ε₁ column gives 1302.282 nm as well; the expected sanity values 1302.3 nm / ε'' ≈ 0.432 are reproduced, gate G1).  Nothing is clamped or extrapolated: requests outside a file's range raise.
* First-order diffraction thresholds at λ_ZE: P = λ/n_glass = **858.7 nm** (glass), P = λ = 1302.3 nm (air).  All screened periods (550–850 nm) are non-diffractive at normal incidence; Stage-2 periods were clipped to 853.7 nm.
* The supplied `Materials.py` points to a non-existent `Materials_data/aSiH.txt`, is cwd-relative and clamps silently outside its table; it is kept under `upstream/` for the record and not used.
* No authoritative fixed a-Si:H fabrication thickness exists in the supplied project (paper: 140/210-nm cuboids; the never-executed TK-BIC notebooks seeded H ≈ 510–600 nm; the activation deck states no thickness), so h was treated as a discrete outer variable (`SOURCE_AUDIT.md` §8).
* Bare-film reference for the new materials (`enz_film_mode.py`): TM ENZ pole at K = G10(P) at 1331–1335 nm with Q 7.35–7.73 for P = 550–850 nm (Drude continuation of the supplied table, rms residual 0.056), and driven p-polarized Berreman absorption of the bare film peaking at 1290 nm.  The design wavelength of this campaign is the task-fixed material crossing 1302.28 nm.

## 3. Preflight gates (Stage 0, `PREFLIGHT.md`)

All ten hard gates passed: (G1/G2) materials as above; (G3) bare air/ITO/glass gives |E_z| = 0 in the ITO
(R = 0.0525, T = 0.9182, A = 0.0293 = F_x); (G4) lossless closure |R+T−1| ≤ 5.8×10⁻¹⁴ with the analytic
propagating-order count reproduced (the lossless-ITO case is evaluated 2 nm off λ_ZE because ε = 0 exactly makes
the homogeneous-layer eigenproblem singular); (G5) F_x+F_y+F_z = 1−R−T to 1.0×10⁻⁴ absolute (2.2×10⁻⁴ relative)
at n_z = 7, shrinking ∝ 1/n_z² to 5×10⁻⁶ at n_z = 31; (G6) Parseval vs real-space |E|² to 1.8×10⁻¹⁰; (G7)
autograd vs central finite differences to 1.3×10⁻⁵ (32-grid) and 2.4×10⁻⁶ (production 128-grid, order [5,5]);
(G8) the hard air ring is exactly empty after filter+projection and after Adam steps; (G9) a deliberately
asymmetric seed keeps S_flip(lr/ud) = 0.43/1.35 through the pipeline and 0.51/1.41 after Adam steps (the
historical projection would give 0), and a static AST check finds no `fliplr/flipud/rot90` outside the asymmetry
metrics; (G10) screen-level convergence: F_z changes 1.5×10⁻⁴ from n_z 7→15 and ≤ 2.6 % from [7,7]→[9,9] for
the reference designs.  Timing on the 4-core CPU: 0.80 s/iteration at [5,5] (1 thread), 5.3 s at [7,7].

Reference designs under the NEW material model at λ_ZE ([7,7], n_z 7):

| design | P/h (nm) | F_z | F_x | F_y | A | R | T | η_z,abs | ⟨|E_z/E_inc|²⟩ | max |E_z/E_inc|² |
|---|---|---|---|---|---|---|---|---|---|---|
| bare ITO film | – | 0 | 0.0293 | 0 | 0.0293 | 0.052 | 0.918 | 0 | 0 | 0 |
| Karimi EDR cuboid 560×500×140 | 850/140 | 0.2619 | 0.0376 | 0.0028 | 0.3024 | 0.192 | 0.506 | 0.866 | 5.46 | 32.3 |
| Karimi MDR cuboid 650×650×210 | 810/210 | 0.1343 | 0.0323 | 0.0008 | 0.1674 | 0.065 | 0.767 | 0.802 | 2.80 | 9.5 |
| direct-Ez winner (old ⟨|E_z|²⟩ campaign) | 850/140 | 0.2377 | 0.2090 | 0.0218 | 0.4686 | 0.331 | 0.201 | 0.507 | 4.96 | 61.3 |
| padded QNM winner | 850/140 | 0.1866 | 0.2377 | 0.0254 | 0.4499 | 0.185 | 0.365 | 0.415 | 3.89 | 51.0 |
| robust-A finalist0 | 800/200 | 0.1772 | 0.0278 | 0.0010 | 0.2060 | 0.070 | 0.724 | 0.860 | 3.70 | 13.1 |
| robust-A finalist1 | 925/240 | 0.1638 | 0.0348 | 0.0084 | 0.2071 | 0.025 | 0.768 | 0.791 | 3.42 | 20.7 |

## 4. Numerical architecture

Vendored upstream torcwa (untouched), complex128 solves, float64 geometry, 128×128 topology grid, Gaussian
filter radius 40 nm, tanh projection with geometric β ramp (1 → 1000; restarted at 8 and 4 for warm starts),
hand-rolled Adam (0.9, 0.999, 10⁻⁸) with cosine learning rate 0.02, [0,1] clamp, hard air ring applied to the raw
variable before the periodic filter and again after projection, orders [5,5] (screen), [7,7] (finalists),
[9,9]/[11,11] (certification); the loss is −F_z and nothing else.  The upstream `Eig` memory-leak patch of the
robust campaign is applied in the adapter (numerically identical).  All histories, raw/hard geometries, masks and
logs are saved per run (`outputs/stage*/runs/<tag>/result.json`, `rho_raw_final.npy`, `rho_hard_binary.npy`).

## 5. Staged search

**Stage 1** (`outputs/stage1/`, 294 runs: P ∈ {550, 650, 750, 825, 850} × h ∈ {100, 150, 200, 250, 300, 400} ×
pad ∈ {4, 8, 12 %} × seeds {333, 1001, 7777}, 80 iterations at [5,5]; plus the 1b extension h ∈ {500, 600} at
P ∈ {750, 825, 850}, pad {8, 12 %}, seeds {333, 1001}).  Best-of-seeds hard-binary F_z (best over pad):

| h \ P | 550 | 650 | 750 | 825 | 850 |
|---|---|---|---|---|---|
| 100 | 0.095 | 0.113 | 0.138 | 0.180 | 0.230 |
| 150 | 0.111 | 0.137 | 0.177 | 0.346 | 0.379 |
| 200 | 0.116 | 0.151 | 0.228 | 0.437 | 0.440 |
| 250 | 0.113 | 0.177 | 0.496 | 0.615 | 0.641 |
| 300 | 0.100 | 0.352 | 0.800 | 0.881 | 0.912 |
| 400 | 0.066 | 0.673 | 0.905 | 0.933 | 0.939 |
| 500 | – | – | 0.941 | 0.946 | 0.940 |
| 600 | – | – | 0.916 | 0.950 | 0.941 |

F_z rises monotonically with height up to ≈ 400–600 nm (where A → 0.98–0.99 saturates it) and with period up
to the substrate diffraction edge; padding matters little on the plateau (best 0.905 / 0.936 / 0.939 for 4 / 8 /
12 % at h = 400).  Seed-to-seed spread is negligible (median 4×10⁻⁴ absolute; one multimodal cell,
P750/h300/pad4 %, spread 0.32).  Landscape figure: `outputs/stage1/landscape.png`.

**Stage 2** (`outputs/stage2/`, 31 runs, 100 iterations at [5,5], final evaluation at [7,7]): the four families
selected by the predeclared rule (top-4 (P,h) cells) were P825/h600, P825/h500, P850/h600 and P750/h500, each
refined at its centre and at P ± 25 nm, h ± 25 nm, pad ± 4 % (warm start from the Stage-1 raw variable, β
restarted at 8); plus three warm-start CONTROLS from the frozen historical designs (direct-Ez winner, robust
P800, robust P925, rescaled onto the best cell) and two extra from-scratch seeds (150 iterations).  Results
(hard binary, [7,7]): the plateau is flat at F_z = 0.942–0.952 for P = 800–854 nm, h = 475–600 nm, pad 8–16 %
(best 0.9521 at P825/h600/pad12 % from the P850 family; A = 0.993, η_z = 0.96); the P750/h500 family reaches
0.929–0.939; the from-scratch seeds at the best cell reach 0.930–0.940; the historical warm starts end far lower
(robust P925 0.878, direct-Ez 0.510, robust P800 0.352): the tall, low-fill (0.25–0.33) from-scratch family is a
qualitatively different geometry class from the historical 140–240-nm designs.

**Stage 3** (`outputs/stage3/`): 150-iteration [7,7] runs of the finalists (three top cells P825/h600/pad12 %,
P825/h575/pad12 %, P825/h600/pad8 %, warm-started from the Stage-1 lineage with β restarted at 4; two diversity
finalists P825/h525 and P750/h500; and one pure from-scratch [7,7] run, seed 8080, at the best cell), final
evaluation at [9,9].  → §7.

## 6. What the source audit contributed (`SOURCE_AUDIT.md`)

Conventions ledger; the two `fliplr` projections (`enz_inverse_design/optimize_enz_overlap.py` l. 128/276 and
the derived notebook cell 2) confirmed absent here; the historical `ito_ez_volume` campaign audited (P850/h140,
y-mirror ON, old materials, F_ENZ 2.285 ≙ F_z ≈ 0.16); validated tolerances carried over; the notebooks' R-direction
bug (`direction='backward'`) not present here; one sign error in this package's bare-film reference solver found
and fixed; the Rayleigh branch points (1289 nm for P = 850) handled explicitly in the pole extraction.

## 7. Certification of the finalists (Stage 4) — *to be filled*

## 8. Physics (Stage 5) — *to be filled*

## 9. Fabrication / locality — *to be filled*

## 10. The eight required statements — *to be filled*
