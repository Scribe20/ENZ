# REPORT — freeform a-Si:H meta-atom maximizing the longitudinal ITO dissipation F_z at the measured ENZ crossing (λ_ZE = 1302.28 nm)

Campaign `enz_jz_inverse_design`, branch `claude/jz-1302-freeform-l2o3ew` (scientific source: read-only branch
`claude/enz-eigenmode-target-u95j8m`).  Every number below is reproducible from the scripts and the json/csv files
named in the text; all figures are under `outputs/`.

> All stages (0–5) are complete; the eight required statements are in §10.  Figures: `outputs/stage1/landscape.png`,
> `outputs/best/{geometry,history,loss_maps,spectra,detuning_map,loss_scaling}.png`, `outputs/stage5/references/reference_spectra.png`,
> per-design figures under `outputs/stage4/<tag>/` and `outputs/stage5/<tag>/`.

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

## 7. Certification of the finalists (Stage 4, `outputs/stage4/CERTIFICATION.md`, `certification.csv|json`)

Certified value = F_z of the hard-binary design at Fourier order [11,11] with 31 midpoint z-slices; every entry
carries its own identity residual F_tot − (1−R−T).

| design (run tag) | P | h | pad | **F_z certified** | F_x | F_y | A = 1−R−T | resid | η_z,abs | R | T | ⟨\|E_z/E_inc\|²⟩ | max \|E_z/E_inc\|² | F_z at [5]/[7]/[9]/[11] | Δ(9→11) | Δ(n_z 7→31) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final3 (P825/h525, from-scratch lineage F1) | 825 | 525 | 12 % | **0.9543** | 0.0282 | 0.0081 | 0.9907 | −1.1e-5 | 0.963 | 0.007 | 0.003 | 19.9 | 103.5 | 0.9529/0.9559/0.9553/0.9543 | −0.11 % | +0.011 % |
| final1 (P825/h575, lineage F0) | 825 | 575 | 12 % | **0.9540** | 0.0280 | 0.0094 | 0.9915 | −1.2e-5 | 0.962 | 0.006 | 0.002 | 19.9 | 106.5 | 0.9559/0.9568/0.9554/0.9540 | −0.15 % | +0.011 % |
| final0 (P825/h600, lineage F2) | 825 | 600 | 12 % | **0.9539** | 0.0278 | 0.0101 | 0.9919 | −1.2e-5 | 0.962 | 0.005 | 0.003 | 19.9 | 111.7 | 0.9564/0.9572/0.9555/0.9539 | −0.18 % | +0.012 % |
| final2 (P825/h600, pad 8 %) | 825 | 600 | 8 % | **0.9536** | 0.0292 | 0.0092 | 0.9921 | −1.2e-5 | 0.961 | 0.005 | 0.003 | 19.9 | 111.0 | 0.9555/0.9569/0.9551/0.9536 | −0.16 % | +0.012 % |
| scratch_s8080 (pure from-scratch [7,7]) | 825 | 600 | 12 % | **0.9418** | 0.0246 | 0.0166 | 0.9830 | −1.2e-5 | 0.958 | 0.009 | 0.008 | 19.7 | 109.3 | 0.9452/0.9522/0.9486/0.9418 | −0.73 % | +0.012 % |
| final4 (P750/h500, family F3) | 750 | 500 | 8 % | **0.9361** | 0.0303 | 0.0118 | 0.9784 | −1.3e-5 | 0.957 | 0.011 | 0.011 | 19.5 | 84.8 | 0.9300/0.9444/0.9406/0.9361 | −0.48 % | +0.013 % |
| FABR (deck thickness h = 970.5, constrained) | 825 | 970.5 | 8 % | **0.9160** | 0.0290 | 0.0050 | 0.9501 | −1.0e-5 | 0.964 | 0.019 | 0.030 | 19.1 | 79.6 | 0.9482/0.9460/0.9326/0.9160 | −1.82 % | +0.010 % |

* The four P825 designs from three different lineages (F0, F1, F2; h 525–600 nm; pad 8–12 %) are certified within
  0.0007 of each other (0.9536–0.9543): the F_z ceiling on this plateau is set by A → 0.99, not by the detailed shape.
* Order convergence: the P825 finalists change by −0.11…−0.18 % from [9,9] to [11,11] (and by < 0.4 % over
  [5,5]…[11,11]); the pure from-scratch and the P750 designs by −0.5…−0.7 %; the 970-nm design by −1.8 % (less
  converged; note the Laurent-rule factorization of the vendored solver converges slowly for tall patterned layers).
  z-quadrature converges to 1×10⁻⁴ (7 → 31 slices); identity residuals are −1×10⁻⁵ (negative, ∝ 1/n_z², as expected
  for the midpoint rule).
* Hotspot watch: max \|E_z/E_inc\|² in the ITO is 98.6 → 103.5 (final3), 107.9 → 111.7 (final0) from [5,5] to [11,11]
  — it grows by a few percent and saturates, and the loss density (`outputs/stage4/<tag>/loss_maps.png`) is two
  broad lobes under the a-Si "legs", not a corner/slot hotspot.  ⟨\|E_z/E_inc\|²⟩_ITO = 19.9 (Karimi EDR cuboid: 5.5).
* Spacer: a 3-nm Al2O3 layer (n = 1.65) between a-Si and ITO (present in the SNU fabricated stack) changes the
  Stage-2 leader from F_z 0.9549 to 0.9565 (A 0.9965 → 0.9980): immaterial.

## 8. Physics (Stage 5, `outputs/stage5/<tag>/`, `outputs/stage5/PHYSICS.md`)

Spectra 1100–1400 nm (the supplied a-Si:H data end at 1400 nm; nothing is extrapolated), order [7,7], every material
at its own tabulated dispersion; scattering poles from AAA fits of r_xx(ω) and t_xx(ω) accepted only when present in
both (relative distance < 2 %), damped, with a Lorentzian peak contribution ≥ 3 %, outside ±3 grid steps of the
Rayleigh branch point (1251.1 nm for P = 825) and inside the sampled window; loss scaling on the field-overlap-
tracked branch with s ∈ {1, ½, ¼, 0.1, 0.03, 0} multiplying Im ε_ITO.

### 8.1 The leading design (final0, P = 825 nm, h = 600 nm) — `outputs/stage5/final0_…/`

| quantity | value |
|---|---|
| F_z(λ_ZE) / A(λ_ZE) | 0.957 / 0.997 (order [7,7]) |
| F_z peak | 0.957 at 1302 nm, FWHM 150 nm (Q_eff ≈ 8.7); A_max = 0.9985 at 1298 nm |
| same geometry, ITO removed | R = 0.9998, T = 1.5×10⁻⁴ at 1314 nm: the a-Si array alone is a broadband near-perfect REFLECTOR (leaky-mode / guided-mode resonance, pole 1318.5 nm, Q 8.4) |
| same geometry, lossless ITO | R ≈ 1 over 1300–1400 nm with narrow features; broad poles at 1266/1275 nm (Q 20/12) plus a comb of high-Q poles (Q 240–1350 at 1321–1396 nm) that are invisible with the real loss |
| bare air/ITO/glass film | A = 0.029 at λ_ZE |
| loaded poles (s = 1) | **1270.9 nm, Q_loaded = 9.05** (γ = 0.0819 rad/fs, dominant, peak fraction 5.8); 1238.2 nm Q 42 (weak); 1152.6 nm Q 15; 1138.8 nm Q 43 |
| loss scaling of the 1271-nm branch | γ(s) = 0.0819 (s=1), 0.0355 (½), 0.0382 (¼), 0.0372 (0.1), 0.0370 (0.03), 0.0374 (0); pole 1271 → 1266 nm; branch overlap ≥ 0.993 at every step (no jump) |
| γ_rad, γ_nr | linear fit: γ_rad 0.0316, γ_nr 0.0412 rad/fs (Q_rad 23.4, Q_nr 18.0, ratio 0.77) but the fit is NOT linear (max residual 20 %); the lossless-limit pole gives **Q_rad = 19.9 (γ_rad 0.0374)** and hence γ_nr(s = 1) = 0.0445 rad/fs, **Q_nr = 15.7**, **γ_rad/γ_nr = 0.84** |
| one-port closure | 4 γ_rad γ_nr / (γ_rad+γ_nr)² = 0.99 (ratio 0.84) … 1.00 (ratio 1): consistent with the observed A_max 0.9985 — the design sits at critical coupling within the uncertainty of the split (ratio 0.77–0.84 from the two estimates) |
| second branch (1238 nm) | γ_rad 0.0136, γ_nr 0.0047 (Q_rad 56, Q_nr 163, ratio 2.9, linear to 1.8 %): an over-coupled, weakly absorbing Si mode |
| multipoles of the induced a-Si current at λ_ZE | ED (p+ikT) 45 % (p alone 9 %, toroidal 14 %), MD 15 %, EQ 39 %, MQ 0.4 %: a mixed ED/EQ/toroidal character, not a pure Mie dipole and not the MD+EQ TK-BIC combination |
| height detuning (h = 360–840 nm, [5,5]) | the absorption band moves red monotonically with h (A_max at 1268 nm for h = 360, 1300 nm for h = 600, 1364 nm for h = 720, 1400 nm for h = 800) and its strength peaks when it coincides with λ_ZE (A_max 0.957 → 0.998 → 0.75); the ITO-free reflection band moves the same way (T_min 1300 → 1330 → …); NO branch splitting / avoided crossing is seen anywhere in the map |

Interpretation.  The tall (≈ 600 nm) low-fill a-Si meta-atom supports a broad (Q ≈ 8) leaky resonance that, on
glass alone, reflects the plane wave almost completely.  Terminating its base with the 23-nm ENZ film converts the
reflector into an absorber: the longitudinal field at the a-Si/ITO interface is amplified by D_z continuity
(|ε_Si/ε_ITO| ≈ 20 in amplitude at λ_ZE), the ITO loss adds a non-radiative rate γ_nr that the optimizer tunes to
the mode's radiative rate (γ_rad/γ_nr ≈ 0.8–1), and T is already blocked by the Si reflector, so the one-port
critical-coupling condition gives A → 1 with η_z,abs = 0.96 of it longitudinal.  The loss-scaling curve is flat for
s ≤ ½ and doubles between s = ½ and 1: the ENZ film is not a weak perturbation of the mode at full loss (its
|ε| ≈ 0.43 is comparable to the field-jump scale), which is why the linear γ(s) decomposition must be replaced by the
lossless-limit estimate.  The bare-film ENZ pole (1331–1335 nm, Q ≈ 7.5) and the driven Berreman peak (1290 nm)
bracket the working point, but the detuning map shows the loaded resonance following the Si resonance continuously
through the ENZ wavelength without an anticrossing, and the loaded and ITO-free Q values are of the same order
(9 vs 8): this is an **ITO-loaded a-Si resonance at critical coupling with an ENZ-enhanced longitudinal loss
channel**, not a hybridized Si–ENZ polariton and not a TK-BIC.  The Karimi-type strong-coupling signature (splitting
larger than the bare linewidth under a length/lattice sweep) is absent, in line with the SNU deck's own estimate
that the coupling (≈ 21 meV) stays below the exceptional-point threshold for this ITO (Γ_ENZ ≈ 122 meV).

### 8.2 Reference designs under the new materials (`outputs/stage5/references/`)

| design | F_z(λ_ZE) | F_z max | A max | loaded poles (nm, Q) | multipoles at λ_ZE |
|---|---|---|---|---|---|
| Karimi EDR cuboid 560×500×140 @ 850 | 0.262 | 0.309 @ 1386 nm | 0.343 @ 1388 nm | 1298 (16.5) | ED 99 % |
| direct-Ez winner (old ⟨\|E_z\|²⟩ campaign, 850/140) | 0.234 | 0.324 @ 1390 nm | 0.468 @ 1302 nm | 1285 (9.9), 1290 (100), 1294 (26), 1297 (60) | ED 98 % |
| robust-A finalist0 (800/200) | 0.177 | 0.357 @ 1214 nm | 0.375 @ 1214 nm | 1201 (9.3), 1217 (24) | ED 85 %, TD 28 %, MD 12 % |

The new family reaches 3.6× the F_z of the best prior reference at λ_ZE and 3× its best value anywhere in the band.

### 8.3 All analysed designs (`outputs/stage5/PHYSICS.md`, `physics_summary.json`)

| design | F_z(λ_ZE) / A(λ_ZE) [7,7] | F_z peak, FWHM | no-ITO reflector (R_max, pole) | dominant loaded pole (nm, Q_loaded) | lossless-limit pole (nm, Q_rad) | γ_rad/γ_nr (lossless-limit est. / linear fit, resid.) | narrow secondary pole | detuning (h) | multipoles at λ_ZE (ED / TD / MD / EQ) |
|---|---|---|---|---|---|---|---|---|---|
| final0 (P825/h600) | 0.957 / 0.997 | 0.957 @ 1302, 150 nm | 0.9998, 1318.5 (Q 8.4) | 1270.9, 9.05 | 1266.2, 19.9 | 0.84 / 0.77 (20 %) | 1238 nm Q 42 (ratio 2.9, resid 1.8 %) | monotonic, peak at h ≈ 600, no anticrossing | 45 / 14 / 15 / 39 % |
| final3 (P825/h525, best certified) | 0.956 / 0.993 | 0.956 @ 1300 | 0.9994, — | 1272.5, 11.3 | 1278.9, 18.0 | 1.66 / 1.75 (9.6 %) | 1261.7 nm Q 173 (Q_rad 355, Q_nr 333, ratio 0.94, resid 1 %) | monotonic, peak at h ≈ 525, no anticrossing | 56 / 11 / 16 / 28 % |
| scratch_s8080 (P825/h600, from scratch) | 0.952 / 0.996 | 0.952 @ 1302 | 0.9979, 1330.6 (Q 8.1) | 1285.7, 9.46 | 1261.4, 28.3 (branches merge at s = 0) | 0.52 / 0.42 (14 %) | 1261.8 nm Q 23 (ratio 4.1) | monotonic, peak at h ≈ 600, no anticrossing | 35 / 12 / 19 / 45 % |
| final4 (P750/h500) | 0.944 / 0.993 | 0.944 @ 1302 | 0.9990, — | 1282.7, 26.9 | 1282.8, 49.7 | **1.19 / 1.19 (1.2 %, clean)** | 1252 nm Q 166 (fit unreliable) | monotonic, peak at h ≈ 500–533, no anticrossing | — |
| FABR (P825/h970.5, deck thickness) | 0.946 / 0.978 | 0.946 @ 1302 | 0.9995, 1304.8 (Q 1415) | 1277.9, 10.5 (not tracked: multi-mode) | — | narrow 1322-nm mode: 0.75 (9 %); 1300-nm mode: fit unreliable (57 %) | many (Q 23–187) | non-monotonic (multi-mode); A_max 0.99 at h ≈ 841 | 58 / 57 / 23 / 4 % (MQ 15 %) |

Common picture across the plateau designs: the same broad ITO-loaded a-Si reflector resonance (Q_loaded 9–11,
lossless-limit Q_rad 18–28), a longitudinal share η_z,abs = 0.96, mixed ED/EQ/toroidal multipole content, a monotonic
red shift of both the loaded absorption band and the ITO-free reflection band with height and NO branch splitting.
The radiative/non-radiative split of these broad, strongly loaded resonances is only semi-quantitative (γ(s) is
non-linear for the P825 family, and two of the tracked secondary branches merge into one at s = 0); it is clean for
the P750/h500 design, where γ_rad/γ_nr = 1.19 ± 0.02 — critical coupling — with a 1.2 % linear residual.  Taken
together with the observed A_max = 0.996–0.9985 at T → 0 (one-port: A_max = 4γ_rγ_nr/(γ_r+γ_nr)² ≥ 0.99 requires
0.8 ≤ γ_r/γ_nr ≤ 1.25), the pure-F_z optimum lies at, or within ~20 % of, γ_rad = γ_nr for every design.  The
970-nm constrained design is a multi-mode structure (a broad Q ≈ 10 pole plus several narrow Q 100–190 modes
near λ_ZE) with a non-monotonic height map, and its narrow 1322-nm branch is itself close to critical coupling
(0.75).

## 9. Fabrication / locality (Stage 4)

| design | fill (cell / active) | components | ring contact | min feature (est.) | min air gap (est.) | −2 px | −1 px | +1 px | +2 px | h −10 % | h −10 nm | h +10 nm | h +10 % | P ×0.98 | P ×1.02 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| final3 (P825/h525) | 0.318 / 0.542 | 1 | none | 45 nm | 97 nm | 0.839 | 0.907 | 0.767 | 0.543 | 0.930 | 0.954 | 0.956 | 0.921 | 0.928 | 0.902 |
| final0 (P825/h600) | 0.280 / 0.477 | 1 | none | 45 nm | 97 nm | 0.837 | 0.879 | 0.808 | 0.643 | 0.930 | 0.956 | 0.955 | 0.906 | 0.926 | 0.886 |
| final1 (P825/h575) | 0.282 / 0.480 | 2 | none | 45 nm | 110 nm | 0.843 | 0.894 | 0.820 | 0.581 | 0.928 | 0.956 | 0.956 | 0.906 | 0.929 | 0.893 |
| final2 (P825/h600, pad 8 %) | 0.275 / 0.386 | 2 | none | 45 nm | 110 nm | 0.834 | 0.906 | 0.796 | 0.521 | 0.927 | 0.956 | 0.955 | 0.899 | 0.928 | 0.887 |
| scratch_s8080 | 0.286 / 0.488 | 2 | none | 58 nm | 97 nm | 0.848 | 0.912 | 0.825 | 0.620 | 0.913 | 0.951 | 0.949 | 0.877 | 0.918 | 0.882 |
| final4 (P750/h500) | 0.442 / 0.621 | 1 | none | 88 nm | 76 nm | 0.836 | 0.893 | 0.782 | 0.699 | 0.878 | 0.940 | 0.944 | 0.910 | 0.890 | 0.816 |
| FABR (h = 970.5) | 0.414 / 0.582 | 1 | 6 px | 122 nm | 71 nm | 0.350 | 0.455 | 0.894 | 0.735 | 0.747 | 0.944 | 0.943 | 0.653 | 0.539 | 0.914 |

(F_z at [7,7]; base values 0.952–0.957.  "±k px" = uniform dilation/erosion of the binary pattern by a k-pixel disk,
1 px = 6.4 nm at P = 825 nm; "min feature/gap" = width at which a morphological opening removes > 2 % of the material/air.)

* Height tolerance is excellent (±10 nm: < 0.2 %; ±10 %: −3 … −5 %); period tolerance is moderate (±2 %: −3 … −7 %;
  at P = 825 nm the (±1,0) glass order stays evanescent up to 858.7 nm, so ×1.02 = 841.5 nm is still non-diffractive).
* Edge tolerance is the weak point: a uniform 6.4-nm erosion/dilation costs 5–20 % and 12.9 nm costs 15–48 %, and
  the plateau designs contain necks of only ≈ 45–60 nm (estimate) in 500–600-nm-tall a-Si (aspect ratio > 8).  The
  deck's fabricated process realized a 182-nm gap at aspect ratio 5.3 with a 40-nm-radius filter (`SOURCE_AUDIT.md`
  §8), so the plateau designs are NOT fabrication-ready as they stand.  This is the pathology the task anticipated:
  the pure-F_z baseline is preserved as such, and a minimum-feature / edge-robust constrained campaign (opening-
  based min-feature ≥ 100–150 nm, erosion/dilation-averaged objective) is the recommended SECOND controlled campaign
  rather than a change of this loss.
* The 970-nm constrained design is the most edge-fragile (−1 px → 0.455) and the least order-converged; its
  lower fill of thin, tall features is not a practical direction.
* All designs are single-component or two-component islands with no contact with the air ring (boundary-isolated
  meta-atom class preserved; the 970-nm design touches the ring on 6 pixels).

## 10. The eight required statements

1. **Best certified F_z = 0.954** (hard-binary design, Fourier order [11,11], 31 z-slices, identity residual
   −1×10⁻⁵): 0.9543 for `final3` (P = 825 nm, h = 525 nm, pad 12 %), with 0.9540 / 0.9539 / 0.9536 for the three
   other P825 finalists (h = 575–600 nm, pad 8–12 %) from independent lineages — i.e. a plateau at F_z = 0.954 ± 0.001,
   A = 0.991–0.992.  The pure from-scratch [7,7] control reaches 0.9418, the P750/h500 family 0.9361, and the
   fabrication-thickness-constrained design (h = 970.5 nm) 0.9160.  Reference designs under the same materials:
   Karimi EDR cuboid 0.262, prior direct-E_z winner 0.234, robust-A finalists 0.16–0.18.
2. **Geometry and materials**: air / freeform a-Si:H (n = 2.9754, k = 0 — extrapolated row of the supplied file) of
   height 525–600 nm and fill 28–32 % on a square 825-nm cell with a hard 12 % (99-nm) air ring / 23-nm ITO
   (ε = 0 + 0.432 i at λ_ZE = 1302.28 nm, `ITO_nk.csv`) / soda-lime glass (n = 1.5165); normal incidence from air,
   x polarization.  The meta-atom is a single asymmetric island (S_flip 0.8–0.9) with two tall "legs" parallel to x
   joined by a bar (`outputs/best/geometry.png`, `outputs/best/rho_hard_binary.npy`).
3. **Longitudinal share**: η_z,abs = F_z / F_tot = **0.96** for every plateau design (F_x ≈ 0.028, F_y ≈ 0.008–0.010,
   F_tot = 0.991 = 1 − R − T); ⟨|E_z/E_inc|²⟩_ITO = 19.9 (Karimi EDR cuboid: 5.5), max |E_z/E_inc|² ≈ 100–112.
4. **Q values**: plateau designs — Q_loaded = 9.1 (final0, pole 1271 nm), 11.3 (final3, 1273 nm), 9.5
   (from-scratch, 1286 nm); lossless-limit Q_rad = 19.9 / 18.0 / 28.3 and Q_nr = 15.7 / 27 / 12.9 (γ_nr = γ(1) − γ_rad);
   the linear γ(s) fits give Q_rad 23 / 18.5 / 31 and Q_nr 18 / 32 / 13 but are not linear (10–20 % residuals).
   P750/h500 design — Q_loaded 26.9, Q_rad 49.7 (lossless limit) = 49.1 (linear fit, 1.2 % residual), Q_nr 58.6.
   F_z(λ) has FWHM ≈ 150 nm (Q_eff ≈ 8.7) for the plateau designs.
5. **Critical coupling**: yes — the pure-F_z objective drove every design to (or within ~20 % of) γ_rad = γ_nr
   without any Q, absorption or coupling term in the loss.  Direct evidence: (a) the P750/h500 design, whose γ(s) is
   linear, gives γ_rad/γ_nr = 1.19 (Q_rad 49.7, Q_nr 58.6); (b) the plateau designs give 0.5–1.7 from two estimators
   (non-linear γ(s), semi-quantitative); (c) all designs reach A_max = 0.996–0.9985 with T → 0, which for a one-port
   resonance (A_max = 4γ_rγ_nr/(γ_r+γ_nr)²) is only possible for 0.8 ≤ γ_rad/γ_nr ≤ 1.25.  Mechanistically this is
   forced: with transmission blocked by the a-Si reflector, F_z ≈ η_z·A can only approach 1 at the critical-coupling
   point, so maximizing F_z selects it.
6. **Mode identity**: the evidence supports an **ITO-loaded a-Si leaky resonance at critical coupling with an
   ENZ-enhanced longitudinal loss channel** — not a Si–ENZ polariton and not a TK-BIC.  Without ITO the same
   geometry is a near-perfect reflector (R = 0.9998, pole 1318 nm, Q 8.4); with lossless ITO the broad pole persists
   (1266 nm, Q 20); the loaded pole (1271 nm, Q 9) tracks the Si resonance continuously through the ENZ wavelength in
   the height-detuning map with no avoided crossing; the multipole content is mixed ED/EQ/toroidal (45/39/14 %).  The
   field IS genuinely ENZ-concentrated (96 % of the absorption is longitudinal, in a 23-nm film, by D_z continuity),
   but there is no hybrid-mode splitting, consistent with the deck's estimate that the coupling stays below the
   exceptional-point threshold for this ITO (Γ_ENZ ≈ 122 meV).
7. **Robustness**: numerically robust — F_z changes by −0.1…−0.2 % from [9,9] to [11,11] and by < 0.4 % over
   [5,5]…[11,11], 1×10⁻⁴ from 7 to 31 z-slices, soft-to-hard binarization changes < 1×10⁻⁴, max |E_z|² saturates with
   order (no corner/slot hotspot), a 3-nm Al2O3 spacer changes F_z by +0.2 %; height-robust (±10 nm: < 0.2 %,
   ±10 %: −3…−5 %); period-sensitive (±2 %: −3…−7 %); **edge-fragile** (±6.4 nm uniform erosion/dilation −5…−20 %,
   ±12.9 nm −15…−48 %) with ≈ 45–60-nm necks in 500–600-nm-tall a-Si.  The pure-F_z baseline is thus preserved as
   such and a minimum-feature / edge-robust constrained campaign is the recommended second controlled campaign.
8. **New geometry family**: yes.  The F_z optimum is a tall (h ≥ 400 nm, plateau 500–600 nm), low-fill (≈ 0.3),
   strongly asymmetric a-Si island on the largest non-diffracting period (825–850 nm) that acts as a critically
   coupled ENZ-terminated reflector — qualitatively different from the 140–240-nm-thick, 0.4–0.6-fill Mie/QNM designs
   of every historical campaign (which, warm-started into the new cell, end at 0.35–0.88).  The family emerged
   independently from three lineages and from pure from-scratch [7,7] runs (0.942–0.954); the deck-thickness
   (970.5-nm) constrained search reaches 0.916 at P = 825 nm but fails at the deck's own 588-nm period (0.054), where
   the high-Q TK-BIC route of the deck lies outside the reach of this gradient search from random starts.

Caveats: (i) the a-Si:H index at 1302 nm comes from the workbook's Sellmeier extrapolation (k = 0); (ii) P = 825 nm
diffracts into the glass for θ > 3.6° (the deck's NA is 0.14, i.e. 8°), so the angular follow-up campaign must
either accept partial substrate diffraction or move below ≈ 790 nm; (iii) the loss-scaling decomposition is
semi-quantitative for these strongly loaded low-Q resonances (non-linear γ(s)).
