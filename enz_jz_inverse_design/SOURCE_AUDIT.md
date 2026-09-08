# Source audit — what the historical ENZ campaigns, notebooks, torcwa and the supplied materials say

Compiled from a direct reading of the read-only source branch `claude/enz-eigenmode-target-u95j8m`
(`enz_inverse_design`, `enz_direct_enz_excitation`, `enz_padding_sideexperiment`, `enz_absorption_campaign`,
`enz_robust_aito_campaign`, `enz_highq_enz_campaign`, `enz_highq_driven_ez_audit`, `enz_target`, the vendored
torcwa), the supplied `Example6.ipynb`, `ENZ_notebook1_TKBIC.ipynb`, `ENZ_notebook2_FiniteQ.ipynb`, the Karimi
et al. paper and the material files, cross-checked by seven independent reader agents (their raw reports are
kept in the session scratchpad; four readers could not run because of a session limit, their questions were
covered by the direct reading and by the Stage-0 gates).

## 1. Conventions ledger (all re-used verbatim in this package)

| item | convention | evidence |
|---|---|---|
| units / time | TORCWA Lorentz–Heaviside, c = ε0 = μ0 = 1, exp(−jωt); Im ε > 0 = loss | `third_party/torcwa/rcwa.py` docstring l. 17–21 |
| source | `source_planewave(amplitude=[1,0], notation='ps')` at θ = 0 is an x-polarized plane wave with \|E_inc\| = 1 ('xy' notation gives \|E_inc\| = √(1+(kx/kz)²) at oblique incidence) | `enz_robust_aito_campaign/forward_multi.py` header; robust PREFLIGHT 1b/1c |
| incident power | P_inc = ½ n_in cosθ \|E0\|² P² | `enz_inverse_design/torcwa_forward.p_inc_cell`, `forward_multi.a_ito_volume` |
| layers | layer 0 = a-Si design layer, layer 1 = ITO (23 nm); `field_xy(1, x, y, z_prop)`, z_prop from the a-Si-side boundary | `torcwa_forward.ez_in_ito`, `rcwa.py` l. 959–1114 |
| Ez in an internal layer | Ez_mn = ε_conv⁻¹ (Ky Hx − Kx Hy); ε_conv is the plain Laurent rule (`_material_conv`, no inverse rule); exact in the uniform ITO layer | `rcwa.py` l. 1040–1090, 1183–1205 |
| R, T | all propagating orders, both output polarizations, p/s coherent combination, evanescent orders zeroed by `power_norm=True` | `forward_multi.rt_all_orders`; robust PREFLIGHT 1b (\|R+T−1\| 2e-14, 60 cases) |
| poles | complex r_xx, t_xx of the (0,0) order with `power_norm=False` (analytic in ω), AAA fits | `enz_absorption_campaign/pole_rt.py`, `enz_highq_driven_ez_audit/poles.py` |
| grid | cell-centred 128×128, x_i = (i+½)P/128; hard air ring: pixel active iff pad ≤ x_i ≤ P − pad | `run_padded.build_mask`, `optimizer.build_pad_mask` |
| optimizer | Example6: FFT Gaussian blur (r = 40 nm), tanh projection with geometric β ramp 1→1000, Adam (0.9, 0.999, 1e-8), lr 0.02 cosine, [0,1] clamp; mask before the filter and after the projection | `optimize_enz_overlap.py`, `enz_robust_aito_campaign/optimizer.py` |
| Eig memory leak | `torch_eig.Eig.forward` keeps its outputs on ctx (reference cycle, ~5 MB per solve); patched in the adapter with detached copies (numerically identical) | `forward_multi.py` header; ported to `forward.py` |

## 2. The two historical `fliplr` projections

`enz_inverse_design/optimize_enz_overlap.py` lines 128 and 276 (`if config.MIRROR_SYMMETRY_Y: rho = (rho + torch.fliplr(rho))/2`,
at initialization and after every Adam step; `MIRROR_SYMMETRY_Y = True` in its `config.py` l. 124) and
`enz_inverse_design/original_pixel_inverse_design.ipynb` cell 2 (the same two lines).  They were inherited by
`enz_padding_sideexperiment` and `enz_direct_enz_excitation` (the historical `ito_ez_volume` campaign ran with
the y-mirror ON: `run_campaign.py` l. 85 records `y_mirror = True`).  The supplied upstream `Example6.ipynb`
itself contains NO symmetry projection.  `enz_robust_aito_campaign/optimizer.py` and this package have none
(static AST check = Stage-0 gate G9).

## 3. The historical `ito_ez_volume` campaign (`enz_direct_enz_excitation`) — audited

* objective F_ENZ = Σ_ITO \|E_z\|² dV / (V_ITO \|E_inc\|²) = ⟨\|E_z/E_inc\|²⟩_ITO, 7 midpoint z-slices, all harmonics (`optimize_enz_overlap.forward_F`, branch `ito_ez_volume`);
* fixed P = 850 nm, h = 140 nm (Karimi EDR thickness), hard 85-nm ring (realized 86.33 nm), seed 333, order [7,7], 150 iterations, y-mirror ON;
* wavelength 1433.488 nm = Re λ of the bare-film ENZ QNM at K = G10(850) of the OLD digitized Karimi ITO (material crossing 1419.6 nm, ε = −0.074 + 0.701i at 1433.5), glass n = 1.4446 (fused-silica assumption, "NOT a paper value");
* result F_ENZ 0.727 → 2.285 (hard binary), +3.2 % over the padded QNM-target winner; A = 0.208, η_z = 0.78; order-robust to 2 %.

Constant-factor audit: F_z of this package = (ω Im ε_ITO d_ITO / 2) · ⟨\|E_z/E_inc\|²⟩ — at fixed wavelength and a
uniform film the two objectives differ by the constant 0.02397 (new material at 1302.28 nm; 0.0707 for the old
material at 1433.5 nm, i.e. the historical F_ENZ 2.285 corresponds to F_z ≈ 0.16).  The new campaign is
distinct through the materials (ε'' 0.432 vs 0.70, n_glass 1.5165 vs 1.4446), the wavelength (1302.3 vs
1433.5 nm), the removed symmetry, the P/h/pad search with several seeds and the component-resolved
certification — not through the constant.

## 4. Validated numerical facts carried over

* volume-loss identity: \|A_vol − (1−R−T)\| ≤ 5e-5 at n_z = 7 (absorption campaign, five references; robust
  campaign at oblique incidence ≤ 3e-5); the residual is negative and scales ~1/n_z² (midpoint rule) — a
  positive or > 1e-3 residual is a red flag (this package: G5 worst 1.0e-4 at n_z = 7, 5e-6 at n_z = 31);
* lossless closure \|R+T−1\| ~ 1e-14 (robust preflight; this package 5.8e-14);
* autograd vs central FD: rel. error ≤ 1e-7 (robust); this package ≤ 1.3e-5 (32-grid) and ≤ 2.4e-6 (production size);
* order convergence of A for the historical designs: ~1 % from [7,7] to [11,11] but individual oblique angles move by up to 0.06;
* warm-started chains beat from-scratch full-order runs by a wide margin in the robust campaign (J 0.35 vs 0.21–0.22): results are path-dependent; seed spread at screen level was negligible (median 1.3e-3); edge criticality ±1 px changed J by 20–30 %.

## 5. Reference designs (frozen, read-only)

| design | path | P / h / pad (nm) | objective | historical metric | under the NEW materials at λ_ZE (this package, PREFLIGHT.md) |
|---|---|---|---|---|---|
| direct-Ez winner | `enz_direct_enz_excitation/outputs/geometries/rho_hard_binary.npy` | 850 / 140 / 86.3 | ⟨\|Ez\|²⟩ at 1433.5 nm | F_ENZ 2.285, A 0.208 | F_z 0.238, A 0.469, η_z 0.51 |
| padded QNM winner | `enz_padding_sideexperiment/outputs/geometries/rho_hard_binary.npy` | 850 / 140 / 86.3 | ±G10 QNM overlap | F_ENZ 2.214, A 0.201 | F_z 0.187, A 0.450 |
| robust A finalist0 | `enz_robust_aito_campaign/outputs/stage4/runs/finalist0_P800_h200_pad0.040_warm/` | 800 / 200 / 31.3 | angle-robust A | A(0°) 0.451, J 0.353 | F_z 0.177, A 0.206 |
| robust A finalist1 | `.../finalist1_P925_h240_pad0.040_warm/` | 925 / 240 / 36.1 | angle-robust A | A(0°) 0.457 | F_z 0.164, A 0.207 |
| Karimi EDR cuboid | analytic 560×500×140 | 850 / 140 / – | paper | – | F_z 0.262, A 0.302, η_z 0.87 |
| Karimi MDR cuboid | analytic 650×650×210 | 810 / 210 / – | paper | – | F_z 0.134, A 0.167 |

## 6. Methods re-used (function → file in this package)

forward solve `build_sim`, all-orders R/T `rt_all_orders`, volume loss `loss_components` (Fourier–Parseval route,
verified against `field_xy` to 2e-10), pad mask `build_pad_mask`, symmetry metrics `s_flip*`, pole extraction
`significant_poles` (+ Rayleigh-anomaly exclusion and duplicate merging, new), loss scaling `track_branch`/`fit_gamma`
(s = 1, .5, .25, .1, .03, 0; field-overlap continuity; dense local rescan for Q > 25), with/without/lossless-ITO
controls `spectrum(with_ito=False | s=0)`, fabrication metrics `fab_metrics`/`sensitivity` (opening-based min
feature/gap, components, ring contact, ±1/±2 px, ±10 nm / ±10 % h, ±2 % P), Cartesian multipoles `multipoles`
(Alaee-normalized; NOT comparable with the notebooks' decomposition, which omits the factor 3 in Q_e).

## 7. Pitfalls found

1. ε_ITO = 0 + 0i exactly at λ_ZE makes a LOSSLESS ITO layer singular in RCWA (homogeneous-layer eigenproblem); lossless controls are evaluated off the crossing (G4: +2 nm; spectra grids avoid ±0.05 nm).
2. Supplied `Materials.py` (both the zip copy and the default-branch copy) points to `Materials_data/aSiH.txt`, which does not exist in the supplied data, silently CLAMPS outside the tabulated range, and is cwd-relative — not used.
3. The a-Si:H file ends at 1400 nm and its rows ≥ 1000 nm are a workbook Sellmeier extrapolation (k = 0); the old repo file is a Cauchy extension to 2000 nm of the same data (n identical at 1302 nm: 2.97540 vs 2.97539; 2.961 at 1500 nm) — usable only as an explicitly flagged extension for spectra beyond 1400 nm (`--asi-extended`).
4. The two ENZ notebooks were never executed (no outputs), use a Drude ITO with the crossing at 1246 nm, a-Si n = 3.61 and SiO2 n = 1.45 (all stale), and take R from `S_parameters(direction='backward')` (= reflection for light from the substrate side) — their A = 1−R−T is wrong for asymmetric stacks.  This package uses `direction='forward'`.
5. First version of `enz_film_mode.py` had the wrong sign in the TM pole condition (caught by the audit against `enz_target/tm_slab_mode.py`); fixed to D = 1 + r12 r23 e^{2i kz2 d}.
6. AAA represents the Rayleigh branch points (λ = n_glass P/√(m²+n²); 1289 nm for P = 850) by clusters of spurious poles — excluded within ±3 grid steps.
7. Historical order convergence: the robust campaign's per-angle A moved up to 0.06 between [7,7] and [11,11]; corner/slot hotspots must be watched through max\|Ez\|² vs order (Stage 4).

## 8. Authoritative fixed a-Si:H thickness?

Text sources give none (Karimi: 140/210-nm cuboids; `enz_inverse_design/config.py` froze 140 nm as "Karimi EDR
thickness"; the robust campaign screened 120–240 nm; the never-executed notebooks seeded H ≈ 510–600 nm).  The
SNU deck `ITO_activation_08_19.pdf` (2026-08-07), whose numbers live only in embedded slide images, DOES fix a
fabricated design: a-Si:H cylinder **h = 970.5 nm**, d = 405.7 nm, P = 588 nm on 23-nm ITO / glass (pp. 13, 19),
with a 3-nm ALD Al2O3 layer between ITO and a-Si:H in the fabricated stack (p. 19; absent from the deck's own
simulation schematic and from the task's stack), EBL with 300-nm ZEP + 30-nm Cr hard mask, SF6/C4F8 etch
(realized gap 182 nm, aspect ratio 5.3), operating point 1291.6 nm at critical coupling (γ_r = γ_nr = 8.54 meV,
A_pk = 0.47, T_min ≈ 0.25), realized NA 0.14 (8°; resonance survives to NA 0.21), and "freeform nonlinear
metasurface" listed as future work (p. 23).  Consequences for this campaign: (i) the free-height search
(100–600 nm, §5 of REPORT.md) stands as the primary discovery; (ii) h = 970.5 nm is respected by an explicit
fabrication-thickness-constrained sub-campaign (`outputs/stage1c`, P ∈ {588, 650, 750, 825} nm) reported
alongside it; (iii) the 3-nm Al2O3 spacer is evaluated as a sensitivity (`analysis.sensitivity`,
`spacer_Al2O3_3nm`): for the Stage-2 leader F_z changes from 0.9549 to 0.9565 (A 0.9965 → 0.9980), i.e. it is
immaterial at this level; (iv) the deck's NA 0.14 and its diffraction-free period 588 nm are the concrete
references for the robustness follow-up.

## 9. What the historical work concluded about critical coupling and mode identity

* High-Q + driven-Ez audit (old material, ε'' = 0.70): no design combined Q_rad ≥ 139 with F_Ez ≥ 3.3; the limit was the ITO loss (Q_nr 5.1–5.7 for ENZ-rich modes); the near-ENZ modes of the padded winners were dark (γ_rad/γ_nr 0.004–0.05, Q_rad 106–1872) while the Si-resonance branches near 1289 nm sat close to γ_rad ≈ γ_nr (1.16, 1.9) but had negligible ENZ participation (η_ENZ,z 0.009); the robust P800 narrow branch was undercoupled (0.53) and identified as an ITO-loaded silicon resonance (no-ITO pole at 1479 nm).
* Mode-identity tests: a genuine ITO-mediated resonance shows no sharp in-window pole without ITO and a lossless-ITO pole continuing the loaded one (`validate_with_without_ito.py`, `compare_padded.py`, stage 17).
* Bare-film ENZ mode for the NEW materials (this package, `enz_film_mode.py`, historical solver convention): complex-ω pole at K = G10(P) at 1331–1335 nm with Q 7.35–7.73 for P = 550–850 (bound; ε_ITO ≈ −0.15 there), ~30 nm red of λ_ZE; the driven p-polarized Berreman absorption of the bare film peaks at 1290 nm (A 0.21–0.29 at 45–70°).  The design wavelength of this campaign is the task-fixed material crossing 1302.28 nm.

## 10. Open items carried into the report

The high-NA goal of the deck implies P < 615 nm (NA 0.6) or < 517.5 nm (all angles) for a diffraction-free
substrate; the normal-incidence F_z optimum found here lies at P = 750–850 nm, where the (±1,0) glass orders
open for θ > 0.9° (P = 850) … 12.7° (P = 750).  This is the subject of the separate robustness follow-up, as
the task prescribes.

## 11. Independent verification of this package (third reader batch)

* torcwa/Fourier route: `forward.ito_fourier_coeffs` is algebraically identical to the internal-layer branch of
  `rcwa.field_xy` (numerically 3.6×10⁻¹⁶); on a random 64×64 test at order [3,3] the Fourier and real-space F_x,
  F_y, F_z, F_tot agree to 7×10⁻¹³, F_tot − (1−R−T) = −1.2×10⁻⁶ (n_z 7) → −1.5×10⁻⁸ (n_z 63), |E_x| = |1+r| at the
  input boundary exactly, E_z ≡ 0 in the ITO of a uniform stack, r/t agree with an independent transfer-matrix
  solution to 10⁻¹⁶; `field_xy` z_prop is measured from the input-side (a-Si) boundary (tangential-E continuity
  pinned to 10⁻¹⁵).  Upstream `rcwa.py` line 5 defines pi = 3.141592652589793 (9th decimal wrong, 3.2×10⁻¹⁰
  relative) — physically irrelevant, but it is the origin of the ~2×10⁻¹⁰ floors quoted in gate G6.  The Laurent
  rule (no inverse rule) makes fields in the patterned layer converge slowly with order (hence the [11,11]
  certification); the p/s power normalization used here is exact and the 'xx'/'yy' route is not for diagonal
  orders (never used).  Gradient caveats (Eig broadening 10⁻¹⁰, missing gauge term) are harmless for RCWA
  losses of asymmetric designs (min eigengap 2.8×10⁻³; autograd vs FD 2×10⁻⁸).
* materials: every number of `materials_audit.json` reproduced exactly by an independent implementation;
  λ_ZE = 1302.282 ± 0.0003 nm (interpolant/rounding spread), ε'' = 0.43195 ± 3×10⁻⁶, glass n = 1.51653 ± 10⁻⁵,
  P_thr = 858.73 ± 0.01 nm; the old-repo ITO is a lossy dielectric at 1302 nm (ε = +0.60 + 0.53 i); fragilities
  outside the campaign window only (glass K_ZERO_TOL 353–367 nm, a-Si k ringing 700–710 nm); supplied
  `Materials.py` additionally has a 2× wavelength-gradient bug (line 52).
* paper: EDR 560×500×140 @ 850, MDR 650×650×210 @ 810, ITO 23 nm, crossing 1410 nm, ENZ mode ≈ 1460 nm,
  Δf 21/26 THz vs 12-THz linewidths, detuning by antenna length (EDR) or lattice (MDR); the SI (fabrication,
  analytic ENZ-mode equation) is not part of the supplied text.
