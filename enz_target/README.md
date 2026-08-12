# Target ENZ eigenmode of the bare air / ITO / glass structure

This package identifies, calculates, validates, visualizes, and saves the
**confined ENZ eigenmode** of a 23 nm ITO film between air and glass — the
target mode `E_z,ENZ` for a future freeform a-Si metasurface inverse design
with objective

```
F(rho) = | ∫_V_ITO  Ez_ENZ^* · Ez_scat(rho) dV |^2 / P_inc,cell
```

No freeform/topology optimization is performed here (by design).

## Headline results

| quantity | value |
|---|---|
| material ENZ wavelength (Re ε = 0, interpolated from CSV) | **1419.59 nm** |
| modal ENZ central wavelength (complex-ω cross-check, zero-vg point) | **≈ 1434 nm** (branch spans 1434–1493 nm for K/k₀ = 1–8) |
| target modal wavelength (max E_z confinement on the bound branch) | **1527 nm** |
| complex in-plane constant at target | **K/k₀ = +5.946 − 12.860 i** |
| ε_ITO at target | −0.590 + 0.845 i |
| air / glass 1/e decay lengths | 41.0 / 41.1 nm |
| E_z localization fraction in ITO | 0.540 |
| pole residual, max BC residual | 2×10⁻¹⁶, 1×10⁻¹⁰ |

The three wavelengths above are deliberately **not** equal: the material zero
crossing is a material property; the mode depends additionally on thickness,
claddings, and momentum (cf. Karimi et al., who report a crossing at 1410 nm
and an ENZ-mode central wavelength ≈ 1460 nm for their sample).

**Honest caveat:** with the supplied ITO data (Im ε ≈ 0.68 at the crossing),
the real-frequency confined ENZ mode is strongly **overdamped in-plane**
(|Im K| > |Re K|): it is evanescently confined in z on both sides but does not
propagate along x (backward-wave character: for the amplitude-decaying
representative, phase and decay directions are antiparallel). This is real
physics of lossy ITO, not a solver artifact; it is the same "inherently dark"
localized ENZ resonance the Karimi antennas couple to via near fields.

## Conventions

- Time: `exp(-iωt)` (identical to Vassant et al. and to TORCWA's `exp(-jωt)`).
- Ansatz: `F(z)·exp(+iKx)·exp(-iωt)`, `K = K' + iK''`.
- `kz_j = sqrt(ε_j k0² − K²)`; proper sheet via Vassant's `Re(kz)+Im(kz) > 0`,
  with sheets explicit everywhere (`tm_slab_mode.kz_branch`).
- Layers: 1 = air (z > d), 2 = ITO (0 < z < d), 3 = glass (z < 0); d = 23 nm
  (paper value); n_glass = 1.4446 (**input parameter** — the papers specify
  SiO₂ but give no number; this is Malitson fused silica at 1.45 µm).

## Dispersion relation

Source-free TM pole condition (derived from Maxwell boundary conditions,
implemented in `tm_slab_mode.py`):

```
D(K,ω) = 1 + r12 · r23 · exp(2i kz2 d) = 0,      r_ij = (ε_j kz_i − ε_i kz_j)/(ε_j kz_i + ε_i kz_j)
```

which is the pole of the three-layer reflection coefficient and is verified
numerically to be equivalent to Eq. (1) of Vassant et al. (= Eq. (1) of the
Karimi SI):

```
1 + (ε1 kz3)/(ε3 kz1) = i tan(kz2 d) [ (ε2 kz3)/(ε3 kz2) + (ε1 kz2)/(ε2 kz1) ]
```

Both forms agree at every found root to ~1e-16. The equation is even in kz2,
so only the cladding sheets matter.

## Mode taxonomy found (all reported, nothing relabeled)

1. **Confined ENZ branch** (proper sheet, decay into both claddings): exists
   only for λ > λ_ZE ≈ 1420 nm (where Re ε < 0), quasi-static character
   K ≈ ln[(ε2−ε1)(ε2−ε3)/((ε2+ε1)(ε2+ε3))]/(2d); overdamped in-plane.
   Below λ_ZE it continues as a virtual/antibound solution (fields grow into
   the claddings) and is flagged as such. **This branch is the target.**
2. **Berreman leaky branch** (improper air sheet): K/k₀ ≈ 0.78–0.86 + 0.1 i,
   inside both light cones, radiative; consistent with the driven p-pol
   absorption peak (50° incidence: absorption peak 1401 nm vs. leaky-branch
   phase-match 1387 nm, well within the broad linewidth). **Not the target** —
   large |E_z| alone was never used as the identification criterion.
3. Higher-order quasi-static replicas (K'' larger by 2π/(2d) steps): strongly
   damped, discarded.

## Target-point selection criterion

On the bound branch the representative wavelength is chosen as the point of
**maximum E_z-localization fraction in the ITO film** (0.540 at 1527 nm).
Alternative centers are reported and saved rather than hidden: the complex-ω
zero-group-velocity point (≈1434 nm), and driven near-field resonance peaks
Im r_p(K,λ) (1430 nm at K = 2k₀ → 1491 nm at K = 8k₀ — the resonance is broad
and K-dependent because the mode is heavily damped). The full branch is saved
in `enz_branch.csv` / inside the NPZ, so the target field can be re-evaluated
at any wavelength with `tm_slab_mode.ModeField`.

## TORCWA capability audit (supplied torcwa-0.1.4.2)

The supplied TORCWA is a **driven** S-matrix RCWA solver. Its eigen-machinery
(`rcwa.py`: `_eigen_decomposition`, line ~1224; `_eigen_decomposition_homogenous`,
line ~1206; `torch_eig.py`: stabilized `torch.linalg.eig`) diagonalizes the
Fourier-space P·Q operator **inside a single layer** to build per-layer
S-matrices — it is *not* a global source-free eigenmode/pole solver for the
full stack. The public API (`add_input_layer`, `add_output_layer`,
`set_incident_angle`, `add_layer`, `solve_global_smatrix`, `S_parameters`,
`source_planewave`, `source_fourier`, `field_xz/yz/xy`) exposes no root
search of det S⁻¹ or pole tracking in complex K or ω. Therefore the eigenmode
is computed with the independent TM pole solver here, and TORCWA is used only
for a driven cross-check: `optional_torcwa_validation.py` compares TORCWA
p-pol R/T of the planar stack against the transfer-matrix solution —
agreement to **5×10⁻¹¹** (validates conventions and material handling).

## Files

```
config.py                    parameters + conventions (single source of truth)
ito_material.py              CSV loading, passivity, zero crossing
tm_slab_mode.py              kz branches, dispersion forms, analytic mode field
solve_enz_dispersion.py      seed scan, complex-K root solve, continuation
validate_mode.py             all validation checks
visualize_enz_mode.py        Figures 1-5
run_all.py                   end-to-end driver (python run_all.py)
optional_torcwa_validation.py  driven TORCWA vs transfer-matrix check
data/ito_digitized_dense_1nm_physical.csv   supplied ITO permittivity
target_enz_mode.npz          THE target mode (fields + branch + metadata)
enz_branch.csv               confined branch vs wavelength + diagnostics
figures/*.png                Figures 1-5
```

## `target_enz_mode.npz` contents

Target point (`wavelength_nm`, `omega_rad_per_s`, `k0_per_nm`,
`K_real_per_nm`, `K_imag_per_nm`), z-grid `z_nm` and complex `Ez`, `Ex`, `Hy`
profiles, layer `kz_*_per_nm`, material/geometry (`eps_ito_*`,
`ito_thickness_nm`, `glass_index` + provenance flag), conventions
(`time_convention`, `K_note`), and the full confined + Berreman branches.

**Normalization:** `∫_ITO |Ez|² dz = 1` with z in nm (per unit area; Ez in
nm^(-1/2)). Eigenmode amplitude is arbitrary — only overlaps/ratios are
meaningful; do not read the profile as a driven-field enhancement.
Reconstruct the 2-D target as `Ez(x,z) = Ez(z)·exp(iKx)` (note `D(K)=D(−K)`;
the +x-amplitude-decaying representative is −K).

### Later TORCWA import sketch (not implemented here, by design)

At the design wavelength, run the a-Si/ITO/glass cell and the bare ITO/glass
reference with `field_xz` (or `field_xy` at z-slices inside the ITO), form
`Ez_scat = Ez_full − Ez_ref`, interpolate the saved `Ez(z)` (and `exp(iKx)`
factor if in-plane phase matching is wanted) onto the same grid, and evaluate
the overlap integral of the objective. All conventions match TORCWA directly.

## Limitations

- ITO permittivity is tabulated at real wavelengths; the complex-ω cross-check
  needs the (excellent, max err 2×10⁻³) Drude fit ε∞ = 3.895,
  ωp = 2.6586 rad/fs, γ = 0.2322 rad/fs — clearly separated from the
  real-ω results, which use the CSV directly.
- Real-ω/complex-K and complex-ω/real-K formulations give different (both
  legitimate) pictures of the same lossy mode; the saved target uses the
  former, as specified.
- n_glass is an assumption (see above); sensitivity can be tested via
  `config.N_GLASS`.
- Local-response, nonmagnetic, planar approximations; no nonlocality or
  thickness-dependent ITO properties.
- The heavy ITO loss makes any single "modal wavelength" convention-dependent;
  all candidate centers are reported, and the branch is saved so downstream
  work is not locked to one choice.
