# `field_plots/` — z-profiles, Ez xy-slice sets and geometry PNGs of the four finalists

**Plot-only deliverable. No FDTD simulation was run for it.** Everything here is produced by
[`../field_stack_plots.py`](../field_stack_plots.py) from the raw FDTDX field data already stored in this
campaign directory (see [`../RUN_SUMMARY.md`](../RUN_SUMMARY.md) for how those runs were made).

Regenerate with:

```
<python-with-numpy-and-matplotlib> field_stack_plots.py
```

(only `numpy` + `matplotlib` are needed; `fdtdx`/`jax` are not).

## 1. Raw input files used

Per design `D ∈ {final3, final1, final0, final2}`, all with the **`prod` tag (the with-ITO runs)**:

| file | what is read from it |
|---|---|
| `D/prod/fields_normalized.npz` | `E_over_Einc[λ, c, ix, iy, iz]` — the complex 3-D volume block (c = Ex, Ey, Ez), plus `z_m`, `dz_m`, `x_m`, `y_m` and the layer index ranges `z_asi_range_idx`, `z_ito_range_idx`, `z_ito_mid_idx`; also the full-height `xz_plane` / `yz_plane` cuts and `z_plane_m` |
| `D/prod/Ez2_slices_aSi_1302.3nm.npz` | `Ez2_over_Einc2[ix, iy, iz]` on **every** stored a-Si:H z cell (used for the all-a-Si:H slice sheet) |
| `D/prod/run_meta.json` | full-domain `z_edges_m` and the layer indices `z_glass0 / z_ito0 / z_asi0 / z_asi1 / z_plane0 / z_plane1`, `dt_s`, grid sizes |
| `D/rho_hard_binary.npy` | the 128×128 hard-binary a-Si:H mask (geometry PNGs, and the per-cell fill fraction used for ε_zz) |
| `materials/material_models.json` | the fitted ε(λ) of ITO / a-Si:H / glass — the same models the simulation used |
| `config/geometry_provenance.json` | P, h, pad, fill fraction, source paths and SHA-256 of each frozen mask |

No other data are used, and none of these files are modified.

## 2. Coordinate system, stack ordering and normalization

Taken from the stored arrays, not assumed:

* **z is the FDTDX domain coordinate.** `z = 0` is the bottom edge of the simulation domain, which lies
  **inside the glass** (the lowest 16 cells, 412.5 nm, are the glass-side PML).
* **+z runs glass → ITO → a-Si:H → air.** The plane wave is launched above the a-Si:H and travels in **−z**,
  so along the propagation direction the stack is a-Si:H → ITO → glass (the ordering used in the user request).
  In the figures the right-hand axis / panel labels also give `z − z(a-Si:H bottom)`, i.e. z measured from the
  ITO / a-Si:H interface, positive upward into the a-Si:H.
* Layer edges for the four designs (identical below the a-Si:H, since only h changes):
  glass `0 … 1726.281 nm`, ITO `1726.281 … 1749.281 nm` (5 cells of 4.6 nm),
  a-Si:H `1749.281 … 1749.281 + h nm`, air above.
* **x, y** are in nm over one period, `0 … 825 nm`, `rho[i, j]`: i → x, j → y. All maps are drawn with
  `origin='lower'`, so x is horizontal and y is vertical, in the same orientation as the geometry PNGs.
* **Normalization is the raw campaign normalization, unchanged and identical for all four designs:**
  `E/E_inc` = phasor divided by the incident Ex phasor of the shared empty reference run (`reference/ref64`).
  Squared magnitudes are therefore `|E_c/E_inc|²`, dimensionless. All colour scales and all curves are
  directly comparable between the four candidates.
* **Dz** is derived from the stored field as `D_z / (ε₀ E_inc) = ε_zz(x, y, z) · (E_z / E_inc)`, so
  `⟨|D_z|²⟩` is in units of `(ε₀ E_inc)²`. `ε_zz` uses the fitted models at the plotted wavelength
  (glass 2.29985, ITO −0.00077 + 0.42866i, a-Si:H 8.85300 at λ_ZE) and, inside the a-Si:H layer, the exact
  per-cell area fill fraction f: `ε_zz = f·ε_aSi + (1−f)·1`. That arithmetic average is the correct
  cell-averaged ε for E_z, because inside the a-Si:H layer the pattern interfaces are **vertical** and E_z
  is the interface-parallel component.
* **Wavelength**: all figures are at **λ_ZE = 1302.282 nm** (ε′_ITO = 0), the first entry of `lam_nm`.
  The stored volume also contains 1294.0, 1298.0, 1306.0 and 1310.0 nm; change `LAM_ZE` / the wavelength
  index in `field_stack_plots.py` to plot those instead.

Sanity check that the ε assignment and the normalization are right: `⟨|D_z|²⟩_xy` is continuous across the
flat glass/ITO interface to **0.15–0.33 %** for all four designs, while `⟨|E_z|²⟩_xy` jumps there by a factor
≈ 29 = |ε_glass/ε_ITO|². In the air cells `⟨|D_z|²⟩` equals `⟨|E_z|²⟩` exactly. Across the ITO / a-Si:H interface
`⟨|D_z|²⟩` steps by ≈ 9 %: that interface is only partly a-Si:H (the rest is air), so the *xy average of a
squared* quantity is not required to be continuous there.

## 3. Files

### Per design — `final3/`, `final1/`, `final0/`, `final2/`

| file | content |
|---|---|
| `zprofile_<D>_1302.3nm.png` | **(A)** three panels sharing the z axis: `⟨\|E/E_inc\|²⟩_xy`, `⟨\|E_z/E_inc\|²⟩_xy`, `⟨\|D_z/(ε₀E_inc)\|²⟩_xy` vs z, log x. Material regions are shaded and labelled (glass / ITO / a-Si:H / air) with dashed boundary lines; the dotted line is the ITO mid-plane |
| `zprofile_<D>_1302.3nm.csv` | the same three curves as numbers, plus `⟨\|Ex\|²⟩`, `⟨\|Ey\|²⟩`, the cell width and the region label of every z cell |
| `Ez2_xy_slices_fullstack_<D>_1302.3nm.png` | **(B1)** 23 `\|E_z/E_inc\|²` xy maps spanning the whole stored block — every glass cell (4), every ITO cell (5), 12 a-Si:H cells spread over the layer, 2 air cells. Panels are ordered **along the propagation direction** (air → a-Si:H → ITO → glass); each is labelled with its z, its region and `Δz` from the a-Si:H bottom, and framed in the region colour. One common log colour scale for the whole figure |
| `Ez2_xy_slices_aSi_all_<D>_1302.3nm.png` | **(B2)** `\|E_z/E_inc\|²` on **all** stored a-Si:H z cells (42 for final3, 46 for final1, 48 for final0/final2), read from `Ez2_slices_aSi_1302.3nm.npz`. Labels give the height above the a-Si:H bottom in nm and in units of h; reading order runs downward, from the a-Si:H top to the ITO. One common log colour scale |
| `Ez2_xz_yz_fullheight_<D>_1302.3nm.png` | **supplementary**: the stored full-height xz and yz cuts plus their line-averaged `\|E_z\|²` vs z. These are the only stored data that reach deep into the glass (see the limitation below). Drawn with `pcolormesh` on the true non-uniform cell edges |

### Shared

| file | content |
|---|---|
| `geometry/geometry_<D>.png` | **(C)** the hard-binary a-Si:H mask of one candidate, 128×128, black = a-Si:H, axes in nm |
| `geometry/geometry_all_four.png` | the four masks side by side |
| `overlay_zprofiles_all_candidates.png` | the three z-profiles of all four candidates on common axes, z shifted so the ITO / a-Si:H interface is at 0 (the a-Si:H heights differ) |
| `manifest.json` | machine-readable record: per design the raw inputs used, the volume-block shape and per-layer cell counts, and every output path |

## 4. Known limitation of the stored data — depth into the glass

The FDTDX **3-D volume detector recorded only 4 glass cells below the ITO**, i.e. **≈ 37 nm of glass**
(cell widths 5.98, 7.77, 10.11, 13.14 nm). There is therefore **no stored 3-D field, and no xy map, deeper
than ~37 nm into the glass** for any of the four candidates — the request for xy slices "somewhat deeper into
the glass" cannot be met from existing data, and would need a re-run with a taller `vol_fields` detector.

What does exist for the deep glass are the stored `xz_plane` / `yz_plane` cuts (122 z cells,
z ≈ 425 … 2790 nm, i.e. ~1.3 µm of glass), but they are 2-D cuts through the cell centre, not xy planes.
They are plotted in `Ez2_xz_yz_fullheight_<D>_1302.3nm.png` and are labelled as supplementary for that reason.

The air side is likewise limited to the 4 stored air cells (≈ 90 nm above the a-Si:H) in the volume block.
