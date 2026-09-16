# METHOD — independent FDTDX validation of the frozen activation finalists

This stage runs AFTER the TORCWA campaign closed and its finalists were frozen (`manifest.json`, written by
`register_finalists.py` from the authoritative `rho_hard_binary.npy` files).  No geometry is re-optimized,
modified or re-selected here; no new objective is introduced; nothing feeds back into the design stage.

## 1. What is reused UNCHANGED from `enz_jz_inverse_design/fdtdx_four_finalists/`

| component | file / routine | status |
|---|---|---|
| material models | `materials/material_models.json` (Drude + Lorentz ADE ITO, lossless Lorentz a-Si:H and glass, fitted by `materials_fit.py` to the SAME supplied files as TORCWA) | reused unchanged; fit errors re-quantified over the operating window in `materials_fit_check.json` |
| scene builder | `fx_sim.build_scene`: periodic P x P cell (Bloch k = 0), 16-cell CPML in z, deep glass (1.25 um bulk + PML) because the (+-1,0) glass orders are evanescent just below their cut-off, rectilinear z mesh (ITO cells 4.6 nm, graded to 12.9 nm), fill-fraction sub-pixel smoothing of the exact 128 x 128 binary on the 64 x 64 FDTD grid (`area_fraction`), TFSF `UniformPlaneSource` (-z, E along x, Gaussian pulse 1300 nm / 13 THz), R and T flux planes, volume / xz / yz phasor detectors, time-domain decay probes, optional early-closing flux detectors (in-run convergence certificate), CFL 0.95 | reused; two hooks added with defaults equal to the old behaviour (below) |
| run loop | `fx_sim.run` (segmented / resumable `custom_fdtd_forward`, persistent JAX compilation cache), `fx_sim.write_outputs` (raw phasors of every detector + `run_meta.json`) | reused unchanged |
| reference normalization | empty-cell reference run of the identical mesh: `T_plane` -> incident spectrum, `R_plane` -> TFSF backward leakage, `inc_field_plane` -> incident Ex phasor at the ITO mid-plane position | reused unchanged (one reference per mesh, i.e. per candidate height) |
| post-processing | `fx_post.plane_flux`, `spectra_of` (T = -S_z(T)/P_inc, R = (S_z(R) - leak)/P_inc, A = 1 - R - T), `geometry`, `normalized_fields` (E / E_inc), `_ScaledPhasors` (dt scaling) | reused unchanged by import |
| FDTDX / JAX | fdtdx 0.6.2, jax 0.10.2 (PyPI; the four-finalists environment used the same versions from a supplied zip), CPU backend | re-installed in `/opt/venv-fdtdx` |

Hooks added to `fx_sim.build_scene` (backward compatible, defaults reproduce the four-finalists runs exactly):
`vol_glass_cells` / `vol_air_cells` (volume-detector extent; the old runs recorded only 4 glass cells, this stage
records 24 ~ 620 nm of glass and 8 air cells), `vol_components` (E only, or E + H), and `prov` / `design_dir`
(the geometry registry and directory, so this stage's frozen masks are used without copying the driver).

## 2. What is specific to this stage

* `register_finalists.py`: freezes the selected TORCWA candidates (uint8 copy, SHA256 of the source file, of
  the uint8 array and of the float64 array used by Stage B, P, h, d_ITO, pad, fill, material provenance, the
  field wavelengths lambda_op(+), lambda_E, the loaded-pole wavelength, lambda_op(-), polarization, incidence),
  draws every geometry from the binary (`geometry_<tag>.png`, `geometry_all_candidates.png`).
* `fx_run.py`: the run wrapper (see its docstring); default 600 fs with early-window flux detectors closed at
  300 and 450 fs, 5 ITO cells (4.6 nm), 64 x 64 in-plane cells, deep volume detector with E and H.
* `fx_fields_plots.py`: everything drawn from the RAW phasor files: normalized complex fields saved with true
  centres and edges; R/T/A vs the TORCWA cold state of Stage B (same materials, [9,9]); artefact checks
  (R > 1, T < 0, A < 0, leakage, early-window vs full-window spectra, end-of-run decay); full-stack z profiles of
  <|E|^2>, <|Ez|^2>, <|Dz|^2> (Dz with the FDTDX material assignment, see the docstring); many-z |Ez|^2 xy maps on
  ONE colour scale (linear and log); xz / yz cuts drawn with `pcolormesh` on the true non-uniform z edges;
  ITO field-enhancement and F_c analogues vs TORCWA; `metadata/plot_manifest.json`.

## 3. Convergence protocol

Simulation time: the with-ITO loaded lines of the finalists have Q_loaded < 100 (energy decay time < 70 fs);
600 fs (> 8 decay times) with early-window detectors at 300 and 450 fs gives the spectrum-vs-time convergence
curve inside the run; the time-domain probes report the end-of-run field relative to the peak.  Spatial
resolution: the four-finalists package established the 5-cell ITO / 64 x 64 in-plane mesh against a 10-cell
ITO mesh (`comparison/mesh_convergence_final3.json`); the same mesh is used and, for the primary candidate, a
10-cell ITO run is repeated when the budget allows.  PML / detector placement: unchanged from the validated
deep-glass configuration (the superseded 350-nm-glass configuration is documented there).  Spectral features
narrower than the DFT resolution of the run length are reported as unresolved, not smoothed.
