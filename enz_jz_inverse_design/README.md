# enz_jz_inverse_design — freeform a-Si:H / 23-nm ITO / glass, maximum longitudinal ITO dissipation at the measured ENZ (λ_ZE ≈ 1302.3 nm)

Independent TORCWA inverse-design campaign (branch `claude/jz-1302-freeform-l2o3ew`, based on the
read-only scientific source branch `claude/enz-eigenmode-target-u95j8m`).  Nothing in the historical
`enz_*` packages or in the vendored upstream torcwa is modified; every change lives in this directory.

## Objective (the only term in the loss)

    F_z = (ω/2) Im[ε_ITO(λ_ZE)] ∫_ITO |E_z(r)|² dV / P_inc,cell,     P_inc,cell = ½ n_air cosθ |E_inc|² P²

TORCWA Lorentz–Heaviside units (c = ε0 = μ0 = 1, exp(−jωt)); plane wave from air, normal incidence,
lab-frame x polarization, |E_inc| = 1; TOTAL driven field in the ITO (all harmonics), midpoint z quadrature
strictly inside the 23-nm film (7 slices during the search, 15/31 for certification).  The same
normalization gives F_x, F_y, F_tot = F_x + F_y + F_z and η_z,abs = F_z / F_tot; the hard validation identity
is F_tot = 1 − R_tot − T_tot (all propagating orders), because ITO is the only lossy layer (a-Si:H k = 0
and glass k = 0 at λ_ZE in the supplied files).

**Mathematical audit.** For a spatially uniform ITO film at a fixed wavelength Im ε_ITO(λ_ZE) is a
constant, so F_z = [ω Im ε_ITO d_ITO / (2 n_air cosθ)] · ⟨|E_z/E_inc|²⟩_ITO is *proportional* to the
volume-averaged |E_z|² objective of the historical `ito_ez_volume` campaign
(`enz_direct_enz_excitation`, `config.OBJECTIVE = "ito_ez_volume"`): with d = 23 nm and ε'' = 0.43195 at
1302.28 nm the constant is ω ε'' d / 2 = 0.02397.  The scientific novelty of this campaign is therefore
NOT the multiplicative constant but: the newly supplied measured/fitted materials, the newly determined
1302.3-nm zero crossing (the old campaigns worked at 1433.5 nm with a digitized Karimi ITO whose crossing
was 1419.6 nm), unrestricted topology without the two historical `fliplr` y-mirror projections, an outer
search over period / height / padding with several from-scratch seeds, correct incident-power
normalization, component-resolved absorption, and post-hoc Q / critical-coupling / spectral certification.

## Layout

| file | role |
|---|---|
| `config.py` | single source of truth (numerics, schedules, screening grids, predeclared tolerances) |
| `materials.py` | direct parsers of the supplied `materials_data/*` (no pandas, no clamping, no extrapolation); λ_ZE by interpolation; Stage-0 material audit |
| `forward.py` | adapter over the vendored upstream torcwa (`../enz_inverse_design/third_party`): stack builder, all-orders R/T, ITO fields (real-space `field_xy` and the exact Parseval/Fourier route), F_x/F_y/F_z, pad mask, reference designs |
| `optimizer.py` | Example6-derived topology optimizer (FFT Gaussian filter, tanh projection ramp, Adam + cosine lr, hard air ring before filter and after projection) — **no symmetry projection** |
| `stage0_preflight.py` | the 10 hard gates → `PREFLIGHT.md`, `outputs/stage0/` |
| `run_stage1.py` | Stage-1 screening driver (P × h × pad × seeds, order [5,5], 4 single-thread processes, restartable) |
| `make_stage2_jobs.py`, `run_jobs.py` | Stage-2 adaptive refinement (predeclared family selection, warm starts, controls) and the generic job runner used for Stages 2–3 |
| `stage4_certify.py` | hard-binary certification: order/z convergence, identity, hotspot watch, fabrication metrics, sensitivities, field/loss maps |
| `stage5_physics.py`, `analysis.py` | spectra, AAA scattering poles (Rayleigh-anomaly exclusion), loss-scaling Q_rad/Q_nr, with/without/lossless-ITO controls, height-detuning map, Cartesian multipoles, reference comparison (`--merge-only` rebuilds the tables) |
| `enz_film_mode.py` | bare air/ITO/glass ENZ-mode reference (complex-ω TM pole with a Drude continuation; driven Berreman absorption) |
| `report_tables.py`, `compact_outputs.py` | machine-readable `outputs/results.csv|json`, `outputs/REPORT_TABLES.md`, `outputs/best/`; uint8 storage of hard binaries |
| `plots.py` | figures |
| `materials_data/` | verbatim copies of the supplied files + `PROVENANCE.md` (sha256) |
| `upstream/` | verbatim copy of the supplied `Example6.ipynb` and `Materials.py` (audited, not used) |
| `outputs/stage*/` | all results (histories in `result.json`, `rho_hard_binary.npy`, `rho_raw_final.npy`, masks, logs, figures, csv/json tables) |

## Reproduce

    cd enz_jz_inverse_design
    python stage0_preflight.py                     # gates; must print "failed gates: none"
    python run_stage1.py --iters 80                # 270 screening runs (restartable)
    python run_stage1.py --iters 80 --h 500 600 --P 750 825 850 --pad 0.08 0.12 --seeds 333 1001   # Stage 1b height extension
    python make_stage2_jobs.py && python run_jobs.py --jobs outputs/stage2/jobs.json --out outputs/stage2
    python make_stage3_jobs.py && python run_jobs.py --jobs outputs/stage3/jobs.json --out outputs/stage3
    python run_jobs.py --jobs outputs/stage1c/jobs.json --out outputs/stage1c    # deck-thickness (970.5 nm) screening
    python run_jobs.py --jobs outputs/stage2c/jobs.json --out outputs/stage2c    # its warm refinement
    python stage4_certify.py --runs <run dirs> --out outputs/stage4
    python stage5_physics.py --runs <run dirs> --out outputs/stage5              # add --no-refs after the first call
    python report_tables.py                                                     # results.csv/json, REPORT_TABLES.md, outputs/best

Environment used here: CPU-only (4 cores), torch 2.14 (CPU), scipy 1.17 (AAA), complex128 solves,
float64 geometry, 128×128 topology grid.

## Headline result

Best certified **F_z = 0.954** (A = 0.991, η_z,abs = 0.96) for a tall (h = 525–600 nm), low-fill (≈ 0.3), asymmetric
freeform a-Si:H meta-atom on an 825-nm cell with a 12 % air ring — an ITO-loaded a-Si leaky resonance brought to
critical coupling (γ_rad/γ_nr ≈ 0.8–1.7, Q_loaded ≈ 9–11) whose longitudinal field is concentrated in the ENZ film
by D_z continuity; 3.6× the Karimi EDR cuboid (0.262) and the prior direct-E_z design (0.234) under the same
materials.  Numerically converged ([9,9]→[11,11] −0.2 %), height-robust, period-sensitive, edge-fragile
(≈ 45–60-nm necks): see `REPORT.md` §7–10.  Best design: `outputs/best/` (`rho_hard_binary.npy`, `geometry.png`,
`loss_maps.png`, `history.png`, `certify.json`, Stage-5 files).

See `REPORT.md` for the results and the eight required final statements, `PREFLIGHT.md` for the gates,
`SOURCE_AUDIT.md` for the audit of the historical sources, `outputs/stage4/CERTIFICATION.md` and
`outputs/stage5/PHYSICS.md` for the certification tables, `outputs/results.csv|json` for every run.
