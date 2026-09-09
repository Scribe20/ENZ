# METHOD — parent-mode (Q_r / eta_Dz) campaign

New research branch `claude/jz-1302-qdz-parent`, package `enz_jz_inverse_design/qdz_parent/`.
Nothing in the completed Jz/Fz, nonlinear-activation or FDTDX packages is modified; every physical
routine is reused from the validated Jz/Fz package one directory up.

## 1. Provenance of reused code

| reused | from | used for |
|---|---|---|
| `forward.build_sim(..., with_ito=False)` | Jz/Fz | the parent stack air / freeform a-Si:H / soda-lime glass, x-pol, normal incidence |
| `forward.rt_all_orders`, `specular_rt_amplitudes`, `p_inc_cell`, `build_pad_mask`, `cell_axes` | Jz/Fz | fluxes, complex r/t for pole extraction, hard air pad, grids |
| `forward.evaluate` (with ITO) | Jz/Fz | the loaded (real-ITO) certification stage only |
| `materials.eps_asi`, `n_glass`, `eps_ito`, `ito_zero_crossing` | Jz/Fz | the supplied measured/fitted dispersions and the authoritative lambda_E |
| `optimizer.gaussian_kernel_fft / filter_rho / project_rho / preprocess / binarization_metric / s_flip*` | Jz/Fz | identical filter (40 nm), tanh projection, beta schedule, symmetry scores |
| `analysis.significant_poles / rayleigh_wavelengths / track_branch / fit_gamma / fab_metrics / min_feature_px` | Jz/Fz | exact poles, Rayleigh exclusion, loss-scaling continuation, fabrication metrics |
| `config.*` | Jz/Fz | dtypes, 128 x 128 grid, orders, Adam constants, ITO thickness |

New code is confined to the parent metrics, the differentiable Q proxy, the constrained loss, and the
campaign drivers.

## 2. Parent system

`air / freeform a-Si:H (height h, density rho) / soda-lime glass`, no ITO layer.  The prospective ITO
interface is the BOTTOM face of the a-Si:H.  Both parent materials are lossless in the supplied data
(max |k| = 0 for a-Si:H and 1e-263 for the glass over 1150-1400 nm) and every solve returns
R + T - 1 = O(1e-13), so **every pole half-width is purely radiative and Q_pole = Q_r** with no
loss-scaling continuation.  No artificial lossless eps = 0 ITO is used anywhere.

lambda_E is taken from the supplied `ITO_nk.csv` at every call (`materials.ito_zero_crossing()`),
never hard-coded: **lambda_E = 1302.281986 nm**, where eps_ITO = -4.7e-16 + 0.43195i.

## 3. Field extraction and the interface quantity

Internal-layer Fourier coefficients follow the same algebra as `torcwa.rcwa.field_xy` (vendored
`rcwa.py` l. 1057-1102), the routine already used by `forward.ito_fourier_coeffs`, generalised to any
layer and additionally returning

    Dz_mn / eps0 = Ky_norm Hx_mn - Kx_norm Hy_mn                (torcwa units: eps0 = mu0 = c = 1)

i.e. the quantity torcwa multiplies by eps_conv^-1 to obtain Ez.  Taking it BEFORE the inverse
convolution removes the inverse-rule error.  Dz is the NORMAL displacement at the horizontal Si/ITO
interface, hence continuous there; it is evaluated on the glass side, where the medium is homogeneous
and no Fourier factorisation error exists.  Audit: the glass-side and in-Si values of the contact
integral agree to 1e-6 (`outputs/phaseA/audit_phaseA.json`, `Dz_glass_over_Dz_inSi`).

Real-space maps are synthesised with an exact zero-padded inverse FFT on the grid **x_j = j P / nx**,
which is the grid on which rho and eps are defined (torcwa FFTs the eps array with that convention).
Verified against torcwa's own `field_xy` to 1.6e-8 on that grid; `forward.cell_axes` is cell-centred
and differs by half a cell, which is why it is not used here.

## 4. Metrics

    eta_Dz = d_ITO * INT_Acontact rho |Dz(x,y,z_int)/eps0|^2 dA / INT_VSi rho eps_Si |E|^2 dV

with d_ITO = 23 nm, z_int the a-Si:H bottom face, and rho the differentiable density during
optimization / the hard-binary mask at certification.  Dimensionless; invariant under any rescaling of
the field amplitude (both integrals are field-quadratic).  It measures the parent mode's ability to
deliver longitudinal displacement to the future ENZ interface and contains no ITO absorption.

Reported alongside (documented alternative, better converged, never substituted silently):

    eta_Dz_P = S_Dz / W_layer,   W_layer = INT_cell eps(x,y)|E|^2 dV = P^2 sum_mn Re[E*_mn . D_mn]

the exact Parseval energy of the whole design layer (no real-space synthesis, no Gibbs error).

Diagnostic only, never optimized:

    U_mid = INT rho eps_Si |E|^2 dV over 0.3 <= z/h <= 0.7, divided by that sub-volume.

An ENZ-eigenmode overlap O_ENZ is NOT reported: the repository's `enz_film_mode.py` solves the
Berreman/ENZ pole of the bare 23-nm film, but the parent campaign defines no ITO layer, so a
normalized complex overlap against that mode would require a mode template rather than a solved
target mode.  Per the campaign rules this is recorded as unavailable rather than invented.

## 5. Radiative-Q handling

The trusted extractor `analysis.significant_poles` is a scipy AAA rational fit of r(omega), t(omega)
over many independent RCWA solves: not differentiable, not cheap.  It is used ONLY as the exact
certification value.  Inside optimization a calibrated differentiable proxy is used.

**Proxy.**  For one resonance on a slow background the driven internal energy obeys
1/W(omega) = a + b x + c x^2 exactly (x = omega - omega_E), so a weighted least-squares quadratic fit
of 1/W_Si over 5 probe frequencies gives, in closed form, omega_r = omega_E - b/(2c),
kappa = 2 sqrt((a - b^2/4c)/c) and Q = omega_r/kappa - fully differentiable, no eigenvalue solve.
W_Si (internal modal energy) is used rather than R or T so that a Fano zero cannot masquerade as high
modal Q.

**Calibration** (`outputs/phaseA/qproxy_calibration.json`, `qproxy_span_selection.json`; 11 exact poles
with Q = 7 - 605 from discs, squares, filtered-random freeforms and two Jz/Fz finalists):

| probe span (x kappa) | 0.125 | 0.25 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|---|
| median \|dQ\|/Q, centred | 0.040 | **0.047** | 0.078 | 0.210 | 0.427 |
| log-corr(Q_proxy, Q_exact) | 0.996 | **0.994** | 0.990 | 0.677 | 0.902 |
| median \|dQ\|/Q, centre offset kappa/4 | 0.100 | **0.114** | 0.148 | 0.265 | 0.494 |

The closed form is exact for an isolated Lorentzian at any span, so all error is background /
neighbouring resonances and grows with span.  **Production span = 0.25 kappa** (median 4.7 %, p90 14 %,
91 % of cases within 20 %).  Valid range: |lambda_r - lambda_E| <~ kappa/4 and Q_target within ~2x of
the true Q, exactly the region the constraints hold.  W_layer performs identically to within 0.01.

## 6. Loss and weight scale audit

    L = -log(eta_Dz + eps) + lambda_Q L_Q + lambda_lambda L_lambda + lambda_shape L_shape

with lambda_geom = 0 because the geometry constraints are structural (128 x 128 grid, hard air pad,
40-nm density filter, tanh projection with the campaign beta schedule, no imposed symmetry, Adam with
the campaign constants).

L_Q and L_lambda are expressed on the fit COEFFICIENTS rather than on the derived Q, because a design
that momentarily has no peak at lambda_E gives c <= 0, where omega/kappa diverges (observed: Q ~ 1e15,
lambda_r -> 0).  With u = c / c_target and c_target = 4a / kappa_target^2,

    L_Q      = [softlog(u)]^2 / 4          (identical to [log(Q/Q_target)]^2 at the peak, since
                                            Q/Q_target = sqrt(u) there; softlog continues log linearly
                                            below u = 0.1 so a flat design is pushed to form a peak)
    L_lambda = [(-b / (2 c_target)) / tol_omega]^2   (the peak-offset penalty, bounded as c -> 0)

Scale rationale, all terms unity at a factor-e / one-tolerance error: the objective changes by 1 when
eta_Dz changes by a factor e; L_Q = 1 when Q is off target by a factor e, symmetric in over- and
under-shoot so a larger Q is never rewarded; L_lambda = 1 at a displacement of tol_nm, set to a
quarter of the target linewidth (0.25 lambda_E / Q_target), so the alignment tightens automatically as
Q_target rises.  Every term and its magnitude is logged each iteration.

**Weight scale audit** (`outputs/phaseA/weight_scale_audit.json`): lambda_Q in {1, 3, 10, 30} from the
final3 h = 525 nm warm start.  NONE creates a resonance at lambda_E - u stays <= 0 and larger weights
merely suppress eta_Dz.  The failure is not weight choice but the starting point: the parent spectrum
is flat at lambda_E and the sensitivity of u to rho vanishes there.  Recorded as a negative result.

## 7. Seed selection (consequence of the audit)

The parent resonance wavelength is set at leading order by the layer height.  `hscan.py` /
`seed_finder.py` scan h for a topology, extract the exact parent poles and score each (topology, h) by

    score = (detuning / linewidth)^2 + [log(Q_pole / Q_target)]^2

i.e. the same two constraints the optimizer then enforces on the topology at fixed h.  Every frontier
run starts from a seed whose parent pole already lies within a fraction of a linewidth of lambda_E.
The heights this selects (380 - 700 nm) extend beyond the Jz/Fz prior of 525 - 600 nm; that prior was
set by absorption, not by a parent resonance at lambda_E.  The expansion is staged and driven by the
measured pole positions, not by a blind sweep.

## 8. Certification (exact, post hoc, no re-optimization)

`certify_parent.py` per hard-binary candidate: exact parent poles at [9,9] with a dense local rescan of
the pole nearest lambda_E; pole position and Q vs order [5,5]/[7,7]/[9,9]; eta_Dz, eta_Dz_P, U_mid,
W_Si, W_layer, R, T at lambda_E AND at lambda_r vs order; the parent R/T/W/eta spectrum; parent field
maps |E|^2, |Ez|^2, |Dz|^2; Cartesian multipole diagnostics of the a-Si:H polarization current with
p_eff = p + i k T (the toroidal term is a diagnostic and is NOT added again to the ED+MD+EQ+MQ
normalisation); geometry/fabrication metrics; SHA256 of the binary.

`hybrid_certify.py` inserts the real measured 23-nm dispersive ITO into selected frozen parents and
reports R/T/A, F_x, F_y, F_z, F_tot, eta_z_abs, <|Ez|^2>, loaded vs parent poles and the loss-scaling
continuation.  F_z is never an optimization objective anywhere in this campaign.

`detuning.py` sweeps the height of one frozen topology, solving BOTH the parent and the loaded system
at every height, assembling the loaded branches around omega_E, and fitting the non-Hermitian two-mode
Hamiltonian ONLY when loaded poles exist on both sides of omega_E over enough of the sweep; otherwise
the negative result is reported and no coupling constant is quoted.  Fit residuals and jacobian-based
uncertainties are always saved.
