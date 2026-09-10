# Fresh no-ITO FDTDX runs

Every number below comes from a new FDTDX simulation. The geometry, materials, source, spatial mesh and monitors are those of the with-ITO production run of the same design; the ITO layer is replaced by the substrate glass and nothing else is changed, except where the table records a longer simulation time or a deeper glass region. R and T are the flux-plane phasors divided by the incident flux of the empty-domain reference run on the same mesh, and A = 1 − R − T. No spectrum is smoothed, clipped, renormalised or otherwise modified after the simulation.

## Acceptance test

Without the ITO every medium in the stack is lossless (the supplied a-Si:H and SiO2 data have k = 0 in band, and the fitted Lorentz models are undamped), so the exact absorption is zero at every wavelength. A run is accepted only if all of the following hold:

* `max |A| = max |1 − R − T| <= 0.01` over the whole 1200–1400 nm grid;
* no wavelength with `R > 1`;
* no wavelength with `T < 0`;
* the spectrum has stopped changing with simulation time: the difference between the longest early DFT window and the full window is `<= 0.01` in both R and T.

The early windows are recorded by the same run — extra flux-plane phasor detectors whose DFT window is closed early — so the convergence certificate needs no second simulation and no post-processing.

## Runs and their acceptance test

| design | run | time | max\|A\| | at | pts >tol | R_max | T_min | window drift R / T | accepted |
|---|---|---|---|---|---|---|---|---|---|
| final3 | `noito6ps` | 6000 fs (461060 steps) | 0.0023 | 1258 nm | 0/101 | 1.2257 | -0.2265 | 0.0340 / 0.0225 | NO |
| final1 | `noito6ps` | 6000 fs (461060 steps) | 0.1349 | 1330 nm | 2/101 | 1.0010 | +0.0000 | 0.0251 / 0.1404 | NO |
| final0 | `noito6ps` | 6000 fs (461060 steps) | 0.0941 | 1340 nm | 5/101 | 1.0005 | +0.0004 | 0.1294 / 0.0724 | NO |
| final2 | `noito6ps` | 6000 fs (461060 steps) | 0.0239 | 1308 nm | 3/101 | 0.9977 | +0.0009 | 0.0968 / 0.1100 | NO |

### Closure residual vs simulated time (windows recorded inside each run)

| design | 3000 fs | full |
|---|---|---|
| final1 | 0.2570 | 0.1349 |
| final0 | 0.1076 | 0.0941 |
| final3 | 0.0290 | 0.0023 |
| final2 | 0.2307 | 0.0239 |

## Files

* `<design>/<tag>/spectra_noito_fresh.csv` — λ, R, T, A and every early-window R/T of that run
* `<design>/<tag>/spectra_noito_fresh.npz` — the same arrays
* `<design>/<tag>/spectra_noito_fresh.png` — spectrum and the closure/convergence check
* `<design>/spectra_noito_accepted.csv|npz` — the accepted run for that design
* `<design>/compare_with_without_ITO.png` — accepted pair
* `comparison/noito_fresh_convergence.json` — the acceptance test above
* `comparison/overlay_RT_noITO_accepted.png` — all accepted no-ITO spectra
