# Stage 17 interpretation - P925_Qt100 with vs without ITO (Karimi Fig. 1(c,d)-style)

All numbers from `spectrum_*.csv`, `resonances_and_poles.json`, `convergence.json`,
`detuning_trajectory_table.csv`, `field_metrics.json` (frozen geometry, no optimization,
order [7,7], ENZ region verified at [9,9] and [11,11]).

## Numerical checks (measured)
- Energy: without ITO max|R+T-1| = 3.2e-10; lossless ITO 1.8e-9; with ITO A = 1-R-T agrees with the
  ITO volume-loss integral to 1.5e-4 (11 wavelengths, 1380-1500 nm).
- Diffraction channels: glass (+-1,0) orders propagate only for lambda < n_glass P = 1336 nm; there T_total
  exceeds T_00 by up to ~0.4. For lambda > 1336 nm T_total = T_00 exactly, so every ENZ-region feature is a
  genuine total-transmittance change, not channel redistribution.
- Order convergence 1400-1470 nm: with-ITO pole 1434.0 / 1432.7 / 1430.9 nm, Q 25.0 / 25.4 / 25.6 at
  [7,7] / [9,9] / [11,11] (max |dT| 0.026); no-ITO Si pole 1465.8 / 1464.3 / 1462.8 nm, Q 58.1 / 58.9 / 59.5.
  The very narrow no-ITO poles at 1337-1341 nm and 1365 nm (Q 30-980) are order-sensitive and are NOT used
  for any conclusion.

## Answers

1. **No-ITO spectrum.** Broad 1200-1700 nm: high transmission (0.7-0.95) except for silicon resonances. In the
   ENZ region there is one dominant, deep and narrow dip at 1474 nm (T_total -> 0.001; pole 1465.8 nm,
   Q 58.1, strong in both r and t), a shallower dip at 1396 nm (pole 1402.9 nm, Q 13.2) and a narrow feature
   near 1364 nm (order-sensitive). At lambda_E the bare structure is transparent: T = 0.68, R = 0.32.
2. **Underlying Si photonic resonance.** 1465.8 nm, Q 58.1, FWHM 25.2 nm = 3.52 THz (certified r/t pole,
   converged to 1462.8 nm at [11,11]). It lies 32 nm to the red of lambda_E.
3. **What the 23-nm ITO changes.** (i) The 1466-nm Si resonance moves to 1434.0 nm (-31.9 nm, +4.5 THz) and its
   Q drops from 58 to 25 (the Stage-16 certification of this pole gives Q_rad 86, Q_nr 34). (ii) The deep
   no-ITO dip (T 0.001) becomes a shallow asymmetric dip (T minimum 0.24 at 1451 nm; A maximum 0.505 at
   1437.6 nm); at lambda_E: T 0.68 -> 0.32, R 0.32 -> 0.18, A 0 -> 0.50. (iii) The second Si mode (1403 nm,
   Q 13) moves to 1356-1359 nm (Q 13-17). (iv) The lossless-ITO control shows the same resonance at 1428 nm
   with Q 93 and T 0.05: the wavelength shift is a dispersive (Re eps_ITO ~ 0) loading effect, the Q
   reduction 93 -> 25 is ITO absorption. The lossless control also exposes a comb of very narrow poles at
   1484-1600 nm (Q 400-1200; ITO-guided/lattice states) that are invisible with the real loss.
4. **Shift, broaden, split, or hybridize?** The resonance SHIFTS (-32 nm) and BROADENS (Q 58 -> 25). It does
   not split: the with-ITO structure has one pole in the 1400-1520 nm window at every height. The two
   with-ITO poles at 1356 and 1434 nm are the ITO-shifted images of two different Si modes (no-ITO 1403 and
   1466 nm), not two hybrid branches of one mode.
5. **Anticrossing.** No. Over h = 220-260 nm the Si pole moves 1447.9 -> 1482.6 nm (+8.7 nm per 10 nm) and
   the with-ITO pole moves 1416.0 -> 1450.9 nm at the same rate with a constant offset of -31.8 +- 0.1 nm and
   nearly constant Q (26.0 -> 24.1); the lower with-ITO pole likewise tracks the 1389 -> 1417 nm Si mode with
   a -40 to -46 nm offset. Nothing is pinned near lambda_E, no gap opens when the shifted resonance crosses
   lambda_E (h ~ 240 nm), and no ENZ-anchored branch appears in the with-ITO map. Caveat: the sweep moves the
   Si resonance by only +-17 nm; a weak avoided crossing with a gap below the pole linewidth (~57 nm FWHM with
   ITO) could not be resolved, but any such gap would be far below the strong-coupling criterion.
6. **Splitting numbers.** Between the two with-ITO poles: Delta_lambda = 78.4 nm, Delta_f = 12.1 THz, larger
   than the no-ITO Si linewidth (3.5 THz) - but this pair does not originate from one mode (item 4), so it is
   not a Rabi splitting. Within the 1466-nm mode itself the splitting is zero.
7. **Longitudinal Ez.** With-ITO 1434-nm branch at its pole: F_Ez 4.23, eta_z 0.60, ITO electric-energy
   fraction 0.20, eta_ENZ,z 0.035, peak |Ez|^2 55. The lower 1356/1359-nm poles: F_Ez 4.66/4.43, eta_z
   0.66/0.64, ITO E-fraction 0.25/0.24 - slightly stronger and more longitudinal. Without ITO, the glass slab at
   the ITO position at the Si resonance has F_Ez 3.98 but eta_z only 0.20: the ITO's small |eps| converts
   the resonant field into a predominantly longitudinal one (D_z continuity) without a large change of the Ez
   magnitude.
8. **Origin of F_Ez = 4.228.** A single ITO-loaded silicon photonic resonance: the 1466-nm Si mode, shifted to
   1434 nm by the ITO's real permittivity and broadened by its absorption, driven at its own pole. It is not
   a Mie/Si-ENZ hybrid doublet; the ITO enters as a perturbative dispersive-absorptive load that carries ~20 %
   of the electric energy and makes the ITO field longitudinal (eta_z 0.6).
9. **Resemblance to Karimi et al. Fig. 1-2.** Partial. Shared: adding the ITO blue-shifts and broadens the
   silicon dip, and the with-ITO dip is shallow where the bare dip was deep. Not shared: Karimi report two
   dips (upper/lower branches) and an avoided crossing versus antenna size; here there is one dip that
   follows the Si mode with a constant offset and no second branch. P925_Qt100 is therefore in the
   weak-coupling / perturbative-loading regime with respect to the ENZ film (whose own resonance is broad,
   Q ~ 5.8), consistent with the Stage 2-9 conclusion that its strong response is an ITO-loaded Si resonance.
