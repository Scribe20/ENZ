# R_TYPE_NUMERICAL_QUALIFICATION

1. Fourier order (hard-binary, [9,9]->[15,15]): R_cross drift <= 0.010
   and phase-error drift <= 9 deg for all six finalists; complex
   Re/Im(r_x), Re/Im(r_y) drift smoothly (convergence_finalists.csv).
   B_P271_H230's phase error IMPROVES to 2 deg at [15,15].
2. Energy: A = 1 - R - T is genuine absorption (k = 0.069); residual
   diffraction accounting exact (specular-only regime verified for all
   P at normal incidence; P271 safe to theta_air = 60.5 deg).
3. Spectral window 600-670 nm: all finalists broad (R_cross >= 0.16-0.29
   across the window; at-target values within 0.03 of peak for A) - no
   fragile high-Q mirror was promoted (none found in the top set).
4. Hard-binary: every promoted geometry is exactly {0,1} x envelope
   mask; SHA256 in the ledger; fill 0.26-0.31; min Si linewidth
   82-110 nm >> 30-nm target; no internal air gap < 24 nm; edge
   clearance > 0 at every PB rotation.
5. Baseline reproduction: TORCWA rectangle matches the paper's reported
   anisotropy (0.394 pi vs ~0.4 pi) and the analytic Fresnel/circular
   checks pass to 8e-5.
