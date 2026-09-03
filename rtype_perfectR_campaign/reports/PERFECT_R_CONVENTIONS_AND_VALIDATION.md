# Perfect-R campaign - definitions, conventions, validations

## 1. The target operator and the fidelity

Unrotated ideal reflective PB half-wave element, principal linear basis:
R_ideal = e^{i psi} diag(1, -1), T_ideal = 0. Circular basis (validated
C = (1/sqrt2)[[1,1],[i,-i]], sigma+ = (x+iy)/sqrt2):
R_circ,ideal = e^{i psi} [[0,1],[1,0]].

Global-phase-invariant coherent fidelity
    F_ideal = |Tr(U^dag R_circ)|^2 / 4 = |Rc01 + Rc10|^2 / 4.
Numerical normalization checks (pr_validate.py):
    ideal diag(1,-1)            F = 1.0000
    ideal x e^{i pi/2}          F = 1.0000   (global phase irrelevant)
    diag(0.8,-0.8)              F = 0.6400   (= |r|^2)
    mirror diag(1,1)            F = 0.0000
    quarter-wave diag(1,-i)     F = 0.5000
For a diagonal Jones matrix F_ideal == R_cross exactly (rectangle:
0.2288 = 0.2288). F differs from the incoherent R_cross only for
non-diagonal (C2/FULL) responses, where it additionally demands phase
coherence of the two helicity-conversion amplitudes.

Axis-invariant variant used as the DISCOVERY objective for C2/FULL:
    F_af = (|Rc01| + |Rc10|)^2 / 4 = max_alpha F(U_alpha),
which credits a motif whose principal axes are rotated from x/y and
still equals 1 only for unit-amplitude linear-eigenaxis half-wave
reflection. The principal-axis angle is recovered afterwards
(eigenchannel analysis, pr_qualify.py eigen).

## 2. Rotated operator and sign conventions (spec sec 1, 24)

U_alpha = [[0, e^{-2i alpha}], [e^{+2i alpha}, 0]] (so that
Tr(U_alpha^dag R) = e^{2ia} Rc01 + e^{-2ia} Rc10). Sign VERIFIED on the
paper rectangle physically rotated by +15 deg: angle(Rc01) shifts by
-33 deg (the measured -2 alpha law of the wide-FOV campaign), and
F(U_+15) = 0.231 > F(U_0) = 0.166 > F(U_-15) = 0.051.

## 3. Azimuth-frame correction for oblique incidence (new finding)

The exact p/s Jones frame rotates with the incidence azimuth phi, so a
motif at orientation alpha, illuminated from azimuth phi, presents the
effective orientation alpha - phi to the p/s frame. The fidelity target
at (theta > 0, phi) is therefore U_{alpha - phi}. Verified on newA:
at phi = 30 deg, U_{-30} recovers the phi = 0 fidelity (0.313 vs
0.314) while U_{+30} gives 0.078; at phi = 45 the uncorrected F is
identically ~0 (cos^2(90 deg)) - a pure frame artifact, not physics.
Edge case: at EXACTLY theta = 0 TORCWA's p/s frame is pinned to x/y
(zero transverse k), so no correction applies there (phi_eff = 0).
The previous campaigns used the incoherent, frame-invariant R_cross at
oblique incidence and are unaffected; every coherent oblique fidelity
in THIS campaign uses the corrected target.

## 4. Rotation of the density during optimization

Rotated PB states (spec sec 24) are produced by differentiable
bilinear resampling (grid_sample) of the SYMMETRIZED latent, followed
by filter + projection WITHOUT re-symmetrization (symmetrizing after
rotation would fold the rotated motif onto its mirror images - a bug
caught and fixed in smoke tests, F_rot 0.008 -> 0.27-0.35 for newA).

## 5. Design space (spec secs 9-15)

- padding 15 nm fixed for every period; r_design = P/2 - 15 nm.
- D2: x and y mirror symmetrization; C2: 180-deg rotation only
  (rho + flip(rho, both axes))/2; FULL: none.
- islands: unconstrained during discovery (count recorded).
- filter radius 15 nm, tanh projection beta 2 -> 16, 96x96 density.
- no multipole quantity anywhere in any objective.
- loss: staged continuation (mirror formation -> operator fidelity),
  augmented penalties on T_tot and R_co caps (0.15/0.08 -> 0.10/0.05
  -> 0.08/0.04 in cont55; finalist tier 0.05/0.02 reported, not forced),
  multipliers x1.6 when violated at 15-iter checkpoints, capped at 40.
  NO absorption penalty at any stage.
- Stage-I solves at order [7,7] (P <= 278) / [9,9] (P >= 300);
  every reported number is hard-binary at [9,9]; finalists to [15,15].

## 6. Diffraction (exact, validated conventions)

sin(theta_open) = lambda/P - n_medium. Glass-side first order opens at
theta_air = 70.3 (P264), 60.5 (P272), 55.1 deg (P278); P <= 252 never.
All device periods are specular-only through the PB-useful range
(<= 50-55 deg). Large-P normal-incidence diagnostic periods (300-400
nm) remain specular at theta = 0 in both media (opening needs P >
434 nm in glass); they are labelled CEILING-ONLY, never device
candidates.

## 7. Materials

a-Si Franta 2013, n + ik = 4.2827 + 0.0687i at 633 nm (no clamping);
glass 1.457. Lossless ceiling branch: same Re(n), k = 0 FROM
INITIALIZATION (spec sec 27) - a causality diagnostic, not a device.
