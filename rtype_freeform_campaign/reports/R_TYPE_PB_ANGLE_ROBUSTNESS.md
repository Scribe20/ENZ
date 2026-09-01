# R_TYPE_PB_ANGLE_ROBUSTNESS

PB rotation (0-180 deg, 15-deg steps, hard-binary rotations, all inside
the fixed circular envelope - zero pixels outside at every angle):
slopes -1.94..-1.98 deg/deg vs ideal -2; phase rms 4.6-10.5 deg;
R_cross rotation ripple 0.33-0.54 (worst B_P271_H230). All six
finalists are valid PB elements. A-method has the best rotation
fidelity (rms <= 7.4 deg).

Angle scan bookkeeping: the sweep variable was the GLASS-side input
angle; the device (air-side) angle is theta_air = asin(1.457 sin
theta_g). Valid range theta_g = 0-40 deg covers theta_air = 0-69.5 deg;
theta_g >= 45 deg rows are total-internal-reflection artifacts
(R=T=0 exactly) and support no claims.

Result (phi = 0/45/90): R_cross falls steeply with angle for every
candidate: at theta_air ~ 22 deg it is 0.12-0.48 (best at phi=90),
by 38-57 deg it is 0.02-0.34. A dedicated Stage-2 worst-case
multi-angle re-optimization (theta_air 0/22/38/53) preserved
normal-incidence quality (A: 0.518; B: 0.495) but lifted oblique
R_cross only to 0.11-0.16. Conclusion: the freeform advantage is
established at and near normal incidence; strong +-60 deg cross-
conversion is not reachable in this single-layer design space and
would need a different architecture (thicker/multilayer or
supercell-level design). No candidate opens diffraction orders within
0-60 deg (P=271 margin: opens at 60.5 deg - flagged as tight).
