# P0750_TRANSMISSION_ZERO_AUDIT — the resonant-mirror state (Phases I-L)

Data: p0750_highres_rt.csv (1300-1360 at 1 nm + 1328-1340 at 0.2 nm,
full observable rows), p0750_t_argand.csv, p0750_component_removal.csv,
p0750_order_tmin.csv, p0750_smatrix_vs_lam.csv; Figures ps_fig3-5.

## I. Resolved minimum
T_min = 0.01077 at 1333.98 nm (0.2-nm core, parabolic; ~20 samples per
FWHM ~ 4 nm). R = 0.9884, A = 0.00087 (real k = 6e-6 loss model) ->
**resonant mirror, not absorptive** (Phase R). Per-order true minima
(0.5-nm scans): 0.0108 [9,9] @1334.0, 0.0072 [11,11] @1333.0,
0.0119 [13,13] @1332.0, 0.0164 [15,15] @1332.0 - depth stable at
~0.01, pole blue-shifting ~1 nm/order (known systematic).

## Polarization anatomy (discovered here)
At T_min: co-pol |t_xx|^2 = 1.8e-4 (|t_xx| = 0.0136 - the complex
amplitude genuinely approaches the origin, from |t_bg| = 0.98);
cross-pol |t_yx|^2 = 0.0106 (flat, non-resonant across the line);
R_xx = 0.950, R_yx = 0.039 (resonant). The total-T floor is therefore
CROSS-POLARIZATION CONVERSION (~1%), not co-pol leakage and not
absorption. Strict statement: a co-polarized transmission near-zero
inside a weakly polarization-converting mirror.

## J. t-plane Argand (Figure 4/5)
At 1334.0: t_bg (0.98) + t_ED walks to 0.48-amplitude; the giant t_MD
(|~2.4|) and t_EQ (|~1.9|) nearly cancel each other; the cumulative
ladder lands at |t| ~ 0.40 (T_model 0.163) with the dotted truncation
arrow closing to the full-TORCWA star at |t_xx| = 0.014 sitting on the
origin. Ladder-to-2nd-order truncation is large at this strongly
resonant point (documented); the exact current integral closes the gap
by construction. The mechanism reading is family-level, not
single-term-level.

## K. Removal tests (T_model at lam_Tmin)
bg only 0.965 | bg+ED 0.230 | bg+MD 11.32 | bg+EQ 6.12 |
bg+MD+EQ 0.908 | bg+ED+EQ 5.52 | bg+ED+MD+EQ 0.163 | full 1.8e-4.
Remove-one: -ED 0.950, -MD 5.52, -EQ 10.45. Verdicts: the my/Qxz pair
is the essential resonant engine (each alone overshoots catastrophically;
together they nearly self-cancel); ED provides the background-cancelling
scale (without it the ladder returns to bg). No family is dispensable
except the 2nd-order term (removal changes nothing at this order).

## L. Internal vs external cancellation
Internal my/Qxz null (min |odd_m + odd_Q| / (|odd_m|+|odd_Q|) = 0.0029)
at 1333.6 nm; T_min at 1334.0 nm: offset +0.4 nm, resolved at 0.2-nm
sampling. The T-zero is NOT the internal-darkness point: the total t
must cancel the direct background, shifting the operating point -
the t-plane analogue of P0550's h = 221 vs 227.5 offset in the r-plane.
