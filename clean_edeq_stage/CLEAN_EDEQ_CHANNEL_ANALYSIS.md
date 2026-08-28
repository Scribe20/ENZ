# CLEAN_EDEQ_CHANNEL_ANALYSIS — periodic radiation-channel analysis of P0550_H0250_seed011

Data: results/p0550_multipole_spectra.csv (161 λ, 1260–1420 nm, 1-nm),
results/p0550_channel_amplitudes.csv, results/p0550_models_tr.csv,
results/p0550_channel_summary.json; contrast: the same files for p0750.
Settings: audited conventions (order [9,9], 48×48×9 grid, canonical
origin, Franta-2013 a-Si, x-pol normal incidence from silica). No new
optimization. Two isolated points (1267, 1269 nm) carry |T+R−1| up to
1.5e-2 (sharp parasitic feature far outside the functional band) and are
excluded from precision statements; everywhere else |T+R−1| < 5e-3.

## 1. Channel formalism (Phase 2–3)

Scattered amplitudes are defined against the bare-stack background
(empty patterned layer): t_sc = t_full − t_bg, r_sc = r_full − r_bg.
Exact induced-current channel integrals E_up = −(Z0/2A)∫J_x e^{−ikz_c}dV
and E_dn (+ikz_c) are computed per wavelength, along with the ladder
terms even_px (0th order), odd_m (m_y), odd_Q (Qe_xz) and the complete
2nd-order integral I2 = ∫z_c²J_x dV (which contains Qm_yz PLUS electric
octupole and mean-radius corrections — labeled "2nd-order", never "MQ").

**Port couplings verified against TORCWA, not assumed** (Phase 2
requirement): the single complex constants g_up = E_up/t_sc and
g_dn = E_dn/r_sc, fitted as band medians:

| coupling | measured | textbook expectation | note |
|---|---|---|---|
| g_up | 1.253 ∠−34.0° | √n_sub = 1.202, ∠−kh/2 = −33.8° | +4.2% modulus |
| g_dn | 1.231 ∠−35.8° | √n_sub, ∠≈−kh/2 | +2.4% modulus |

Within 1300–1400 nm the per-λ |g_up| varies only 1.239–1.269 (±1.2%):
the constant-g layered-port picture is solid in the functional band; the
larger deviations (max 36%) occur only at the scan edges / the parasitic
1267-nm feature. (The P0750 reference gives g_up = 1.391 ∠−37.0° — the
coupling is geometry-dependent at the several-% level via the layered
background, which is precisely why it must be measured, not assumed.)

Channel-normalized amplitudes (TORCWA t-units): A_X^top = X/g_up,
A_X^bot = ±X/g_dn with the parity sign (odd terms flip for downward
radiation). Consequence: Δφ_bot = Δφ_top + 180° exactly — ED–EQ
interference that is constructive in one port is destructive in the
other. This parity structure, not any resonance, is the root of the
directionality below.

## 2. Ladder and model fidelity (Phase 4)

Reconstruction errors vs full TORCWA, band-averaged, per unit incident
amplitude (error_t_abs / error_r_abs); relative-to-channel-mean values in
the JSON (the r-channel relative numbers are inflated by the small |r|
across this low-R band):

| model | t error (abs) | r error (abs) | reading |
|---|---|---|---|
| X: exact integral + background | 0.048 | 0.018 | induced-current + bare-background picture reproduces full TORCWA to ~5% over 160 nm — framework validated |
| A: ED only | 0.88 | 0.98 | hopeless — ED alone explains nothing |
| B: ED+EQ | 0.28 | 0.19 | captures the structure, 2–3× short |
| C: ED+EQ+MD | 0.105 | 0.155 | good |
| D: C + full 2nd order | 0.114 | 0.144 | marginal further change |
| noED (EQ+MD+2nd) | 0.98 | 0.85 | removing ED destroys everything |
| noEQ (ED+MD+2nd) | 1.19 | 1.26 | removing EQ destroys everything |

Reading (Phase 4 verdict): the far field is genuinely ED–EQ-mediated —
deleting either family is catastrophic — but **ED+EQ alone is NOT
quantitatively sufficient**: m_y (a 5% family fraction) contributes
materially to the odd channel (t error 0.28 → 0.105 when included). The
same lesson as the Stage-A audit, now on the bright side: the odd channel
is always the m_y + Qe_xz pair. Median ladder truncation residual vs the
exact integral: 6.3% (up port).

## 3. Frozen-state interference structure (Phase 5)

At λ0 = 1332.5 nm: |A_ED^top| = 1.04, |A_EQ^top| = 1.19 (ratio 1.15 —
amplitude-matched to 15%), Δφ_top = −33.9° → ξ_top = 0.957 (nearly fully
constructive up), while Δφ_bot = +146.1° → ξ_bot = 0.299 (strongly
destructive down). Scattered-power directionality (port-impedance
corrected, P_top = |E_up|², P_bot = n_sub|E_dn|²): η_dir = +0.78 at λ0,
0.77–0.89 over the clean band, maximum 0.989 at 1414 nm. R stays
0.085–0.165 over the clean band; T 0.83–0.91.

**R minimum**: R = 0.0132 at 1414 nm, where Δφ_bot = 171° (near-perfect
intra-scattered cancellation, ξ_bot = 0.144) AND the residual scattered
wave partially cancels the background (|r_sc| = 0.111 vs |r_bg| = 0.187,
|r_total| = 0.115). The suppression is therefore **primarily
intra-scattered ED–EQ(+MD) destructive interference, with a secondary
background-assisted component** — per the integrity rule, "ED–EQ
mediated" is justified because removing either ED or EQ from the
reconstruction destroys the cancellation (§2), but the honest full
statement includes m_y and the background term.

**Cancellation condition (Phase 11, derived not assumed)**: for the
bottom port, r_total = r_bg + [even − odd]/g_dn ≈ 0. At 1414 nm the
measured terms give |even−odd|/g_dn = 0.111 at the phase that opposes
r_bg = 0.187∠(arg r_bg); exact zero requires the scattered term to reach
−r_bg — i.e. BOTH Δφ_bot → ~180° AND a magnitude condition tied to the
background, NOT the naive free-space Δφ = 180° alone. The naive condition
is close here only because |r_bg| is small (bare silica–air Fresnel).

## 4. P0550 vs P0750 (Phase 7): composition vs interference

| | P0550 (clean reference) | P0750 (dark reference) |
|---|---|---|
| composition at λ0/pole | f_ED 0.491 / f_EQ 0.457 / f_MD 0.052 / f_MQ 0.0004 | f_ED 0.072 / f_MD 0.638 / f_EQ 0.237 / f_MQ 0.053 |
| purity | px&#124;ED = 1.00, Qxz&#124;EQ = 0.85 | my&#124;MD = 0.97, Qxz&#124;EQ = 0.97, px&#124;ED = 0.20 |
| bandwidth | ≥121 nm dominance band; 73 nm clean band | FWHM 3.7 nm (Q = 358) |
| odd-channel structure | ED–EQ amplitude-matched; parity-split ports (ξ_top 0.96 / ξ_bot 0.30) | m_y↔Q_xz internally anti-phased (net odd 0.2% of parts) |
| model errors (abs, t) | ED-only 0.88 → ED+EQ 0.28 → +MD 0.105 | ED-only 0.80 → ED+EQ **1.01** → +MD 0.27 |
| η_dir | +0.78 at λ0 (forward), up to 0.989 | −0.30 at the pole (backward-leaning) |
| R at operating point | 0.085–0.165 (min 0.013) | ≥ 0.38 (reflective resonance) |

The P0750 row "ED+EQ worse than ED-only" is the signature of its
anti-phased m_y/Q_xz pair: including Q_xz without its cancelling partner
m_y misrepresents the field. The two structures thus demonstrate two
distinct design dimensions — P0550: composition control with parity-split
port interference (bright, broadband, forward); P0750: emergent
phase-locked intra-channel cancellation (dark, narrow). Interpretations
kept separate per the stage contract.

## 5. Phase-9 sensitivity result (design-dimension independence)

At λ0, autograd gradients on the 64×64 density give
cos∠(∇φ_ED–EQ, ∇log(C_ED/C_EQ)) = 0.24 — the phase and balance gradients
are largely independent directions in design space. Knob comparison
(merit = deg of Δφ per 0.01 change of log-balance):
thickness −0.81°/nm at ~2e-4 balance/nm (merit 47.7) ≫ Gram–Schmidt
pixel direction (12.8) ≫ index (2.1) ≫ scale (0.57). β = h selected for
the Phase-10 sweep; reaching Δφ_bot = 180° at λ0 predicts h ≈ 208 nm.
