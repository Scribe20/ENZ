# CLEAN_EDEQ_FUNCTION_REPORT — what the clean balanced p_x/Qe_xz state is actually useful for

Stage-B report. References: P0550_H0250_seed011 (composition reference,
frozen; snapshot + SHA256 in results/p0550_frozen_snapshot.json) and
P0750_H0250_seed011 (dark-state contrast). Methods and channel formalism:
CLEAN_EDEQ_CHANNEL_ANALYSIS.md. Prior art: CLEAN_EDEQ_NOVELTY_AUDIT.md.
Machine-readable: p0550_multipole_spectra.csv, p0550_channel_amplitudes.csv,
p0550_models_tr.csv, p0550_phase_sweep.csv, p0550_robustness.csv,
p0550_angle_scan.csv, p0550_ypol_rt.csv, p0550_gradient_diag.json,
p0550_channel_summary.json (+ p0750 counterparts), figures/figA…L.
No topology optimization was launched in this stage.

## Headline

The frozen P0550 state is already a **broadband forward-directional
(Kerker-type) scatterer** (η_dir = 0.77–0.89 over its clean band), and a
single interpretable knob — slab thickness — steers the ED–EQ channel
phase through exact backward cancellation while preserving the clean
composition. At h = 225–235 nm the family reaches the joint state the
stage was designed to look for: **clean balanced ED–EQ composition AND
broadband reflection suppression simultaneously** (R ≤ 0.05 over
95–130 nm with η ≥ 0.91 and the clean criterion co-holding over
115–125 nm). Composition and radiative phase behave as approximately
independent design dimensions (measured gradient angle: cos = 0.24).

## Phase-23 decision gate

**GO — generalized Kerker / broadband directional-scattering branch**
(Branch A + C: reflection suppression with top-port redistribution),
with the thickness-tuned variant h ≈ 227 nm as the working point, and a
secondary future option on the field-gradient branch (modest 8.4×
interface gradient enhancement). The active phase-switching branch is a
**conditional no** at realistic index perturbations (trimming, not
switching — see Q15). No new topology optimization is required for the
primary branch (Phase 24 not triggered).

## The 20 required answers (Phase 22)

**1. Is P0550 still a genuine clean balanced ED–EQ state under final
settings?** Yes. Revalidated against the audited ledger to ≤1.3e-4
absolute in every field (float32 solver noise; snapshot + checksums
frozen). Class: clean_balanced_ED_EQ.

**2. Exact fractions at λ0 = 1332.5 nm:** f_ED = 0.491, f_EQ = 0.457,
f_MD = 0.052, f_MQ = 0.0004 (complete exact partition, sum = 1 exact;
toroidal kept diagnostic-only, CT/C_ED = 6.6e-4).

**3. Over what range is it genuinely balanced?** B_ED_EQ ≥ 0.8 over
141 nm (1279–1420, non-contiguous); the full clean criterion
(f_ED, f_EQ ≥ 0.2, sum ≥ 0.8, both purities ≥ 0.8) holds contiguously
over **1314–1387 nm (73 nm)** on the frozen geometry; the dominance
band f_ED+f_EQ ≥ 0.8 spans 1299–1420 (121 nm, edge-limited).

**4. p_x dominant inside ED throughout?** Yes — px|ED ≥ 0.9996 across
the entire 1260–1420 nm scan (never drops).

**5. Qe_xz dominant inside EQ?** Yes in-band: Qxz|EQ ≥ 0.80 over
1314–1387 (max 0.86); it degrades outside (0.62 at 1290, 0.78 at 1400 —
this is what terminates the clean band, not the fractions).

**6. Actual channel-normalized complex phase relation:** with port
couplings measured against TORCWA (g_up = 1.253∠−34.0°,
g_dn = 1.231∠−35.8°; ±1.2% stable in-band; NOT free-space values —
geometry-dependent at the few-% level), the ED–EQ relative phase at λ0
is Δφ_top = −33.9° and, by the exact parity flip, Δφ_bot = +146.1°.
Amplitudes |A_ED| = 1.04, |A_EQ| = 1.19 (matched to 15%). The phase
drifts smoothly by ~25° across the band.

**7. Does the frozen geometry already show Kerker-like directionality?**
Yes: scattered-power directionality η_dir = +0.78 at λ0, 0.77–0.89 over
the clean band (port-impedance-corrected: P_top = |E_up|²,
P_bot = n_sub|E_dn|²), peaking at 0.989 at 1414 nm where R = 0.013.
R = 0.085–0.165 in-band on the frozen h = 250.

**8. Can ED+EQ reconstruct the full TORCWA R/T quantitatively?**
Partially. Per-unit-incident amplitude errors: ED-only 0.88/0.98 (t/r),
ED+EQ 0.28/0.19, ED+EQ+MD 0.105/0.155, +full-2nd-order 0.114/0.144;
the exact current integral + bare background reaches 0.048/0.018 —
validating the framework. ED+EQ captures the structure but is 2–3×
short quantitatively.

**9. How important are MD and MQ despite small fractions?** MD is
materially important: adding m_y (a 5% family) cuts the t-error from
0.28 to 0.105 — the odd channel is always the m_y + Q_xz pair (the same
lesson as the Stage-A dark state, here on the bright side). The full
2nd-order term (which contains Qm_yz plus octupole corrections) changes
little (f_MQ = 4e-4). Honest statement: the response is ED–EQ-mediated
(removing either is catastrophic: errors ≥ 0.85) but not ED–EQ-exclusive.

**10. Balance broadband but interference narrowband?** On the FROZEN
geometry: balance is broadband and the useful interference is
broadband-partial (η ≥ 0.77 everywhere in-band) with the deep
cancellation only near the band edge (1414 nm). On the THICKNESS-TUNED
family (h = 225–235): **both are broadband simultaneously** — R ≤ 0.05
over 95–130 nm with clean composition over 115–125 nm and η ≥ 0.91.

**11. Can a local perturbation tune phase without destroying balance?**
Yes — this is the stage's central positive result. Sensitivity analysis
first (Phase 9): cos∠(∇φ, ∇balance) = 0.24 over the 64² design space;
knob comparison ranks thickness at −0.81°/nm with ~2e-4 balance/nm
(merit 47.7) above pixel-direction (12.8), index (2.1), scale (0.57).
The h-sweep then shows Δφ_bot sweeping −153° → +122° (through ±180° at
h ≈ 221 nm) while f_ED+f_EQ ≈ 0.99 and px|ED = 1.00 throughout;
B dips to 0.75 at the crossing and recovers to 0.85–0.93 by h = 227–250.

**12. Does a true cancellation point occur?** Yes. At h ≈ 221,
λ0: ξ_bot = 0.018 (the coherent ED+EQ bottom-port sum collapses to 2%
of its parts). The total-field R minimum sits at h ≈ 227.5 (R = 0.020),
offset from the naive 180° point because the substrate background term
shifts the true zero — measured offset ≈ 9° / +6 nm of h. On the frozen
geometry the same physics gives R = 0.013 at 1414 nm.

**13. Which physical channel is suppressed?** The backward/specular
reflection channel (bottom port, into the substrate). Transmission is
correspondingly enhanced (T = 0.96–0.98 at the tuned points) — forward
redistribution, not absorption (a-Si loss k = 6e-6 is negligible).

**14. Is the suppression ED–EQ interference, background-assisted, or
other?** Primarily intra-scattered ED–EQ(+m_y) destructive interference
— removing ED or EQ from the reconstruction destroys the cancellation
(the mandated test) — with a secondary background-assisted component:
at the frozen R-min, |r_sc| = 0.111 against |r_bg| = 0.187 combine to
|r| = 0.115; the exact cancellation condition is
r_bg + [even−odd]/g_dn ≈ 0, not Δφ = 180° alone.

**15. Can a small index perturbation switch the state?** No — it trims.
Δn/n = ±1.4% (≈ ±0.05 in n, generous for real active materials) moves
R at λ0 by ∓18/+11% relative (0.160 → 0.130/0.177) and translates the
band edge; composition is preserved (B ≥ 0.91 in-band except a
band-edge crossing point). The broadband balance is index-ROBUST — the
flip side is weak switchability. Genuine switching would need the
resonant (P0750-like) branch or an ENZ overlay (Phase 14: not pursued —
the simple test did not justify it).

**16. Is P0550 significantly more robust than the P0750 dark state?**
Yes, dramatically. P0550 clean-band function survives every tested
perturbation: corner rounding and ±5 nm thickness are free (B_min ≥
0.92); ±10 nm thickness and ±1.4% index translate the working point but
preserve composition; the worst axis is 1-px (±8.6 nm) lateral etch bias
(band re-centers; worst-λ balance dips to 0.34/0.78 at fixed λ).
Function is angle-robust to 10° (in-band R_mean unchanged at 0.133;
B_min 0.73 at 10°, no new diffraction orders below θ = 10°). The P0750
dark resonance under a single-pixel erosion moved 38 nm ≈ 8 linewidths
(completely detuned). Low-Q broadband balance is demonstrably a
robustness feature.

**17. Does Qe_xz produce a useful field-gradient distribution?**
Modestly. The symmetrized gradient (∂E_x/∂z + ∂E_z/∂x)/2 that the Q_xz
interaction samples reaches 8.4× the incident-planewave gradient scale
at the open top interface (volume p99: 3.4×) — a real, broadband,
surface-accessible gradient platform, far below resonant-cavity records.
Promising for robustness-first gradient-coupling experiments; no
light–matter enhancement claims made.

**18. Most natural application of this geometry?** A robust, broadband,
polarization-selective transmissive Huygens-type element: near-total
forward scattering (R ≤ 0.05 over ≥ 95 nm at h ≈ 227–235) built from a
verified balanced p_x/Qe_xz state — i.e., a quadrupole-assisted
anti-reflection / directional layer whose multipolar content is
certified, with the thickness family as a phase-trim dimension, plus a
secondary role as a gradient-field substrate for future higher-order
coupling.

**19. What is genuinely novel?** (CLEAN_EDEQ_NOVELTY_AUDIT.md) Not
ED–EQ Kerker or broadband Kerker per se. The defensible core: freeform
synthesis of a verified broadband balanced p_x/Qe_xz state; the
measured near-orthogonality of composition and radiative phase as
design dimensions with an interpretable knob sweeping phase through
cancellation at preserved composition; the joint
clean-composition + broadband-cancellation state; and the two-reference
composition-vs-interference demonstration from one design framework.

**20. What claims are not justified?** "First/novel ED–EQ generalized
Kerker" (exists); "broadband Kerker" as a standalone novelty (exists);
active switching (only trimming shown); any quantum/emitter enhancement
(diagnostic only); ED–EQ *exclusivity* of the far field (m_y matters);
free-space Kerker formulas (the background-corrected condition differs
by a measured 9°); any statement at 1267–1269 nm (energy-flagged
points); and any high-Q framing of P0550 (contract: its virtue is the
opposite).


## Phase-18 note — metagrating extension (analysis only, no modification)

At λ0 = 1332.5 nm with P = 550 nm, all nonzero diffraction orders are
closed (first substrate order requires P > λ0/n_sub = 922.7 nm; first air
order requires P > λ0 = 1332.5 nm), and they remain closed to θ = 10°.
A 2×1 supercell (P_x = 1100 nm) would open the substrate ±1 orders only
(air orders still closed): a natural future metagrating platform in
which the p_x/Q_xz radiation-pattern asymmetry (their different angular
lobes and parities) could bias +1 vs −1 substrate diffraction. A 3×1
supercell (1650 nm) opens air ±1 as well. This is recorded as a future
direction only — the current zero-order broadband-Kerker functionality
already justifies the branch without structural modification.

## Integrity notes

Never equated: ED≈EQ with Kerker (cancellation demonstrated + removal
test passed); low Q with poor design (robustness quantified); raw
multipole phases were never compared (all phases channel-normalized with
TORCWA-verified couplings); the substrate background enters the
cancellation condition explicitly; port-impedance corrections applied to
all power ratios; P0550 and P0750 interpretations kept separate
throughout.
