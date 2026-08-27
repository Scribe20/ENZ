# AUDIT_FINAL_ANSWERS — the 15 questions of audit §22

Sources: MULTIPOLE_FAMILY_AUDIT.md (families, purities, classification),
Q_EXTRACTION_AUDIT.md (method audit + refits), results/candidate_ledger_v2.csv,
results/families_at_wavelengths.csv, results/q_validation_v2*.csv,
results/audit/{needle_forensics.json,champion_matrices.json,
channel_recon_champion.csv,mqspec_*.csv}, results/figures/*.png.
No new optimization was run anywhere in this audit.

**1. Which candidates are genuinely ED–EQ dominant with all four families
separated?**
At λ0 = 1332.5 nm, f_ED + f_EQ ≥ 0.80 holds for eight candidates, but six
of them are ED-dominated (f_EQ < 0.2). Genuinely ED–EQ dominant AND
co-excited: **P0550_H0250_seed011** (f_ED+f_EQ = 0.948) and
**P0750_H0350_seed011** (0.872). P0750_H0350_seed029 just misses (0.785,
with f_MQ = 0.071). The champion is NOT ED–EQ dominant (f_ED+f_EQ = 0.375
at λ0, 0.309 at its pole).

**2. Which are genuinely balanced ED–EQ states?**
Exactly two at λ0 (criterion: f_ED+f_EQ ≥ 0.80, each ≥ 0.20, px|ED ≥ 0.80,
Qxz|EQ ≥ 0.80): **P0550_H0250_seed011** (balance B = 0.93) and
**P0750_H0350_seed011** (B = 0.43, ED-leaning). P0650_H0350_seed011
reaches B = 0.99 only at 1292.5 nm — the scan-window edge, not λ0.

**3. Is P0550_H0250_seed011 truly a clean balanced p_x/Q_xz state?**
YES — it survives the complete 4-family partition: f_ED = 0.491,
f_EQ = 0.457, f_MD = 0.052, f_MQ = 0.0004; px|ED = 1.000, Qxz|EQ = 0.847;
and the balanced condition holds over a ≥74-nm band (1298.5–1372.5 nm) —
broadband co-excitation, not a resonant coincidence. Caveat kept from the
family audit: within its small EQ remainder, Qxy carries 15%; and its
my|MD = 0.78 of an immaterial 5% family.

**4. Champion (P0750_H0250_seed011) family fractions at its pole?**
f_ED = 0.072, f_MD = 0.638, f_EQ = 0.237, f_MQ = 0.053 (order [9,9] at
1330.25 nm), order-robust: 0.070/0.640/0.241/0.048 at [11,11]. The
"high-Q ED–EQ resonance" of Stage A is in fact **MD-majority**.

**5. Is its MD component specifically m_y?**
YES: my|MD = 0.971 at [9,9], 0.947 at [11,11] — precisely the component
that is channel-degenerate with Q_xz in the x-polarized specular channel
(∫z J_x → m_y − (iω/6)Q_xz). The degeneracy predicted in the channel
derivation is what materialized.

**6. Is the narrow pole Q_xz-like, m_y-like, or hybrid?**
**Hybrid, m_y-leaning by radiation weight** (C_my : C_Qxz ≈ 2.7 : 1 at the
pole), and the formal channel reconstruction shows the two terms
anti-phase-locked at 179–180° across the entire resonance with their
magnitude ratio crossing 1.00 at 1332.3 nm, where the net odd channel
collapses to 0.2% of its parts. The mode's radiative darkness IS internal
m_y↔Q_xz destructive interference. "EQ dark mode" (Stage A) is corrected
to "dark m_y/Q_xz hybrid odd mode".

**7. Corrected, spectrally resolved Q values?**
Adaptive joint t+r shared-pole refits (Q_RESOLVED = all gates pass):
* P0550_H0350_seed011: **Q = 280.0 ± 0.1** @ 1331.90 nm — RESOLVED
* P0650_H0350_seed011: **Q = 267.4 ± 0.1** @ 1331.94 nm — RESOLVED
* Champion P0750_H0250_seed011: **Q = 357.6 ± 1.2** @ 1330.38 nm —
  RESOLVED (v1 said 559.5: a −36% correction)
* Needle P0750_H0350_seed029: **Q = 276.2 ± 0.6** @ 1309.80 nm — RESOLVED
* P0550_H0150_seed011: fit 881–903, UNRESOLVED (energy gate; see Q13)
* P0750_H0250_seed029: fit 543–545, UNRESOLVED (energy gate, marginal)
* Remaining 12 candidates: no qualifying resonance (broad/no fit).

**8. Which Stage-A Q values survive adaptive spectral refinement?**
As numbers: all five 100 < Q < 1e5 fits reproduce within −3% to −10%
except the champion (−36%). As certified claims: only the four RESOLVED
rows above. The needle's 1.4e7 does NOT survive (→ 276). Trajectory Q's
(620→165) survive as a TREND (the α = 1.0 cross-check bounds the 0.25-nm
method error at ~3%) but are not precision values.

**9. Which survive higher Fourier order?**
Champion: 357.6 → 346.6 → 336.6 at [9,9]→[11,11]→[13,13] (−3% per step,
shrinking; pole drift 1330.4→1329.3→1328.4 nm), Q_RESOLVED at every
order; family fractions order-stable. Needle: 276→262 with pole
1309.8→1303.9 nm, RESOLVED at both. The two energy-gate failures are
fit-stable across orders (+2.5%, +0.5%) but their energy violation GROWS
with order (0.041→0.236; 0.0075→0.0212) — they fail certification at all
tested orders. ([15,15] champion row + grid/origin matrices:
results/audit/champion_matrices.json.)

**10. Is P0750_H0350_seed029 a real ultrahigh-Q pole or an artifact?**
**Artifact — definitively.** Adaptive refinement to 5e-4-nm local steps
finds an ordinary Q ≈ 276 (RESOLVED, [9,9]) / 262 ([11,11]) resonance; no
sub-nm feature exists in any refined window; a complex-frequency Newton
probe honestly failed to converge (recorded, moot); single-pixel erosion
shifts the resonance smoothly (1271.7 nm, Q ~ 171 indicative). The 1.4e7
came from the v1 constant-background t-only fit on a 1-nm grid.

**11. Were Q_rad, Q_loaded, Q_abs independent or algebraically inferred?**
Q_rad and Q_loaded: independently fitted (separate spectra). **Q_abs:
purely algebraic inference** 1/Q_abs = 1/Q_loaded − 1/Q_rad, never
validated against P_abs = (ωε0/2)∫Im(ε)|E|²dV — neither in Stage A nor
in this audit. Every Q_abs statement carries this label; the benign
Stage-A conclusion ("absorption not limiting at k = 1e-4") remains
plausible because Q_loaded ≈ Q_rad within a few %, which is a direct
(non-inferred) observation.

**12. Does the final Q agree between two independent extraction methods?**
For every RESOLVED candidate, yes: v1 single-pole t-only fit vs audit
joint t+r shared-pole fit agree within 3–10%; champion also has the
independent 0.25-nm trajectory fit (≈370 vs 357.6, 3%); needle and
champion agree across two Fourier orders within 5%/3%. The two UNRESOLVED
candidates also agree across methods/orders — method agreement is
necessary but was correctly NOT accepted as sufficient (energy gate).

**13. Does the periodic energy balance close at resonance?**
For the four RESOLVED candidates: yes — worst in-window |T+R−1| =
9.0e-4 (needle), 2.4e-3, 2.4e-3, 2.6e-3 (champion) against the 5e-3 gate.
For P0550_H0150_seed011: NO (0.041 at [9,9], worsening to 0.236 at
[11,11]); P0750_H0250_seed029: NO (0.0075 → 0.0212). Those two Q values
are therefore reported as apparent, uncertified.

**14. Which Stage-A claims remain valid?**
* Prescribed-multipole freeform synthesis works; the objective's exact
  moments are what the qualification finds (no proxy false positive).
* Two candidates hold genuine clean balanced p_x+Q_xz co-excitation at
  λ0; P0550_H0250_seed011 is broadband-balanced (Q3).
* A sharp, order-robust resonance emerged with Q never optimized;
  corrected Q = 357.6 ± 1.2 (RESOLVED at three orders).
* The causal trajectory conclusion: Q tracks the dark mode's radiative
  darkness, NOT ED–EQ spectral alignment; Δφ_rad never approaches
  destructive. (Strengthened by the audit: the darkness mechanism is now
  identified — internal m_y/Q_xz anti-phase cancellation.)
* The needle was correctly flagged "unresolved, not claimed" in Stage A.
* Toroidal kept diagnostic-only — correct (exact kernels already contain
  it inside p).

**15. Which claims must be weakened or corrected?**
* Champion re-labeled: "bright-ED background + dark-EQ mode" → bright-ED
  background + **dark m_y/Q_xz-hybrid odd mode (MD-majority by radiation
  weight)**; it is NOT an ED–EQ-dominant state (f_ED+f_EQ = 0.31 at pole).
* Champion Q: 559.5 → **357.6 ± 1.2** (v1 method −36% error).
* "ED-only control Q ≈ 905" → apparent Q ~ 900, **uncertified** (energy
  violation grows with order); same for the 597 (→ ~545, uncertified).
* Needle Q ~ 1.4e7: **voided** (ordinary Q ≈ 276).
* All Stage-A 3-family fractions (EDEQ_frac/MD_frac): superseded by the
  4-family partition; P0650_H0350_seed029 is disqualified from any
  dipole/EQ narrative (f_MQ = 0.30 at λ0, 0.49 at pole, Qm_yz-dominated).
* Any Q_abs number: algebraic inference only (Q11).
* Detuning-trajectory Q values: trend-quality, not precision; α = 1.0
  point superseded by 357.6.

Per §22: Stage A2 may begin only now that these answers are recorded.
