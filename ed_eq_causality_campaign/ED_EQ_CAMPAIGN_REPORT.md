# ED_EQ_CAMPAIGN_REPORT — Stage A (pilot + causal trajectory)

Contract: SCIENTIFIC_CONTRACT.md (frozen before discovery). Methods and
gates: METHOD_VALIDATION.md, ED_EQ_CHANNEL_DERIVATION.md. Data:
results/candidate_ledger.csv, results/qualify/*, results/detuning/*,
results/detuning_trajectory.csv, results/q_summary.csv.

## Verdict (contract §34): **CONDITIONAL GO — with a mechanistic redirect**

The pilot proved the *method* (prescribed-multipole freeform synthesis
works; a genuine bright-ED/dark-EQ high-Q Fano resonance emerged with Q
never optimized) and produced a clean causal trajectory — but the
trajectory does **not** support the headline hypothesis in the tested
family: Q tracks the EQ mode's radiative *darkness*, not ED–EQ spectral
alignment, and the ED–EQ channel phase never approaches the destructive
condition. Continuation should redirect to (a) a two-parameter family
decoupling alignment from scale, and/or (b) families where Δφ_rad crosses
±180°. This negative-on-the-headline result is preserved as a primary
finding, per contract.

## What was run (all gates passed before discovery)

* Material: a-Si Franta-2013 (n = 3.652 + 6e-6i at 1332.5 nm; brackets
  Pierce 3.51 / Karaman 3.81; no clamping — the previous campaign's
  clamped-material mistake is structurally excluded).
* Frozen target: p_x + Qe_xz (derived: the unique EQ partner of p_x in
  the x-polarized specular channel; m_y is channel-degenerate with Q_xz
  and is always reported alongside).
* Objective: F_ED_EQ = ½[log S_px + log S_Qxz] on EXACT differentiable
  current multipoles (torch closed-form Alaee kernels — no surrogate gap;
  validated to 2.6e-7 vs the corrected MENP port; channel identity vs
  TORCWA validated to 1.3%). Q, linewidth, T/R, phase: excluded.
* Pilot: 18 runs, P ∈ {550,650,750} × h ∈ {150,250,350} × 2 seeds,
  specular-only regime (n_sub·P/λ₀ ≤ 0.82), no symmetry operations.
* Qualification: 81-point λ scans ×3 material scenarios per candidate,
  exact moments, channel amplitudes, pole-fit Q (recorded only).
* Causal test: 13-point uniform in-plane-scale detuning trajectory of the
  champion, all-observable.

## The 30 questions (Stage-A answers)

1. **Did we create the targeted p_x + Q_ij state?** Yes. Multiple pilot
   candidates hold genuine simultaneous p_x and Qe_xz content at λ₀; the
   champion (P0750_H0250_seed011) realizes it as a sharp EQ-driven
   resonance on an ED background; others realize broad co-excitation.
2. **Which EQ component?** Qe_xz (derived, then confirmed: purity up to
   1.00 within EQ).
3. **p_x dominant within ED?** Champion at λ₀: px = 53% of |p|² (py/pz
   contamination present); the purest co-excited candidates: 94–100%.
4. **Q_xz dominant within EQ?** Champion: 97%; best candidates 85–100%.
5. **ED+EQ globally significant?** Champion: ED+EQ = 39% with MD = 61% at
   λ₀ (m_y rides the same odd channel — expected from the derivation);
   purest co-excited: ED+EQ up to 95%, MD 5%.
6. **Peaks overlapping?** Champion: EQ feature at 1331 nm vs λ₀ within
   1.5 nm; broad-overlap candidates: splitting 7–10 nm inside much wider
   lines (split/FWHM 0.09–0.12).
7. **Resolved splitting:** see ledger (0–80 nm across the ensemble).
8. **Linewidths:** champion FWHM_EQ 2.25–8.75 nm along the trajectory;
   broad candidates ≥ 80 nm.
9. **Was Q in the objective?** No — never (contract-enforced; the FoM code
   contains no Q/linewidth/T/R/phase terms).
10. **What Q emerged?** Champion Q_rad ≈ 620…165 along the trajectory
    (≈370 at the λ₀-aligned point); loaded Q with k = 1e-4 within a few
    % of Q_rad (absorption not limiting at this Q scale). One pilot
    candidate carries an unresolved sub-nm needle (fitted Q > 1e6 —
    UNRESOLVED at 1-nm sampling; flagged, not claimed).
11. **Higher than controls?** Mixed: ED-only control fitted Q ≈ 905 (a
    different narrow feature), broad co-excited controls have no
    resonance at all (no Q), detuned EQ-heavy candidates Q ≈ 290. High Q
    does NOT preferentially attach to ED–EQ-overlap candidates.
12. **Does high Q correlate with reduced leakage?** Yes trivially through
    the linewidth (Q ~ 1/FWHM confirmed), but the leakage reduction is
    governed by EQ darkness (small α ⇒ darker EQ ⇒ narrower line), not by
    ED–EQ alignment.
13. **ED–EQ relative radiation phase:** Δφ_rad drifts −116° → −104° over
    the whole trajectory — never near ±180° (destructive).
14. **Does the outgoing-field reconstruction show destructive ED–EQ
    interference?** No. The exact channel integral is validated (1.3%),
    and the phase observable shows no cancellation regime in this family.
15. **Which channel is suppressed?** On resonance the transmission
    channel is suppressed (T → 0.01–0.35, R → 0.65–0.99) — reflective
    resonance, not radiative cancellation.
16. **Local / lattice / substrate / symmetry?** Largely LOCAL multipolar:
    the resonance survives substrate removal (T-dip persists, shifted
    1335→1305 nm, weakened) — unlike the previous campaign's
    substrate-bound modes; safely below all diffraction cutoffs; no
    symmetry imposed or present.
17. **One mixed mode vs two coupled modes / Fano / EIT / quasi-BIC /
    Kerker?** Best described as a **bright-ED-background + dark-EQ-mode
    Fano resonance**. Not claimed as EIT (no group-delay analysis yet);
    not claimed as quasi-BIC (no leakage-zero parameter identified —
    §26 criteria not met); not Kerker (no channel cancellation).
18. **Does the detuning trajectory support the proposed mechanism?**
    NO for the headline (alignment→destructive-interference→Q): Q is
    monotone in α, not peaked at alignment. YES for an alternative,
    well-defined mechanism: Q is set by the EQ state's radiative
    coupling, which the scale parameter tunes smoothly (λ/P controls
    darkness).
19. **Q_rad rises as P_rad falls?** Yes in the linewidth sense (γ_rad
    falls monotonically with decreasing α; stored-energy proxy U rises
    in lockstep, 622→1329).
20. **Fourier-order convergence:** resonance position drifts 1336→1332 nm
    and depth stays ≤0.03 across [7,7]→[13,13] — robust (contrast with
    the prior campaign's fragile champions).
21. **Integration-grid convergence:** inherited from validated gates
    (moments ≤3% grid drift at pilot settings); full grid matrix on the
    finalist deferred to continuation.
22. **Origin variation:** cell-center convention; per the previous
    campaign's finding, in-plane lattice-direction shifts mix dipoles and
    quadrupoles — Stage-A conclusions rest on the canonical origin;
    full origin matrix deferred to continuation.
23. **Does realistic loss preserve the state?** Yes: Q_loaded within a
    few % of Q_rad at k = 1e-4 (absorption negligible at Q ~ 10²-10³).
24. **What limits loaded Q?** Radiative leakage of the EQ mode (not
    absorption) at this Q scale.
25. **Fabrication sensitivity:** not yet tested (deferred; §32 of the
    instructions remains open for the continuation stage).
26. **Does multipole-aware beat the field-proxy strategy?** Yes,
    directly: the previous campaign's |Ex|²-proxy produced a false
    positive (EQ/toroidal content mistaken for ED); here the objective
    targets the actual moments, and the qualification confirms the
    intended content — no false positive occurred.
27. **Supported claim:** freeform inverse design with an exact
    differentiable current-multipole objective can synthesize prescribed
    same-band p_x + Qe_xz states on demand; in the resulting
    scale-detuning family, emergent radiative Q is governed by the EQ
    state's darkness and is NOT enhanced by ED–EQ spectral alignment;
    the ED–EQ channel phase stays ~70–75° away from destructive.
28. **Tempting but unsupported claims:** "ED–EQ destructive interference
    creates high Q" (not observed); "quasi-BIC" (criteria unmet); "EIT"
    (unmet); any claim resting on the unresolved Q>1e6 needle.
29. **Novelty (see NOVELTY_GAP_ANALYSIS.md):** EQ-driven quasi-BICs via
    symmetry-broken canonical geometries are established; multipole-aware
    inverse design exists (DDA-based current-multipole objectives). The
    defensible distinctives here: exact-kernel differentiable multipole
    objectives inside freeform RCWA topology optimization, the
    Q-never-optimized causal design with a controlled detuning
    trajectory, and the resulting mechanistic finding (darkness-governed
    Q; alignment insufficient).
30. **Single strongest next step:** a two-parameter geometry family
    (e.g., scale × one interpretable deformation) that (i) holds the EQ
    linewidth fixed while sweeping ED–EQ alignment, and (ii) drives
    Δφ_rad through ±180°, to test the interference hypothesis where it
    can actually bite; plus resolving the sub-nm needle candidate.

## Machine-readable outputs
candidate_ledger.csv; qualify/*/spectra_{main,lossless,lossy}.csv (=
multipole_spectra + radiation_channels per candidate: complex moments,
t/r amplitudes, exact channel integrals, T/R); q_summary.csv;
detuning_trajectory.csv (+ per-alpha CSVs); fit_inspection.png;
detuning_trajectory.png. Convergence/robustness rows beyond the checks
reported above are deferred to the continuation stage and marked as such.
