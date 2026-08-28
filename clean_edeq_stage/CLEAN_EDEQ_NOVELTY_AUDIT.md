# CLEAN_EDEQ_NOVELTY_AUDIT — prior art vs the Stage-B results

Fresh targeted literature check (web, 2026-08) against the eight mandated
categories. Sources are listed at the end; closest prior art named per
claim.

## What is established (and therefore NOT claimable as novel)

1. **ED–EQ generalized Kerker scattering exists.** Complete-transmission
   conditions from ED+EQ co-excitation are known for Drude-like spheres
   (NJP 2020, "Characteristics of electric quadrupole…"), and Kerker-type
   constructive/destructive interference is the working principle of
   Huygens metasurfaces (PR Materials 4, 125202 (2020)).
2. **Broadband generalized Kerker exists.** Mie voids give broadband
   directional scattering across the visible (PRB 2025); "proper
   superposition of electric multipoles" broadband-Kerker designs exist;
   transverse/generalized Kerker reviewed in arXiv:1808.10708.
3. **Quadrupole-based Huygens metasurfaces exist** — cross-shaped
   resonators using EQ resonances for phase-gradient transmission control
   (Opt. Lett. 45, 4847 (2020)). Canonical geometry, analytic design.
4. **Multipole-aware inverse design exists.** Multipole decomposition
   integrated with adjoint optimization (e.g. nonlinear-metasurface SFG
   work, 2024); multi-objective topology optimization selecting emission
   direction; DDA current-multipole objectives (Stage-A audit).
5. **Active multipole switching exists** — anapole↔dipole switching via
   GST structuring (Nat. Commun. 2018), photoswitchable anapole
   metasurfaces; not composition-preserving ED–EQ phase control, but the
   "active multipolar interference switching" headline is taken.
6. Bright/dark mode interference, Fano, EIT metasurfaces: standard.

## Defensible novelty (each tied to Stage-B data)

1. **Freeform synthesis of a broadband balanced p_x/Qe_xz state.** The
   clean balanced criterion (f_ED, f_EQ ≥ 0.2 each, sum ≥ 0.8, purities
   ≥ 0.8 under the complete exact 4-family partition) holds over 73 nm on
   the frozen geometry and 115–125 nm at the tuned thickness — no
   canonical-geometry ED–EQ Huygens work reports a *verified exact
   multipole partition* of this breadth, and the state was synthesized
   from a prescribed-multipole objective, not diagnosed post hoc.
2. **Quantitative separation of multipolar composition and radiative
   phase as design dimensions.** Measured, not asserted: the design-space
   gradients of ED–EQ channel phase and ED/EQ balance are nearly
   orthogonal (cos = 0.24 at λ0 over the 64² density space), and a single
   interpretable knob (thickness, −0.81°/nm at ~2×10⁻⁴ balance/nm)
   sweeps the bottom-port phase through exact cancellation (−153°→+122°,
   crossing 180° at h ≈ 221 nm) while f_ED+f_EQ stays ≈ 0.99 and
   px|ED = 1.00. We found no prior demonstration of composition-preserving
   continuous multipolar-phase steering with an explicit orthogonality
   measurement.
3. **The joint state: clean balanced ED–EQ composition AND broadband
   directional cancellation simultaneously** (h = 225–235 nm: R ≤ 0.05
   over 95–130 nm, η_dir ≥ 0.91, clean criterion co-holding over
   115–125 nm) on a substrate-supported freeform platform — with the
   attribution passing the strict removal test (deleting ED or EQ from
   the reconstruction destroys the response) and the cancellation
   condition derived with the measured background term (true R-zero sits
   9° off the naive 180°).
4. **The two-reference contrast as a designed demonstration**: the same
   freeform framework produced a composition-controlled bright broadband
   forward scatterer (P0550) and an interference-locked narrow dark state
   (P0750, m_y/Q_xz anti-phase), analyzed under identical normalization —
   an explicit empirical case that these are separable capabilities.

## What must NOT be claimed

"First ED–EQ Kerker" (exists), "first broadband Kerker" (exists),
"quadrupole Huygens metasurface" (exists), "active multipolar switching"
(exists in anapole form; our index test shows trimming-level phase
sensitivity only — see function report), any high-Q framing for P0550
(contract), and any claim resting on the two energy-flagged wavelengths
(1267–1269 nm, outside the band).

## Verdict

The publishable novelty is the **separable-design-dimension result**
(#2, supported by #1 and #3) with the two-reference contrast (#4) as the
narrative frame — not the existence of ED–EQ directionality.

Sources:
- [Characteristics of electric quadrupole and… (NJP 2020)](https://iopscience.iop.org/article/10.1088/1367-2630/ab6cde/pdf)
- [Kerker-type scattering in an ultrathin silicon Huygens metasurface (PRM 2020)](https://link.aps.org/doi/10.1103/PhysRevMaterials.4.125202)
- [Mie voids as broadband directional light sources (PRB)](https://journals.aps.org/prb/abstract/10.1103/v2b8-j986)
- [Generalized Kerker effects review (arXiv:1808.10708)](https://arxiv.org/pdf/1808.10708)
- [Quadrupole-based Huygens metasurface (Opt. Lett. 45, 4847)](https://opg.optica.org/ol/abstract.cfm?uri=ol-45-17-4847)
- [Inverse design of nonlinear metasurfaces / multipole-adjoint (2024)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11501459/)
- [Empowering Metasurfaces with Inverse Design (ACS Photonics)](https://pubs.acs.org/doi/10.1021/acsphotonics.1c01850)
- [Active control of anapole states via GST (Nat. Commun. 2018)](https://www.nature.com/articles/s41467-018-08057-1)
- [Photoswitchable Anapole Metasurfaces](https://www.researchgate.net/publication/357167569_Photoswitchable_Anapole_Metasurfaces)
- [Inverse design of metasurfaces with non-local interactions (npj Comput. Mater.)](https://www.nature.com/articles/s41524-020-00369-5)
