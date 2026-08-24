# NOVELTY_GAP_ANALYSIS — ED–EQ causality campaign (Stage A audit)

Targeted literature check (web search, 2026-08; see sources in the session
report) against the six mandated categories:

1. **EQ-driven high-Q / quasi-BIC metasurfaces** — established. Recent
   examples: EQ-dominated quasi-BIC trapped modes in THz free-standing
   metasurfaces (Cojocari et al., Adv. Photon. Res. 2026); EQ-mode
   quasi-BICs from symmetry-broken rectangular-hole / tilted-gap
   geometries; ED+MQ / MD+EQ interference quasi-BICs with Q ~ 650.
   Common denominator: CANONICAL geometries + deliberate symmetry
   breaking; the multipole content is diagnosed post hoc, not prescribed
   as the optimization target.
2. **ED–EQ collective EIT / ultrahigh-Q** — established as a physics
   concept; our Stage-A result does not claim EIT.
3. **Inverse design with desired multipole amplitude/phase** —
   partially established: DDA-based topology optimization with
   current-multipole objectives and multipole-decomposition-informed
   adjoint design exist. Differentiable-RCWA freeform design is
   established (TensorFlow/torch RCWA AD).
4. **Generalized Kerker with ED and EQ** — established concept; not
   claimed here (no channel cancellation observed).
5. **Dipole–quadrupole interference in Fano/anapole/quasi-BIC systems** —
   established.
6. **Freeform/topology-optimized nonlocal metasurfaces** — established.

## Defensible distinctives of this campaign (as supported by actual data)

* **Exact-kernel differentiable current-multipole objective inside a
  freeform RCWA topology loop**: the optimization target is the exact
  finite-size Alaee moments themselves (validated against the original
  MENP implementation), not a field proxy and not a long-wavelength
  surrogate — and the resulting states verify as prescribed (no false
  positives, unlike the prior field-proxy campaign).
* **Causal design**: Q was excluded from the objective and the selection,
  then measured independently, then interrogated with a controlled
  one-parameter detuning trajectory. To our knowledge the
  "Q-never-optimized, mechanism-attributed-afterwards" protocol is not
  standard in the metasurface inverse-design literature.
* **The mechanistic finding itself**: in the synthesized bright-ED/
  dark-EQ family, emergent Q is governed by EQ radiative darkness and is
  NOT enhanced by ED–EQ spectral alignment (phase stays ~70–75° from
  destructive). This is an honest, transferable negative-on-the-headline
  result about when ED–EQ co-excitation does and does not buy Q.

## What must NOT be claimed
"ED and EQ at the same wavelength" (exists), "ED–EQ gives high Q"
(exists as a concept; and our data do not even support it in the tested
family), "quasi-BIC"/"EIT" labels (criteria unmet at Stage A).

Verdict: the novelty case rests on the synthesis-method + causal-protocol
+ mechanistic-finding triple, not on the existence of ED–EQ states.
A publication-grade claim requires the Stage-A2 two-parameter study
(alignment decoupled from darkness; phase driven through destructive).
