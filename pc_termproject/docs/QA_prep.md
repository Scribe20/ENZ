# Expected Q&A (with rigorous answers) — numbers refer to the C4v final design (see RESULTS.md for exact values)

**Q1. Why does this geometry open both TE and TM gaps?**
Because it contains both ingredients that the two polarisations need, at scales that put their gaps at the same frequency.
The large Si block is an isolated high-index island: TM (Ez, continuous) modes localise their Ez inside it and consecutive
TM bands correspond to island resonances with an increasing number of nodal lines (band 3: dipole-like, 98 % of the electric
energy in Si at X; band 4: quadrupole-like at Γ, 92 %). The Dz discontinuity at the block boundary makes those two
resonances have very different effective indices → wide TM 3-4 gap. The thin veins make the dielectric connected, which is
what TE (in-plane E, discontinuous) needs: TE band 2 keeps |D|² inside the block ring (92 % in Si), and the next band is
forced into the veins/air (44 % in Si) → TE 2-3 gap. The veins are thin enough not to let the TM air band penetrate.

**Q2. Why is the target frequency inside the gap?**
The gap position is controlled by the average index (fill 0.41) and the block size; the optimiser maximised the *centred*
score, so the two limiting edges — TE band 2 maximum at Γ (lower edge) and TM band 4 minimum at Γ (upper edge) — sit at
equal distances (0.0220 and 0.0217 in a/λ) from 0.40903. The score is a min of two margins, so at the optimum they are equal.

**Q3. Why C4v symmetry?**
The lattice is square and the grader samples half of the Brillouin zone, so any symmetry is allowed. C4v was used because
(i) it reduces the design space 8× (1176 orbit variables), (ii) band extrema then sit at Γ, X, M, so the irreducible wedge
of the official grid is exact (verified: identical extrema to 1e-14), (iii) it makes the crystal isotropic in the two lattice
directions. Removing the constraint was tested: the unconstrained refinement gains only +0.14 % and differs in 4 pixels.

**Q4. What happens if the fill fraction changes?**
Uniform 1-pixel erosion (fill 0.379) or dilation (0.450) closes the gap: the average index shifts the whole band structure
by ≈ 3 % and, more importantly, ±1 px is ±26 % of the vein width, which moves the TE edges. The parametric map shows the same:
rods 0.32 + veins 0.10 (fill 0.40) already drops to 0.4 %. The design sits in a narrow valley of fill ≈ 0.41 because the
target frequency is fixed; wider gaps exist at other frequencies but they would not contain 1550 nm.

**Q5. How sensitive is the result to pixelisation?**
The grader itself is pixel-based (Fourier sums over 96 × 96 pixel centres) so the design *is* the pixel map; the question
is whether individual pixels matter. Random flips of 1 % of the boundary pixels change the score by [see RESULTS.md];
single boundary-orbit flips change band edges by ≈ 1–2 × 10⁻³, i.e. ≈ 0.5–1 % of score. Features are all ≥ 7 px wide, above
the resolution of the 361-plane-wave basis (≈ 5 px), so there is no sub-pixel exploitation.

**Q6. Why is this better than a circular rod?**
A rod lattice (any r) has no TE gap: the Ex,y discontinuity compensates the index contrast (PC-III p.28) and the field
cannot stay inside an isolated rod. The example r = 0.2a in the project sheet scores 0 for that reason. Our Stage A sweep of
21 rod radii and 3 rod shapes confirms it. The rod+vein hybrid r = 0.32, w = 0.08 already gives 9.5 %; the optimised block
shape (chamfered square rather than a circle) adds another 1.1 % by balancing the two edges.

**Q7. How was the PWEM implementation validated?**
(1) Homogeneous medium: exact free-photon dispersion to 1e-16; (2) 1-D stripe limit against the analytic Bloch/transfer-matrix
dispersion; (3) ε = 8.9, r = 0.2a rods reproduce the textbook TM gap 0.32–0.44; (4) the project's own Figure 1 (rod example)
was digitised: all TM and TE band values at Γ, X, M agree with the solver to ±0.001 — this also identified the grader's
formulation (TM: ε-matrix, TE: 1/ε coefficients); (5) a MATLAB mirror of the solver, executed in Octave, reproduces the
Python numbers exactly. Finite-difference checks validated the sensitivities to 4 digits.

**Q8. Why can TE and TM behave differently?**
Different boundary conditions at ε jumps. TM: Ez continuous, Dz jumps — energy D·E can be concentrated in Si without
penalty, giving strongly different effective indices between consecutive bands (large gaps for islands). TE: E_t continuous
but E_n jumps, D_n continuous — a mode crossing an interface must pay for the discontinuity; only along connected dielectric
paths can the in-plane E stay in Si (large gaps for networks).

**Q9. Is the optimum robust to basis truncation?**
Yes in the relevant direction: Mmax 9 → 11 → 13 → 15 gives 10.62 → 10.60 → 10.60 → 10.60 %. The TM edges are converged
at Mmax = 5 (ε-matrix form), the TE edges of the 1/ε form drift by ≈ 0.004 and move *away* from the target. Below the official
basis (Mmax 5, 7) the TE error is so large that the gap closes — this is a property of the grader's method, not of the design.
Both alternative formulations (all-ε, all-1/ε) were also evaluated (table in the appendix).

**Q10. How do you know the optimiser did not exploit numerical artefacts?**
(a) filter radius ≥ 2 px and boundary-only discrete moves → no dithering / sub-resolution features; (b) the score is
unchanged on a 41 × 41 fine k-grid; (c) the score is stable for larger bases; (d) the band-edge modes are physically
interpretable (island resonances, vein modes) and the same gap appears in the purely parametric rod+vein structure (9.5 %)
that was never touched by the optimiser.

**Q11. What exactly did AI contribute?**
See the AI-use disclosure: the AI wrote the PWEM solver, the optimiser, the MATLAB script and the analysis scripts, ran
the sweeps, and drafted the physical interpretation and slides; the student defined the task, reviewed and verified the
reasoning, and is responsible for the submitted content. The official evaluator was not executed by the AI (no MATLAB in
its environment); the numbers presented as official must be reproduced by the student with pc_evaluate.p.

**Q12. Why TM 3-4 and TE 2-3 rather than the fundamental gaps?**
At a/λ = 0.409 with 41 % Si the lowest TM band already reaches 0.25 at X; the TM 1-2 gap of such a lattice is around
0.25–0.35. The TE 1-2 gap requires thick veins, which kills any TM gap. The pair scan (all combinations of TM 1–3 and TE 1–3
from gray starts) found only TM3-4/TE1-2 (≤ 4.7 %) and TM3-4/TE2-3 (≥ 10 %) feasible.

**Q13. Could the gap be wider at a different frequency?**
Yes — the width is not the objective. Without the target constraint the same family gives wider gaps at higher a/λ
(smaller fill), but the score would be 0 if 0.409 leaves the gap. The score rewards centring.

**Q14. What limits further improvement?**
Both limiting edges are at Γ: TE2 max and TM4 min. Any change that raises TM4 at Γ (smaller block) lowers TE2 at Γ less
than it raises... — the LP-based refinement finds no combination of pixel moves (up to 3 px deep) that raises the min
margin, i.e. the design is a local optimum for this topology; other topologies (diagonal veins, rings, frames, holes with
rods) were all worse in Stage A.
