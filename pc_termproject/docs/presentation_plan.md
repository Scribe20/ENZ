# Presentation plan (main ≤ 10 pages incl. cover, appendix ≤ 5 pages) — hourglass structure (SW lecture p.14)

## Main slides
1. **Cover** — title, name/ID, AI-use statement line ("AI (Claude) used for coding, optimisation, analysis, slides – see p.10").
2. **What is known** — 2D PC, Bloch theorem → PWEM bands; rods give TM gaps, veins give TE gaps (PC-III p.15–16 figures re-drawn
   with own solver); Dz-discontinuity vs Ex,y-discontinuity argument in one picture.
3. **Problem** — fixed target a/λ = 0.409, fixed materials and 96×96 pixels; the scoring metric (centred margin, not width);
   Figure: rod example (score 0) and why: no TE gap; one-inclusion families all fail (Stage A summary map).
4. **Hypothesis / design principle** — hybrid island + thin network; higher-order TM gap (3-4) + TE 2-3; balance the two
   limiting band edges around the target.
5. **Method** — independent PWEM (formulation identified from Figure 1: TM eps-matrix, TE 1/ε), validation table
   (analytic 1D, textbook rods, digitised figure); staged search: A parametric families → B gradient max-min (Hellmann–
   Feynman sensitivities = field intensities) → exact pixel refinement → C verification.
6. **Results 1: candidate evolution** — table/graph: rods 0 → rods+veins 9.5 → gradient + refinement 10.6 (C4v) → 10.75 (free);
   professor's example 4.03 for comparison.
7. **Results 2: final structure and band diagram** — geometry (nm axes), TE/TM bands Γ-X-M-Γ with the gap shaded, exact
   gap edges, margins, score.
8. **Discussion 1: why it works** — band-edge mode fields (TM3/TM4 Ez, TE2/TE3 |D|²/ε), energy-in-Si percentages, which
   edge limits the gap and how block size / vein width move each edge (ablation plot: score vs r and w).
9. **Discussion 2: robustness** — score vs Mmax (both formulations), fine k-grid, pixel erosion/dilation, symmetry-free
   comparison; what fails (holes, crosses, rings, TE1-2 pairs) and why.
10. **Conclusion + AI disclosure + consultation statement** — design principle in one sentence, final score, limitations
    (official evaluator not run by me; MATLAB mirror only), disclosure.

## Appendix
A1. PWEM equations and Fourier coefficients; k-grid; score definition; validation numbers.
A2. Sensitivity formulas and the max-min LP; filter/projection; discrete refinement algorithm.
A3. Stage A family map (all 19 families, best score each, gap pairs).
A4. Convergence / formulation tables; wedge vs full grid; runtime.
A5. Additional candidates (thin-node grid 4.7 %, diagonal veins 3.3 %, unconstrained 10.75 %) and field profiles.
