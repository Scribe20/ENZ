# Final submission checklist (to be executed by the student in MATLAB R2022a or later)

## Files
- [ ] `Nano_opt_code.m` — from `deliverable/`; rename nothing inside except the two variables at the top if needed.
- [ ] `Nano_P_<initial>_<student>.mat` — produced by running `Nano_opt_code.m`; contains ONLY `epsr` (96 × 96 double,
      values exactly 1.0 and 12.1104). Rename the file (and the `save` line) with your initial and student number.
- [ ] ZIP named `Nano_<initial>_<student>.zip` containing exactly the two files above (no folders, no extra files).
- [ ] `Nano_P_<initial>_<student>.pptx` and `.pdf` (main ≤ 10 pages incl. cover, appendix ≤ 5 pages).

## Verification steps in MATLAB (the only OFFICIAL numbers)
1. `cd` to a folder containing `Nano_opt_code.m` and `pc_evaluate.p`.
2. `run('Nano_opt_code.m')` — should finish in < 2 minutes; it prints the independent PWEM check for the seed and the final
   design and saves the `.mat` file. Check `whos('-file','Nano_P_<initial>_<student>.mat')` shows only `epsr`.
3. `R = pc_evaluate('Nano_P_<initial>_<student>.mat')` — record the printed score and gap edges; compare with the
   SURROGATE-OFFICIAL prediction in RESULTS.md (expected agreement ≈ 1e-3 in a/λ, ≈ 0.1–0.3 % absolute in score).
   If the official score differs by more than ~0.5 % absolute, tell me: it would mean the evaluator's formulation differs
   from the one identified from Figure 1, and the design should be re-optimised against the correct one.
4. Also run `pc_evaluate` on `example_2_rod.mat` and confirm 0.00 % and the Figure-1 band diagram.
5. Confirm `version` is R2022a or later (the .p file declares compatibility R2022a).

## Design validity (verified here for the final design)
- [x] size(epsr) = [96 96]
- [x] every element is exactly 1.0 or 12.1104 (checked in Python and in Octave after reload)
- [x] periodic tiling is implicit (pixel-centred grid; the PWEM uses the periodic Fourier sums)
- [x] the target a/λ = 0.40903 lies inside the complete TE+TM gap (SURROGATE-OFFICIAL)
- [x] evaluated with Mmax = 9 (361 PW) and the 21 × 11 half-BZ grid; identical on the C4v wedge and on a 41 × 41 fine grid
- [x] stable for Mmax 9–15; not a symmetry, k-sampling or dithering artefact
- [ ] OFFICIAL score from pc_evaluate.p — NOT available in this environment (no MATLAB); must be done by the student
