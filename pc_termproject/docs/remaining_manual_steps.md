# What you (the student) must still do manually

1. **Run the official evaluator.** MATLAB was not available to the AI session, so no result in this repository is an
   output of `pc_evaluate.p`. Run `Nano_opt_code.m` then `pc_evaluate('Nano_P_<initial>_<student>.mat')` in MATLAB
   (R2022a+). Report the printed score. Everything labelled SURROGATE-OFFICIAL is expected to match to ~0.1–0.3 %.
2. **Fill in your initial and student number** in the file names (`Nano_opt_code.m` has a `save(...)` line and a comment
   header; the `.mat`, `.zip`, `.pptx`, `.pdf` names follow the project sheet).
3. **Build the slides** from `docs/presentation_plan.md` using the figures in `figs/` (main ≤ 10 pages, appendix ≤ 5).
   Export to PDF. Put the AI-use disclosure (`docs/AI_disclosure.md`) and the consultation statement on the last slide.
4. **Rehearse the Q&A** with `docs/QA_prep.md`; be able to re-derive the band-edge argument from the field plots.
5. If the official score deviates strongly from the prediction, re-run `verify.py` with the alternative formulations
   (`--form eta` / `eps`) to see which one the evaluator uses, and re-optimise (`run_opt.py` / `run_refine.py`) against it.
