# Nanophotonics term project — complete TE+TM band gap at 1550 nm (square lattice, Si/air, 96 × 96 pixels)

Start with **RESULTS.md** (final design, verification, candidate table), then `docs/physics_explanation.md`,
`docs/presentation_plan.md`, `docs/QA_prep.md`, `docs/AI_disclosure.md`, `docs/submission_checklist.md`,
`docs/remaining_manual_steps.md`, and `docs/optimization_log.md` (full history with hypotheses and tests).

Submission files: `deliverable/Nano_opt_code.m` + `deliverable/Nano_P_INITIAL_STUDENTID.mat` (rename with your initial / id).
Independent MATLAB check: `matlab/pc_pwem_official.m` (run `R = pc_pwem_official(epsr)`).

Python tools (numpy/scipy): `pwem.py` solver, `topopt.py` optimiser, `verify.py`, `analyze_design.py`; see RESULTS.md §4.
IMPORTANT: no result here comes from `pc_evaluate.p` (MATLAB unavailable); see RESULTS.md "Provenance".
