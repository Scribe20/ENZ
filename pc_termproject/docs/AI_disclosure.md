# AI-use disclosure (truthful summary; to be included on the last slide)

AI tool: Claude (Anthropic), used through Claude Code in an autonomous session directed by the student.

| area | what the AI did | what the student did |
|---|---|---|
| Coding | Wrote the Python PWEM solver (`pwem.py`), the gradient/LP optimiser (`topopt.py`), the discrete refinement, the verification and plotting scripts, the MATLAB mirror (`pc_pwem_official.m`) and the submission script `Nano_opt_code.m` | Provided the task, the lecture material and the evaluator; must run `Nano_opt_code.m` and `pc_evaluate.p` in MATLAB and report the official score |
| Optimisation strategy | Proposed and executed the staged search (topology families → gradient max-min → exact pixel refinement), chose the band pair TM 3-4 / TE 2-3, ran all sweeps | Set the objective (official score) and the constraints; reviews the choices |
| Numerical analysis | Validated the solver (analytic limits, textbook rods, digitised Figure 1), identified the grader's formulation, convergence / k-grid / pixel-robustness checks, MATLAB–Python cross-check in Octave | Must reproduce the key numbers with pc_evaluate.p (MATLAB was not available to the AI) |
| Physical interpretation | Drafted the causal explanation (island resonances for TM, connected veins for TE, band-edge field analysis) based on lectures PC-I–III | Must understand and defend the explanation; all statements are checkable from the figures and code |
| Presentation | Drafted the slide plan, figures, Q&A answers and this disclosure | Prepares and delivers the presentation |

No web search or external online sources were used; only the uploaded lecture notes, the project sheet, the two example
files, and the AI's own derivations and computations. Nobody else was consulted.
