"""Appends the physics synthesis (facts / derived / interpretation /
uncertainty) and the four-question answer to REPORT.md.  Numbers are read
from the stage outputs where available; run after stage15_report.py."""
import numpy as np
import pandas as pd

import common as cm

t3 = pd.read_csv(cm.OUT / "stage3_branches_table.csv")
s16 = cm.jload(cm.OUT / "stage16/stage16_results.json")
loss = cm.jload(cm.OUT / "stage7_loss_sweep.json")
leak = cm.jload(cm.OUT / "stage8_leakage.json")
d1 = cm.jload(cm.OUT / "stage1_driven_lambdaE.json")
surv = cm.jload(cm.OUT / "stage11_survivors.json")


def row(branch):
    return t3[t3["branch"] == branch].iloc[0]


p925, p800n, p800b, qnm, fenz = row("P925 robust|1447nm"), row("P800 robust|1438nm"), row("P800 robust|1515nm"), row("padded QNM|1459nm"), row("padded F_ENZ|1466nm")
best16 = max(s16, key=lambda r: (r["F_Ez"] if (r.get("Q_rad") and np.isfinite(r["Q_rad"]) and r["Q_rad"] >= 80) else 0))
leak_best = max((r for fam, rr in leak.items() if fam != "parent" for r in rr), key=lambda r: r["F_Ez"])
qnm_loss = loss["padded QNM|1459nm"]; p925_loss = loss["P925 robust|1447nm"]
sc = surv["P800 robust"]["supercell"]; sc9 = surv["P925 robust"]["supercell"]; scq = surv["padded QNM"]["supercell"]

L = ["", "---", "", "## 15. Synthesis (written after all stages; numbers from the stage outputs)", "",
     "### 15.1 Measured facts", "",
     f"1. Certified loss-scaling decomposition (field-overlap tracked, [7,7], dense local rescans): P925 target-near branch 1447 nm Q_loaded {p925['Q_loaded']:.1f}, "
     f"Q_rad {p925['Q_rad']:.0f}, Q_nr {p925['Q_nr']:.0f}, gamma_rad/gamma_nr {p925['gamma_ratio']:.2f}; P800 narrow 1437.6 nm Q_loaded {p800n['Q_loaded']:.1f}, Q_rad {p800n['Q_rad']:.0f}, "
     f"Q_nr {p800n['Q_nr']:.0f}, ratio {p800n['gamma_ratio']:.2f} (order-converged: 1437.6/1436.9/1436.6 nm, Q 73.8/75.3/75.8 at [7,7]/[9,9]/[11,11]); P800 broad Q_rad {p800b['Q_rad']:.0f}, Q_nr {p800b['Q_nr']:.1f}; "
     f"padded QNM ENZ-band branch Q_rad {qnm['Q_rad']:.0f} (see caveat), Q_nr {qnm['Q_nr']:.1f}, ratio {qnm['gamma_ratio']:.3f}; padded F_ENZ Q_rad {fenz['Q_rad']:.0f}, Q_nr {fenz['Q_nr']:.1f}, ratio {fenz['gamma_ratio']:.3f} "
     "(historical 106 / 0.048 reproduced).",
     f"2. Modal ENZ participation at the lossless-auxiliary pole (eta_ENZ,z; ITO E-energy fraction): padded QNM {qnm['eta_ENZ_z_modal']:.2f} ({qnm['ito_E_frac_modal']:.2f}), padded F_ENZ {fenz['eta_ENZ_z_modal']:.2f} ({fenz['ito_E_frac_modal']:.2f}), "
     f"P800 broad {p800b['eta_ENZ_z_modal']:.2f} ({p800b['ito_E_frac_modal']:.2f}), P925 1447 nm {p925['eta_ENZ_z_modal']:.2f} ({p925['ito_E_frac_modal']:.2f}), P800 narrow {p800n['eta_ENZ_z_modal']:.3f} ({p800n['ito_E_frac_modal']:.2f}).",
     f"3. Driven at lambda_E (real ITO loss): F_Ez padded QNM {d1['padded QNM']['F_Ez']:.2f}, padded F_ENZ {d1['padded F_ENZ']['F_Ez']:.2f}, P800 {d1['P800 robust']['F_Ez']:.2f}, P925 {d1['P925 robust']['F_Ez']:.2f}; "
     f"A_ITO 0.20 / 0.21 / 0.45 / 0.46; eta_z driven 0.78 / 0.78 / 0.34 / 0.62.",
     f"4. Loss sweep (Stage 7): reducing the ITO loss raises F_Ez at the pole monotonically for every ENZ-rich branch (padded QNM: "
     + " -> ".join(f"{r['F_Ez_pole']:.0f}" for r in qnm_loss) + f" for s = 1 -> 0; P925 1447 nm: " + " -> ".join(f"{r['F_Ez_pole']:.0f}" for r in p925_loss)
     + ") while A_ITO at the pole peaks (0.44 for padded QNM) where gamma_rad/(s gamma_nr) ~ 1.",
     f"5. Leakage tuning of the padded QNM parent (Stage 8, 28 geometries incl. the corrected notch family): F_Ez(lambda_E) stays in 1.99-2.53 (best {leak_best['tag']}: {leak_best['F_Ez']:.2f}) although the fitted Q_rad varies from ~90 to ~860; "
     "no Pareto-superior state.",
     f"6. Stage 16 epsilon-constraint designs: P925 seed -> certified Q_rad 86-106 with F_Ez 4.0-4.2 for all three Q_target values (best {best16['tag']}: F_Ez {best16['F_Ez']:.2f}, Q_rad {best16['Q_rad']:.0f}, ratio {best16['gamma_ratio']:.2f}, eta_ENZ,z modal {best16['eta_ENZ_z_modal']:.3f}); "
     "padded-QNM seed -> F_Ez 1.9-2.2, Q_rad 92-111 (one run uncertifiable: negative fitted gamma_rad). Both fliplr projections absent (S_flip 0.14-0.94).",
     f"7. Locality: 2x2 supercell with one neighbour eroded by 2 px changes A/F_Ez by {100*(sc['supercell_one_neighbour_eroded_2px']['A']/sc['single_cell_order55']['A']-1):+.0f}%/{100*(sc['supercell_one_neighbour_eroded_2px']['F_Ez']/sc['single_cell_order55']['F_Ez']-1):+.0f}% (P800), "
     f"{100*(sc9['supercell_one_neighbour_eroded_2px']['A']/sc9['single_cell_order55']['A']-1):+.0f}%/{100*(sc9['supercell_one_neighbour_eroded_2px']['F_Ez']/sc9['single_cell_order55']['F_Ez']-1):+.0f}% (P925), "
     f"{100*(scq['supercell_one_neighbour_eroded_2px']['A']/scq['single_cell_order55']['A']-1):+.0f}%/{100*(scq['supercell_one_neighbour_eroded_2px']['F_Ez']/scq['single_cell_order55']['F_Ez']-1):+.0f}% (padded QNM); identical-cell supercells reproduce the single cell exactly.",
     "", "### 15.2 Derived quantities", "",
     "- For every branch that actually lives in the ITO (eta_ENZ,z modal ~ 0.4, ITO E-energy fraction ~ 0.9), Q_nr = 5.1-5.7: the ENZ film's own absorption sets the loaded linewidth, so Q_rad = 10^2-10^3 means gamma_rad/gamma_nr = 0.03-0.006 (dark). "
     "The only branches with gamma_rad/gamma_nr >= 0.3 are those with low ITO participation (P800 narrow: 10-15 % ITO E-energy; P925 1447 nm: ~0.3 ITO/Si ratio) - their Q_nr of 30-110 is what makes them accessible.",
     "- Driven F_Ez at fixed material loss is maximal near gamma_rad ~ gamma_nr (Stage 7); for ENZ-rich modes that optimum sits at Q_rad ~ Q_nr ~ 5, i.e. exactly the low-Q regime of the historical designs. Their F_Ez ~ 2.2 at lambda_E is a non-resonant near-field background: opening the radiative channel 7x (Stage 8) or constraining Q_rad (Stage 16, padded-QNM seed) leaves it unchanged.",
     "- The Pareto frontier of certified states in (Q_rad, F_Ez) is therefore: {P925-family: Q_rad ~ 85-106, F_Ez 4.0-4.2, eta_ENZ,z ~ 0.09-0.11} -- {P800 narrow: Q_rad 206, F_Ez 2.2, eta 0.024} -- {ENZ-rich dark parents: Q_rad 110-180 (or 500-1800 on the second sub-branch), F_Ez 2.2, eta ~ 0.4}. No point has Q_rad >= 139 together with F_Ez >= 3.3.",
     "", "### 15.3 Physical interpretation", "",
     "- P925's large driven Ez is an ITO-loaded silicon photonic resonance (no-ITO pole 1479 nm, Q 56, field overlap 0.78-0.88 with the with-ITO 1447-nm branch) whose ~30 % ITO participation gives Q_nr ~ 30 and a coupling ratio ~0.35; it is the best available compromise, not an ENZ mode made bright.",
     "- The P800 narrow pole is a Q ~ 75 silicon slab mode of mixed ED + MQ character that barely touches the ITO; its coherent forward/backward multipole sums are constructive and the S-matrix shows an absorption resonance (A 0.16 -> 0.51, R nearly constant), so no Kerker/BIC mechanism is claimed.",
     "- The padded QNM and F_ENZ parents are genuinely ENZ-rich high-Q_rad modes, but with the measured ITO (Im eps ~ 0.70 at lambda_E) they are intrinsically overdamped by absorption; making them bright would require gamma_rad ~ 0.12 rad/fs, i.e. destroying the high Q_rad. This is a material limit of the ENZ film, not a topology limit.",
     "- Consequently the desired regime (Q_rad >= O(10^2), F_Ez >> 2.2, eta_ENZ,z high, accessible) does not exist in the air-padded a-Si/ITO/glass class with this ITO; the reachable trade-off is quantified by the frontier above.",
     "", "### 15.4 The four questions kept separate", "",
     "1. Does a high-Q ENZ-rich MODE exist? YES - padded QNM / padded F_ENZ / P800 broad branches (Q_rad 116-181 certified, second sub-branch of the padded QNM up to ~10^3 on some tracks), eta_ENZ,z modal ~ 0.41-0.43, ITO E-energy fraction ~ 0.9.",
     "2. Can free-space light strongly EXCITE it? NO - gamma_rad/gamma_nr = 0.03-0.04 (0.006 on the higher-Q sub-branch); the loaded Q is 4.9-5.5, set by the ITO loss.",
     "3. Does excitation produce large longitudinal Ez in ITO? Only at the non-resonant background level (F_Ez ~ 2.2). Large driven Ez (4.0-4.2, peak |Ez|^2 up to 48) is obtained only by the moderately ENZ-loaded P925 family with Q_rad ~ 100.",
     "4. Does it drive the nonlinear material strongly? Unanswerable with validated inputs: no nonlinear ITO model or pulse spectrum exists in the repository; the linear proxies (A_ITO, A/d absorbed-energy density at fixed intensity, F_Ez, p95/p99 Ez^2, eta_ENZ,z) all rank P925 first, P800 second (by A) and the ENZ-rich parents last.",
     "", "### 15.5 Unresolved uncertainty", "",
     "- The padded QNM parent's ENZ-band loaded pole is a merged doublet (dense scan: 1459 and 1503 nm at s = 1); the two sub-branches separate under loss scaling to Q_rad ~ 139 (1513-nm line) and ~500-1870 (1466-nm line, the historical value). Which sub-branch is followed depends on the loss-level set; both are dark. The historical Q_rad = 1872 therefore describes one sub-branch of a doublet, not 'the' parent mode.",
     "- P925's 1338-nm branch has a non-monotonic gamma(s) (fit residual 0.30) and is reported only as radiatively dominated.",
     "- The Stage 16 differentiable Q_rad proxy (lossless ITO-field linewidth at lambda_E) saturated near Q_rad ~ 100 for the P925 family regardless of Q_target; a stricter proxy (e.g. explicit pole tracking inside the loop) would be needed to test whether Q_rad >= 300 is reachable at F_Ez ~ 4.",
     "- The (40 deg, 45 deg) padded-QNM poles with Q ~ 950 / 170 found in Stage 5 are uncertified (no dense rescan at oblique incidence).",
     "- Angular robustness of the P925 branch: it persists in the phi = 0 plane to 40 deg (Q ~ 20) but degrades in the phi = 90 plane (Q 7-8, field overlap ~0.5); the P800 narrow pole persists only in the phi = 0 plane.",
     "", "### 15.6 Recommendation", "",
     "- Stop broad topology searches in this material class for the high-Q + high-Ez goal: the limiting factor is the ENZ film's Im(eps), and the certified frontier is F_Ez ~ 4.2 at Q_rad ~ 100 (P925 family, best Stage-16 design) versus F_Ez ~ 2.2 at Q_rad ~ 10^2-10^3 (dark ENZ-rich parents).",
     "- If higher Q with real ENZ character is required, the physically meaningful levers are material ones (lower-loss ENZ film, thinner ITO / different doping) or a deliberately overcoupled hybrid (Q_rad ~ Q_nr ~ 5-30) accepted as low-Q; the P925 family with h ~ 260-270 nm and P ~ 960-980 nm (Stage 6: F_Ez 4.7-5.3 at Q_rad 22-69) is the pragmatic direction for maximal driven Ez.",
     ]
with open(cm.HERE / "REPORT.md", "a") as f:
    f.write("\n".join(L) + "\n")
print("synthesis appended")
