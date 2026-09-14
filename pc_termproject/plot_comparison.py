"""Comparison figure: rod example (project sheet, 0 %), parametric seed rods+veins (9.5 %), final design (10.63 %)."""
import numpy as np
from pwem import *; from plots import *; import stageA as SA
import matplotlib.pyplot as plt
UP = "/root/.claude/uploads/0544bbbf-a4a5-500d-b67c-8cecae91e9bb/"
cases = [("rod example r = 0.2a (project sheet)", load_mat(UP + "15485546-example_2_rod.mat")),
         ("seed: rods r = 0.32a + veins w = 0.08a", mask_to_eps(SA.fam_rod_veins(0.32, 0.08))),
         ("final optimised design", mask_to_eps(np.load('runs/ref_rv33/mask.npy')))]
fig, axs = plt.subplots(2, 3, figsize=(15, 8.5), gridspec_kw=dict(height_ratios=[1, 1.15]))
for j, (title, eps) in enumerate(cases):
    r = evaluate(eps, nbands=12)
    plot_geometry(eps, axs[0, j], title=f"{title}\nfill = {r['fill']:.3f}")
    band_diagram(eps, axs[1, j], gap=r['gap'], title=f"TE red, TM blue | score {100*r['score']:.2f}%", ylim=(0, 0.7))
    print(summarize(r, title))
plt.tight_layout(); plt.savefig('figs/comparison_rod_seed_final.png', dpi=110)
