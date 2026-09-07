"""Stage 1: common driven-field harness at lambda_E (normal incidence) for
every candidate + same-normalization field maps."""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as cm

torch.set_num_threads(4)
cm.FIG.mkdir(parents=True, exist_ok=True)
res, maps = {}, {}
for name in cm.CANDIDATES:
    rho, P, h = cm.load_candidate(name)
    m = cm.driven_metrics(rho, P, h, want_field=True)
    maps[name] = (m.pop("Ez_mid"), m.pop("E_mid"), P)
    e = cm.energy_participation(rho, P, h, cm.LAMBDA_E, 1.0)
    m["energy_lambda_E"] = e
    res[name] = m
    print(f"{name:18s} A={m['A_rt']:.4f} (vol {m['A_vol']:.4f}) F_Ez={m['F_Ez']:.3f} "
          f"F_Etot={m['F_Etot']:.3f} eta_z={m['eta_z']:.3f} peak={m['peak_Ez2']:.2f} "
          f"p95={m['Ez2_p95']:.2f} p99={m['Ez2_p99']:.2f} conc50={m['conc50_Ez2']:.3f} "
          f"etaENZ={e['eta_ENZ_z']:.3e} itoE={e['ito_E_energy_fraction']:.3f} U={e['U']:.3e}",
          flush=True)
cm.jdump(res, cm.OUT / "stage1_driven_lambdaE.json")
np.savez(cm.OUT / "stage1_maps.npz", **{k.replace(" ", "_").replace("/", "_") + "_Ez": v[0]
                                          for k, v in maps.items()})
names = ["padded QNM", "padded F_ENZ", "P800 robust", "P925 robust"]
vmax = max(np.abs(maps[n][0]).max() for n in names)
fig, axs = plt.subplots(2, 4, figsize=(17, 8))
for i, n in enumerate(names):
    Ez, E, P = maps[n]
    im = axs[0, i].imshow(np.abs(Ez).T, origin="lower", cmap="magma", vmin=0, vmax=vmax,
                          extent=[0, P, 0, P])
    axs[0, i].set_title(f"{n}: |Ez/E_inc| ITO mid-plane, lambda_E\nF_Ez={res[n]['F_Ez']:.2f} eta_z={res[n]['eta_z']:.2f}", fontsize=9)
    Et = np.sqrt(np.abs(E[0]) ** 2 + np.abs(E[1]) ** 2)
    im2 = axs[1, i].imshow(Et.T, origin="lower", cmap="viridis", vmin=0, vmax=vmax,
                           extent=[0, P, 0, P])
    axs[1, i].set_title("|E_t/E_inc| (in-plane)", fontsize=9)
    rho, _, _ = cm.load_candidate(n)
    for ax in axs[:, i]:
        ax.contour(np.linspace(0, P, 128), np.linspace(0, P, 128), rho.numpy().T,
                   levels=[0.5], colors="w", linewidths=0.6)
fig.colorbar(im, ax=axs[0, :], shrink=0.8); fig.colorbar(im2, ax=axs[1, :], shrink=0.8)
fig.savefig(cm.FIG / "fig6_field_maps_same_normalization.png", dpi=150, bbox_inches="tight")
print("[stage1] done")
