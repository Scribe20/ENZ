"""Stage 9: origin of the P925 driven Ez - with-ITO / lossless-ITO / no-ITO
spectra, poles, branch field overlaps, Fourier content, multipoles."""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as cm
import forward_multi as fm
import poles as pl
import multipole as mp

torch.set_num_threads(4)
NAME = "P925 robust"
rho, P, h = cm.load_candidate(NAME)
LAMS = pl.SCAN_DENSE
CTRL = {"with_ITO": dict(with_ito=True, s=1.0), "lossless_ITO": dict(with_ito=True, s=0.0),
        "no_ITO": dict(with_ito=False, s=1.0)}


def ez_map_ctrl(lam, with_ito, s, n=48):
    with torch.no_grad():
        sim = fm.build_sim(rho, P, h, lam=lam, with_ito=with_ito, ito_loss_scale=s)
        x, y = fm.cell_axes(P, n)
        if with_ito:
            E, _ = sim.field_xy(1, x, y, cm.D_ITO / 2)
        else:                                  # glass slab where the ITO would be
            E, _ = sim.field_xy(sim.layer_N, x, y, cm.D_ITO / 2)
    return E[2].numpy(), E


res = {}
for key, c in CTRL.items():
    A, R, T, FEz = [], [], [], []
    with torch.no_grad():
        for lam in LAMS:
            sim = fm.build_sim(rho, P, h, lam=lam, with_ito=c["with_ito"], ito_loss_scale=c["s"])
            Rr, Tt = fm.rt_all_orders(sim)
            A.append(float(1 - Rr - Tt)); R.append(float(Rr)); T.append(float(Tt))
            x, y = fm.cell_axes(P, 48)
            lay = 1 if c["with_ito"] else sim.layer_N
            E, _ = sim.field_xy(lay, x, y, cm.D_ITO / 2)
            FEz.append(float((E[2].abs() ** 2).mean()))
    r, t = pl.rt_scan(rho, P, h, c["s"], LAMS, with_ito=c["with_ito"])
    sig = pl.significant_poles(LAMS, r, t)
    for p in sig:
        p["local_refine"] = pl.local_refine(rho, P, h, c["s"], p) if (p["Q"] > 25 and c["with_ito"]) else None
    res[key] = dict(lam=LAMS.tolist(), A=A, R=R, T=T, FEz_mid=FEz,
                    poles=[{k: (v if k != "omega" else [v.real, v.imag]) for k, v in p.items()} for p in sig])
    print(f"[{key}] poles: " + ", ".join(f"{p['lambda_nm']:.1f}nm/Q{p['Q']:.1f}(peak r{p['peak_r']:.2f})" for p in sig), flush=True)

# branch identity across controls: overlap of the mid-plane Ez maps at each pole
maps = {}
for key, c in CTRL.items():
    for p in res[key]["poles"]:
        if 1300 <= p["lambda_nm"] <= 1600:
            maps[f"{key}@{p['lambda_nm']:.0f}"] = ez_map_ctrl(p["lambda_nm"], c["with_ito"], c["s"])[0]
maps["with_ITO@lambda_E"] = ez_map_ctrl(cm.LAMBDA_E, True, 1.0)[0]
keys = list(maps)
ovl = {a: {b: pl.overlap(maps[a], maps[b]) for b in keys} for a in keys}
res["branch_overlap"] = ovl
for a in keys:
    print(f"  overlap {a:24s}: " + " ".join(f"{b.split('@')[0][:4]}@{b.split('@')[1]}:{ovl[a][b]:.2f}" for b in keys), flush=True)

# Fourier content of Ez in the ITO at lambda_E and at the nearest with-ITO pole
def fourier(lam, s=1.0, n=96, nz=7):
    with torch.no_grad():
        sim = fm.build_sim(rho, P, h, lam=lam, ito_loss_scale=s)
        x, y = fm.cell_axes(P, n)
        acc = 0.0
        for zp in (np.arange(nz) + 0.5) * cm.D_ITO / nz:
            E, _ = sim.field_xy(1, x, y, float(zp))
            acc = acc + np.abs(np.fft.fft2(E[2].numpy()) / n ** 2) ** 2
    acc /= acc.sum()
    return {f"({a},{b})": float(acc[a % n, b % n]) for a in range(-2, 3) for b in range(-2, 3)}


near = min(res["with_ITO"]["poles"], key=lambda p: abs(p["lambda_nm"] - cm.LAMBDA_E))
res["fourier"] = dict(lambda_E=fourier(cm.LAMBDA_E), pole=fourier(near["lambda_nm"]),
                      G10_over_k0=cm.LAMBDA_E / P, K_ENZ_over_k0=1.6865)
res["multipoles"] = dict(lambda_E=mp.moments(rho, P, h, cm.LAMBDA_E), pole=mp.moments(rho, P, h, near["lambda_nm"]))
res["nearest_with_ITO_pole"] = near
cm.jdump(res, cm.OUT / "stage9_p925_controls.json")

fig, axs = plt.subplots(1, 3, figsize=(17, 4.5))
for key in CTRL:
    axs[0].plot(LAMS, res[key]["A"], label=f"{key} A"); axs[1].plot(LAMS, res[key]["T"], label=f"{key} T")
    axs[2].plot(LAMS, res[key]["FEz_mid"], label=f"{key} <|Ez|^2> mid-plane")
for ax in axs:
    ax.axvline(cm.LAMBDA_E, ls="--", c="k", lw=.8); ax.set_xlabel("wavelength (nm)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
axs[0].set_ylabel("A"); axs[1].set_ylabel("T_total"); axs[2].set_ylabel("<|Ez/E_inc|^2> (ITO / glass slab)")
fig.suptitle("Stage 9 - P925 controls: with ITO / lossless ITO / no ITO")
fig.savefig(cm.FIG / "stage9_p925_controls.png", dpi=150, bbox_inches="tight")
fig, axs = plt.subplots(1, min(len(keys), 6), figsize=(4 * min(len(keys), 6), 4))
for ax, k in zip(np.atleast_1d(axs), keys[:6]):
    ax.imshow(np.abs(maps[k]).T, origin="lower", cmap="magma", extent=[0, P, 0, P]); ax.set_title(k, fontsize=9)
fig.savefig(cm.FIG / "stage9_p925_branch_maps.png", dpi=130, bbox_inches="tight")
print("[stage9] done")
