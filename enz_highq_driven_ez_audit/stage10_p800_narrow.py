"""Stage 10: physics of the P800 narrow (~1437.6 nm, Q_loaded ~ 74) branch
versus the broad (~1490-1510 nm) branch: fields, ITO longitudinal
participation, Fourier composition, multipole channel amplitudes, and the
DIRECT S-matrix evidence for open-channel behaviour (|r|, |t| around the
pole, residues in r and t)."""
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
NAME = "P800 robust"
rho, P, h = cm.load_candidate(NAME)
br = cm.jload(cm.OUT / "stage2_branches.json")[NAME]
narrow = min(br, key=lambda b: abs(b["rows"][0]["lambda_pole"] - 1437.6))
broad = max(br, key=lambda b: b["rows"][0]["lambda_pole"])
res = dict(narrow_branch=narrow["branch"], broad_branch=broad["branch"],
           narrow_fit=narrow["fit"], broad_fit=broad["fit"])
out = {}
for tag, b in (("narrow", narrow), ("broad", broad)):
    lam1 = b["rows"][0]["lambda_pole"]
    d1 = cm.driven_metrics(rho, P, h, lam=lam1, want_field=True)
    e1 = cm.energy_participation(rho, P, h, lam1, 1.0)
    lossless = b["fit"].get("lossless_pole")
    e0 = cm.energy_participation(rho, P, h, lossless["lambda_nm"], 0.0) if lossless else None
    d0 = cm.driven_metrics(rho, P, h, lam=lossless["lambda_nm"], s=0.0, want_field=True) if lossless else None
    fz = {}
    with torch.no_grad():
        sim = fm.build_sim(rho, P, h, lam=lam1)
        x, y = fm.cell_axes(P, 96)
        acc = 0.0
        for zp in (np.arange(7) + 0.5) * cm.D_ITO / 7:
            E, _ = sim.field_xy(1, x, y, float(zp))
            acc = acc + np.abs(np.fft.fft2(E[2].numpy()) / 96 ** 2) ** 2
        acc /= acc.sum()
        fz = {f"({a},{bb})": float(acc[a % 96, bb % 96]) for a in range(-2, 3) for bb in range(-2, 3)}
    out[tag] = dict(lambda_pole=lam1, Q_loaded=b["rows"][0]["Q"],
                    driven_at_pole={k: v for k, v in d1.items() if k not in ("Ez_mid", "E_mid")},
                    participation_loaded=e1, participation_lossless=e0,
                    driven_lossless_at_pole=({k: v for k, v in d0.items() if k not in ("Ez_mid", "E_mid")} if d0 else None),
                    fourier_Ez_ITO=fz, multipoles=mp.moments(rho, P, h, lam1),
                    Ez_map=d1["Ez_mid"], Ez_map_lossless=(d0["Ez_mid"] if d0 else None))
    print(f"[{tag}] pole {lam1:.1f} nm Q={b['rows'][0]['Q']:.1f}: F_Ez={d1['F_Ez']:.3f} A={d1['A_rt']:.3f} eta_z={d1['eta_z']:.3f} "
          f"etaENZ(loaded)={e1['eta_ENZ_z']:.3e} itoE={e1['ito_E_energy_fraction']:.3f} "
          f"| lossless: etaENZ={(e0 or {}).get('eta_ENZ_z', float('nan')):.3e} itoE={(e0 or {}).get('ito_E_energy_fraction', float('nan')):.3f} "
          f"F_Ez={(d0 or {}).get('F_Ez', float('nan')):.2f}", flush=True)
    mpl = out[tag]["multipoles"]
    print(f"   multipole power fractions {({k: round(v, 3) for k, v in mpl['power_fractions'].items()})}; "
          f"forward cancellation ratio {mpl['forward']['cancellation_ratio']:.3f}, backward {mpl['backward']['cancellation_ratio']:.3f}", flush=True)
res["overlap_narrow_broad"] = pl.overlap(out["narrow"]["Ez_map"], out["broad"]["Ez_map"])
# direct S-matrix evidence around the narrow pole: dense |r|^2, |t|^2, A
lam0 = out["narrow"]["lambda_pole"]; fw = lam0 / out["narrow"]["Q_loaded"]
lams = np.linspace(lam0 - 4 * fw, lam0 + 4 * fw, 61)
r, t = pl.rt_scan(rho, P, h, 1.0, lams)
Rs, Ts, As = [], [], []
with torch.no_grad():
    for lam in lams:
        sim = fm.build_sim(rho, P, h, lam=lam)
        Rr, Tt = fm.rt_all_orders(sim)
        Rs.append(float(Rr)); Ts.append(float(Tt)); As.append(float(1 - Rr - Tt))
i0 = int(np.argmin(np.abs(lams - lam0)))
res["s_matrix_evidence"] = dict(lams=lams.tolist(), r_abs=np.abs(r).tolist(), t_abs=np.abs(t).tolist(),
                                R=Rs, T=Ts, A=As, at_pole=dict(R=Rs[i0], T=Ts[i0], A=As[i0]),
                                off_resonance=dict(R=Rs[0], T=Ts[0], A=As[0]),
                                residues=dict(peak_r=narrow["rows"][0]["peak_r"], peak_t=narrow["rows"][0]["peak_t"]),
                                note="open-channel statement must rest on these amplitudes, not on multipole powers")
print(f"   at pole: R={Rs[i0]:.3f} T={Ts[i0]:.3f} A={As[i0]:.3f}; off (blue edge): R={Rs[0]:.3f} T={Ts[0]:.3f} A={As[0]:.3f}", flush=True)
res["narrow"] = {k: v for k, v in out["narrow"].items() if not k.startswith("Ez_map")}
res["broad"] = {k: v for k, v in out["broad"].items() if not k.startswith("Ez_map")}
cm.jdump(res, cm.OUT / "stage10_p800_narrow.json")
np.savez(cm.OUT / "stage10_maps.npz", narrow=out["narrow"]["Ez_map"], broad=out["broad"]["Ez_map"])
fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
vmax = max(np.abs(out["narrow"]["Ez_map"]).max(), np.abs(out["broad"]["Ez_map"]).max())
for ax, tag in zip(axs[:2], ("narrow", "broad")):
    im = ax.imshow(np.abs(out[tag]["Ez_map"]).T, origin="lower", cmap="magma", vmin=0, vmax=vmax, extent=[0, P, 0, P])
    ax.set_title(f"P800 {tag} branch {out[tag]['lambda_pole']:.1f} nm (Q={out[tag]['Q_loaded']:.1f}): |Ez/E_inc| ITO mid-plane", fontsize=9)
fig.colorbar(im, ax=axs[:2], shrink=.8)
axs[2].plot(lams, Rs, label="R"); axs[2].plot(lams, Ts, label="T"); axs[2].plot(lams, As, label="A")
axs[2].axvline(cm.LAMBDA_E, ls="--", c="k", lw=.8); axs[2].set_xlabel("wavelength (nm)"); axs[2].legend(); axs[2].grid(alpha=.3)
axs[2].set_title("direct S-matrix evidence around the narrow pole")
fig.savefig(cm.FIG / "stage10_p800_narrow.png", dpi=150, bbox_inches="tight")
print("[stage10] done")
