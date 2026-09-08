"""Stage 17: Karimi-Fig.1(c,d)-style WITH-ITO vs WITHOUT-ITO transmittance
comparison for the frozen P925_Qt100 geometry (no optimization).

Cases (identical Si geometry, period, height, padding, polarization,
incidence, glass):  A = with 23-nm ITO (measured dispersive eps);
                    B = without ITO;  C = lossless ITO (Im eps -> 0).
Quantities per wavelength: R_total, T_total (all propagating orders,
coherent p/s sum), T_00, A = 1 - R - T, complex r_xx/t_xx amplitudes (for
the AAA pole cross-check).  Broad 1200-1700 nm @ 1 nm, fine 1350-1520 nm
@ 0.25 nm, extra 0.1 nm within +-6 nm of lambda_E.  Order [7,7]; ENZ
region re-verified at [9,9] and [11,11].  Detuning sweep h = 220..260 nm.
"""
import hashlib
import time

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks, peak_widths
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as cm
import forward_multi as fm
import poles as pl

torch.set_num_threads(4)
OUT = cm.OUT / "stage17_qt100_ito_control"
OUT.mkdir(parents=True, exist_ok=True)
LOGF = OUT / "stage17.log"
GEO_FILE = cm.OUT / "stage16/P925_Qt100/rho_hard_binary.npy"
P, H, PAD_FRAC = 925.0, 240.0, 0.04
LAM_E, LAM_POLE = cm.LAMBDA_E, 1433.985
C_NM_THZ = 299792.458          # nm * THz
CASES = {"with_ITO": dict(with_ito=True, s=1.0), "without_ITO": dict(with_ito=False, s=1.0),
         "lossless_ITO": dict(with_ito=True, s=0.0)}


def log(s):
    print(s, flush=True)
    with open(LOGF, "a") as f:
        f.write(s + "\n")


rho_np = np.load(GEO_FILE)
assert rho_np.shape == (128, 128) and set(np.unique(rho_np)) <= {0.0, 1.0}
rho = torch.as_tensor(rho_np, dtype=cm.GEO)
log(f"[geometry] {GEO_FILE.relative_to(cm.ROOT)} sha256={hashlib.sha256(GEO_FILE.read_bytes()).hexdigest()} fill={rho_np.mean():.4f} "
    f"P={P} h={H} pad={PAD_FRAC*P:.2f} nm")


def solve(lam, case, h=H, order=cm.ORDER):
    c = CASES[case]
    with torch.no_grad():
        sim = fm.build_sim(rho, P, h, lam=lam, order=order, with_ito=c["with_ito"], ito_loss_scale=c["s"])
        R, T, tab = fm.rt_all_orders(sim, per_order=True)
        T00 = next((o["T"] for o in tab if o["m"] == 0 and o["n"] == 0), 0.0)
        R00 = next((o["R"] for o in tab if o["m"] == 0 and o["n"] == 0), 0.0)
        rr = sim.S_parameters(orders=[0, 0], direction="forward", port="reflection", polarization="xx",
                              ref_order=[0, 0], power_norm=False)
        tt = sim.S_parameters(orders=[0, 0], direction="forward", port="transmission", polarization="xx",
                              ref_order=[0, 0], power_norm=False)
    return dict(lam=lam, R=float(R), T=float(T), A=float(1 - R - T), T00=T00, R00=R00,
                n_orders=len(tab), r_xx=complex(rr.ravel()[0]), t_xx=complex(tt.ravel()[0]))


def spectrum(lams, case, h=H, order=cm.ORDER, tag=""):
    rows = []
    t0 = time.time()
    for i, lam in enumerate(lams):
        rows.append(solve(float(lam), case, h, order))
        if i % 100 == 0:
            log(f"  [{tag or case}] {lam:.2f} nm  R={rows[-1]['R']:.4f} T={rows[-1]['T']:.4f} A={rows[-1]['A']:.4f} T00={rows[-1]['T00']:.4f} ({time.time()-t0:.0f}s)")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. spectra (broad + fine + ultrafine), all three cases
# ---------------------------------------------------------------------------
lams_broad = np.arange(1200.0, 1700.001, 1.0)
lams_fine = np.arange(1350.0, 1520.001, 0.25)
lams_ultra = np.arange(LAM_E - 6.0, LAM_E + 6.001, 0.1)
lams_all = np.unique(np.round(np.concatenate([lams_broad, lams_fine, lams_ultra]), 4))
spec = {}
for case in CASES:
    df = spectrum(lams_all, case, tag=case)
    df["case"] = case
    spec[case] = df
    df.drop(columns=["r_xx", "t_xx"]).assign(re_r=df["r_xx"].apply(lambda z: z.real), im_r=df["r_xx"].apply(lambda z: z.imag),
                                              re_t=df["t_xx"].apply(lambda z: z.real), im_t=df["t_xx"].apply(lambda z: z.imag)
                                              ).to_csv(OUT / f"spectrum_{case}.csv", index=False)
    conserv = (df["R"] + df["T"] + df["A"] - 1).abs().max()
    log(f"[{case}] {len(df)} wavelengths; max|R+T+A-1| = {conserv:.1e}; "
        + (f"max|R+T-1| = {(df['R']+df['T']-1).abs().max():.1e} (lossless control)" if case != "with_ITO" else f"A range {df['A'].min():.4f}..{df['A'].max():.4f}"))

# A cross-check vs the ITO volume integral (with ITO) at 11 wavelengths
xc = []
for lam in np.linspace(1380, 1500, 11):
    with torch.no_grad():
        sim = fm.build_sim(rho, P, H, lam=float(lam), order=cm.ORDER)
        A_rt = float(fm.a_ito(sim)[0])
        v = fm.a_ito_volume(sim, P, float(lam), 0.0, n_xy=96, n_z=7)
    xc.append(dict(lam=float(lam), A_rt=A_rt, A_vol=v["A_vol"], diff=A_rt - v["A_vol"]))
pd.DataFrame(xc).to_csv(OUT / "A_volume_crosscheck.csv", index=False)
log(f"[A cross-check] max |A_rt - A_vol| = {max(abs(x['diff']) for x in xc):.1e}")


# ---------------------------------------------------------------------------
# 2. resonance identification (spectral features) + pole cross-check
# ---------------------------------------------------------------------------
def features(df, key, sign, prominence=0.02):
    lam = df["lam"].values; y = df[key].values * sign
    idx, props = find_peaks(y, prominence=prominence)
    w = peak_widths(y, idx, rel_height=0.5)
    out = []
    for k, i in enumerate(idx):
        fwhm = (w[0][k]) * np.median(np.diff(lam[max(i - 5, 0):i + 5]))
        out.append(dict(lambda_nm=float(lam[i]), value=float(df[key].values[i]), prominence=float(props["prominences"][k]),
                        fwhm_nm=float(fwhm), Q_spectral=float(lam[i] / fwhm) if fwhm > 0 else np.nan))
    return out


feat, polelist = {}, {}
for case, df in spec.items():
    fine = df[(df["lam"] >= 1250) & (df["lam"] <= 1650)].sort_values("lam")
    feat[case] = dict(T_dips=features(fine, "T", -1.0), A_peaks=(features(fine, "A", 1.0) if case == "with_ITO" else []),
                      R_peaks=features(fine, "R", 1.0))
    sub = df[(df["lam"] >= 1250) & (df["lam"] <= 1650) & (np.isclose(df["lam"] % 1.0, 0))].sort_values("lam")
    pls = pl.significant_poles(sub["lam"].values, sub["r_xx"].values, sub["t_xx"].values)
    for p in pls:
        p["local_refine"] = pl.local_refine(rho, P, H, CASES[case]["s"], p) if (p["Q"] > 25 and CASES[case]["with_ito"]) else None
    polelist[case] = pls
    log(f"[{case}] T dips: " + ", ".join(f"{f['lambda_nm']:.1f} nm (T={f['value']:.3f}, FWHM {f['fwhm_nm']:.1f}, Q~{f['Q_spectral']:.1f})" for f in feat[case]["T_dips"]))
    log(f"[{case}] r/t poles (1-nm scan, AAA): " + ", ".join(f"{p['lambda_nm']:.1f} nm Q={p['Q']:.1f} peak(r,t)=({p['peak_r']:.2f},{p['peak_t']:.2f}) rt={p['rt_rel_diff']:.1e}"
                                                     + (f" refine={p['local_refine']['converged']}" if p.get('local_refine') else "") for p in pls))
cm.jdump(dict(features=feat, poles={c: [{k: (v if k != "omega" else [v.real, v.imag]) for k, v in p.items()} for p in pls] for c, pls in polelist.items()}),
         OUT / "resonances_and_poles.json")

# Si resonance without ITO = pole in the ENZ region nearest lambda_E with the largest peak fraction
def enz_region(pls, lo=1350, hi=1520):
    return [p for p in pls if lo <= p["lambda_nm"] <= hi]

si_poles = enz_region(polelist["without_ITO"])
si = max(si_poles, key=lambda p: p["peak_r"] + p["peak_t"]) if si_poles else None
hyb = sorted(enz_region(polelist["with_ITO"]), key=lambda p: p["lambda_nm"])
lossless_hyb = sorted(enz_region(polelist["lossless_ITO"]), key=lambda p: p["lambda_nm"])
split = None
if len(hyb) >= 2:
    lo_, hi_ = hyb[0], hyb[-1]
    split = dict(lambda_lower=lo_["lambda_nm"], lambda_upper=hi_["lambda_nm"], delta_lambda_nm=hi_["lambda_nm"] - lo_["lambda_nm"],
                 delta_f_THz=C_NM_THZ / lo_["lambda_nm"] - C_NM_THZ / hi_["lambda_nm"],
                 si_linewidth_nm=(si["lambda_nm"] / si["Q"]) if si else None,
                 si_linewidth_THz=(C_NM_THZ / si["lambda_nm"] / si["Q"]) if si else None,
                 lower_Q=lo_["Q"], upper_Q=hi_["Q"])
log(f"[Si resonance without ITO] {si}")
log(f"[with-ITO ENZ-region poles] {[(round(p['lambda_nm'],1), round(p['Q'],1)) for p in hyb]}  split: {split}")


# ---------------------------------------------------------------------------
# 3. convergence of the ENZ region at [9,9] and [11,11]
# ---------------------------------------------------------------------------
lams_conv = np.arange(1400.0, 1470.001, 1.0)
conv = {}
for case in ("with_ITO", "without_ITO"):
    for od in ([9, 9], [11, 11]):
        df = spectrum(lams_conv, case, order=od, tag=f"{case} {od}")
        df.drop(columns=["r_xx", "t_xx"]).to_csv(OUT / f"conv_{case}_{od[0]}.csv", index=False)
        pls = pl.significant_poles(df["lam"].values, df["r_xx"].values, df["t_xx"].values)
        ref = spec[case][spec[case]["lam"].isin(lams_conv)].sort_values("lam")
        conv[f"{case}_{od[0]}"] = dict(max_dT=float(np.max(np.abs(df["T"].values - ref["T"].values))),
                                      max_dA=float(np.max(np.abs(df["A"].values - ref["A"].values))),
                                      poles=[dict(lambda_nm=p["lambda_nm"], Q=p["Q"]) for p in pls])
        log(f"[conv {case} {od}] max|dT| vs [7,7] = {conv[f'{case}_{od[0]}']['max_dT']:.4f}; poles {[(round(p['lambda_nm'],1), round(p['Q'],1)) for p in pls]}")
cm.jdump(conv, OUT / "convergence.json")


# ---------------------------------------------------------------------------
# 4. detuning sweep in h (anticrossing map)
# ---------------------------------------------------------------------------
H_SWEEP = [220.0, 230.0, 240.0, 250.0, 260.0]
lams_map = np.arange(1350.0, 1520.001, 1.0)
tmap = {c: [] for c in ("with_ITO", "without_ITO")}
traj = {c: [] for c in ("with_ITO", "without_ITO")}
for h in H_SWEEP:
    for case in ("with_ITO", "without_ITO"):
        if h == H:
            df = spec[case][spec[case]["lam"].isin(lams_map)].sort_values("lam")
        else:
            df = spectrum(lams_map, case, h=h, tag=f"{case} h={h:.0f}")
        tmap[case].append(df["T"].values)
        pls = pl.significant_poles(df["lam"].values, df["r_xx"].values, df["t_xx"].values)
        dips = features(df, "T", -1.0, prominence=0.03)
        traj[case].append(dict(h=h, poles=[dict(lambda_nm=p["lambda_nm"], Q=p["Q"], peak=p["peak_r"] + p["peak_t"]) for p in pls],
                               T_dips=dips))
        log(f"[sweep h={h:.0f} {case}] poles {[(round(p['lambda_nm'],1), round(p['Q'],1)) for p in pls]} dips {[round(d['lambda_nm'],1) for d in dips]}")
        pd.DataFrame(dict(lam=df["lam"].values, T=df["T"].values, R=df["R"].values, A=df["A"].values)).to_csv(OUT / f"sweep_{case}_h{h:.0f}.csv", index=False)
cm.jdump(traj, OUT / "detuning_trajectories.json")


# ---------------------------------------------------------------------------
# 5. field maps + ITO metrics at the identified branches
# ---------------------------------------------------------------------------
def glass_slab_metrics(lam, n_xy=96, n_z=7):
    """No-ITO control: fields in the glass slab occupying the ITO position."""
    with torch.no_grad():
        sim = fm.build_sim(rho, P, H, lam=lam, order=cm.ORDER, with_ito=False)
        x, y = fm.cell_axes(P, n_xy)
        E = []
        for zp in (np.arange(n_z) + 0.5) * cm.D_ITO / n_z:
            Ev, _ = sim.field_xy(sim.layer_N, x, y, float(zp))
            E.append(torch.stack(list(Ev), 0))
        E = torch.stack(E, 0)
    E2 = (E.abs() ** 2).sum(1); Ez2 = E[:, 2].abs() ** 2
    return dict(F_Ez=float(Ez2.mean()), F_Etot=float(E2.mean()), eta_z=float(Ez2.sum() / E2.sum()),
                Ez_mid=E[n_z // 2, 2].numpy(), E_mid=E[n_z // 2].numpy())


branches = {}
if si:
    branches["no-ITO Si resonance"] = ("without_ITO", si["lambda_nm"])
for i, p in enumerate(hyb):
    branches[f"with-ITO branch {i+1} ({p['lambda_nm']:.1f} nm)"] = ("with_ITO", p["lambda_nm"])
branches["with-ITO at lambda_E"] = ("with_ITO", LAM_E)
fmaps, fmet = {}, {}
for name, (case, lam) in branches.items():
    if case == "with_ITO":
        m = cm.driven_metrics(rho, P, H, lam=lam, want_field=True)
        en = cm.energy_participation(rho, P, H, lam, 1.0)
        fmaps[name] = (m.pop("Ez_mid"), m.pop("E_mid"))
        fmet[name] = dict(case=case, lam=lam, F_Ez=m["F_Ez"], F_Etot=m["F_Etot"], eta_z=m["eta_z"], A=m["A_rt"], T=m["T"],
                          peak_Ez2=m["peak_Ez2"], ito_E_energy_fraction=en["ito_E_energy_fraction"], eta_ENZ_z=en["eta_ENZ_z"])
    else:
        g = glass_slab_metrics(lam)
        fmaps[name] = (g.pop("Ez_mid"), g.pop("E_mid"))
        fmet[name] = dict(case=case, lam=lam, **g, note="glass slab at the ITO position (no ITO present)")
    log(f"[fields] {name}: {({k: (round(v, 4) if isinstance(v, float) else v) for k, v in fmet[name].items()})}")
cm.jdump(fmet, OUT / "field_metrics.json")
np.savez(OUT / "field_maps.npz", **{k.replace(" ", "_").replace("(", "").replace(")", ""): v[0] for k, v in fmaps.items()})

# ---------------------------------------------------------------------------
# 6. figures
# ---------------------------------------------------------------------------
cols = {"with_ITO": "tab:orange", "without_ITO": "tab:blue", "lossless_ITO": "tab:green"}
def vlines(ax):
    ax.axvline(LAM_E, ls="--", c="k", lw=.9, label="lambda_E = 1433.488 nm")
    ax.axvline(LAM_POLE, ls=":", c="tab:red", lw=.9, label="certified real-ITO pole 1433.985 nm")

fig, ax = plt.subplots(figsize=(10, 5))
for case, df in spec.items():
    d = df[np.isclose(df["lam"] % 1.0, 0)].sort_values("lam")
    ax.plot(d["lam"], d["T"], c=cols[case], label=f"{case.replace('_', ' ')}: T_total")
vlines(ax); ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("power transmittance T_total (all orders)"); ax.set_ylim(0, 1)
ax.set_title("P925_Qt100 (frozen): transmittance with vs without the 23-nm ITO (normal incidence, lab-x, order [7,7])"); ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.savefig(OUT / "P925_Qt100_transmittance_with_vs_without_ITO.png", dpi=160, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
for case, df in spec.items():
    d = df[(df["lam"] >= 1350) & (df["lam"] <= 1520)].sort_values("lam")
    ax.plot(d["lam"], d["T"], c=cols[case], label=f"{case.replace('_', ' ')}: T_total")
for p in hyb:
    ax.axvline(p["lambda_nm"], c="tab:orange", lw=.6, alpha=.6)
if si:
    ax.axvline(si["lambda_nm"], c="tab:blue", lw=.6, alpha=.6)
vlines(ax); ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("T_total"); ax.set_ylim(0, 1)
ax.set_title("ENZ-region zoom (0.25-nm sampling; thin lines = r/t poles: blue no-ITO, orange with-ITO)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.savefig(OUT / "P925_Qt100_transmittance_ENZ_zoom.png", dpi=160, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
d = spec["with_ITO"][np.isclose(spec["with_ITO"]["lam"] % 1.0, 0)].sort_values("lam")
ax.plot(d["lam"], d["T"], label="T_total"); ax.plot(d["lam"], d["R"], label="R_total"); ax.plot(d["lam"], d["A"], label="A_ITO = 1-R-T")
vlines(ax); ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("power fraction"); ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=.3)
ax.set_title("P925_Qt100 with ITO: T, R, A"); fig.savefig(OUT / "P925_Qt100_T_R_A_with_ITO.png", dpi=160, bbox_inches="tight"); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
for case in ("with_ITO", "without_ITO"):
    d = spec[case][np.isclose(spec[case]["lam"] % 1.0, 0)].sort_values("lam")
    axs[0].plot(d["lam"], d["T"], c=cols[case], label=f"{case}: T_total"); axs[0].plot(d["lam"], d["T00"], c=cols[case], ls="--", label=f"{case}: T_00")
    axs[1].plot(d["lam"], d["n_orders"], c=cols[case], label=case)
vlines(axs[0]); axs[0].set_xlabel("wavelength (nm)"); axs[0].set_ylabel("T"); axs[0].legend(fontsize=8); axs[0].grid(alpha=.3); axs[0].set_title("T_total vs zeroth-order T_00")
axs[1].set_xlabel("wavelength (nm)"); axs[1].set_ylabel("# propagating orders (air+glass channels)"); axs[1].grid(alpha=.3); axs[1].legend(fontsize=8)
axs[1].axvline(fm.N_GLASS * P, ls=":", c="k"); axs[1].text(fm.N_GLASS * P + 3, axs[1].get_ylim()[1] * 0.9, "n_glass P (glass (+-1,0) cutoff)", fontsize=8)
fig.savefig(OUT / "P925_Qt100_Ttotal_vs_T00.png", dpi=160, bbox_inches="tight"); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
for ax, case in zip(axs, ("with_ITO", "without_ITO")):
    im = ax.imshow(np.array(tmap[case]), aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1,
                   extent=[lams_map[0], lams_map[-1], H_SWEEP[0] - 5, H_SWEEP[-1] + 5])
    for tr in traj["without_ITO"]:
        for p in tr["poles"]:
            ax.plot(p["lambda_nm"], tr["h"], "o", mfc="none", mec="w", ms=7 if p["peak"] > 0.3 else 4)
    for tr in traj["with_ITO"]:
        for p in tr["poles"]:
            ax.plot(p["lambda_nm"], tr["h"], "x", c="tab:red", ms=7 if p["peak"] > 0.3 else 4)
    ax.axvline(LAM_E, ls="--", c="w", lw=.9); ax.set_xlabel("wavelength (nm)"); ax.set_title(f"T_total map, {case.replace('_', ' ')} (white o: no-ITO poles; red x: with-ITO poles; size ~ pole weight)")
axs[0].set_ylabel("Si height h (nm)"); fig.colorbar(im, ax=axs, shrink=.85, label="T_total")
fig.savefig(OUT / "P925_Qt100_anticrossing_map.png", dpi=160, bbox_inches="tight"); plt.close(fig)

names = list(fmaps)
vmax_z = max(np.abs(fmaps[n][0]).max() for n in names); vmax_e = max(np.sqrt((np.abs(fmaps[n][1]) ** 2).sum(0)).max() for n in names)
fig, axs = plt.subplots(2, len(names), figsize=(4.2 * len(names), 8), squeeze=False)
for j, n in enumerate(names):
    Ez, E = fmaps[n]
    im0 = axs[0, j].imshow(np.abs(Ez).T, origin="lower", cmap="magma", vmin=0, vmax=vmax_z, extent=[0, P, 0, P])
    im1 = axs[1, j].imshow(np.sqrt((np.abs(E) ** 2).sum(0)).T, origin="lower", cmap="viridis", vmin=0, vmax=vmax_e, extent=[0, P, 0, P])
    for ax in axs[:, j]:
        ax.contour(np.linspace(0, P, 128), np.linspace(0, P, 128), rho_np.T, levels=[0.5], colors="w", linewidths=.5)
    axs[0, j].set_title(f"{n}\n|Ez/E_inc| F_Ez={fmet[n]['F_Ez']:.2f} eta_z={fmet[n]['eta_z']:.2f}", fontsize=8); axs[1, j].set_title("|E/E_inc|", fontsize=8)
fig.colorbar(im0, ax=axs[0, :], shrink=.8); fig.colorbar(im1, ax=axs[1, :], shrink=.8)
fig.savefig(OUT / "P925_Qt100_hybrid_field_maps.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------------------
# 7. report
# ---------------------------------------------------------------------------
def fl(v, nd=2):
    return "n/a" if v is None else f"{v:.{nd}f}"

wI = spec["with_ITO"]; wo = spec["without_ITO"]; ll = spec["lossless_ITO"]
atE = {c: spec[c].iloc[(spec[c]["lam"] - LAM_E).abs().argmin()] for c in spec}
L = ["# Stage 17 - P925_Qt100: transmittance with vs without ITO (Karimi Fig. 1(c,d)-style control)", "",
     f"Frozen geometry `{GEO_FILE.relative_to(cm.ROOT)}` (sha256 {hashlib.sha256(GEO_FILE.read_bytes()).hexdigest()[:16]}...), P = {P:.0f} nm, h = {H:.0f} nm, "
     f"padding {PAD_FRAC*P:.1f} nm, normal incidence, lab-x polarization, |E_inc| = 1, TORCWA complex128 order [7,7]; T_total/R_total sum all propagating orders "
     "(coherent p/s), A = 1 - R - T.  No optimization was performed.", "",
     "## Numerical checks", "",
     f"- Energy conservation: without ITO max|R+T-1| = {(wo['R']+wo['T']-1).abs().max():.1e}; lossless ITO max|R+T-1| = {(ll['R']+ll['T']-1).abs().max():.1e}; "
     f"with ITO A in [{wI['A'].min():.4f}, {wI['A'].max():.4f}], A vs ITO volume integral max diff {max(abs(x['diff']) for x in xc):.1e} (11 wavelengths).",
     f"- Order convergence 1400-1470 nm: " + "; ".join(f"{k}: max|dT| = {v['max_dT']:.4f}, poles {[(round(p['lambda_nm'],1), round(p['Q'],1)) for p in v['poles']]}" for k, v in conv.items()),
     f"- Sampling: 1 nm broad (1200-1700), 0.25 nm fine (1350-1520), 0.1 nm within +-6 nm of lambda_E ({len(lams_all)} wavelengths per case).",
     "", "## Values at lambda_E", "", "| case | T_total | T_00 | R_total | A |", "|---|---|---|---|---|"]
for c, r in atE.items():
    L.append(f"| {c} | {r['T']:.4f} | {r['T00']:.4f} | {r['R']:.4f} | {r['A']:.4f} |")
L += ["", "## Resonances (spectral features) and r/t poles", ""]
for c in spec:
    L.append(f"- **{c}** T dips: " + ", ".join(f"{f['lambda_nm']:.1f} nm (T {f['value']:.3f}, FWHM {f['fwhm_nm']:.1f} nm, Q~{f['Q_spectral']:.1f})" for f in feat[c]["T_dips"]) or "- none")
    L.append(f"  poles: " + ", ".join(f"{p['lambda_nm']:.1f} nm (Q {p['Q']:.1f}, peak r/t {p['peak_r']:.2f}/{p['peak_t']:.2f}, r/t diff {p['rt_rel_diff']:.1e}"
                                     + (f", refine {'ok' if p['local_refine']['converged'] else 'NOT converged'}" if p.get('local_refine') else "") + ")" for p in polelist[c]))
L += ["", f"- Underlying Si resonance without ITO (largest-weight pole in 1350-1520 nm): "
      + (f"{si['lambda_nm']:.1f} nm, Q {si['Q']:.1f}, FWHM {si['lambda_nm']/si['Q']:.1f} nm ({C_NM_THZ/si['lambda_nm']/si['Q']:.2f} THz)" if si else "none found"),
      f"- With-ITO poles in 1350-1520 nm: {[(round(p['lambda_nm'],1), round(p['Q'],1)) for p in hyb]}; lossless-ITO poles: {[(round(p['lambda_nm'],1), round(p['Q'],1)) for p in lossless_hyb]}",
      f"- Splitting: " + (f"lower {split['lambda_lower']:.1f} nm (Q {split['lower_Q']:.1f}), upper {split['lambda_upper']:.1f} nm (Q {split['upper_Q']:.1f}); "
                          f"Delta_lambda = {split['delta_lambda_nm']:.1f} nm, Delta_f = {split['delta_f_THz']:.2f} THz vs no-ITO Si linewidth {fl(split['si_linewidth_nm'], 1)} nm ({fl(split['si_linewidth_THz'])} THz)"
                          if split else "only one with-ITO pole in the ENZ region - no splitting"),
      "", "## Detuning sweep (h = 220-260 nm)", ""]
for tr_wo, tr_w in zip(traj["without_ITO"], traj["with_ITO"]):
    L.append(f"- h = {tr_wo['h']:.0f} nm: no-ITO poles {[(round(p['lambda_nm'],1), round(p['Q'],1)) for p in tr_wo['poles']]}; with-ITO poles {[(round(p['lambda_nm'],1), round(p['Q'],1)) for p in tr_w['poles']]}")
L += ["", "![map](P925_Qt100_anticrossing_map.png)", "", "## Field metrics at the branches", "",
      "| branch | case | lambda | F_Ez | F_Etot | eta_z | A | ITO E-energy fraction | eta_ENZ,z |", "|---|---|---|---|---|---|---|---|---|"]
for n, m in fmet.items():
    L.append(f"| {n} | {m['case']} | {m['lam']:.1f} | {m['F_Ez']:.2f} | {m['F_Etot']:.2f} | {m['eta_z']:.2f} | {fl(m.get('A'), 3)} | {fl(m.get('ito_E_energy_fraction'), 3)} | {fl(m.get('eta_ENZ_z'), 4)} |")
L += ["", "![fields](P925_Qt100_hybrid_field_maps.png)", "", "## Figures", "",
      "A `P925_Qt100_transmittance_with_vs_without_ITO.png`, B `P925_Qt100_transmittance_ENZ_zoom.png`, C `P925_Qt100_T_R_A_with_ITO.png`, "
      "D `P925_Qt100_Ttotal_vs_T00.png`, E `P925_Qt100_anticrossing_map.png`, F `P925_Qt100_hybrid_field_maps.png`; data: `spectrum_*.csv`, `conv_*.csv`, `sweep_*.csv`, "
      "`resonances_and_poles.json`, `detuning_trajectories.json`, `field_metrics.json`, `A_volume_crosscheck.csv`.",
      "", "## Interpretation", "", "(written by hand after inspection - see INTERPRETATION.md)"]
(OUT / "REPORT.md").write_text("\n".join(L) + "\n")
log("[stage17] done")
