"""Post-optimization validation: Karimi Fig. 1(c,d)-STYLE with/without-ITO
comparison for the FINAL frozen inverse-designed freeform metasurface.

Mandatory workflow implemented here (steps 6-16 of the task):
  * hard binarization of the final design (threshold 0.5) + soft-vs-hard
    forward check,
  * RCWA convergence verification at the design wavelength ([7,7]->[9,9]
    ->[11,11]),
  * geometry frozen, then two driven simulations that differ ONLY by the
    23-nm ITO film:
        Case A: air / a-Si(140 nm, final binary rho) / glass
        Case B: air / a-Si(140 nm, SAME rho) / ITO(23 nm) / glass
  * broad power-normalized T/R/A spectra (1200-1700 nm, 2 nm) using the
    MEASURED real-frequency ITO permittivity (never the QNM continuation),
    summing transmitted power over ALL glass-propagating orders (the
    (+-1,0)/(0,+-1) orders propagate in glass for lambda < n_glass*Lambda
    ~= 1228 nm),
  * fine scans + resonance extraction (lambda, FWHM, Q, T_min),
  * paper-like EDR benchmark (560 x 500 x 140 nm cuboid, same period),
  * field maps at every relevant resonance (|Ez|, |E|^2, Re Ez, |H|),
  * Fourier-channel comparison (m,n) in {(+-1,0),(0,+-1),(+-1,+-1),(+-2,0)}
    with and without ITO,
  * figures T1-T4 + summary CSV.

Power-normalization audit: torcwa S_parameters(power_norm=True) multiplies
by sqrt(kz_out/kz_in) and the polarization factors (rcwa.py ~l.355-395) so
|S|^2 is a power ratio; this was previously verified against a transfer
matrix to 5e-11 (optional_torcwa_validation.py).  Evanescent orders return
0 by the built-in cutoff, so summing a fixed order list is safe.

a-Si is lossless in this band (measured k = 0 above the gap), so in Case A
the absorption A = 1-R-T must vanish (numerical consistency check), and in
Case B the entire absorption is ITO absorption.

Run:  python validate_with_without_ito.py          (after the optimization)
"""

import json

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import target_mode
import torcwa_forward as fwd
import objective as obj

C_NM_FS = 299.792458
OUT = config.OUT_DIR
FIG = OUT / "figures"

C_BLUE, C_ORANGE, C_AQUA, C_VIOLET, C_MAGENTA = \
    "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": "#e3e2df", "grid.linewidth": 0.6, "font.size": 10.5,
    "lines.linewidth": 2.0, "legend.frameon": False})

GLASS_ORDERS = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]
FOURIER_CHANNELS = [(1, 0), (-1, 0), (0, 1), (0, -1),
                    (1, 1), (1, -1), (-1, 1), (-1, -1), (2, 0), (-2, 0)]


def _save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"figure written: {FIG/name}")


# ---------------------------------------------------------------------------
# forward model with optional ITO layer
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(config.TORCWA_DIR))
import torcwa  # noqa: E402


def build_sim(rho, lam_nm, with_ito, order=None):
    """Stack builder; rho=None -> unpatterned a-Si layer of eps=1 (air)."""
    order = order or config.FOURIER_ORDER
    eps_asi = fwd.eps_asi_of_lambda(lam_nm)
    eps_ito = fwd.eps_ito_of_lambda(lam_nm)      # measured real-axis CSV
    sim = torcwa.rcwa(freq=1.0 / lam_nm, order=order,
                      L=[config.PX_NM, config.PY_NM],
                      dtype=config.SIM_DTYPE, device=config.DEVICE)
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=complex(config.N_GLASS) ** 2)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    if rho is None:
        sim.add_layer(thickness=config.ASI_THICKNESS_NM, eps=1.0)
    else:
        eps_layer = rho * (complex(eps_asi) - 1.0) + 1.0
        sim.add_layer(thickness=config.ASI_THICKNESS_NM,
                      eps=eps_layer.to(config.SIM_DTYPE))
    if with_ito:
        sim.add_layer(thickness=float(config.ITO_THICKNESS_NM), eps=eps_ito)
    sim.solve_global_smatrix()
    sim.source_planewave(amplitude=[1.0, 0.0], direction="forward")
    return sim


def power_RT(sim):
    """Power T (summed over glass-propagating orders) and specular R."""
    T = 0.0
    for o in GLASS_ORDERS:
        for pol in ("xx", "yx"):
            t = sim.S_parameters(orders=o, direction="forward",
                                 port="transmission", polarization=pol,
                                 ref_order=[0, 0])
            T += float(np.abs(t.cpu().numpy().ravel()[0]) ** 2)
    R = 0.0
    for pol in ("xx", "yx"):
        r = sim.S_parameters(orders=[0, 0], direction="forward",
                             port="reflection", polarization=pol,
                             ref_order=[0, 0])
        R += float(np.abs(r.cpu().numpy().ravel()[0]) ** 2)
    return R, T


def spectrum(rho, lams, with_ito, order=None, tag=""):
    R, T = [], []
    with torch.no_grad():
        for i, lam in enumerate(lams):
            r, t = power_RT(build_sim(rho, lam, with_ito, order))
            R.append(r), T.append(t)
            if i % 40 == 0:
                print(f"  [{tag}] lambda {lam:7.1f}: T = {t:.4f}, R = {r:.4f}")
    R, T = np.array(R), np.array(T)
    return {"lam": np.asarray(lams), "R": R, "T": T, "A": 1 - R - T}


# ---------------------------------------------------------------------------
# resonance extraction
# ---------------------------------------------------------------------------
def find_dips(sp, prominence=0.02):
    from scipy.signal import find_peaks
    idx, props = find_peaks(-sp["T"], prominence=prominence)
    return [(float(sp["lam"][i]), float(sp["T"][i])) for i in idx]


def refine_dip(rho, lam0, with_ito, half=20.0, step=0.25, tag=""):
    lams = np.arange(lam0 - half, lam0 + half + step / 2, step)
    sp = spectrum(rho, lams, with_ito, tag=f"{tag} fine")
    i = int(np.argmin(sp["T"]))
    lam_res, T_min = float(sp["lam"][i]), float(sp["T"][i])
    # FWHM of the dip in 1-T relative to the local baseline
    depth = np.percentile(sp["T"], 90) - T_min
    half_level = T_min + depth / 2
    below = sp["T"] <= half_level
    if below.any():
        li = np.where(below)[0]
        fwhm = float(sp["lam"][li[-1]] - sp["lam"][li[0]])
    else:
        fwhm = np.nan
    Q = lam_res / fwhm if fwhm and fwhm > 0 else np.nan
    return {"lambda_nm": lam_res, "T_min": T_min, "fwhm_nm": fwhm, "Q": Q,
            "R_at_res": float(sp["R"][i]), "A_at_res": float(sp["A"][i]),
            "fine": sp}


# ---------------------------------------------------------------------------
# field maps and Fourier channels
# ---------------------------------------------------------------------------
def field_maps(rho, lam, with_ito, tag, y_slice_frac=0.5):
    sim = build_sim(rho, lam, with_ito)
    x = torch.as_tensor((np.arange(340) + 0.5) / 340 * config.PX_NM,
                        dtype=config.GEO_DTYPE, device=config.DEVICE)
    z_top = config.ASI_THICKNESS_NM + float(config.ITO_THICKNESS_NM) + 150.0
    z = torch.linspace(-150.0, z_top, 341, dtype=config.GEO_DTYPE,
                       device=config.DEVICE)
    with torch.no_grad():
        E, H = sim.field_xz(x, z, config.PY_NM * y_slice_frac)
    Ez = E[2].cpu().numpy(); E2 = sum(np.abs(c.cpu().numpy())**2 for c in E)
    Hn = np.sqrt(sum(np.abs(c.cpu().numpy())**2 for c in H))
    fig, axs = plt.subplots(1, 4, figsize=(15.5, 4.0), sharey=True)
    ext = [0, config.PX_NM, float(z[0]), float(z[-1])]
    panels = [(np.abs(Ez), r"$|E_z|$", "Blues", None),
              (E2, r"$|E|^2$", "Blues", None),
              (Ez.real, r"Re $E_z$", "RdBu_r",
               np.nanpercentile(np.abs(Ez.real), 99.5)),
              (Hn, r"$|H|$", "Blues", None)]
    for ax, (v, t, cm, vmax) in zip(axs, panels):
        im = ax.imshow(v.T, origin="lower", cmap=cm, aspect="auto",
                       extent=ext, vmin=-vmax if vmax else None, vmax=vmax)
        ax.axhline(0, color=INK, lw=0.8)
        ax.axhline(config.ASI_THICKNESS_NM, color=INK, lw=0.8)
        if with_ito:
            ax.axhline(config.ASI_THICKNESS_NM + config.ITO_THICKNESS_NM,
                       color=C_ORANGE, lw=1.2)
        ax.set_title(t, fontsize=10); ax.set_xlabel("x (nm)"); ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.85)
    axs[0].set_ylabel("z (nm)  (0 = a-Si top)")
    fig.suptitle(f"{tag}: fields at lambda = {lam:.1f} nm "
                 f"({'with' if with_ito else 'no'} ITO; orange line = ITO)",
                 y=1.02)
    _save(fig, f"fieldmap_{tag}.png")
    return sim


def fourier_channels(rho, lam, with_ito):
    """|Ez| Fourier amplitudes in the 23-nm slab below the a-Si layer.

    With ITO that slab IS the ITO; without ITO it is the corresponding
    glass region, so the comparison probes the same spatial channel.
    """
    sim = build_sim(rho, lam, with_ito)
    n = 96
    x = torch.as_tensor((np.arange(n) + 0.5) / n * config.PX_NM,
                        dtype=config.GEO_DTYPE, device=config.DEVICE)
    y = torch.as_tensor((np.arange(n) + 0.5) / n * config.PY_NM,
                        dtype=config.GEO_DTYPE, device=config.DEVICE)
    layer = 1 if with_ito else None
    amps = {}
    zs = (np.arange(config.Z_SAMPLES_ITO) + 0.5) \
        * config.ITO_THICKNESS_NM / config.Z_SAMPLES_ITO
    with torch.no_grad():
        acc = 0.0
        for zp in zs:
            if with_ito:
                E, _ = sim.field_xy(1, x, y, float(zp))
            else:                       # output layer (glass), depth below a-Si
                E, _ = sim.field_xy(sim.layer_N, x, y, float(zp))
            F = np.fft.fft2(E[2].cpu().numpy()) / n ** 2
            acc = acc + np.abs(F) ** 2
    acc = np.sqrt(acc / len(zs))
    for (m, mn) in [(c, c) for c in FOURIER_CHANNELS] + [((0, 0), (0, 0))]:
        amps[str(m)] = float(acc[m[0] % n, m[1] % n])
    return amps


# ---------------------------------------------------------------------------
def main():
    torch.set_num_threads(config.N_THREADS)
    tgt = target_mode.load_target_npz()
    lam_E = float(tgt["wavelength_nm"])
    config.ITO_THICKNESS_NM = float(tgt["ito_thickness_nm"])
    config.N_GLASS = float(tgt["glass_index"])
    if config.EPS_ASI is None:
        config.EPS_ASI = fwd.eps_asi_of_lambda(lam_E)
    lam_ze = 1419.59

    # ---- step 6: hard binarization + forward check ------------------------
    rho_soft = np.load(OUT / "geometries" / "rho_proj_final.npy")
    rho_hard = (rho_soft > 0.5).astype(float)
    np.save(OUT / "geometries" / "rho_hard_binary.npy", rho_hard)
    rho_soft_t = torch.as_tensor(rho_soft, dtype=config.GEO_DTYPE)
    rho_hard_t = torch.as_tensor(rho_hard, dtype=config.GEO_DTYPE)
    print(f"[binarize] gray pixels changed by threshold: "
          f"{int((np.abs(rho_hard - rho_soft) > 1e-6).sum())} / {rho_soft.size}"
          f"; fill fraction = {rho_hard.mean():.3f}")

    # overlap FoM soft vs hard at the design wavelength
    x, yg = fwd.grid_axes()
    zp = target_mode.ito_z_slices(config.ITO_THICKNESS_NM,
                                  config.Z_SAMPLES_ITO)
    Tp, dV = target_mode.build_target_field(tgt, x.cpu().numpy(),
                                            yg.cpu().numpy(), zp, "+x")
    Tm, _ = target_mode.build_target_field(tgt, x.cpu().numpy(),
                                           yg.cpu().numpy(), zp, "-x")
    with torch.no_grad():
        Ez_ref = fwd.ez_in_ito(fwd.build_solved_sim(
            None, lam_E, fwd.eps_ito_of_lambda(lam_E), config.N_GLASS),
            x, yg, zp)
        Fvals = {}
        for name, r in (("soft", rho_soft_t), ("hard", rho_hard_t)):
            sim = fwd.build_solved_sim(r, lam_E,
                                       fwd.eps_ito_of_lambda(lam_E),
                                       config.N_GLASS)
            Ez = fwd.ez_in_ito(sim, x, yg, zp) - Ez_ref
            F, _d = obj.enz_objective(Tp, Ez, dV, fwd.p_inc_cell(),
                                      target_minus=Tm, direction="bidir")
            Fvals[name] = float(F)
    print(f"[binarize] F_bidir soft = {Fvals['soft']:.4e}, "
          f"hard = {Fvals['hard']:.4e} "
          f"(change {abs(Fvals['hard']-Fvals['soft'])/Fvals['soft']:.2%})")

    # ---- step 7: RCWA convergence at the design wavelength ----------------
    print("[convergence] T at lambda_E, with ITO, hard-binary design:")
    conv = {}
    for od in ([7, 7], [9, 9], [11, 11]):
        with torch.no_grad():
            _r, t = power_RT(build_sim(rho_hard_t, lam_E, True, order=od))
        conv[str(od)] = t
        print(f"    order {od}: T = {t:.5f}")
    c79 = abs(conv["[9, 9]"] - conv["[7, 7]"]) / conv["[9, 9]"]
    c911 = abs(conv["[11, 11]"] - conv["[9, 9]"]) / conv["[11, 11]"]
    print(f"    relative change [7,7]->[9,9]: {c79:.2%}, "
          f"[9,9]->[11,11]: {c911:.2%}")

    # ---- steps 8-10: frozen geometry, broad spectra -----------------------
    lams = np.arange(1200.0, 1700.5, 2.0)
    print("[spectra] Case A: freeform / glass (no ITO)")
    spA = spectrum(rho_hard_t, lams, False, tag="A")
    print("[spectra] Case B: freeform / ITO / glass")
    spB = spectrum(rho_hard_t, lams, True, tag="B")
    print(f"[consistency] max|A(lambda)| no-ITO = "
          f"{np.abs(spA['A']).max():.2e} (a-Si and glass lossless -> ~0)")

    # ---- EDR paper-like benchmark ----------------------------------------
    nx, ny = config.NX_DESIGN, config.NY_DESIGN
    xg = (np.arange(nx) + 0.5) / nx * config.PX_NM
    ygg = (np.arange(ny) + 0.5) / ny * config.PY_NM
    X, Y = np.meshgrid(xg, ygg, indexing="ij")
    rho_edr = (((np.abs(X - config.PX_NM / 2) < 280.0)
                & (np.abs(Y - config.PY_NM / 2) < 250.0)).astype(float))
    rho_edr_t = torch.as_tensor(rho_edr, dtype=config.GEO_DTYPE)
    print("[spectra] EDR-like cuboid 560x500x140, no ITO")
    spEA = spectrum(rho_edr_t, lams, False, tag="EDR-A")
    print("[spectra] EDR-like cuboid, with ITO")
    spEB = spectrum(rho_edr_t, lams, True, tag="EDR-B")

    np.savez(OUT / "histories" / "with_without_ito_spectra.npz",
             lam=lams, freeform_noITO_T=spA["T"], freeform_noITO_R=spA["R"],
             freeform_ITO_T=spB["T"], freeform_ITO_R=spB["R"],
             edr_noITO_T=spEA["T"], edr_ITO_T=spEB["T"],
             edr_noITO_R=spEA["R"], edr_ITO_R=spEB["R"])

    # ---- steps 11-12: dips + fine scans ----------------------------------
    rows = []
    dip_sets = {}
    for tag, sp, rho_t, w_ito in (("freeform_noITO", spA, rho_hard_t, False),
                                  ("freeform_ITO", spB, rho_hard_t, True),
                                  ("EDR_noITO", spEA, rho_edr_t, False),
                                  ("EDR_ITO", spEB, rho_edr_t, True)):
        dips = find_dips(sp)
        print(f"[dips] {tag}: {[f'{l:.0f} nm (T={t:.3f})' for l, t in dips]}")
        refined = []
        for lam0, _ in dips:
            r = refine_dip(rho_t, lam0, w_ito, tag=tag)
            r["case"] = tag
            refined.append(r)
            rows.append({k: v for k, v in r.items() if k != "fine"})
        dip_sets[tag] = refined
    pd.DataFrame(rows).to_csv(OUT / "histories" / "resonance_summary.csv",
                              index=False)
    print(f"[saved] {OUT/'histories'/'resonance_summary.csv'}")

    # ---- figures T1-T4 ----------------------------------------------------
    qnm_ref = lam_E
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(lams, spA["T"], color=C_BLUE,
            label="optimized freeform / glass (no ITO)")
    ax.plot(lams, spB["T"], color=C_ORANGE,
            label="optimized freeform / 23-nm ITO / glass")
    ax.axvline(lam_ze, color=C_VIOLET, ls="--", lw=1.1)
    ax.axvline(qnm_ref, color=C_AQUA, ls="--", lw=1.1)
    ax.text(lam_ze - 6, 0.06, f"material ENZ {lam_ze:.0f} nm", rotation=90,
            fontsize=8.5, color=C_VIOLET)
    ax.text(qnm_ref + 3, 0.06, f"bare-film QNM $G_{{10}}$ {qnm_ref:.0f} nm",
            rotation=90, fontsize=8.5, color=C_AQUA)
    ax.axvline(1460, color=INK2, ls=":", lw=0.9)
    ax.text(1462, 0.06, "Karimi ~1460 nm (their sample, reference only)",
            rotation=90, fontsize=8, color=INK2)
    for r in dip_sets["freeform_noITO"]:
        ax.annotate(f"{r['lambda_nm']:.0f}", (r["lambda_nm"], r["T_min"]),
                    textcoords="offset points", xytext=(-4, -14),
                    color=C_BLUE, fontsize=9)
    for r in dip_sets["freeform_ITO"]:
        ax.annotate(f"{r['lambda_nm']:.0f}", (r["lambda_nm"], r["T_min"]),
                    textcoords="offset points", xytext=(-4, -14),
                    color=C_ORANGE, fontsize=9)
    ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("power transmittance T")
    ax.set_ylim(0, 1.02)
    ax.set_title("Fig. T1 - transmittance of the final inverse-designed "
                 "freeform metasurface\nwith and without the 23-nm ITO layer "
                 "(same frozen geometry; analogous to Karimi Fig. 1(c,d))")
    ax.legend(loc="lower left", fontsize=9)
    _save(fig, "T1_freeform_with_without_ITO.png")

    fig, axs = plt.subplots(1, 2, figsize=(12.6, 4.4), sharey=True)
    for ax, sp, t in ((axs[0], spA, "no ITO"), (axs[1], spB, "with ITO")):
        ax.plot(lams, sp["T"], color=C_BLUE, label="T")
        ax.plot(lams, sp["R"], color=C_ORANGE, label="R")
        ax.plot(lams, sp["A"], color=C_AQUA, label="A = 1-R-T")
        ax.set_xlabel("wavelength (nm)"); ax.set_title(f"final freeform, {t}")
    axs[0].set_ylabel("power fraction"); axs[0].legend(fontsize=9)
    axs[1].text(1210, 0.9, "A = ITO absorption\n(only lossy layer)",
                fontsize=9, color=C_AQUA)
    fig.suptitle("Fig. T2 - R/T/A of the final freeform design", y=1.02)
    _save(fig, "T2_RTA.png")

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for r in dip_sets["freeform_noITO"]:
        ax.plot(r["fine"]["lam"], r["fine"]["T"], color=C_BLUE)
    for r in dip_sets["freeform_ITO"]:
        ax.plot(r["fine"]["lam"], r["fine"]["T"], color=C_ORANGE)
    ax.axvline(lam_ze, color=C_VIOLET, ls="--", lw=1.0)
    ax.axvline(qnm_ref, color=C_AQUA, ls="--", lw=1.0)
    for r in dip_sets["freeform_noITO"] + dip_sets["freeform_ITO"]:
        col = C_BLUE if "no" in r["case"] else C_ORANGE
        ax.annotate(f"{r['lambda_nm']:.1f} nm\nQ={r['Q']:.0f}"
                    if np.isfinite(r["Q"]) else f"{r['lambda_nm']:.1f} nm",
                    (r["lambda_nm"], r["T_min"]),
                    textcoords="offset points", xytext=(6, -4), fontsize=8.5,
                    color=col)
    ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("T")
    ax.set_title("Fig. T3 - fine scans around all resonances "
                 "(blue: no ITO, orange: with ITO)")
    _save(fig, "T3_fine_scans.png")

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(lams, spEA["T"], color=C_BLUE, label="EDR-like cuboid / glass")
    ax.plot(lams, spEB["T"], color=C_ORANGE,
            label="EDR-like cuboid / 23-nm ITO / glass")
    ax.axvline(lam_ze, color=C_VIOLET, ls="--", lw=1.0)
    ax.axvline(qnm_ref, color=C_AQUA, ls="--", lw=1.0)
    ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("T"); ax.set_ylim(0, 1.02)
    ax.set_title("Fig. T4 - paper-like EDR benchmark (560x500x140 nm cuboid, "
                 "$\\Lambda$=850 nm)\nour materials; NOT a reproduction of "
                 "the published experimental spectrum")
    ax.legend(loc="lower left", fontsize=9)
    _save(fig, "T4_EDR_benchmark.png")

    # ---- step 13: field maps at the relevant resonances -------------------
    for r in dip_sets["freeform_noITO"]:
        field_maps(rho_hard_t, r["lambda_nm"], False,
                   f"freeform_noITO_{r['lambda_nm']:.0f}nm")
    for r in dip_sets["freeform_ITO"]:
        field_maps(rho_hard_t, r["lambda_nm"], True,
                   f"freeform_ITO_{r['lambda_nm']:.0f}nm")

    # ---- step 14: Fourier channels ---------------------------------------
    lam_probe = (dip_sets["freeform_ITO"][0]["lambda_nm"]
                 if dip_sets["freeform_ITO"] else lam_E)
    fc = {"lambda_probe_nm": lam_probe,
          "with_ITO": fourier_channels(rho_hard_t, lam_probe, True),
          "no_ITO": fourier_channels(rho_hard_t, lam_probe, False),
          "note": ("rms |Ez| Fourier amplitude over the 23-nm slab below the "
                   "a-Si (ITO when present, glass otherwise)")}
    with open(OUT / "histories" / "fourier_channels.json", "w") as f:
        json.dump(fc, f, indent=1)
    print("[fourier] |Ez| channel amplitudes at "
          f"{lam_probe:.1f} nm (with / without ITO):")
    for ch in ["(0, 0)"] + [str(c) for c in FOURIER_CHANNELS]:
        w = fc["with_ITO"][ch]; wo = fc["no_ITO"][ch]
        print(f"    {ch:9s}: {w:.3e} / {wo:.3e}  "
              f"(ratio {w/max(wo,1e-30):.2f})")

    print("\n[done] all validation outputs saved.")
    return dip_sets, conv, Fvals


if __name__ == "__main__":
    main()
