"""FDTDX field plotting + TORCWA cross-check for the frozen validation candidates, from the RAW phasor files
(no FDTD is re-run here).  Reuses the validated post-processing routines of the four-finalists package
(fx_post.plane_flux / spectra_of / geometry / normalized_fields, fx_sim.eps_model / area_fraction).

    /opt/venv-fdtdx/bin/python fx_fields_plots.py --design <tag> [--run prod] [--ref prod]

Per candidate (all under <tag>/):
  raw_fields/fields_normalized.npz      complex E/E_inc (and H/E_inc if recorded) on the volume block + true x/y/z
                                        centres AND edges, layer indices, boundaries, normalization, conventions
  spectra/spectra_<tag>.{csv,npz,png}   FDTDX R, T, A vs TORCWA ([9,9] Stage-B cold state) + convergence/artefact checks
  profiles/zprofile_<tag>.{csv,npz,png} <|E|^2>_xy(z), <|Ez|^2>_xy(z), <|Dz|^2>_xy(z) over air / a-Si:H / ITO / glass
  xy_slices/Ez2_xy_slices_<tag>_{lin,log}.png + .npz   many-z |Ez|^2 maps, COMMON colour scale
  xz_yz/Ez2_xz_<tag>.png, Ez2_yz_<tag>.png             full-height cuts drawn with the TRUE cell edges (pcolormesh)
  metadata/plot_manifest.json           provenance of every figure

Conventions.  z = 0 at the bottom of the computational domain (inside the glass PML); +z points UP from the
glass through the ITO and the a-Si:H into the air; the source propagates towards -z.  Phasors are in the
exp(-i w t) convention (same as TORCWA); E is normalized by the incident Ex phasor of the empty-cell reference
run at the ITO mid-plane position, so |E/E_inc|^2 is directly comparable with TORCWA's |E_inc| = 1 fields.
|Ez|^2 = Ez * conj(Ez).  D_z is reconstructed as eps_zz(x, y, z, lambda) * Ez (eps0 omitted, i.e. in units of
eps0 V/m) with the material assignment of the FDTDX model: fitted eps_glass(lambda) in the glass cells, fitted
eps_ITO(lambda) in the ITO cells, 1 in the air cells, and in the patterned a-Si:H cells the zz component of the
sub-pixel-smoothed tensor, which for vertical (in-plane-normal) interfaces is the arithmetic mean
f eps_aSiH + (1 - f) with f the exact area fill of the cell (fx_sim.area_fraction); the horizontal layer
interfaces coincide with cell edges, so no z-mixing occurs.
"""
import argparse, hashlib, json, os, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parent
FX = HERE.parent.parent / "fdtdx_four_finalists"
sys.path.insert(0, str(FX))
import fx_sim                                   # noqa: E402
import fx_post                                  # noqa: E402

C0 = 299792458.0
MANIFEST = HERE / "manifest.json"


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_run(d):
    d = Path(d)
    if not (d / "phasors.npz").exists():
        raise FileNotFoundError(d / "phasors.npz")
    z = np.load(d / "phasors.npz"); meta = json.load(open(d / "run_meta.json"))
    return dict(z=fx_post._ScaledPhasors(z, meta["dt_s"]), meta=meta, dir=d, raw=z)


def eps_zz_block(run, g, lam_nm, fill):
    """Per-cell eps_zz on the volume block (nx, ny, nz_vol) for one wavelength."""
    idx = g["idx"]; zv0, zv1 = idx["z_vol0"], idx["z_vol1"]
    nz = zv1 - zv0; n = run["meta"]["nxy"]
    e = np.ones((n, n, nz), complex)
    e_g = complex(fx_sim.eps_model("glass", lam_nm * 1e-9)); e_i = complex(fx_sim.eps_model("ITO", lam_nm * 1e-9)); e_a = complex(fx_sim.eps_model("aSiH", lam_nm * 1e-9))
    for k in range(nz):
        kz = zv0 + k
        if kz < idx["z_ito0"]:
            e[:, :, k] = e_g
        elif kz < idx["z_asi0"]:
            e[:, :, k] = e_g if run["meta"]["no_ito"] else e_i
        elif kz < idx["z_asi1"]:
            e[:, :, k] = fill * e_a + (1.0 - fill)
        else:
            e[:, :, k] = 1.0
    return e


def region_of(kz, idx):
    if kz < idx["z_ito0"]:
        return "glass"
    if kz < idx["z_asi0"]:
        return "ITO"
    if kz < idx["z_asi1"]:
        return "a-Si:H"
    return "air"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True); ap.add_argument("--run", default="prod"); ap.add_argument("--ref", default="prod")
    ap.add_argument("--torcwa-scan", default=None, help="Stage-B stageB_scan.npz of the same candidate (default: ../outputs/candidates/<tag>/)")
    a = ap.parse_args()
    tag = a.design
    man = json.load(open(MANIFEST)); cand = man["candidates"][tag]
    run = load_run(HERE / tag / "raw_fields" / a.run); ref = load_run(HERE / "reference" / f"{tag}_{a.ref}")
    for sub in ("raw_fields", "spectra", "profiles", "xy_slices", "xz_yz", "metadata"):
        (HERE / tag / sub).mkdir(parents=True, exist_ok=True)
    g = fx_post.geometry(run); idx = g["idx"]; meta = run["meta"]
    comps = meta.get("vol_components", ["Ex", "Ey", "Ez"])
    F = fx_post.normalized_fields(run, ref)                 # vol: (nlam, ncomp, nx, ny, nz), xz: (nlam, ncomp, nx, nz), yz: (nlam, ncomp, ny, nz)
    lam = F["lam"]; n = meta["nxy"]
    rho = np.load(HERE / tag / "geometry" / "rho_hard_binary.npy"); fill = fx_sim.area_fraction(rho, n)
    zv0, zv1 = idx["z_vol0"], idx["z_vol1"]
    ze_v = g["ze"][zv0:zv1 + 1]; zc_v = g["zc"][zv0:zv1]; dz_v = g["dz"][zv0:zv1]
    xe = g["dxy"] * np.arange(n + 1); xc = g["xc"]
    z_ito_bot, z_ito_top, z_asi_top = g["ze"][idx["z_ito0"]], g["ze"][idx["z_asi0"]], g["ze"][idx["z_asi1"]]
    prov = dict(raw_phasors=str(run["dir"] / "phasors.npz"), raw_sha256=sha_file(run["dir"] / "phasors.npz"), run_meta=str(run["dir"] / "run_meta.json"),
                reference_phasors=str(ref["dir"] / "phasors.npz"), reference_sha256=sha_file(ref["dir"] / "phasors.npz"),
                geometry_file=str(HERE / tag / "geometry" / "rho_hard_binary.npy"), geometry_sha256_file=sha_file(HERE / tag / "geometry" / "rho_hard_binary.npy"),
                geometry_sha256_uint8=cand["sha256_uint8_array"], wavelengths_nm=[float(l) for l in lam],
                normalization="E / E_inc, E_inc = incident Ex phasor of the empty-cell reference at the ITO mid-plane position; |E_inc| = 1 equivalent",
                time_convention="exp(-i w t)", coordinates="z = 0 at the domain bottom (glass PML); +z upward glass -> ITO -> a-Si:H -> air; source propagates -z; x, y cell-centred, period P",
                boundaries_nm=dict(ITO_bottom=z_ito_bot * 1e9, ITO_top=z_ito_top * 1e9, aSiH_top=z_asi_top * 1e9, volume_block=[ze_v[0] * 1e9, ze_v[-1] * 1e9]),
                n_ito_cells=meta["n_ito"], dz_ito_nm=meta["dz_ito_m"] * 1e9, field_components_recorded=comps, E_inc_nonuniformity=[float(x) for x in F["E0_nonuniformity"]],
                material_eps_at_lams={l: dict(ITO=[complex(fx_sim.eps_model("ITO", l * 1e-9)).real, complex(fx_sim.eps_model("ITO", l * 1e-9)).imag],
                                              aSiH=complex(fx_sim.eps_model("aSiH", l * 1e-9)).real, glass=complex(fx_sim.eps_model("glass", l * 1e-9)).real) for l in [float(x) for x in lam]})
    # ---- 1. raw normalized complex fields on the block ----------------------------------------------------
    iE = [comps.index(c) for c in ("Ex", "Ey", "Ez")]
    E = F["vol"][:, iE]
    save = dict(lam_nm=lam, E_over_Einc=E.astype(np.complex64), x_centers_m=xc, y_centers_m=xc, x_edges_m=xe, y_edges_m=xe, z_centers_m=zc_v, z_edges_m=ze_v, dz_m=dz_v,
                z_edges_full_domain_m=g["ze"], layer_index_in_block=dict(ito0=idx["z_ito0"] - zv0, asi0=idx["z_asi0"] - zv0, asi1=idx["z_asi1"] - zv0, n_block=zv1 - zv0),
                layer_index_full=dict(z_ito0=idx["z_ito0"], z_asi0=idx["z_asi0"], z_asi1=idx["z_asi1"], z_vol0=zv0, z_vol1=zv1, z_T=idx["z_T"], z_R=idx["z_R"], z_src=idx["z_src"]),
                boundaries_nm=np.array([z_ito_bot, z_ito_top, z_asi_top]) * 1e9, E_inc_phasor=F["E0"], fill_fraction_per_cell=fill, geometry_sha256_uint8=cand["sha256_uint8_array"],
                note=json.dumps(dict(prov, material_eps_at_lams=None)))
    if all(c in comps for c in ("Hx", "Hy", "Hz")):
        save["H_over_Einc"] = F["vol"][:, [comps.index(c) for c in ("Hx", "Hy", "Hz")]].astype(np.complex64)
        save["H_note"] = "H phasors (FDTDX eta0-normalized H) divided by the same E_inc phasor"
    np.savez_compressed(HERE / tag / "raw_fields" / "fields_normalized.npz", **save)
    # ---- 2. spectra vs TORCWA + checks ----------------------------------------------------------------------
    s = fx_post.spectra_of(run, ref)
    win = (s["lam"] >= 1252.0) & (s["lam"] <= 1400.0)
    checks = dict(R_max=float(s["R"][win].max()), T_min=float(s["T"][win].min()), A_min=float(s["A"][win].min()), A_max=float(s["A"][win].max()),
                  closure_note="A = 1 - R - T by definition; artefacts show up as R > 1, T < 0 or A < 0",
                  leak_over_Pinc_max=float(np.abs(s["leak_over_Pinc"]).max()), window_nm=[1252.0, 1400.0])
    conv = {}
    for name in [k.split("/")[0] for k in run["z"].files if k.startswith("T_plane_early") and k.endswith("/phasor")]:
        sfx = name.replace("T_plane", "")
        Se = fx_post.plane_flux(run["z"][f"T_plane{sfx}/phasor"], g["dxy"]); Re_ = fx_post.plane_flux(run["z"][f"R_plane{sfx}/phasor"], g["dxy"])
        Te = -Se / s["P_inc"]; Rr = (Re_ - s["P_inc"] * s["leak_over_Pinc"]) / s["P_inc"]
        conv[sfx] = dict(max_abs_dT_vs_full=float(np.abs(Te - s["T"])[win].max()), max_abs_dR_vs_full=float(np.abs(Rr - s["R"])[win].max()),
                         dT_at_lams={f"{l:.1f}": float(np.interp(l, s["lam"], Te) - np.interp(l, s["lam"], s["T"])) for l in lam})
    checks["early_window_convergence"] = conv
    dec = {}
    for k in ("probe_asi", "probe_ito"):
        if f"{k}/fields" in run["z"].files:
            f = run["z"][f"{k}/fields"]; amp = np.abs(f).max(axis=1); dec[k] = dict(end_over_peak=float(amp[-50:].max() / amp.max()), peak_time_fs=float(np.argmax(amp) * meta["dt_s"] * 1e15))
    checks["decay_probes"] = dec
    checks["time_simulated_fs"] = meta["time_s"] * 1e15; checks["steps"] = meta["n_steps_run"]; checks["final_max_abs_E"] = meta["max_abs_E_final"]
    tor = None
    tpath = Path(a.torcwa_scan) if a.torcwa_scan else HERE.parent / "outputs" / "candidates" / tag / "stageB_scan.npz"
    if tpath.exists():
        z = np.load(tpath); tor = dict(lam=z["lam"], T=z["T0"], R=z["R0"], A=z["A0"], Fz=z["Fz0"], Ftot=z["Ftot0"], Ez2=z["mean_Ez2_0"], order=[int(x) for x in z["order"]], file=str(tpath))
        ov = (s["lam"] >= tor["lam"].min()) & (s["lam"] <= tor["lam"].max()) & win
        Ti = np.interp(s["lam"][ov], tor["lam"], tor["T"]); Ri = np.interp(s["lam"][ov], tor["lam"], tor["R"]); Ai = np.interp(s["lam"][ov], tor["lam"], tor["A"])
        checks["torcwa_vs_fdtdx"] = dict(order=tor["order"], max_abs_dT=float(np.abs(s["T"][ov] - Ti).max()), rms_dT=float(np.sqrt(np.mean((s["T"][ov] - Ti) ** 2))),
                                         max_abs_dR=float(np.abs(s["R"][ov] - Ri).max()), max_abs_dA=float(np.abs(s["A"][ov] - Ai).max()),
                                         lam_T_min=dict(fdtdx=float(s["lam"][ov][np.argmin(s["T"][ov])]), torcwa=float(tor["lam"][np.argmin(tor["T"])])),
                                         lam_A_max=dict(fdtdx=float(s["lam"][ov][np.argmax(s["A"][ov])]), torcwa=float(tor["lam"][np.argmax(tor["A"])])),
                                         at_lams={f"{l:.1f}": dict(T_fdtdx=float(np.interp(l, s["lam"], s["T"])), T_torcwa=float(np.interp(l, tor["lam"], tor["T"])),
                                                                   R_fdtdx=float(np.interp(l, s["lam"], s["R"])), R_torcwa=float(np.interp(l, tor["lam"], tor["R"])),
                                                                   A_fdtdx=float(np.interp(l, s["lam"], s["A"])), A_torcwa=float(np.interp(l, tor["lam"], tor["A"]))) for l in lam})
    with open(HERE / tag / "spectra" / f"spectra_{tag}.csv", "w") as f:
        f.write("lambda_nm,R_fdtdx,T_fdtdx,A_fdtdx,T_torcwa,R_torcwa,A_torcwa\n")
        for i in range(len(s["lam"])):
            tt = [np.interp(s["lam"][i], tor["lam"], tor[k]) if tor is not None else np.nan for k in ("T", "R", "A")]
            f.write(f"{s['lam'][i]:.2f},{s['R'][i]:.6f},{s['T'][i]:.6f},{s['A'][i]:.6f},{tt[0]:.6f},{tt[1]:.6f},{tt[2]:.6f}\n")
    np.savez(HERE / tag / "spectra" / f"spectra_{tag}.npz", lam_nm=s["lam"], R=s["R"], T=s["T"], A=s["A"], P_inc=s["P_inc"], **({f"torcwa_{k}": tor[k] for k in ("lam", "T", "R", "A")} if tor else {}))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for k, c in (("R", "b"), ("T", "r"), ("A", "k")):
        ax.plot(s["lam"], s[k], c + "-", lw=1.4, label=f"{k} FDTDX")
        if tor is not None:
            ax.plot(tor["lam"], tor[k], c + "--", lw=1.0, label=f"{k} TORCWA [{tor['order'][0]},{tor['order'][1]}]")
    for l in lam:
        ax.axvline(l, color="g", ls=":", lw=0.6)
    ax.axvline(1251.2, color="m", ls=":", lw=0.6, label="glass Rayleigh 1251 nm"); ax.set_xlabel("wavelength [nm]"); ax.set_ylabel("R, T, A"); ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    ax.set_title(f"{tag}: FDTDX ({meta['nxy']}^2 x {idx['n_z']} cells, {meta['n_ito']} ITO cells, {meta['time_s']*1e15:.0f} fs) vs TORCWA cold state", fontsize=9)
    fig.tight_layout(); fig.savefig(HERE / tag / "spectra" / f"spectra_{tag}.png", dpi=150); plt.close(fig)
    # ---- 3. z profiles ----------------------------------------------------------------------------------
    prof = {}
    fig, axs = plt.subplots(3, 1, figsize=(8, 9.5), sharex=True)
    for il, l in enumerate(lam):
        Ez = E[il, 2]; E2 = (np.abs(E[il]) ** 2).sum(0); Ez2 = np.abs(Ez) ** 2
        ezz = eps_zz_block(run, g, l, fill); Dz2 = np.abs(ezz * Ez) ** 2
        p = dict(z_nm=zc_v * 1e9, z_minus_ITO_top_nm=(zc_v - z_ito_top) * 1e9, E2=E2.mean((0, 1)), Ez2=Ez2.mean((0, 1)), Dz2=Dz2.mean((0, 1)),
                 region=[region_of(zv0 + k, idx) for k in range(len(zc_v))])
        prof[f"{l:.3f}"] = p
        for ax, k, lab in zip(axs, ("E2", "Ez2", "Dz2"), (r"$\langle|E/E_{inc}|^2\rangle_{xy}$", r"$\langle|E_z/E_{inc}|^2\rangle_{xy}$", r"$\langle|D_z/(\epsilon_0 E_{inc})|^2\rangle_{xy}$")):
            ax.plot(p["z_nm"], p[k], "-o", ms=2.5, lw=1.0, label=f"{l:.1f} nm"); ax.set_ylabel(lab, fontsize=9)
        with open(HERE / tag / "profiles" / f"zprofile_{tag}_{l:.1f}nm.csv", "w") as f:
            f.write("z_nm,z_minus_ITO_top_nm,region,mean_E2,mean_Ez2,mean_Dz2\n")
            for k in range(len(zc_v)):
                f.write(f"{p['z_nm'][k]:.4f},{p['z_minus_ITO_top_nm'][k]:.4f},{p['region'][k]},{p['E2'][k]:.6e},{p['Ez2'][k]:.6e},{p['Dz2'][k]:.6e}\n")
    for ax in axs:
        for zb, lab in ((z_ito_bot, "glass | ITO"), (z_ito_top, "ITO | a-Si:H"), (z_asi_top, "a-Si:H | air")):
            ax.axvline(zb * 1e9, color="k", ls="--", lw=0.7)
        ax.axvspan(z_ito_bot * 1e9, z_ito_top * 1e9, color="orange", alpha=0.25); ax.grid(alpha=0.3); ax.set_yscale("log")
    axs[0].legend(fontsize=7, ncol=3); axs[-1].set_xlabel("z [nm]  (0 = domain bottom in the glass PML; +z upward; shaded = 23-nm ITO; dashed = material boundaries)")
    sec = axs[0].secondary_xaxis("top", functions=(lambda z: z - z_ito_top * 1e9, lambda z: z + z_ito_top * 1e9)); sec.set_xlabel("z - z(ITO top) [nm]")
    fig.suptitle(f"{tag}: full-stack xy-averaged profiles (FDTDX raw fields; log scale, common physical normalization)", fontsize=9)
    fig.tight_layout(); fig.savefig(HERE / tag / "profiles" / f"zprofile_{tag}.png", dpi=150); plt.close(fig)
    np.savez(HERE / tag / "profiles" / f"zprofile_{tag}.npz", **{f"{k}_{l}": np.asarray(v) for l, p in prof.items() for k, v in p.items() if k != "region"},
             region=np.array(prof[f"{lam[0]:.3f}"]["region"]), lam_nm=lam)
    # ---- 4. many-z |Ez|^2 xy maps (common scales) ----------------------------------------------------------
    a0, a1 = idx["z_asi0"] - zv0, idx["z_asi1"] - zv0; i0, i1 = idx["z_ito0"] - zv0, idx["z_asi0"] - zv0
    n_asi = a1 - a0
    asi_sel = sorted({a1 - 1, a0 + int(0.75 * n_asi), a0 + n_asi // 2, a0 + n_asi // 4, a0 + 1, a0})
    ito_sel = list(range(i0, i1))
    glass_sel = [k for k in (i0 - 1, i0 - 2, i0 - 4, i0 - 8, i0 - 16) if k >= 0]
    air_sel = [k for k in (a1, a1 + 3) if k < zv1 - zv0]
    sel = sorted(set(air_sel + asi_sel + ito_sel + glass_sel), reverse=True)        # top (air) -> bottom (glass)
    slices = {}
    for il, l in enumerate(lam):
        Ez2 = np.abs(E[il, 2]) ** 2
        maps = np.stack([Ez2[:, :, k] for k in sel]); vmax = float(maps.max()); vmin_log = max(vmax * 1e-4, 1e-12)
        slices[f"{l:.3f}"] = dict(k=sel, z_nm=[float(zc_v[k] * 1e9) for k in sel], region=[region_of(zv0 + k, idx) for k in sel], maps=maps.astype(np.float32), vmax=vmax)
        for scale in ("lin", "log"):
            nc = 6; nr = int(np.ceil(len(sel) / nc))
            fig, axs = plt.subplots(nr, nc, figsize=(3.1 * nc, 3.3 * nr))
            for ax, k, m in zip(axs.flat, sel, maps):
                im = ax.pcolormesh(xe * 1e9, xe * 1e9, m.T, cmap="inferno", shading="flat", **(dict(vmin=0, vmax=vmax) if scale == "lin" else dict(norm=LogNorm(vmin=vmin_log, vmax=vmax))))
                ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"z = {zc_v[k]*1e9:.1f} nm  {region_of(zv0 + k, idx)}\n(z - ITO top = {(zc_v[k]-z_ito_top)*1e9:+.1f} nm)  max {m.max():.2f}", fontsize=7)
            for ax in list(axs.flat)[len(sel):]:
                ax.axis("off")
            fig.colorbar(im, ax=list(axs.flat), shrink=0.6, pad=0.01, label=r"$|E_z/E_{inc}|^2$" + (" (common linear scale)" if scale == "lin" else " (common log scale, 4 decades)"))
            fig.suptitle(f"{tag}: |Ez|^2 xy maps at lambda = {l:.2f} nm from the FDTDX volume phasors; panels top (air) to bottom (glass); ONE colour scale (max {vmax:.2f})", fontsize=9)
            fig.savefig(HERE / tag / "xy_slices" / f"Ez2_xy_slices_{tag}_{l:.1f}nm_{scale}.png", dpi=130); plt.close(fig)
    np.savez_compressed(HERE / tag / "xy_slices" / f"Ez2_xy_slices_{tag}.npz", lam_nm=lam, x_edges_m=xe, y_edges_m=xe,
                        **{f"maps_{l}": v["maps"] for l, v in slices.items()}, **{f"z_nm_{l}": np.array(v["z_nm"]) for l, v in slices.items()},
                        **{f"region_{l}": np.array(v["region"]) for l, v in slices.items()}, source="raw_fields/fields_normalized.npz (derived from phasors.npz)")
    # ---- 5. xz / yz cuts with the TRUE cell edges ---------------------------------------------------------
    zp0, zp1 = idx["z_plane0"], idx["z_plane1"]; ze_p = g["ze"][zp0:zp1 + 1]
    for il, l in enumerate(lam):
        for cut, arr, xlabel in (("xz", F["xz"][il], "x [nm] (y = P/2)"), ("yz", F["yz"][il], "y [nm] (x = P/2)")):
            Ez2 = np.abs(arr[comps.index("Ez")]) ** 2; E2 = (np.abs(arr[iE]) ** 2).sum(0)
            fig, axs = plt.subplots(1, 2, figsize=(11, 6.5))
            for ax, m, lab in zip(axs, (Ez2, E2), (r"$|E_z/E_{inc}|^2$", r"$|E/E_{inc}|^2$")):
                im = ax.pcolormesh(xe * 1e9, ze_p * 1e9, m.T, cmap="inferno", shading="flat", norm=LogNorm(vmin=max(m.max() * 1e-4, 1e-12), vmax=m.max()))
                for zb in (z_ito_bot, z_ito_top, z_asi_top):
                    ax.axhline(zb * 1e9, color="w", ls="--", lw=0.7)
                ax.annotate("ITO 23 nm", (5, z_ito_top * 1e9 + 8), color="w", fontsize=7)
                ax.set_xlabel(xlabel); ax.set_ylabel("z [nm] (true cell edges; +z up)"); ax.set_title(lab + " (log)", fontsize=9); plt.colorbar(im, ax=ax, pad=0.01)
            fig.suptitle(f"{tag}: {cut} cut at lambda = {l:.2f} nm, full height (PML to PML), pcolormesh on the non-uniform z mesh", fontsize=9)
            fig.tight_layout(); fig.savefig(HERE / tag / "xz_yz" / f"Ez2_{cut}_{tag}_{l:.1f}nm.png", dpi=130); plt.close(fig)
    ito_cells_z = [(g["ze"][k] * 1e9, g["ze"][k + 1] * 1e9) for k in range(idx["z_ito0"], idx["z_asi0"])]
    checks["ito_layer_check"] = dict(ito_cell_edges_nm=ito_cells_z, thickness_nm=float((z_ito_top - z_ito_bot) * 1e9), expected_nm=23.0)
    # ---- 6. ITO field enhancement / F_c analogue vs TORCWA ---------------------------------------------------
    fe = {}
    for il, l in enumerate(lam):
        I2 = np.abs(E[il]) ** 2; w = 2 * np.pi * C0 / (l * 1e-9); e_i = complex(fx_sim.eps_model("ITO", l * 1e-9))
        dV = g["dxy"] ** 2 * dz_v[i0:i1]
        Fc = [(w / C0) * e_i.imag * np.sum(I2[c, :, :, i0:i1] * dV[None, None, :]) / g["P"] ** 2 for c in range(3)]
        d = dict(mean_Ez2_ito_fdtdx=float(I2[2, :, :, i0:i1].mean()), max_Ez2_ito_fdtdx=float(I2[2, :, :, i0:i1].max()), Fz_fdtdx=Fc[2], Ftot_fdtdx=float(sum(Fc)),
                 eta_z_fdtdx=(Fc[2] / sum(Fc) if sum(Fc) > 0 else None), A_fdtdx_flux=float(np.interp(l, s["lam"], s["A"])))
        if tor is not None:
            d.update(mean_Ez2_ito_torcwa=float(np.interp(l, tor["lam"], tor["Ez2"])), Fz_torcwa=float(np.interp(l, tor["lam"], tor["Fz"])), Ftot_torcwa=float(np.interp(l, tor["lam"], tor["Ftot"])))
        fe[f"{l:.3f}"] = d
    checks["ito_field_enhancement"] = fe
    json.dump(checks, open(HERE / tag / "spectra" / f"checks_{tag}.json", "w"), indent=1)
    pm = dict(candidate=tag, provenance=prov, figures=dict(
        spectra=str(HERE / tag / "spectra" / f"spectra_{tag}.png"), zprofile=str(HERE / tag / "profiles" / f"zprofile_{tag}.png"),
        xy_slices=[str(p) for p in sorted((HERE / tag / "xy_slices").glob("*.png"))], xz_yz=[str(p) for p in sorted((HERE / tag / "xz_yz").glob("*.png"))],
        geometry=str(HERE / tag / "geometry" / f"geometry_{tag}.png")),
        colour_scales=dict(xy_slices="common per wavelength across all panels; linear (0..max) and log (4 decades) versions", xz_yz="log, per panel", zprofile="log y, common physical normalization"),
        field_definitions=dict(E2="|Ex|^2+|Ey|^2+|Ez|^2 with |u|^2 = u conj(u)", Ez2="Ez conj(Ez)", Dz="eps_zz(x,y,z,lambda) Ez, eps0 omitted; per-cell eps_zz per the docstring (fill-weighted arithmetic mean in the patterned a-Si:H cells)"),
        checks=str(HERE / tag / "spectra" / f"checks_{tag}.json"), raw_normalized=str(HERE / tag / "raw_fields" / "fields_normalized.npz"))
    json.dump(pm, open(HERE / tag / "metadata" / "plot_manifest.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in checks.items() if k not in ("ito_field_enhancement", "early_window_convergence")}, indent=1, default=str)[:3000])
    print("field enhancement:", json.dumps(fe, indent=None)[:1500])
    print("early-window convergence:", json.dumps(conv, indent=None)[:800])


if __name__ == "__main__":
    main()
