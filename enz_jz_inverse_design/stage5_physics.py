"""Stage 5: physics certification of finalist designs.

For each run directory:
  * dense spectra 1100-1400 nm (a-Si:H data end at 1400 nm; no extrapolation)
    at order [7,7] with the supplied dispersion of every material:
      - full structure (s = 1):  F_z, F_x, F_y, A, R, T, r_xx, t_xx
      - same geometry, ITO removed (with_ito = False)
      - same geometry, lossless ITO (s = 0; grid avoids eps = 0 exactly)
      - bare air/ITO/glass film (no a-Si)
  * scattering poles (AAA on r and t, accepted only if present in both,
    damped, significant) at s = 1 and for the ITO-free structure
  * loss-scaling branch tracking s in {1, .5, .25, .1, .03, 0} of the poles
    nearest lambda_ZE -> gamma(s) = gamma_rad + s gamma_nr -> Q_rad, Q_nr,
    gamma_rad/gamma_nr (critical-coupling test, post hoc)
  * height-detuning map A(lambda, h) with ITO and T(lambda, h) without ITO
    at order [5,5] + pole branches vs h (avoided-crossing test)
  * reference spectra (Karimi EDR cuboid, direct-Ez winner) under the new materials

    python stage5_physics.py --runs <run_dir> [...] --out outputs/stage5
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analysis as an
import config
import forward as fwd
import materials as mat
import plots

GEO = config.GEO_DTYPE


def run_candidate(run_dir, out, lam_ze, log, order, lam_step, h_frac, n_h, detune_order):
    run_dir = Path(run_dir)
    res = json.load(open(run_dir / "result.json"))
    P, h, tag = res["P"], res["h"], res["tag"]
    rho = torch.as_tensor(np.load(run_dir / "rho_hard_binary.npy"), dtype=GEO)
    o = out / tag
    o.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lams = an.lam_grid(1100.0, 1400.0, lam_step, avoid=lam_ze)
    log(f"== {tag}: P={P:.0f} h={h:.0f} | {len(lams)} wavelengths, order {order} ==")
    cache = {}
    log("  spectrum with ITO (s=1)")
    cache[1.0] = an.spec_arrays(an.spectrum(rho, P, h, lams, order, s=1.0, log=log))
    log("  spectrum without ITO")
    sp_no = an.spec_arrays(an.spectrum(rho, P, h, lams, order, with_ito=False, log=log))
    log("  spectrum lossless ITO (s=0)")
    cache[0.0] = an.spec_arrays(an.spectrum(rho, P, h, lams, order, s=0.0, log=log))
    log("  bare ITO film")
    sp_bare = an.spec_arrays(an.spectrum(None, P, h, lams, [1, 1], s=1.0))
    np.savez_compressed(o / "spectra.npz", lam=lams,
                        **{f"with_{k}": v for k, v in cache[1.0].items() if k != "lam"},
                        **{f"noito_{k}": v for k, v in sp_no.items() if k != "lam"},
                        **{f"lossless_{k}": v for k, v in cache[0.0].items() if k != "lam"},
                        **{f"bare_{k}": v for k, v in sp_bare.items() if k != "lam"})
    sw = cache[1.0]
    i_ze = int(np.argmin(np.abs(lams - lam_ze)))
    peaks = dict(lam_ZE=lam_ze, Fz_at_ZE=float(sw["Fz"][i_ze]), A_at_ZE=float(sw["A"][i_ze]),
                 Fz_max=float(sw["Fz"].max()), lam_Fz_max=float(lams[int(sw["Fz"].argmax())]),
                 A_max=float(sw["A"].max()), lam_A_max=float(lams[int(sw["A"].argmax())]),
                 T_min_noITO=float(sp_no["T"].min()), lam_T_min_noITO=float(lams[int(sp_no["T"].argmin())]),
                 R_max_noITO=float(sp_no["R"].max()), bare_A_at_ZE=float(sp_bare["A"][i_ze]))
    # FWHM of the Fz(lambda) peak (half-max crossings around the maximum)
    fz = sw["Fz"]; im = int(fz.argmax()); half = fz[im] / 2
    lo = im
    while lo > 0 and fz[lo] > half:
        lo -= 1
    hi = im
    while hi < len(fz) - 1 and fz[hi] > half:
        hi += 1
    peaks["Fz_fwhm_nm"] = float(lams[hi] - lams[lo]); peaks["Q_eff_Fz"] = float(lams[im] / max(lams[hi] - lams[lo], 1e-9))
    log(f"  peaks: {json.dumps(peaks)}")

    # ---- poles ------------------------------------------------------------
    poles_with = an.significant_poles(lams, sw["r"], sw["t"])
    poles_no = an.significant_poles(lams, sp_no["r"], sp_no["t"])
    poles_lossless = an.significant_poles(lams, cache[0.0]["r"], cache[0.0]["t"])
    for lab, pl in (("with ITO", poles_with), ("no ITO", poles_no), ("lossless ITO", poles_lossless)):
        log(f"  poles {lab}: " + "; ".join(f"{p['lambda_nm']:.1f} nm Q={p['Q']:.1f} (peak r/t {p['peak_r']:.2f}/{p['peak_t']:.2f})" for p in pl) if pl else f"  poles {lab}: none")
    # loss scaling of the (up to 2) with-ITO poles nearest lambda_ZE
    near = sorted(poles_with, key=lambda p: abs(p["lambda_nm"] - lam_ze))[:2]
    tracks = []
    for p in near:
        rows = an.track_branch(rho, P, h, complex(p["omega_re"], p["omega_im"]), lams, order=order,
                               log=log, tag=f"{tag}|{p['lambda_nm']:.0f}nm", spectra_cache=cache)
        fit = an.fit_gamma(rows)
        tracks.append(dict(start_pole=p, rows=rows, fit=fit))
        log(f"  fit: {json.dumps({k: v for k, v in fit.items() if k not in ('lossless_pole',)})}")
    an.jdump(dict(tag=tag, P=P, h=h, peaks=peaks, poles_with_ito=poles_with, poles_no_ito=poles_no,
                  poles_lossless_ito=poles_lossless, loss_scaling=tracks), o / "poles.json")
    # spectra at intermediate s for the figure
    np.savez_compressed(o / "spectra_loss_levels.npz", lam=lams, **{f"s{s}_A": cache[s]["A"] for s in cache},
                        **{f"s{s}_Fz": cache[s]["Fz"] for s in cache})
    plots.spectra_plot({"with ITO": sw, "no ITO": sp_no, "lossless ITO": cache[0.0], "bare ITO film": sp_bare},
                       o / "spectra.png", lam_ze, title=f"{tag}: P={P:.0f} h={h:.0f}, order {order}")
    # gamma(s) figure
    if tracks:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for tr in tracks:
            S = [r["s"] for r in tr["rows"]]; G = [r["gamma"] for r in tr["rows"]]; L = [r["lambda_nm"] for r in tr["rows"]]
            ax[0].plot(S, G, "o-", label=f"pole {tr['start_pole']['lambda_nm']:.0f} nm")
            f = tr["fit"]
            if "gamma_rad" in f:
                ss = np.linspace(0, 1, 10); ax[0].plot(ss, f["gamma_rad"] + ss * f["gamma_nr"], "--", lw=0.8)
            ax[1].plot(S, L, "o-")
        ax[0].set_xlabel("ITO loss scale s"); ax[0].set_ylabel("gamma = -Im(omega) (rad/fs)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
        ax[1].set_xlabel("s"); ax[1].set_ylabel("pole wavelength (nm)"); ax[1].axhline(lam_ze, color="k", ls="--", lw=0.6); ax[1].grid(alpha=.3)
        fig.suptitle(f"{tag}: loss scaling"); fig.tight_layout(); fig.savefig(o / "loss_scaling.png", dpi=140); plt.close(fig)

    # ---- height detuning map ----------------------------------------------
    hs = np.linspace((1 - h_frac) * h, (1 + h_frac) * h, n_h)
    lams_d = an.lam_grid(1100.0, 1400.0, 4.0, avoid=lam_ze)
    A_map, Fz_map, T_no_map = [], [], []
    branches = []
    log(f"  detuning map: {n_h} heights x {len(lams_d)} wavelengths, order {detune_order}")
    for hh in hs:
        sp = an.spec_arrays(an.spectrum(rho, P, float(hh), lams_d, detune_order, s=1.0))
        spn = an.spec_arrays(an.spectrum(rho, P, float(hh), lams_d, detune_order, with_ito=False))
        A_map.append(sp["A"]); Fz_map.append(sp["Fz"]); T_no_map.append(spn["T"])
        pw = an.significant_poles(lams_d, sp["r"], sp["t"]); pn = an.significant_poles(lams_d, spn["r"], spn["t"])
        branches.append(dict(h=float(hh), with_ito=[(p["lambda_nm"], p["Q"]) for p in pw], no_ito=[(p["lambda_nm"], p["Q"]) for p in pn],
                             lam_A_max=float(lams_d[int(sp["A"].argmax())]), A_max=float(sp["A"].max()),
                             lam_Fz_max=float(lams_d[int(sp["Fz"].argmax())]), Fz_max=float(sp["Fz"].max())))
        log(f"    h={hh:6.1f}: A_max {sp['A'].max():.3f} @ {lams_d[int(sp['A'].argmax())]:.0f} nm; Fz_max {sp['Fz'].max():.3f} @ {lams_d[int(sp['Fz'].argmax())]:.0f}; "
            f"poles with ITO {[round(p['lambda_nm']) for p in pw]}; no ITO {[round(p['lambda_nm']) for p in pn]}")
    A_map, Fz_map, T_no_map = np.array(A_map), np.array(Fz_map), np.array(T_no_map)
    np.savez_compressed(o / "detuning.npz", hs=hs, lam=lams_d, A=A_map, Fz=Fz_map, T_noITO=T_no_map)
    an.jdump(branches, o / "detuning_branches.json")
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, Z, lab in zip(axs, (A_map, Fz_map, T_no_map), ("A with ITO", "F_z with ITO", "T without ITO")):
        im = ax.imshow(Z, origin="lower", aspect="auto", extent=[lams_d[0], lams_d[-1], hs[0], hs[-1]], cmap="viridis")
        for b in branches:
            for (l, q) in b["with_ito"]:
                ax.plot(l, b["h"], "r.", ms=3)
            for (l, q) in b["no_ito"]:
                ax.plot(l, b["h"], "w.", ms=3)
        ax.axvline(lam_ze, color="k", ls="--", lw=0.6); ax.axhline(h, color="c", ls=":", lw=0.6)
        ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("a-Si height h (nm)"); ax.set_title(f"{lab} (red: poles with ITO, white: poles without ITO)", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"{tag}: height detuning, order {detune_order}"); fig.tight_layout(); fig.savefig(o / "detuning_map.png", dpi=140); plt.close(fig)
    summary = dict(tag=tag, P=P, h=h, peaks=peaks, poles_with_ito=poles_with, poles_no_ito=poles_no,
                   poles_lossless_ito=poles_lossless, loss_scaling=[dict(start_lambda=t["start_pole"]["lambda_nm"], **t["fit"]) for t in tracks],
                   wall_s=time.time() - t0)
    an.jdump(summary, o / "physics.json")
    return summary


def reference_spectra(out, lam_ze, log, order, lam_step):
    refs = fwd.load_references()
    lams = an.lam_grid(1100.0, 1400.0, lam_step, avoid=lam_ze)
    o = out / "references"; o.mkdir(parents=True, exist_ok=True)
    specs, rows = {}, []
    for name in ("Karimi EDR cuboid (560x500x140, P850)", "direct-Ez winner (P850,h140,pad86)", "robust A finalist0 (P800,h200,pad4%)"):
        rho, P, h, _ = refs[name]
        log(f"  reference spectrum: {name}")
        sp = an.spec_arrays(an.spectrum(rho, P, h, lams, order, s=1.0))
        specs[name] = sp
        pl = an.significant_poles(lams, sp["r"], sp["t"])
        rows.append(dict(name=name, P=P, h=h, Fz_at_ZE=float(sp["Fz"][np.argmin(np.abs(lams - lam_ze))]),
                         Fz_max=float(sp["Fz"].max()), lam_Fz_max=float(lams[int(sp["Fz"].argmax())]),
                         A_max=float(sp["A"].max()), lam_A_max=float(lams[int(sp["A"].argmax())]),
                         poles=[(p["lambda_nm"], p["Q"]) for p in pl]))
        np.savez_compressed(o / f"{name.split(' (')[0].replace(' ', '_')}.npz", lam=lams, **{k: v for k, v in sp.items() if k != "lam"})
    plots.spectra_plot(specs, o / "reference_spectra.png", lam_ze, title=f"reference designs under the new materials, order {order}")
    an.jdump(rows, o / "reference_summary.json")
    return rows


def write_md(summaries, refs, out, lam_ze):
    L = ["# Stage 5 - physics certification", "",
         f"lambda_ZE = {lam_ze:.2f} nm. Spectra 1100-1400 nm (a-Si:H data end at 1400 nm; nothing is extrapolated). "
         "Poles: AAA rational fits of r_xx(omega) and t_xx(omega), accepted only when present in both with relative distance < 2%, "
         "damped, and with Lorentzian peak fraction >= 3% of the observable. Loss scaling: gamma(s) = gamma_rad + s gamma_nr on the "
         "field-overlap-tracked branch, s = scale of Im(eps_ITO).", "",
         "## Working point and peaks", "",
         "| design | Fz(lam_ZE) | Fz_max @ lam | FWHM (nm) / Q_eff | A(lam_ZE) | A_max @ lam | no-ITO T_min @ lam | bare-film A(lam_ZE) |", "|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        p = s["peaks"]
        L.append(f"| {s['tag']} | {p['Fz_at_ZE']:.4f} | {p['Fz_max']:.4f} @ {p['lam_Fz_max']:.0f} | {p['Fz_fwhm_nm']:.0f} / {p['Q_eff_Fz']:.1f} | {p['A_at_ZE']:.4f} | "
                 f"{p['A_max']:.4f} @ {p['lam_A_max']:.0f} | {p['T_min_noITO']:.3f} @ {p['lam_T_min_noITO']:.0f} | {p['bare_A_at_ZE']:.4f} |")
    L += ["", "## Poles (order [7,7])", "", "| design | with ITO (lambda nm, Q) | without ITO | lossless ITO |", "|---|---|---|---|"]
    for s in summaries:
        f = lambda pl: "; ".join(f"{p['lambda_nm']:.1f} (Q {p['Q']:.1f})" for p in pl) or "none"
        L.append(f"| {s['tag']} | {f(s['poles_with_ito'])} | {f(s['poles_no_ito'])} | {f(s['poles_lossless_ito'])} |")
    L += ["", "## Q / loss-scaling table", "",
          "| design | tracked pole (nm) | Q_loaded | Q_rad | Q_nr | gamma_rad (rad/fs) | gamma_nr (rad/fs) | gamma_rad/gamma_nr | linearity resid | lossless pole | levels |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        for t in s["loss_scaling"]:
            if "gamma_rad" not in t:
                L.append(f"| {s['tag']} | {t['start_lambda']:.1f} | - | - | - | - | - | - | {t.get('error')} | - | {t.get('n')} |"); continue
            lp = t.get("lossless_pole")
            lp_s = (f"{lp['lambda_nm']:.1f} nm, Q {lp['Q']:.1f}" if lp else "n/a")
            L.append(f"| {s['tag']} | {t['start_lambda']:.1f} | {t['Q_loaded']:.2f} | {t['Q_rad']:.2f} | {t['Q_nr']:.2f} | {t['gamma_rad']:.5f} | {t['gamma_nr']:.5f} | "
                     f"**{t['gamma_ratio']:.3f}** | {t['linearity_resid']:.3f} | {lp_s} | {t['n_levels']} |")
    L += ["", "## Reference designs under the new materials", "", "| design | P | h | Fz(lam_ZE) | Fz_max @ lam | A_max @ lam | poles (nm, Q) |", "|---|---|---|---|---|---|---|"]
    for r in refs:
        L.append(f"| {r['name']} | {r['P']:.0f} | {r['h']:.0f} | {r['Fz_at_ZE']:.4f} | {r['Fz_max']:.4f} @ {r['lam_Fz_max']:.0f} | {r['A_max']:.4f} @ {r['lam_A_max']:.0f} | "
                 + "; ".join(f"{l:.0f} ({q:.1f})" for l, q in r["poles"]) + " |")
    (out / "PHYSICS.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=str(config.OUT / "stage5"))
    ap.add_argument("--order", type=int, nargs=2, default=config.ORDER_FULL)
    ap.add_argument("--detune-order", type=int, nargs=2, default=config.ORDER_SCREEN)
    ap.add_argument("--lam-step", type=float, default=2.0)
    ap.add_argument("--h-frac", type=float, default=0.4)
    ap.add_argument("--n-h", type=int, default=13)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-refs", action="store_true")
    a = ap.parse_args()
    fwd.set_threads(a.threads)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lam_ze, _ = mat.ito_zero_crossing()
    logf = open(out / "physics.log", "a")

    def log(*x):
        s = " ".join(str(v) for v in x); print(s, flush=True); print(s, file=logf, flush=True)
    summaries = [run_candidate(r, out, lam_ze, log, list(a.order), a.lam_step, a.h_frac, a.n_h, list(a.detune_order)) for r in a.runs]
    refs = [] if a.no_refs else reference_spectra(out, lam_ze, log, list(a.order), a.lam_step)
    an.jdump(dict(candidates=summaries, references=refs), out / "physics_summary.json")
    write_md(summaries, refs, out, lam_ze)
    log("[stage5] done")


if __name__ == "__main__":
    main()
