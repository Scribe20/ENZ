"""Stage 4: hard-binary certification of finalist designs.

For every run directory given: Fourier-order x z-sampling convergence table
(orders [5,5]..[11,11], n_z 7/15/31) with the loss identity at every point,
max|Ez|^2 vs order (corner/slot hotspot watch), fabrication/locality metrics,
erosion/dilation, height and period sensitivity, real-space field and loss
maps, figures.  The CERTIFIED value is F_z at order [11,11], n_z = 31.

    python stage4_certify.py --runs <run_dir> [<run_dir> ...] --out outputs/stage4
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import analysis as an
import config
import forward as fwd
import materials as mat
import plots

ORDERS = ([5, 5], [7, 7], [9, 9], [11, 11])
NZS = (7, 15, 31)


def certify(run_dir, out, lam, log=print):
    run_dir = Path(run_dir)
    res = json.load(open(run_dir / "result.json"))
    P, h, pad = res["P"], res["h"], res["pad_frac"]
    rho = np.load(run_dir / "rho_hard_binary.npy")
    M = np.load(run_dir / "design_mask.npy")
    tag = res["tag"]
    o = out / tag
    o.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log(f"== {tag}: P={P:.0f} h={h:.0f} pad={pad:.2f} (fill {rho.mean():.3f}) ==")
    conv = an.convergence_table(rho, P, h, lam, ORDERS, NZS)
    for r in conv:
        log(f"   order {r['order']:2d}: A={r['A']:.5f} Fz nz7/15/31 = {r['Fz_nz7']:.5f}/{r['Fz_nz15']:.5f}/{r['Fz_nz31']:.5f} "
            f"resid nz31 {r['resid_nz31']:+.1e} max|Ez|^2={r['max_Ez2']:.1f}")
    cert = conv[-1]
    fz11, fz9, fz7, fz5 = (r["Fz_nz31"] for r in conv[::-1])
    fab = an.fab_metrics(rho, P, M)
    log(f"   fab: {json.dumps(fab)}")
    sens = an.sensitivity(rho, P, h, lam, M, order=config.ORDER_FULL)
    log("   sensitivity (Fz at [7,7]): base %.4f | " % sens["base"]["Fz"] +
        " ".join(f"{k}: {v['Fz']:.4f}" for k, v in sens["morph"].items()) + " | " +
        " ".join(f"{k}: {v['Fz']:.4f}" for k, v in sens["height"].items()) + " | " +
        " ".join(f"{k}: {v['Fz']:.4f}" for k, v in sens["period"].items()))
    maps = an.field_maps(rho, P, h, lam, order=[9, 9])
    np.savez_compressed(o / "fields_ito_order9.npz", E_ito=maps["E_ito"], loss_density_xyz=maps["loss_density_xyz"],
                        E_si_mid=maps["E_si_mid"], E_xz=maps["E_xz"], z_xz=maps["z_xz"], x=maps["x"], P=P, h=h, lam=lam)
    plots.loss_maps_plot(maps, rho, P, h, o / "loss_maps.png", title=f"{tag}: P={P:.0f} h={h:.0f}, lambda_ZE={lam:.1f} nm, order [9,9]")
    plots.geometry_plot([rho], o / "geometry.png", titles=[f"{tag} hard binary (fill {rho.mean():.3f})"], P=P)
    if (run_dir / "result.json").exists():
        plots.history_plot(res, o / "history.png")
    np.save(o / "rho_hard_binary.npy", rho)
    summary = dict(tag=tag, run_dir=str(run_dir), P=P, h=h, pad_frac=pad, lam=lam,
                   Fz_certified=cert["Fz_nz31"], Fx_certified=cert["Fx_nz7"], Fy_certified=cert["Fy_nz7"],
                   Ftot_certified=cert["Ftot_nz31"], A_certified=cert["A"], R_certified=cert["R"], T_certified=cert["T"],
                   eta_z_abs=cert["Fz_nz31"] / cert["Ftot_nz31"], identity_resid_certified=cert["resid_nz31"],
                   mean_Ez2=cert["mean_Ez2"], max_Ez2=cert["max_Ez2"],
                   Fz_by_order_nz31=dict(zip(("5", "7", "9", "11"), (fz5, fz7, fz9, fz11))),
                   rel_change_9_to_11=(fz11 - fz9) / fz11, rel_change_7_to_11=(fz11 - fz7) / fz11,
                   rel_change_nz7_to_31_at11=(cert["Fz_nz31"] - cert["Fz_nz7"]) / cert["Fz_nz31"],
                   maxEz2_by_order=[r["max_Ez2"] for r in conv],
                   convergence=conv, fabrication=fab, sensitivity=sens,
                   Fz_soft_optimizer=res["Fz_soft"], Fz_hard_optimizer=res["Fz_hard"], order_optimizer=res["order"],
                   wall_s=time.time() - t0)
    an.jdump(summary, o / "certify.json")
    return summary


def write_md(summaries, out):
    L = ["# Stage 4 - hard-binary certification", "",
         "Certified value = F_z of the hard-binary design at order [11,11], n_z = 31 (midpoint quadrature).  "
         "All quantities at lambda_ZE with the supplied materials.", "",
         "| design | P | h | pad | F_z cert | F_x | F_y | F_tot | A=1-R-T | resid | eta_z,abs | R | T | <|Ez|^2> | max|Ez|^2 | Fz [5]/[7]/[9]/[11] | d(9->11) | d(nz7->31) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        fo = s["Fz_by_order_nz31"]
        L.append(f"| {s['tag']} | {s['P']:.0f} | {s['h']:.0f} | {s['pad_frac']:.2f} | **{s['Fz_certified']:.4f}** | {s['Fx_certified']:.4f} | {s['Fy_certified']:.4f} | "
                 f"{s['Ftot_certified']:.4f} | {s['A_certified']:.4f} | {s['identity_resid_certified']:+.1e} | {s['eta_z_abs']:.3f} | {s['R_certified']:.3f} | {s['T_certified']:.3f} | "
                 f"{s['mean_Ez2']:.2f} | {s['max_Ez2']:.1f} | {fo['5']:.4f}/{fo['7']:.4f}/{fo['9']:.4f}/{fo['11']:.4f} | {100*s['rel_change_9_to_11']:+.2f}% | {100*s['rel_change_nz7_to_31_at11']:+.3f}% |")
    L += ["", "## max|Ez/E_inc|^2 vs order (hotspot watch)", "", "| design | [5,5] | [7,7] | [9,9] | [11,11] |", "|---|---|---|---|---|"]
    for s in summaries:
        L.append(f"| {s['tag']} | " + " | ".join(f"{v:.1f}" for v in s["maxEz2_by_order"]) + " |")
    L += ["", "## Fabrication / locality", "",
          "| design | fill (cell) | fill (active) | components | largest comp. | boundary contact px | ring px | min feature (nm, est.) | min air gap (nm, est.) | enclosed air regions |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        f = s["fabrication"]
        L.append(f"| {s['tag']} | {f['fill_cell']:.3f} | {f['fill_active']:.3f} | {f['n_components']} | {f['largest_component_frac']:.3f} | {f['boundary_contact_px']} | "
                 f"{f['material_in_ring_px']} | {f['min_feature_nm_est']:.0f} | {f['min_air_gap_nm_est']:.0f} | {f['n_enclosed_air_regions']} |")
    L += ["", "## Sensitivity (F_z at [7,7], n_z 7; base = hard binary)", "",
          "| design | base | -2 px | -1 px | +1 px | +2 px | h-10% | h-10nm | h+10nm | h+10% | P x0.98 | P x1.02 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        se = s["sensitivity"]
        m = list(se["morph"].values()); hh = list(se["height"].values()); pp = list(se["period"].values())
        L.append(f"| {s['tag']} | {se['base']['Fz']:.4f} | " + " | ".join(f"{v['Fz']:.4f}" for v in m) + " | " +
                 " | ".join(f"{v['Fz']:.4f}" for v in hh) + " | " + " | ".join(f"{v['Fz']:.4f}" for v in pp) + " |")
    (out / "CERTIFICATION.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=str(config.OUT / "stage4"))
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    fwd.set_threads(a.threads)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lam, _ = mat.ito_zero_crossing()
    logf = open(out / "certify.log", "a")

    def log(*x):
        s = " ".join(str(v) for v in x); print(s, flush=True); print(s, file=logf, flush=True)
    summaries = [certify(r, out, lam, log) for r in a.runs]
    summaries.sort(key=lambda s: -s["Fz_certified"])
    an.jdump(summaries, out / "certification.json")
    import csv
    keys = ["tag", "P", "h", "pad_frac", "Fz_certified", "Fx_certified", "Fy_certified", "Ftot_certified", "A_certified",
            "identity_resid_certified", "eta_z_abs", "R_certified", "T_certified", "mean_Ez2", "max_Ez2",
            "rel_change_9_to_11", "rel_change_7_to_11", "rel_change_nz7_to_31_at11", "Fz_hard_optimizer"]
    with open(out / "certification.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for s in summaries:
            w.writerow({k: s[k] for k in keys})
    write_md(summaries, out)
    log(f"[stage4] done: " + ", ".join(f"{s['tag']} Fz={s['Fz_certified']:.4f}" for s in summaries))


if __name__ == "__main__":
    main()
