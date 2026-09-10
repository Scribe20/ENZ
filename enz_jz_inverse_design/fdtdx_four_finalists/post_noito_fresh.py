"""Post-processing of the FRESH no-ITO runs (fx_sim.py --no-ito --time-fs 6000 --conv-check-fs 3000).

R, T and A = 1 - R - T are computed from the actual FDTDX flux phasors exactly as for the with-ITO
production runs (fx_post.spectra_of): T = -S_z(T plane)/P_inc, R = (S_z(R plane) - TFSF leakage)/P_inc,
with S_z = 1/2 Re(Ex Hy* - Ey Hx*) integrated over the cell, P_inc from the empty-domain reference run.
NOTHING is smoothed, clipped, renormalised or otherwise altered: the numbers written are the ones the
simulation produced.

Convergence is judged on two quantities that the simulation itself supplies:
  * the lossless-closure residual A = 1 - R - T (every material is lossless without the ITO, so the
    exact value is 0 at every wavelength) - this also catches R > 1 and T < 0;
  * the difference between the 3-ps and 6-ps DFT windows recorded in the SAME run, i.e. whether the
    spectrum has stopped changing with simulation time.
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fx_post as fp                       # noqa: E402

DESIGNS = ["final3", "final1", "final0", "final2"]
LAM_ZE = fp.LAM_ZE
OUT = HERE / "comparison"


def spectra_from(run, ref, suffix=""):
    """R, T, A from the flux phasors of one detector pair (suffix '' or '_early')."""
    dxy = run["meta"]["dxy_m"]
    P_inc = -fp.plane_flux(ref["z"]["T_plane/phasor"], dxy)
    leak = fp.plane_flux(ref["z"]["R_plane/phasor"], dxy)
    S_T = fp.plane_flux(run["z"][f"T_plane{suffix}/phasor"], dxy)
    S_R = fp.plane_flux(run["z"][f"R_plane{suffix}/phasor"], dxy)
    T = -S_T / P_inc
    R = (S_R - leak) / P_inc
    return dict(lam=run["z"]["lam_spec_m"] * 1e9, R=R, T=T, A=1 - R - T, P_inc=P_inc,
                leak_over_Pinc=leak / P_inc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="noito6ps")
    ap.add_argument("--ref", default="ref64_fresh")
    ap.add_argument("--designs", nargs="*", default=DESIGNS)
    ap.add_argument("--tol", type=float, default=0.01, help="acceptance tolerance on |A| and on the window difference")
    a = ap.parse_args()
    ref = fp.load_run("reference", a.ref)
    assert ref is not None, f"reference run {a.ref} missing"
    OUT.mkdir(exist_ok=True)
    report, overlay = {}, {}
    for d in a.designs:
        run = fp.load_run(d, a.tag)
        if run is None:
            print(f"  {d}: run {a.tag} missing"); continue
        full = spectra_from(run, ref, "")
        early = spectra_from(run, ref, "_early") if f"R_plane_early/phasor" in run["z"].files else None
        lam = full["lam"]
        outd = run["dir"]
        with open(outd / "spectra_noito_fresh.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["lambda_nm", "R", "T", "A_closure_residual", "R_3ps_window", "T_3ps_window"])
            for i in range(len(lam)):
                w.writerow([f"{lam[i]:.2f}", f"{full['R'][i]:.6f}", f"{full['T'][i]:.6f}", f"{full['A'][i]:.6f}",
                            f"{early['R'][i]:.6f}" if early else "", f"{early['T'][i]:.6f}" if early else ""])
        np.savez(outd / "spectra_noito_fresh.npz", lam_nm=lam, R=full["R"], T=full["T"], A=full["A"],
                 P_inc=full["P_inc"], leak_over_Pinc=full["leak_over_Pinc"],
                 **({"R_early": early["R"], "T_early": early["T"], "A_early": early["A"]} if early else {}),
                 time_fs=run["meta"]["time_s"] * 1e15, conv_check_fs=run["meta"].get("conv_check_fs"),
                 n_steps=run["meta"]["n_steps_run"], order="FDTDX, no ITO, fresh run")
        dR = np.abs(full["R"] - early["R"]).max() if early else float("nan")
        dT = np.abs(full["T"] - early["T"]).max() if early else float("nan")
        rep = dict(design=d, time_fs=run["meta"]["time_s"] * 1e15, n_steps=run["meta"]["n_steps_run"],
                   conv_check_fs=run["meta"].get("conv_check_fs"),
                   max_abs_A=float(np.abs(full["A"]).max()), lam_max_abs_A=float(lam[int(np.argmax(np.abs(full["A"])))]),
                   rms_A=float(np.sqrt(np.mean(full["A"] ** 2))),
                   n_points_absA_gt_tol=int((np.abs(full["A"]) > a.tol).sum()), n_points=int(len(lam)),
                   R_max=float(full["R"].max()), lam_R_max=float(lam[int(np.argmax(full["R"]))]),
                   T_min=float(full["T"].min()), lam_T_min=float(lam[int(np.argmin(full["T"]))]),
                   n_points_R_gt_1=int((full["R"] > 1.0).sum()), n_points_T_lt_0=int((full["T"] < 0.0).sum()),
                   max_window_diff_R=float(dR), max_window_diff_T=float(dT),
                   max_leak_over_Pinc=float(np.abs(full["leak_over_Pinc"]).max()),
                   R_at_lamZE=float(np.interp(LAM_ZE, lam, full["R"])), T_at_lamZE=float(np.interp(LAM_ZE, lam, full["T"])),
                   A_at_lamZE=float(np.interp(LAM_ZE, lam, full["A"])))
        rep["passes"] = bool(rep["max_abs_A"] <= a.tol and rep["n_points_R_gt_1"] == 0 and rep["n_points_T_lt_0"] == 0
                             and (not early or max(dR, dT) <= a.tol))
        report[d] = rep; overlay[d] = full
        print(f"  {d}: max|A| = {rep['max_abs_A']:.4f} @ {rep['lam_max_abs_A']:.0f} nm (rms {rep['rms_A']:.4f}, "
              f"{rep['n_points_absA_gt_tol']}/{rep['n_points']} points > {a.tol}); R_max = {rep['R_max']:.4f}, "
              f"T_min = {rep['T_min']:+.4f}; 3ps->6ps window drift: dR {dR:.4f} dT {dT:.4f}  -> "
              f"{'PASS' if rep['passes'] else 'NOT CONVERGED'}")
        # per-design figure
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
        axs[0].plot(lam, full["R"], "b-", label="R"); axs[0].plot(lam, full["T"], "r-", label="T")
        axs[0].plot(lam, full["A"], "k-", lw=1, label="A = 1 − R − T (exact value 0: no lossy material)")
        if early:
            axs[0].plot(lam, early["R"], "b--", lw=0.8, alpha=0.7, label="R (3-ps DFT window)")
            axs[0].plot(lam, early["T"], "r--", lw=0.8, alpha=0.7, label="T (3-ps DFT window)")
        axs[0].axvline(LAM_ZE, color="0.5", ls=":", lw=0.8); axs[0].set_xlabel("λ [nm]"); axs[0].set_ylabel("R, T, A")
        axs[0].grid(alpha=0.3); axs[0].legend(fontsize=7); axs[0].set_ylim(-0.05, 1.05)
        axs[0].set_title(f"{d} without ITO — fresh FDTDX run, {rep['time_fs']:.0f} fs ({rep['n_steps']} steps)", fontsize=9)
        axs[1].plot(lam, full["A"], "k-", label="A (full window)")
        if early:
            axs[1].plot(lam, early["A"], "g--", lw=0.9, label="A (3-ps window)")
            axs[1].plot(lam, full["R"] - early["R"], "b:", lw=0.9, label="R(6 ps) − R(3 ps)")
        axs[1].axhline(0, color="0.6", lw=0.8); axs[1].axhline(a.tol, color="r", ls=":", lw=0.8)
        axs[1].axhline(-a.tol, color="r", ls=":", lw=0.8)
        axs[1].set_xlabel("λ [nm]"); axs[1].set_ylabel("closure residual / window drift"); axs[1].grid(alpha=0.3)
        axs[1].legend(fontsize=7); axs[1].set_title("energy-conservation and time-convergence check", fontsize=9)
        fig.tight_layout(); fig.savefig(outd / "spectra_noito_fresh.png", dpi=150); plt.close(fig)
    if overlay:
        fig, ax = plt.subplots(figsize=(8, 5))
        for d, s in overlay.items():
            ax.plot(s["lam"], s["R"], label=f"{d} R")
            ax.plot(s["lam"], s["T"], ls="--", label=f"{d} T")
        ax.axvline(LAM_ZE, color="k", ls=":", lw=0.8); ax.set_xlabel("λ [nm]"); ax.set_ylabel("R, T")
        ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
        ax.set_title("FDTDX without ITO (fresh runs) — R and T", fontsize=10)
        fig.tight_layout(); fig.savefig(OUT / "overlay_RT_noITO_fresh.png", dpi=160); plt.close(fig)
    json.dump(report, open(OUT / "noito_fresh_convergence.json", "w"), indent=1)
    print(f"\n[noito fresh] {sum(r['passes'] for r in report.values())}/{len(report)} designs pass at tol {a.tol}")


if __name__ == "__main__":
    main()
