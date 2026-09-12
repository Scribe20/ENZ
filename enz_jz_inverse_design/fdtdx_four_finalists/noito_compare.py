"""Assemble the ACCEPTED fresh no-ITO runs with their with-ITO partners.

Reads comparison/accepted_runs.json:

    {"final3": {"noito": ["noito6ps_deepglass", "ref64_deepglass"],
                "prod":  ["prod_deepglass",     "ref64_deepglass"]}, ...}

i.e. per design the [run tag, empty-domain reference tag] of the without-ITO and the with-ITO run.
The two runs of a pair must share the same mesh, materials, source and monitors; the reference is the
run that supplies P_inc and the TFSF leakage for the same mesh.  R, T and A = 1 - R - T come straight
from the FDTDX flux phasors (fx_post.plane_flux); nothing is smoothed, clipped or renormalised.
"""
import csv, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fx_post as fp                                   # noqa: E402
from post_noito_fresh import spectra_from              # noqa: E402

LAM_ZE = fp.LAM_ZE
COMP = HERE / "comparison"


def main():
    acc = json.load(open(COMP / "accepted_runs.json"))
    refs, rows, ov = {}, [], {}
    for d, pair in acc.items():
        got = {}
        for kind, (tag, rtag) in pair.items():
            if rtag not in refs:
                refs[rtag] = fp.load_run("reference", rtag)
                assert refs[rtag] is not None, f"reference run {rtag} missing"
            run = fp.load_run(d, tag)
            assert run is not None, f"{d}/{tag} missing"
            got[kind] = (run, spectra_from(run, refs[rtag]))
            rows.append(dict(design=d, kind=kind, tag=tag, reference=rtag,
                             # actual simulated time (steps_done * dt), never the nominal --time-fs:
                             # a run may be accepted before its nominal end
                             time_fs=int(run["meta"].get("steps_done", run["meta"]["n_steps_run"]))
                             * run["meta"]["dt_s"] * 1e15,
                             nominal_time_fs=run["meta"]["time_s"] * 1e15,
                             complete=bool(run["meta"].get("complete", True)),
                             n_steps=run["meta"]["n_steps_run"],
                             n_z=run["meta"]["idx"]["n_z"], nxy=run["meta"]["nxy"],
                             glass_extra_nm=run["meta"].get("glass_extra_nm", 0.0),
                             R_ZE=float(np.interp(LAM_ZE, got[kind][1]["lam"], got[kind][1]["R"])),
                             T_ZE=float(np.interp(LAM_ZE, got[kind][1]["lam"], got[kind][1]["T"])),
                             A_ZE=float(np.interp(LAM_ZE, got[kind][1]["lam"], got[kind][1]["A"])),
                             R_max=float(got[kind][1]["R"].max()), T_min=float(got[kind][1]["T"].min()),
                             max_abs_A=float(np.abs(got[kind][1]["A"]).max())))
        if "noito" not in got:
            continue
        b = got["noito"][1]
        ov[d] = b
        with open(HERE / d / "spectra_noito_accepted.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["lambda_nm", "R", "T", "A_closure_residual"])
            for i in range(len(b["lam"])):
                w.writerow([f"{b['lam'][i]:.2f}", f"{b['R'][i]:.6f}", f"{b['T'][i]:.6f}", f"{b['A'][i]:.6f}"])
        np.savez(HERE / d / "spectra_noito_accepted.npz", lam_nm=b["lam"], R=b["R"], T=b["T"], A=b["A"],
                 P_inc=b["P_inc"], leak_over_Pinc=b["leak_over_Pinc"],
                 run_tag=pair["noito"][0], reference_tag=pair["noito"][1],
                 time_fs=got["noito"][0]["meta"]["time_s"] * 1e15)
        if "prod" not in got:
            continue
        a = got["prod"][1]
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
        axs[0].plot(a["lam"], a["R"], "b-", label="R with ITO"); axs[0].plot(b["lam"], b["R"], "b--", label="R without ITO")
        axs[0].plot(a["lam"], a["T"], "r-", label="T with ITO"); axs[0].plot(b["lam"], b["T"], "r--", label="T without ITO")
        axs[1].plot(a["lam"], a["A"], "k-", label="A with ITO")
        axs[1].plot(b["lam"], b["A"], "k--", label="A without ITO (exact value 0: every medium lossless)")
        for ax in axs:
            ax.axvline(LAM_ZE, color="0.5", ls=":", lw=0.8); ax.set_xlabel("λ [nm]"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        axs[0].set_ylabel("R, T"); axs[1].set_ylabel("A"); axs[0].set_ylim(-0.05, 1.05)
        fig.suptitle(f"{d}: with vs without ITO — FDTDX, identical mesh/materials/source/monitors "
                     f"({got['prod'][0]['meta']['time_s']*1e15:.0f} fs / {got['noito'][0]['meta']['time_s']*1e15:.0f} fs)",
                     fontsize=10)
        fig.tight_layout(); fig.savefig(HERE / d / "compare_with_without_ITO.png", dpi=150); plt.close(fig)
    if ov:
        fig, ax = plt.subplots(figsize=(8, 5))
        for d, s in ov.items():
            ax.plot(s["lam"], s["R"], label=f"{d} R"); ax.plot(s["lam"], s["T"], ls="--", label=f"{d} T")
        ax.axvline(LAM_ZE, color="k", ls=":", lw=0.8); ax.set_xlabel("λ [nm]"); ax.set_ylabel("R, T")
        ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
        ax.set_title("FDTDX without ITO — accepted fresh runs", fontsize=10)
        fig.tight_layout(); fig.savefig(COMP / "overlay_RT_noITO_accepted.png", dpi=160); plt.close(fig)
    with open(COMP / "spectra_table_noito_accepted.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(rows, open(COMP / "spectra_table_noito_accepted.json", "w"), indent=1)
    for r in rows:
        print(f"{r['design']:7s} {r['kind']:6s} {r['tag']:22s} {r['time_fs']/1000:6.2f} ps"
              + ("  " if r["complete"] else "* ") + " "
              f"R_ZE={r['R_ZE']:.4f} T_ZE={r['T_ZE']:.4f} A_ZE={r['A_ZE']:+.4f}  "
              f"R_max={r['R_max']:.4f} T_min={r['T_min']:+.4f} max|A|={r['max_abs_A']:.4f}")


if __name__ == "__main__":
    main()
