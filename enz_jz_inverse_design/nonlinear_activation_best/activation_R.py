"""Reflection-mode activation quantification (same TTM heatmaps, observable <R> instead of <T>) and the
deck-style LINEAR-energy-axis maps.  Re-uses make_figures.py; writes *_Rmode files and fig2_linear_axis*.png.

    python activation_R.py [--tags base G05 G2] [--suffix ""] [--lam-ops 1292]
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import materials as mat            # noqa: E402
import ttm_activation as ttm       # noqa: E402
import make_figures as mf          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=["base", "G05", "G2"])
    ap.add_argument("--lam-ops", type=float, nargs="*", default=[])
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    lam_ze, _ = mat.ito_zero_crossing()
    H = mf.load_heat(a.tags[0]); Hs = {t: mf.load_heat(t) for t in a.tags[1:]}; Hs = {t: h for t, h in Hs.items() if h is not None}
    mf.P_CELL[0] = H["P"]
    # ---- swapped tables: observable R in the "T" slot
    def swap(h):
        h2 = dict(h); h2["T"], h2["R"] = h["R"], h["T"]; return h2
    HR = swap(H); HRs = {t: swap(h) for t, h in Hs.items()}
    mf.SFX[0] = a.suffix + "_Rmode"
    lam_ops_j, swing = mf.select_lam_ops(HR, lam_ze, extra=a.lam_ops)
    lam_ops = [float(H["lam"][j]) for j in lam_ops_j]
    print("R-mode selected lambda_op:", lam_ops)
    Q = []
    for j in lam_ops_j:
        q = mf.quantify(H["lam"][j], H["I"], H["E"], HR["T"][j], HR["R"][j], H["A"][j], H["Fz"][j], H["Te_peak"][j], H["valid_trust"][j], H["valid_model"][j],
                        Tsens={t: h["T"][j] for t, h in HRs.items()})
        q = {("observable" if k == "lam_op_nm" else k): v for k, v in q.items()} if False else q
        q["observable"] = "R (keys named T_* refer to R in this file)"
        Q.append(q)
    mf.fig34_activation(HR, HRs, lam_ops_j, Q, obs="R")
    with open(mf._p(mf.OUT / "threshold_summary.csv"), "w", newline="") as f:
        keys = sorted(set().union(*[q.keys() for q in Q]), key=lambda k: (k != "lam_op_nm", k))
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for q in Q:
            w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in q.items()})
    with open(mf._p(mf.OUT / "activation_curves.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lambda_op_nm", "I_peak_Wcm2", "E_cell_J", "R_avg", "T_avg", "A_avg", "Te_peak_K", "I_refl_out_Wcm2", "valid_trust(Te<=6000K)", "valid_model(Te<=8000K)"] + [f"R_avg_{t}" for t in Hs])
        for j in lam_ops_j:
            for i in range(len(H["I"])):
                w.writerow([f"{H['lam'][j]:.0f}", f"{H['I'][i]:.4e}", f"{H['E'][i]:.4e}", f"{H['R'][j,i]:.6f}", f"{H['T'][j,i]:.6f}", f"{H['A'][j,i]:.6f}", f"{H['Te_peak'][j,i]:.1f}",
                            f"{H['R'][j,i]*H['I'][i]:.4e}", int(H["valid_trust"][j, i]), int(H["valid_model"][j, i])] + [f"{h['R'][j,i]:.6f}" for h in Hs.values()])
    json.dump(dict(lam_ze=lam_ze, lam_ops=lam_ops, quant=Q, G=H["G"], observable="R"), open(mf._p(mf.OUT / "activation_summary.json"), "w"), indent=1, default=float)
    # per-lambda swing table for R
    with open(mf._p(mf.OUT / "swing_vs_lambda_op.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["lambda_op_nm", "R_cold", "R_at_I_max_trust", "dR_trust", "I_max_trust_Wcm2"])
        for j in range(len(H["lam"])):
            ok = np.where(H["valid_trust"][j])[0]
            w.writerow([f"{H['lam'][j]:.0f}", f"{H['R'][j,0]:.6f}", f"{H['R'][j,ok[-1]]:.6f}", f"{swing[j]:+.6f}", f"{H['I'][ok[-1]]:.3e}"])
    for q in Q:
        print(json.dumps({k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in q.items() if not k.startswith("fit_") or k.endswith("rmse") or k.endswith("norm")}, default=str))

    # ---- deck-style linear-energy-axis maps (T and R) over the trusted range
    mf.SFX[0] = a.suffix
    lam, I, E = H["lam"], H["I"], H["E"]
    jz = int(np.argmin(abs(lam - lam_ze)))
    okz = np.where(H["valid_trust"][jz])[0]; E_max = E[okz[-1]]
    Elin = np.linspace(0, E_max, 121)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.2))
    for ax, key, lab, log in zip(axs, ("T", "R", "A"), ("⟨T⟩", "⟨R⟩", "⟨A⟩"), (True, False, False)):
        M = np.full((len(lam), len(Elin)), np.nan)
        for j in range(len(lam)):
            v = np.interp(Elin, E, H[key][j], left=H[key][j, 0])
            Tep = np.interp(Elin, E, H["Te_peak"][j], left=300.0)
            v[Tep > ttm.TE_MAX_MODEL] = np.nan
            M[j] = v
        im = ax.pcolormesh(Elin * 1e15, lam, np.ma.masked_invalid(M), shading="nearest", cmap="viridis" if key != "T" else "magma", norm=LogNorm(1e-3, 1) if log else None)
        Tmap = np.array([np.interp(Elin, E, H["Te_peak"][j], left=300.0) for j in range(len(lam))])
        ax.contour(Elin * 1e15, lam, Tmap, levels=[ttm.TE_TRUST], colors="w", linestyles="--"); ax.contour(Elin * 1e15, lam, Tmap, levels=[ttm.TE_MAX_MODEL], colors="r")
        ax.axhline(lam_ze, color="w", ls=":", lw=0.8); ax.axhline(mat.n_glass(1250.0) * H["P"], color="0.7", ls="-.", lw=0.8)
        ax.set_xlabel(f"E_cell per {H['P']:.0f}-nm cell [fJ]  (linear axis, deck style)"); ax.set_ylabel("λ_op [nm]"); ax.set_title(lab, fontsize=10)
        plt.colorbar(im, ax=ax, pad=0.01)
    fig.suptitle("Deck-style transfer-function scan on a linear energy axis (masked: T_e,peak > 8000 K; white dashed: 6000 K)", fontsize=10)
    fig.tight_layout(); fig.savefig(mf._p(mf.FIG / "fig2_linear_axis.png"), dpi=160); plt.close(fig)
    print("[activation_R] done")


if __name__ == "__main__":
    main()
