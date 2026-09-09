"""Collect every optimization run + certification into the campaign tables and the primary plots.

  outputs/tables/runs.csv            one row per optimization run (settings + final proxy values)
  outputs/tables/candidates.csv      one row per CERTIFIED hard-binary candidate (exact pole Q_r, eta_Dz, U_mid, ...)
  outputs/figures/pareto_Qr_etaDz.png    the primary result: certified Q_r  vs  eta_Dz
  outputs/figures/Qr_vs_Umid.png         Q_r vs U_mid
  outputs/figures/lambdar_vs_Qr.png      lambda_r vs Q_r (spectral alignment achieved)
  outputs/figures/histories.png          optimization histories (eta, Q proxy, loss terms)
  outputs/figures/geometries.png         representative certified geometries
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import parent_fwd as pf                       # noqa: E402

TAB = HERE / "outputs" / "tables"; FIG = HERE / "outputs" / "figures"
LAM_E = pf.lambda_E()


def run_rows():
    rows = []
    for p in sorted(HERE.glob("*/*/result.json")) + sorted(HERE.glob("*/*/*/result.json")):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if r.get("campaign") != "qdz_parent":
            continue
        H = r["history"]
        rows.append(dict(tag=r["tag"], stage=p.parent.parent.name, dir=str(p.parent.relative_to(HERE)),
                         P=r["P"], h=r["h"], Q_target=r["Q_target"], pad=r["pad_frac"], seed=r["seed"],
                         warm=r["warm_start"], n_iter=r["n_iter"], order=str(r["order"]),
                         eta_soft=r["soft"]["eta_Dz"], eta_hard=r["hard"]["eta_Dz"],
                         Q_proxy_hard=r["hard"].get("Q_proxy"), lam_r_proxy_hard=r["hard"].get("lambda_r_proxy"),
                         curvature_u_hard=r["hard"].get("curvature_u"), U_mid_hard=r["hard"]["U_mid"],
                         L_obj_end=H["L_obj"][-1], L_Q_end=H["L_Q"][-1], L_lam_end=H["L_lam"][-1],
                         u_end=H["curvature_u"][-1], u_max=max(H["curvature_u"]),
                         acquired=bool(max(H["curvature_u"]) > 0.1),
                         eta_gain=H["eta_Dz"][-1] / max(H["eta_Dz"][0], 1e-30),
                         fill=r["fill_fraction"], wall_s=r["wall_s"]))
    return rows


def cand_rows():
    rows = []
    for p in sorted(HERE.glob("outputs/certified/*/certification.json")) + sorted(HERE.glob("baselines/*/certification.json")):
        c = json.load(open(p))
        mE = c.get("metrics", {}).get("lambda_E", {}).get("by_order", [{}])[-1]
        mR = c.get("metrics", {}).get("lambda_r", {}).get("by_order", [{}])[-1]
        mp = c.get("multipoles", {}).get("lambda_E", {})
        rows.append(dict(tag=c["tag"], kind=("baseline" if "baselines" in str(p) else "optimized"),
                         P=c["P"], h=c["h"], sha256=c["sha256"][:16], fill=c["fill_fraction"],
                         Q_r=c.get("Q_r"), lambda_r=c.get("lambda_r"), detuning_nm=c.get("detuning_nm"),
                         detuning_linewidths=c.get("detuning_in_linewidths"),
                         eta_Dz_lamE=mE.get("eta_Dz"), eta_Dz_P_lamE=mE.get("eta_Dz_P"), U_mid_lamE=mE.get("U_mid"),
                         eta_Dz_lamr=mR.get("eta_Dz"), U_mid_lamr=mR.get("U_mid"),
                         R_lamE=mE.get("R"), T_lamE=mE.get("T"), W_Si_lamE=mE.get("W_Si"),
                         order_conv_rel=c.get("metrics", {}).get("lambda_E", {}).get("order_convergence_rel"),
                         n_parent_poles=len(c.get("parent_poles", [])),
                         pole_refine_converged=c.get("local_refine", {}).get("converged"),
                         Q_proxy=c.get("q_proxy_at_certification", {}).get("Q_proxy"),
                         Q_proxy_rel_err=c.get("q_proxy_at_certification", {}).get("rel_err_vs_pole"),
                         ED=mp.get("frac_ED_eff"), MD=mp.get("frac_MD"), EQ=mp.get("frac_EQ"),
                         MQ=mp.get("frac_MQ"), TD_diag=mp.get("TD_diag_over_sum"),
                         min_feature_px=c.get("geometry", {}).get("min_feature_px"),
                         energy_resid=c.get("parent_energy_resid"), dir=str(p.parent.relative_to(HERE))))
    return rows


def write_csv(rows, path):
    if not rows:
        return
    keys = list({k for r in rows for k in r})
    keys = [k for k in rows[0]] + [k for k in keys if k not in rows[0]]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main():
    TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    R, C = run_rows(), cand_rows()
    write_csv(R, TAB / "runs.csv"); write_csv(C, TAB / "candidates.csv")
    json.dump(dict(runs=R, candidates=C, lam_E=LAM_E), open(TAB / "campaign.json", "w"), indent=1, default=float)
    print(f"[collect] {len(R)} runs -> runs.csv ; {len(C)} certified candidates -> candidates.csv")
    ok = [c for c in C if c["Q_r"] and c["eta_Dz_lamE"] is not None]
    if ok:
        # ---- primary Pareto plot
        fig, ax = plt.subplots(figsize=(8.2, 5.6))
        opt = [c for c in ok if c["kind"] == "optimized"]; bas = [c for c in ok if c["kind"] == "baseline"]
        if opt:
            qt = sorted({round(float(str(c["tag"]).split("_Q")[1].split("_")[0])) if "_Q" in str(c["tag"]) else 0 for c in opt})
            cmap = plt.get_cmap("viridis")
            for c in opt:
                q = float(str(c["tag"]).split("_Q")[1].split("_")[0]) if "_Q" in str(c["tag"]) else 0
                col = cmap(qt.index(round(q)) / max(len(qt) - 1, 1)) if qt else "C0"
                ax.scatter(c["Q_r"], c["eta_Dz_lamE"], s=70, color=col, edgecolor="k", zorder=3,
                           label=f"Q_target = {q:.0f}" if f"Q_target = {q:.0f}" not in [t.get_text() for t in ax.get_legend_handles_labels()[1]] else None)
        for c in bas:
            ax.scatter(c["Q_r"], c["eta_Dz_lamE"], s=90, marker="s", facecolor="none", edgecolor="tab:red", zorder=4)
            ax.annotate(c["tag"], (c["Q_r"], c["eta_Dz_lamE"]), fontsize=7, color="tab:red",
                        xytext=(4, 4), textcoords="offset points")
        pts = sorted([(c["Q_r"], c["eta_Dz_lamE"]) for c in ok])
        front, best = [], -1
        for q, e in sorted(pts, key=lambda z: -z[0]):
            if e > best:
                front.append((q, e)); best = e
        if len(front) > 1:
            ax.plot(*zip(*sorted(front)), "k--", lw=1, alpha=0.6, label="non-dominated set")
        ax.set_xscale("log"); ax.set_xlabel("certified radiative $Q_r$ of the parent mode (exact pole)")
        ax.set_ylabel(r"$\eta_{Dz}$  (normalized longitudinal interface participation)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_title(f"Parent-mode trade-off at $\\lambda_E$ = {LAM_E:.2f} nm (squares: existing Jz/Fz and cylinder baselines)", fontsize=10)
        fig.tight_layout(); fig.savefig(FIG / "pareto_Qr_etaDz.png", dpi=170); plt.close(fig)
        # ---- Q vs U_mid and lambda_r vs Q_r
        for yk, yl, fn in (("U_mid_lamE", r"$U_{mid}$ (central-slab $\epsilon|E|^2$ density)", "Qr_vs_Umid.png"),
                           ("lambda_r", r"certified $\lambda_r$ [nm]", "lambdar_vs_Qr.png")):
            fig, ax = plt.subplots(figsize=(7.4, 5))
            for c in ok:
                m = "s" if c["kind"] == "baseline" else "o"
                ax.scatter(c["Q_r"], c[yk], s=70, marker=m, facecolor=("none" if m == "s" else "C0"),
                           edgecolor=("tab:red" if m == "s" else "k"))
            if yk == "lambda_r":
                ax.axhline(LAM_E, color="k", ls=":", lw=1, label=f"$\\lambda_E$ = {LAM_E:.2f} nm"); ax.legend(fontsize=8)
            ax.set_xscale("log"); ax.set_xlabel("certified $Q_r$"); ax.set_ylabel(yl); ax.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(FIG / fn, dpi=170); plt.close(fig)
    # ---- histories
    runs = sorted(HERE.glob("*/*/result.json"))
    runs = [p for p in runs if json.load(open(p)).get("campaign") == "qdz_parent"]
    if runs:
        fig, axs = plt.subplots(2, 2, figsize=(13, 7.5))
        for p in runs:
            r = json.load(open(p)); H = r["history"]; lab = r["tag"]
            axs[0, 0].plot(H["eta_Dz"], label=lab)
            axs[0, 1].plot(H["curvature_u"], label=lab)
            axs[1, 0].plot(H["L_Q"], label=lab)
            axs[1, 1].plot(H["U_mid"], label=lab)
        for ax, t in zip(axs.flat, (r"$\eta_{Dz}$", r"fit curvature $u=c/c_{target}$ (>0: a peak exists at $\lambda_E$)",
                                    r"$L_Q$", r"$U_{mid}$")):
            ax.set_xlabel("iteration"); ax.set_title(t, fontsize=10); ax.grid(alpha=0.3)
        axs[0, 1].axhline(0, color="k", lw=0.8); axs[0, 1].axhline(1, color="g", ls=":", lw=0.8)
        axs[0, 0].legend(fontsize=6, ncol=2)
        fig.tight_layout(); fig.savefig(FIG / "histories.png", dpi=150); plt.close(fig)
    print(f"[collect] figures -> {FIG}")


if __name__ == "__main__":
    main()
