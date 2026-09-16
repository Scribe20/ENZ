"""SELECTION - transparent multi-metric comparison of every evaluated candidate (no scalar collapse).

Reads outputs/candidates/<tag>/stageA.json + stageB.json and produces
    outputs/tables/candidates.csv / .json      one row per candidate, every metric, failed ones included
    outputs/tables/pareto.json                 non-dominated set on the activation-relevant axes
    outputs/tables/threshold_sweep.json        survivors vs (T_bg_min, C_res_min, Ftot_min, eta_z_min)
    outputs/tables/definition_sweep.json       T_bg re-derived for several (k_excl, D_op) from the stored spectra
    outputs/tables/correlations.json           Spearman / Pearson: Q_r, eta_Dz, eta_Dz_P, T_bg vs dT_max_pos, S_T
    outputs/figures/*.png
Nothing here re-solves Maxwell's equations; everything is derived from the stored stage outputs.
"""
import argparse, csv, json
from pathlib import Path

import numpy as np

import common as cm                       # noqa: F401
import parent_background as pb            # noqa: E402

TAB = cm.OUT / "tables"; FIG = cm.OUT / "figures"
ROOT = [cm.OUT / "candidates"]


GUARD_R_NM, GUARD_EDGE_NM = 20.0, 5.0


def interior_point(tag, B, root):
    """Post-hoc operating point away from the window edges: argmax dT with lambda >= lambda_R + GUARD_R_NM and
    lambda <= lam_hi - GUARD_EDGE_NM (from stageB_scan.npz; the Stage-B argmax may sit exactly at the safe-window
    edge, where dT is still rising towards the Rayleigh anomaly or the end of the material data)."""
    f = root / tag / "stageB_scan.npz"
    if not B or not f.exists():
        return {}
    z = np.load(f); lam = z["lam"]
    lo = max(B["safe_window"][0] - B.get("m_R_nm", 10.0) + GUARD_R_NM, lam.min()); hi = B["grid"]["lam_hi"] - GUARD_EDGE_NM
    m = (lam >= lo) & (lam <= hi)
    if not m.any():
        return {}
    i = np.where(m)[0][int(np.argmax(z["dT"][m]))]
    win = (lam >= B["safe_window"][0]) & (lam <= B["safe_window"][1])
    return dict(lam_op_int=float(lam[i]), dT_int=float(z["dT"][i]), S_T_int=float(z["S_T"][i]), T0_int=float(z["T0"][i]), T1_int=float(z["T1"][i]),
                A0_int=float(z["A0"][i]), Ftot0_int=float(z["Ftot0"][i]), eta_z_int=float(z["eta_z0"][i]), dR_int=float(z["dR"][i]),
                T0_loaded_median=float(np.median(z["T0"][win])), T0_loaded_max=float(z["T0"][win].max()), interior_window=[float(lo), float(hi)])


def row_of(tag, A, B):
    c = (B or A)["meta"]["candidate"]
    cert = c.get("cert") or {}
    bg = (A or {}).get("background", {}); rs = (A or {}).get("resonance", {}) or {}
    pm = (A or {}).get("parent_metrics_here", {})
    pts = (B or {}).get("points", {}); pos = pts.get("lambda_op_pos", {}); neg = pts.get("lambda_op_neg", {}); pE = pts.get("lambda_E", {})
    cc = (B or {}).get("critical_coupling_estimate", {}) or {}
    lin = ((B or {}).get("linearity_direction") or [{}])[0]
    interior = interior_point(tag, B, ROOT[0])
    conv = [x for x in (B or {}).get("order_convergence", []) if pos and abs(x["lam"] - pos.get("lam_nm", -1)) < 1e-6]
    conv = conv[0] if conv else {}
    hi = (conv.get("by_order") or [{}])[-1]
    eta_E = cert.get("eta_Dz_lamE", pm.get("lambda_E", {}).get("eta_Dz"))
    etaP_E = cert.get("eta_Dz_P_lamE", pm.get("lambda_E", {}).get("eta_Dz_P"))
    poles_here = (A or {}).get("poles", [])
    near = rs.get("pole") or {}
    return dict(tag=tag, kind=c["kind"], certified=c.get("certified"), P=c["P"], h=c["h"], Q_target=c.get("Q_target"),
                fill=(B or A)["meta"].get("fill_fraction"), rho_sha256=((B or A)["meta"].get("rho_sha256") or "")[:16],
                Q_r=cert.get("Q_r", near.get("Q")), lambda_r=cert.get("lambda_r", near.get("lam")), fwhm_r=cert.get("fwhm_r_nm", near.get("fwhm")),
                Q_source=("certified exact pole" if cert.get("Q_r") else ("exact pole computed here (not certified)" if near else "none")),
                eta_Dz_lamE=eta_E, eta_Dz_P_lamE=etaP_E, eta_source=("certification" if cert.get("eta_Dz_lamE") is not None else "computed here"),
                eta_regression_rel=(A or {}).get("regression_vs_certification", {}).get("rel_diff"),
                T_parent_lamE=(A or {}).get("T_parent_lamE"), R_parent_lamE=(A or {}).get("R_parent_lamE"),
                T_bg=bg.get("T_bg_median"), T_bg_min=bg.get("T_bg_min"), T_bg_n=bg.get("n_points"), T_bg_defined=bg.get("defined"),
                R_bg=bg.get("R_bg_median"), T_at_lambda_r=rs.get("T_at_lambda_r"), C_res=rs.get("C_res_center"), C_res_ext=rs.get("C_res_ext"),
                res_depth=(rs.get("local") or {}).get("depth"),
                lam_op_pos=pos.get("lam_nm"), dT_max_pos=pos.get("dT"), S_T_pos=pos.get("S_T"), T0_pos=pos.get("T0"), T1_pos=pos.get("T1"),
                contrast_pos=pos.get("contrast_ratio"), dR_pos=pos.get("dR"), dA_pos=pos.get("dA"), channel_pos=(pos.get("channel") or {}).get("label"),
                A0_pos=pos.get("A0"), Ftot0_pos=pos.get("Ftot0"), Fz0_pos=pos.get("Fz0"), eta_z_pos=pos.get("eta_z0"), mean_Ez2_pos=pos.get("mean_Ez2_0"),
                rayleigh_margin_nm_pos=(pos.get("rayleigh_margin") or {}).get("margin_nm"),
                at_window_edge_pos=(bool(pos.get("lam_nm") is not None and (pos["lam_nm"] >= ((B or {}).get("grid") or {}).get("lam_hi", 1400.0) - 2.0
                                                                          or pos["lam_nm"] <= ((B or {}).get("safe_window") or [0])[0] + 2.0))),
                loaded_Q_pos=pos.get("pole_Q_cold"), loaded_pole_shift_nm=pos.get("pole_shift_nm"),
                unresolved_pos=pos.get("numerically_unresolved"), dT_order_change=conv.get("dT_change_last_two"), dT_at_highest_order=hi.get("dT"),
                highest_order=(hi.get("order") or [None])[0], linear_response=lin.get("linear_response"), S_T_spread_rel=lin.get("S_T_spread_rel"),
                frac_dT_from_Re=lin.get("fraction_dT_from_real_part"),
                **interior,
                lam_op_neg=neg.get("lam_nm"), dT_max_neg=neg.get("dT"), T0_neg=neg.get("T0"), channel_neg=(neg.get("channel") or {}).get("label"),
                dT_lamE=pE.get("dT"), T0_lamE=pE.get("T0"), A0_lamE=pE.get("A0"), Fz0_lamE=pE.get("Fz0"), eta_z_lamE=pE.get("eta_z0"),
                A_to_R_dominant=((B or {}).get("global_scan") or {}).get("A_to_R_dominant"),
                Ftot0_max=((B or {}).get("global_scan") or {}).get("Ftot0_max"), T0_max=((B or {}).get("global_scan") or {}).get("T0_max_in_window"),
                gamma_nr_over_gamma_r_est=cc.get("gamma_nr_over_gamma_r"), Q_loaded_est=cc.get("Q_loaded_est"), Q_loaded_exact=cc.get("Q_loaded_exact_nearest"),
                eta_Dz_P_critical=cc.get("eta_Dz_P_for_critical_coupling"),
                multipole_dominant=(max(cert.get("multipoles_lamE", {}).items(), key=lambda kv: kv[1])[0] if cert.get("multipoles_lamE") else None),
                abs_delta_eps_pos=pos.get("abs_delta_eps"), Te=((B or {}).get("perturbation") or {}).get("Te_K"),
                order_B=((B or {}).get("order") or [None])[0], order_A=((A or {}).get("order_spectrum") or [None])[0],
                identity_resid_max=((B or {}).get("global_scan") or {}).get("max_identity_resid"),
                wall_s=((A or {}).get("wall_s") or 0) + ((B or {}).get("wall_s") or 0))


def dominated(r, others, keys):
    """r is dominated if some other row is >= on every key and > on at least one (None counts as -inf)."""
    v = lambda x, k: (-np.inf if x.get(k) is None else x[k])
    for o in others:
        if o is r:
            continue
        ge = all(v(o, k) >= v(r, k) for k in keys); gt = any(v(o, k) > v(r, k) for k in keys)
        if ge and gt:
            return True
    return False


def spearman(x, y):
    from scipy.stats import spearmanr, pearsonr
    m = np.array([(a is not None and b is not None and np.isfinite(a) and np.isfinite(b)) for a, b in zip(x, y)])
    if m.sum() < 4:
        return dict(n=int(m.sum()), spearman=None, pearson=None)
    xa = np.array([float(a) for a, k in zip(x, m) if k]); ya = np.array([float(b) for b, k in zip(y, m) if k])
    s = spearmanr(xa, ya); p = pearsonr(xa, ya)
    return dict(n=int(m.sum()), spearman=float(s.statistic), spearman_p=float(s.pvalue), pearson=float(p.statistic), pearson_p=float(p.pvalue))


def buildup_table(rows_out):
    """Hypothesis test on the CERTIFIED parents (independent of Stage B): does a larger exact Q_r buy internal
    build-up (U_mid, W_Si at lambda_r) at the expense of interface participation (eta_Dz, eta_Dz_P)?"""
    import candidates as cd
    from scipy.stats import spearmanr
    rows = []
    for c in cd.registry(include=("certified", "baselines")):
        ce = cm.jload(c["cert_path"])
        if not ce.get("Q_r"):
            continue
        mE = ce["metrics"]["lambda_E"]["by_order"][-1]; mR = ce["metrics"]["lambda_r"]["by_order"][-1]
        rows.append(dict(tag=c["tag"], Q_r=ce["Q_r"], lambda_r=ce["lambda_r"], detuning_lw=ce["detuning_in_linewidths"],
                         eta_Dz_lamE=mE["eta_Dz"], eta_Dz_P_lamE=mE["eta_Dz_P"], U_mid_lamE=mE["U_mid"], W_Si_lamE=mE["W_Si"],
                         eta_Dz_lamr=mR["eta_Dz"], eta_Dz_P_lamr=mR["eta_Dz_P"], U_mid_lamr=mR["U_mid"], W_Si_lamr=mR["W_Si"], T_parent_lamE=mE["T"], T_parent_lamr=mR["T"]))
    out = dict(rows=rows, correlations={})
    for subset, sel in (("all_certified_with_pole", rows), ("aligned_abs_detuning_lt_1p5_lw", [r for r in rows if abs(r["detuning_lw"]) < 1.5])):
        Q = np.log([r["Q_r"] for r in sel]); d = {}
        for k in ("U_mid_lamr", "W_Si_lamr", "eta_Dz_lamr", "eta_Dz_P_lamr", "U_mid_lamE", "eta_Dz_lamE", "T_parent_lamE"):
            v = [r[k] for r in sel]
            if len(sel) >= 4:
                s_ = spearmanr(Q, v); d[k] = dict(spearman_vs_logQ=float(s_.statistic), p=float(s_.pvalue))
        out["correlations"][subset] = dict(n=len(sel), **d)
    cm.jdump(out, TAB / "certified_Q_vs_buildup.json")
    return out


def write_csv(rows, path):
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()
    root = Path(a.root) if a.root else cm.OUT / "candidates"
    ROOT[0] = root
    TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    rows, full = [], {}
    for d in sorted(root.iterdir()):
        A = cm.jload(d / "stageA.json") if (d / "stageA.json").exists() else None
        B = cm.jload(d / "stageB.json") if (d / "stageB.json").exists() else None
        if A is None and B is None:
            continue
        rows.append(row_of(d.name, A, B)); full[d.name] = dict(A=A, B=B)
    if not rows:
        print("no candidates found"); return
    write_csv(rows, TAB / "candidates.csv"); cm.jdump(rows, TAB / "candidates.json")
    # ---- Pareto (activation-relevant axes; larger is better everywhere) ------------------------------
    keys = ("dT_int", "T_bg", "C_res_ext", "Ftot0_int", "eta_z_int", "T0_loaded_median", "Q_r", "eta_Dz_lamE")
    designs = [r for r in rows if r["kind"] != "reference"]
    for r in rows:
        r["dominated_activation_axes"] = dominated(r, designs, keys) if r in designs else None
    front = [r["tag"] for r in designs if not r["dominated_activation_axes"]]
    cm.jdump(dict(axes=list(keys), non_dominated=front,
                  dominated=[r["tag"] for r in designs if r["dominated_activation_axes"]]), TAB / "pareto.json")
    # ---- threshold sweep (sweepable hyperparameters, not constants) ----------------------------------
    sweep = []
    for Tb in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
        for Cr in (0.05, 0.1, 0.2, 0.3):
            for Fm in (0.02, 0.05, 0.1, 0.2):
                for Ez in (0.0, 0.5, 0.7, 0.8):
                    surv = [r["tag"] for r in designs if (r["T_bg"] or 0) >= Tb and (r["C_res_ext"] or 0) >= Cr and (r.get("Ftot0_int") or 0) >= Fm
                            and (r.get("eta_z_int") or 0) >= Ez and (r.get("dT_int") or 0) > 0]
                    best = max(surv, key=lambda t: next(r["dT_int"] for r in designs if r["tag"] == t), default=None)
                    sweep.append(dict(T_bg_min=Tb, C_res_min=Cr, Ftot_min=Fm, eta_z_min=Ez, n_survivors=len(surv), survivors=surv, best_dT=best))
    cm.jdump(sweep, TAB / "threshold_sweep.json")
    # ---- definition sweep of T_bg from the stored spectra --------------------------------------------
    dsw = {}
    for tag, fb in full.items():
        A = fb["A"]
        if not A or "spectrum" not in A:
            continue
        s = A["spectrum"]; lam = np.array(s["lam"]); T = np.array(s["T"]); R = np.array(s["R"])
        dsw[tag] = {}
        for k in (1.0, 2.0, 3.0, 5.0):
            for D in (30.0, 50.0, 80.0):
                b = pb.background_set(lam, T, R, A["poles"], A["lam_E"], A["lambda_R_top"], D_op=D, m_R=A["parameters"]["m_R_nm"], k_excl=k)
                dsw[tag][f"k{k:g}_D{D:g}"] = dict(T_bg=b["T_bg_median"], n=b["n_points"], defined=b["defined"])
    cm.jdump(dsw, TAB / "definition_sweep.json")
    # ---- correlations ---------------------------------------------------------------------------------
    cor = {}
    for xk in ("Q_r", "eta_Dz_lamE", "eta_Dz_P_lamE", "T_bg", "C_res_ext", "gamma_nr_over_gamma_r_est", "Ftot0_pos", "eta_z_pos"):
        for yk in ("dT_max_pos", "S_T_pos", "dT_max_neg", "loaded_Q_pos"):
            cor[f"{xk}__vs__{yk}"] = spearman([r[xk] for r in designs], [r[yk] for r in designs])
    cm.jdump(cor, TAB / "correlations.json")
    bt = buildup_table(rows)
    print("certified parents, spearman vs log Q_r:", json.dumps(bt["correlations"], indent=None))
    # ---- markdown table --------------------------------------------------------------------------------
    cols = [("tag", "{}"), ("kind", "{}"), ("Q_target", "{}"), ("Q_r", "{:.0f}"), ("eta_Dz_lamE", "{:.3f}"), ("eta_Dz_P_lamE", "{:.3f}"), ("T_bg", "{:.2f}"),
            ("C_res_ext", "{:.2f}"), ("T0_loaded_median", "{:.2f}"), ("lam_op_int", "{:.1f}"), ("dT_int", "{:+.4f}"), ("S_T_int", "{:+.3f}"), ("T0_int", "{:.3f}"),
            ("Ftot0_int", "{:.3f}"), ("eta_z_int", "{:.2f}"), ("lam_op_pos", "{:.1f}"), ("dT_max_pos", "{:+.4f}"), ("at_window_edge_pos", "{}"), ("channel_pos", "{}"),
            ("dT_max_neg", "{:+.4f}"), ("gamma_nr_over_gamma_r_est", "{:.1f}"), ("loaded_Q_pos", "{:.0f}"), ("unresolved_pos", "{}"), ("dominated_activation_axes", "{}")]
    def fmt(v, f):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "-"
        try:
            return f.format(v)
        except Exception:
            return str(v)
    lines = ["| " + " | ".join(k for k, _ in cols) + " |", "|" + "---|" * len(cols)]
    for r in sorted(rows, key=lambda r: -(r.get("dT_int") if r.get("dT_int") is not None else -9)):
        lines.append("| " + " | ".join(fmt(r.get(k), f) for k, f in cols) + " |")
    (TAB / "candidates.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nnon-dominated on {keys}: {front}")
    # ---- figures ------------------------------------------------------------------------------------------
    if not a.no_fig:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        def sc(ax, xk, yk, logx=False):
            for r in rows:
                if r[xk] is None or r[yk] is None:
                    continue
                m = {"qdz_certified": "o", "qdz_uncertified": "^", "jzfz_baseline": "s", "reference": "*"}[r["kind"]]
                col = "tab:red" if r.get("A_to_R_dominant") else ("tab:green" if (r["T_bg"] or 0) > 0.6 else "tab:gray")
                ax.scatter(r[xk], r[yk], marker=m, s=70, color=col, edgecolor="k", zorder=3)
                ax.annotate(r["tag"].replace("fr_", "").replace("pilot_", "p_"), (r[xk], r[yk]), fontsize=6, xytext=(3, 3), textcoords="offset points")
            if logx:
                ax.set_xscale("log")
            ax.set_xlabel(xk); ax.set_ylabel(yk); ax.grid(alpha=0.3); ax.axhline(0, color="k", lw=0.6)
        fig, axs = plt.subplots(2, 3, figsize=(16, 9))
        sc(axs[0, 0], "Q_r", "dT_max_pos", logx=True); sc(axs[0, 1], "eta_Dz_lamE", "dT_max_pos"); sc(axs[0, 2], "T_bg", "dT_max_pos")
        sc(axs[1, 0], "gamma_nr_over_gamma_r_est", "dT_max_pos", logx=True); sc(axs[1, 1], "Ftot0_pos", "dT_max_pos"); sc(axs[1, 2], "eta_z_pos", "dT_max_pos")
        fig.suptitle("Stage B: max positive real-ITO dT (Te perturbation) vs parent / heating metrics  (red: absorption->reflection dominant; green: T_bg > 0.6)", fontsize=10)
        fig.tight_layout(); fig.savefig(FIG / "dT_vs_metrics.png", dpi=150); plt.close(fig)
        # spectra panels for the top 6 by dT_max_pos and the references
        top = [r["tag"] for r in sorted(designs, key=lambda r: -(r["dT_max_pos"] or -9))[:6]] + [r["tag"] for r in rows if r["kind"] == "reference"]
        top = [t for t in top if (root / t / "stageB_scan.npz").exists()]
        fig, axs = plt.subplots(max(len(top), 1), 1, figsize=(9, 2.6 * max(len(top), 1)), sharex=True)
        for ax, tag in zip(np.atleast_1d(axs), top):
            z = np.load(root / tag / "stageB_scan.npz") if (root / tag / "stageB_scan.npz").exists() else None
            if z is None:
                continue
            ax.plot(z["lam"], z["T0"], "k-", lw=1.2, label="T cold"); ax.plot(z["lam"], z["T1"], "r--", lw=1.0, label="T perturbed")
            ax.plot(z["lam"], z["A0"], "b-", lw=0.8, label="A cold"); ax.plot(z["lam"], z["Fz0"], "b:", lw=0.8, label="Fz cold")
            ax2 = ax.twinx(); ax2.plot(z["lam"], z["dT"], "g-", lw=1.0, label="dT"); ax2.axhline(0, color="g", lw=0.4); ax2.set_ylabel("dT", color="g")
            A = full[tag]["A"]
            if A and A.get("spectrum"):
                ax.plot(A["spectrum"]["lam"], A["spectrum"]["T"], color="0.6", lw=0.8, ls="-.", label="T parent (no ITO)")
            ax.axvline(full[tag]["B"]["lam_E"], color="k", ls=":", lw=0.6)
            for rl in full[tag]["B"]["rayleigh"]:
                if 1160.0 <= rl["lam"] <= 1400.0:
                    ax.axvline(rl["lam"], color="m", ls=":", lw=0.6)
            ax.set_title(tag, fontsize=9); ax.set_ylim(0, 1); ax.set_xlim(1160.0, 1400.0); ax.grid(alpha=0.3)
            if ax is np.atleast_1d(axs)[0]:
                ax.legend(fontsize=6, ncol=5, loc="upper left")
        np.atleast_1d(axs)[-1].set_xlabel("wavelength [nm]")
        fig.tight_layout(); fig.savefig(FIG / "spectra_top.png", dpi=140); plt.close(fig)
        print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
