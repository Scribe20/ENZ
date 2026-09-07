"""Stage 4: Pareto analysis (no weighted score)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as cm

df = pd.read_csv(cm.OUT / "stage3_branches_table.csv")
d1 = cm.jload(cm.OUT / "stage1_driven_lambdaE.json")
df = df[np.isfinite(df["Q_rad"])].copy()
df["logQrad"] = np.log10(df["Q_rad"].clip(lower=1e-3))


def pareto(points, keys):
    idx = []
    for i, p in points.iterrows():
        dominated = any(all(q[k] >= p[k] for k in keys) and any(q[k] > p[k] for k in keys)
                        for j, q in points.iterrows() if j != i)
        if not dominated:
            idx.append(i)
    return points.loc[idx]


fronts = {
    "logQrad_vs_FEz_lambdaE": pareto(df, ["logQrad", "F_Ez_lambdaE"]),
    "logQrad_vs_FEz_pole": pareto(df, ["logQrad", "F_Ez_pole"]),
    "logQrad_vs_etaENZ_modal": pareto(df.dropna(subset=["eta_ENZ_z_modal"]), ["logQrad", "eta_ENZ_z_modal"]),
    "FEz_vs_A_lambdaE": pareto(df, ["F_Ez_lambdaE", "A_lambdaE"]),
}
hist_FEz = 2.2
q_hi = float(np.percentile(df["Q_rad"], 75))
upper_right = df[(df["Q_rad"] >= max(100.0, q_hi)) & (df["F_Ez_lambdaE"] >= 1.5 * hist_FEz)]
summary = dict(
    highest_Q_Ez_rich_parent=df.sort_values("Q_rad", ascending=False).iloc[0]["branch"],
    strongest_driven_Ez=df.sort_values("F_Ez_lambdaE", ascending=False).iloc[0]["branch"],
    best_accessible_highQ=(df[df["Q_rad"] >= 100].sort_values("gamma_ratio", ascending=False).iloc[0]["branch"]
                           if (df["Q_rad"] >= 100).any() else None),
    knee=None, Q_rad_75pct=q_hi, historical_FEz=hist_FEz,
    upper_right_candidates=upper_right["branch"].tolist(),
    fronts={k: v["branch"].tolist() for k, v in fronts.items()})
# knee: maximize product of normalized ranks on the logQ-FEz front
f = fronts["logQrad_vs_FEz_lambdaE"]
if len(f):
    lq = (f["logQrad"] - df["logQrad"].min()) / max(df["logQrad"].max() - df["logQrad"].min(), 1e-9)
    fe = (f["F_Ez_lambdaE"] - df["F_Ez_lambdaE"].min()) / max(df["F_Ez_lambdaE"].max() - df["F_Ez_lambdaE"].min(), 1e-9)
    summary["knee"] = f.iloc[int(np.argmax(lq.values * fe.values))]["branch"]
cm.jdump(summary, cm.OUT / "stage4_pareto.json")
print(summary)

fig, axs = plt.subplots(2, 2, figsize=(13, 10))
def sc(ax, x, y, xl, yl, title, front=None):
    for _, r in df.iterrows():
        pad = cm.CANDIDATES[r["candidate"]][3]
        ax.scatter(r[x], r[y], s=40 + 400 * min(r["gamma_ratio"], 1), c="tab:orange" if pad > 0 else "tab:blue",
                   alpha=.7, edgecolor="k")
        ax.annotate(r["branch"].replace(" robust", "").replace("padded ", "pad "), (r[x], r[y]), fontsize=7,
                    textcoords="offset points", xytext=(4, 3))
    if front is not None and len(front):
        fr = front.sort_values(x); ax.plot(fr[x], fr[y], "k--", lw=.8)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title, fontsize=10); ax.grid(alpha=.3)
sc(axs[0, 0], "logQrad", "F_Ez_lambdaE", "log10 Q_rad", "F_Ez driven at lambda_E",
   "Fig.1 log Q_rad vs driven F_Ez (marker size ~ gamma_rad/gamma_nr; orange = padded)", fronts["logQrad_vs_FEz_lambdaE"])
sc(axs[0, 1], "logQrad", "eta_ENZ_z_modal", "log10 Q_rad", "eta_ENZ,z modal (lossless aux.)",
   "Fig.2 log Q_rad vs modal ENZ participation", fronts["logQrad_vs_etaENZ_modal"])
sc(axs[1, 0], "F_Ez_lambdaE", "A_lambdaE", "F_Ez(lambda_E)", "A_ITO(lambda_E)", "Fig.3 F_Ez vs A_ITO", fronts["FEz_vs_A_lambdaE"])
sc(axs[1, 1], "gamma_ratio", "F_Ez_lambdaE", "gamma_rad / gamma_nr", "F_Ez(lambda_E)", "Fig.4 accessibility vs driven F_Ez")
axs[1, 1].set_xscale("log")
fig.savefig(cm.FIG / "fig1-4_pareto.png", dpi=150, bbox_inches="tight")
print("[stage4] done")
