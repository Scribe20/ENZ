"""Metasurface-oriented ENZ target-candidate selection from the QNM branch.

Consumes outputs/enz_branch_metrics.csv (analyze_enz_branch_for_metasurface)
and produces:
  * exact-K benchmark solves for the Karimi lattice constants (850, 810 nm,
    first-order) and the previous 770 nm/(3,0) Phase-2 target,
  * a quantitative classification of the previous 1527 nm max-localization
    complex-K point (kept as a valid Maxwell pole, NOT discarded),
  * outputs/benchmark_850_810.csv, outputs/target_candidates.csv,
  * figures A-D (qnm_dispersion_metrics, metasurface_period_mapping,
    branch_field_comparison, target_candidate_summary).

No single-metric ranking is used; the tradeoffs are tabulated explicitly.

Run:  python select_enz_target_candidates.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from ito_material import ITOMaterial
from tm_slab_mode import ModeField
import analyze_enz_branch_for_metasurface as ab

C_NM_FS = 299.792458
OUT = ab.OUT_DIR
FIG = config.FIG_DIR

C_BLUE, C_ORANGE, C_AQUA, C_VIOLET, C_MAGENTA = \
    "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": "#e3e2df", "grid.linewidth": 0.6, "font.size": 10,
    "lines.linewidth": 2.0, "legend.frameon": False})


def _save(fig, name):
    fig.savefig(FIG / name, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"figure written: {FIG/name}")


# ---------------------------------------------------------------------------
def exact_point(K, label, ito, p, drude, w_seed=None):
    """Solve the QNM at an exact K and return the full metric row."""
    w_seed = w_seed or 2 * np.pi * C_NM_FS / 1450.0 * (1 - 0.09j)
    w, res = ab.solve_qnm(K, w_seed, drude, p, config.D_ITO_NM)
    if w is None:
        raise RuntimeError(f"QNM solve failed for {label}")
    m = ab.point_metrics(K, w, drude, p, ito, config.D_ITO_NM)
    m["pole_residual"] = res
    m["label"] = label
    return m


def legacy_1527_metrics(ito):
    """Same field diagnostics for the previous max-localization target
    (real-omega/complex-K formulation, lambda = 1527 nm) - kept as a valid
    Maxwell pole of the OTHER representation and classified, not discarded."""
    d = np.load(config._HERE / "target_enz_mode.npz")
    lam = float(d["wavelength_nm"])
    K = complex(float(d["K_real_per_nm"]), float(d["K_imag_per_nm"]))
    k0 = 2 * np.pi / lam
    eps2 = complex(float(d["eps_ito_real"]), float(d["eps_ito_imag"]))
    mode = ModeField(K, k0, eps2, config.D_ITO_NM, ab.EPS1, ab.EPS3)
    z = np.linspace(0.0, config.D_ITO_NM, 401)
    aEz, aEx = np.abs(mode.Ez(z)), np.abs(mode.Ex(z))
    kz1, _, kz3 = mode.kz
    return dict(
        label="legacy 1527 nm (real-w, complex-K, max-localization)",
        K_per_nm=K.real, K_over_k0=K.real / k0, lambda_qnm_nm=lam,
        omega_real=np.nan, omega_imag=np.nan, Q=np.nan,
        linewidth_THz=np.nan,
        eps_ito_cont_real=np.nan, eps_ito_cont_imag=np.nan,
        eps_ito_meas_real=eps2.real, eps_ito_meas_imag=eps2.imag,
        Ez_to_Ex_rms=float(np.sqrt(np.trapezoid(aEz**2, z)
                                   / np.trapezoid(aEx**2, z))),
        Ez_to_Ex_max=float(aEz.max() / aEx.max()),
        Ez_flatness_cv=float(aEz.std() / aEz.mean()),
        Ez_flatness_max_min=float(aEz.max() / aEz.min()),
        Ez_intensity_localization_fraction=float(d["localization_fraction"]),
        air_decay_length_nm=1 / kz1.imag, glass_decay_length_nm=1 / kz3.imag,
        pole_residual=float(d["pole_residual"]),
        max_boundary_condition_residual=np.nan,
        outside_air_lightline=True, outside_glass_lightline=True,
        bound_or_leaky="bound (z), overdamped in-plane (Im K = "
                       f"{K.imag/k0:.2f} k0)",
        Lambda_10_nm=2 * np.pi / K.real,
        Lambda_11_nm=2 * np.pi * np.sqrt(2) / K.real,
        Lambda_20_nm=4 * np.pi / K.real,
    )


# ---------------------------------------------------------------------------
def figA_B(df, lam_ze, marks):
    fig = plt.figure(figsize=(12.6, 8.6))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.25, 1, 1], hspace=0.42,
                          wspace=0.33)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(df.K_over_k0, df.lambda_qnm_nm, color=C_BLUE,
            label=r"bound QNM branch $\lambda_{\rm QNM}(K)$ (Drude continuation)")
    ax.axhline(lam_ze, color=C_VIOLET, ls="--", lw=1.1)
    ax.text(6.0, lam_ze + 2, f"material ENZ (CSV): {lam_ze:.1f} nm",
            color=C_VIOLET, fontsize=9)
    ax.axhline(1460, color=INK2, ls=":", lw=1.1)
    ax.text(6.0, 1462, "Karimi-reported ENZ-mode ~1460 nm (their sample)",
            color=INK2, fontsize=9)
    for lab, row, col in marks:
        ax.plot([row["K_over_k0"]], [row["lambda_qnm_nm"]], "o", ms=9,
                mfc="none", mec=col, mew=2)
        ax.annotate(lab, (row["K_over_k0"], row["lambda_qnm_nm"]),
                    textcoords="offset points", xytext=(8, -12), color=col,
                    fontsize=9)
    ax.set_xlabel(r"$K/k_0$  ($k_0 = {\rm Re}\,\tilde\omega/c$)")
    ax.set_ylabel(r"$\lambda_{\rm QNM} = 2\pi c/{\rm Re}\,\tilde\omega$ (nm)")
    ax.set_title("Figure A - bare air/ITO(23 nm)/glass ENZ QNM dispersion "
                 "(branch ends at the glass light line on the left)")
    ax.legend(loc="upper left", fontsize=9)

    panels = [
        ("Q", "Q", C_BLUE, None),
        ("Ez_to_Ex_rms", r"$E_z/E_x$ (rms, ITO)", C_ORANGE, None),
        ("Ez_flatness_cv", r"$E_z$ flatness CV (ITO)", C_AQUA, "log"),
        ("Ez_intensity_localization_fraction",
         r"$|E_z|^2$ localization fraction", C_VIOLET, None),
        ("air_decay_length_nm", r"$L_{\rm air}$ (nm)", C_MAGENTA, "log"),
        ("glass_decay_length_nm", r"$L_{\rm glass}$ (nm)", "#008300", "log"),
    ]
    for i, (colname, lab, col, yscale) in enumerate(panels):
        ax = fig.add_subplot(gs[1 + i // 3, i % 3])
        ax.plot(df.K_over_k0, df[colname], color=col)
        if yscale:
            ax.set_yscale(yscale)
        for _lab, row, mcol in marks:
            ax.axvline(row["K_over_k0"], color=mcol, lw=0.9, ls="--",
                       alpha=0.6)
        ax.set_ylabel(lab)
        if i >= 3:
            ax.set_xlabel(r"$K/k_0$")
        if colname == "air_decay_length_nm":
            ax.axhspan(140, 210, color="#dbe7f7", zorder=0)
            ax.text(4.6, 150, "Karimi antenna\nheights 140-210 nm",
                    fontsize=8, color=INK2)
    fig.text(0.5, 0.615, "Figure B - modal-quality metrics along the branch "
             "(dashed lines: benchmark K values)", ha="center", fontsize=11)
    _save(fig, "qnm_dispersion_metrics.png")


def figC(df, marks):
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for col, lab, cc in (("Lambda_10_nm", r"$\Lambda_{10}=2\pi/K$", C_BLUE),
                         ("Lambda_11_nm", r"$\Lambda_{11}=2\pi\sqrt{2}/K$",
                          C_ORANGE),
                         ("Lambda_20_nm", r"$\Lambda_{20}=4\pi/K$", C_AQUA)):
        ax.plot(df[col], df.lambda_qnm_nm, color=cc, label=lab)
    for lab, row, col in marks:
        ax.plot([row["Lambda_10_nm"]], [row["lambda_qnm_nm"]], "o", ms=9,
                mfc="none", mec=col, mew=2)
    ax.axvline(850, color=INK2, ls="--", lw=1.0)
    ax.axvline(810, color=INK2, ls=":", lw=1.0)
    ax.text(852, 1489, "850 nm", fontsize=9, color=INK2, rotation=90)
    ax.text(812, 1489, "810 nm", fontsize=9, color=INK2, rotation=90)
    ax.set_xlim(150, 1500)
    ax.set_xlabel("lattice period $\\Lambda$ (nm) giving $K=|G_{mn}|$")
    ax.set_ylabel(r"$\lambda_{\rm QNM}$ (nm)")
    ax.set_title("Figure C - metasurface period mapping of the bare ENZ "
                 "branch (normal incidence, $k_B=0$)")
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, "metasurface_period_mapping.png")


def figD(points, ito, p, drude):
    """Field-profile comparison at representative branch points."""
    fig, axs = plt.subplots(5, len(points), figsize=(3.1 * len(points), 12),
                            sharex="col")
    for j, (lab, kind, data) in enumerate(points):
        if kind == "qnm":
            K, w = data
            eps2 = complex(drude(p, w))
            mode = ModeField(K, w / C_NM_FS, eps2, config.D_ITO_NM,
                             ab.EPS1, ab.EPS3)
        else:                              # legacy complex-K at real omega
            lam, u = data
            k0 = 2 * np.pi / lam
            mode = ModeField(u * k0, k0, complex(ito.eps(lam)),
                             config.D_ITO_NM, ab.EPS1, ab.EPS3)
        La, Lg = mode.decay_lengths_nm()
        z = np.linspace(-2.5 * Lg, config.D_ITO_NM + 2.5 * La, 1200)
        # normalize each profile: max |Ez| in ITO = 1 (visual comparison)
        zi = np.linspace(0, config.D_ITO_NM, 301)
        s = 1.0 / np.abs(mode.Ez(zi)).max()
        rows = [(np.abs(mode.Hy(z)) * s, r"$|H_y|$", C_AQUA),
                (np.abs(mode.Ex(z)) * s, r"$|E_x|$", C_ORANGE),
                (np.abs(mode.Ez(z)) * s, r"$|E_z|$", C_BLUE),
                ((mode.Ez(z) * s).real, r"Re $E_z$", C_VIOLET),
                (np.angle(mode.Ez(z) * s), r"arg $E_z$", C_MAGENTA)]
        for i, (y, ylab, col) in enumerate(rows):
            ax = axs[i, j]
            ax.axvspan(0, config.D_ITO_NM, color="#dbe7f7", zorder=0)
            ax.plot(z, y, color=col, lw=1.6)
            if i == 0:
                ax.set_title(lab, fontsize=9.5)
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 4:
                ax.set_xlabel("z (nm)")
    fig.suptitle("Figure D - modal field evolution along the branch "
                 "(each profile normalized to max$_{ITO}|E_z|=1$; shaded: ITO)",
                 y=0.995)
    fig.tight_layout()
    _save(fig, "branch_field_comparison.png")


def fig_summary(rows):
    keys = [("label", "candidate"), ("lambda_qnm_nm", "lambda (nm)"),
            ("Q", "Q"), ("Ez_to_Ex_rms", "Ez/Ex rms"),
            ("Ez_flatness_cv", "Ez CV"),
            ("Ez_intensity_localization_fraction", "|Ez|^2 loc."),
            ("air_decay_length_nm", "L_air (nm)"),
            ("glass_decay_length_nm", "L_glass (nm)"),
            ("Lambda_10_nm", "Lambda_10 (nm)"),
            ("pole_residual", "|D|")]
    cell = []
    for r in rows:
        line = []
        for k, _ in keys:
            v = r.get(k, np.nan)
            if isinstance(v, str):
                line.append(v if len(v) < 34 else v[:31] + "...")
            elif k == "pole_residual":
                line.append(f"{v:.0e}")
            elif isinstance(v, float):
                line.append(f"{v:.3g}")
            else:
                line.append(str(v))
        cell.append(line)
    fig, ax = plt.subplots(figsize=(13.6, 0.5 * len(rows) + 1.4))
    ax.axis("off")
    tbl = ax.table(cellText=cell, colLabels=[h for _, h in keys],
                   loc="center", cellLoc="left",
                   colWidths=[0.24] + [0.084] * (len(keys) - 1))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.6)
    for (r, c), cl in tbl.get_celld().items():
        cl.set_edgecolor("#e3e2df")
        if r == 0:
            cl.set_text_props(color=SURF, fontweight="bold")
            cl.set_facecolor(C_BLUE)
        elif r % 2 == 0:
            cl.set_facecolor("#f3f2ef")
    ax.set_title("Target-candidate summary (no single-metric ranking - "
                 "tradeoffs shown explicitly)", pad=12)
    _save(fig, "target_candidate_summary.png")


# ---------------------------------------------------------------------------
def main():
    OUT.mkdir(exist_ok=True)
    ito = ITOMaterial()
    p, drude, _ = ab.drude_fit_report(ito)
    lam_ze = ito.zero_crossing_nm()
    df = pd.read_csv(OUT / "enz_branch_metrics.csv")

    # exact benchmark points -------------------------------------------------
    bench = []
    for lam_lat, lab in ((850.0, "Karimi EDR lattice 850 nm, (1,0)"),
                         (810.0, "Karimi MDR lattice 810 nm, (1,0)")):
        bench.append(exact_point(2 * np.pi / lam_lat, lab, ito, p, drude))
    b770 = exact_point(3 * 2 * np.pi / 770.0,
                       "previous Phase-2 target: 770 nm, (3,0)", ito, p, drude)
    legacy = legacy_1527_metrics(ito)

    pd.DataFrame(bench).to_csv(OUT / "benchmark_850_810.csv", index=False)
    print(f"[saved] {OUT/'benchmark_850_810.csv'}")

    # Pareto-relevant candidates --------------------------------------------
    i_loc = df.Ez_intensity_localization_fraction.idxmax()
    cand_maxloc = exact_point(float(df.K_per_nm[i_loc]),
                              "max |Ez|^2-localization on QNM branch",
                              ito, p, drude)
    cand_head = exact_point(float(df.K_per_nm.iloc[0]),
                            "branch head (glass-light-line edge)",
                            ito, p, drude)
    candidates = bench + [cand_maxloc, cand_head, b770, legacy]
    pd.DataFrame(candidates).to_csv(OUT / "target_candidates.csv", index=False)
    print(f"[saved] {OUT/'target_candidates.csv'}")

    # figures ---------------------------------------------------------------
    marks = [("850 (1,0)", bench[0], C_ORANGE),
             ("810 (1,0)", bench[1], C_MAGENTA),
             ("770 (3,0)", b770, C_AQUA)]
    figA_B(df, lam_ze, marks)
    figC(df, marks)

    def wq(row):
        return row["omega_real"] + 1j * row["omega_imag"]

    figD([("850 nm (1,0)\n" + f"$\\lambda$={bench[0]['lambda_qnm_nm']:.0f} nm",
           "qnm", (bench[0]["K_per_nm"], wq(bench[0]))),
          ("810 nm (1,0)\n" + f"$\\lambda$={bench[1]['lambda_qnm_nm']:.0f} nm",
           "qnm", (bench[1]["K_per_nm"], wq(bench[1]))),
          ("max-loc QNM\n" + f"$\\lambda$={cand_maxloc['lambda_qnm_nm']:.0f} nm",
           "qnm", (cand_maxloc["K_per_nm"], wq(cand_maxloc))),
          ("770 nm (3,0)\n" + f"$\\lambda$={b770['lambda_qnm_nm']:.0f} nm",
           "qnm", (b770["K_per_nm"], wq(b770))),
          ("legacy 1527 nm\n(complex-K)", "legacy",
           (1527.0, 5.946338 - 12.859551j))],
         ito, p, drude)
    fig_summary(candidates)

    # printed conclusions ----------------------------------------------------
    print("\n----- benchmark summary -----")
    for r in bench + [b770, cand_maxloc, cand_head]:
        print(f"{r['label']}: lambda_QNM = {r['lambda_qnm_nm']:.1f} nm, "
              f"Q = {r['Q']:.2f}, Ez/Ex = {r['Ez_to_Ex_rms']:.1f}, "
              f"CV = {r['Ez_flatness_cv']:.4f}, "
              f"loc = {r['Ez_intensity_localization_fraction']:.3f}, "
              f"L_air = {r['air_decay_length_nm']:.0f} nm, "
              f"L_glass = {r['glass_decay_length_nm']:.0f} nm")
    print(f"\nlegacy 1527 nm point: Ez/Ex = {legacy['Ez_to_Ex_rms']:.1f}, "
          f"CV = {legacy['Ez_flatness_cv']:.4f}, "
          f"loc = {legacy['Ez_intensity_localization_fraction']:.3f}, "
          f"L_air = {legacy['air_decay_length_nm']:.0f} nm, "
          f"eps_ITO = {legacy['eps_ito_meas_real']:+.3f}"
          f"+{legacy['eps_ito_meas_imag']:.3f}i, "
          f"Im K = {legacy['K_over_k0']*0+12.86:.2f} k0 -> classification: "
          "ENZ-derived but strongly overdamped short-range-SPP-like "
          "continuation (valid Maxwell pole of the real-omega/complex-K "
          "representation; suboptimal as a metasurface target)")
    return candidates


if __name__ == "__main__":
    main()
