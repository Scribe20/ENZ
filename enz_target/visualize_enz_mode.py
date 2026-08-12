"""Figures 1-5 for the ENZ target-mode analysis.

Palette / style: light surface, recessive grid, series colors
blue #2a78d6, orange #eb6834, aqua #1baf7a, violet #4a3aa7, magenta #e87ba4.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from tm_slab_mode import ModeField
from solve_enz_dispersion import k0_of, EPS1, EPS3

C_BLUE, C_ORANGE, C_AQUA, C_VIOLET, C_MAGENTA = \
    "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": "#e3e2df", "grid.linewidth": 0.6, "font.size": 10.5,
    "axes.titlesize": 11.5, "lines.linewidth": 2.0, "legend.frameon": False,
})


def _save(fig, name):
    path = config.FIG_DIR / name
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"figure written: {path}")


# ---------------------------------------------------------------- Figure 1
def fig1_epsilon(ito, lam_ze):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(ito.wl, ito.eps_re, color=C_BLUE, label=r"Re $\varepsilon_{\rm ITO}$")
    ax.plot(ito.wl, ito.eps_im, color=C_ORANGE, label=r"Im $\varepsilon_{\rm ITO}$")
    ax.axhline(0.0, color=INK2, lw=1.0, ls="-")
    ax.axhline(-1.0, color=INK2, lw=0.8, ls=":")
    ax.text(1208, -0.93, r"Re $\varepsilon=-1$", color=INK2, fontsize=9)
    ax.axvline(lam_ze, color=C_VIOLET, lw=1.2, ls="--")
    ax.annotate(rf"material ENZ: Re $\varepsilon=0$"
                f"\n$\\lambda_{{ZE}}$ = {lam_ze:.1f} nm",
                xy=(lam_ze, 0.0), xytext=(1435, -0.55),
                arrowprops=dict(arrowstyle="->", color=C_VIOLET), color=C_VIOLET)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Relative permittivity")
    ax.set_title("ITO permittivity (recommended-physical columns of supplied CSV)")
    ax.legend(loc="upper right")
    _save(fig, "ito_epsilon.png")


# ---------------------------------------------------------------- Figure 2
def fig2_dispersion(enz, berreman, cplx_omega, lam_ze, target):
    """enz/berreman: dict with wl, u.  cplx_omega: dict with u_K, lam_re.
    target: (wl_E, u_E)."""
    wl_E, u_E = target
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.4))

    ax = axs[0]
    ax.plot(enz["u"].real, enz["wl"], color=C_BLUE, label="confined ENZ branch")
    ax.plot(berreman["u"].real, berreman["wl"], color=C_ORANGE,
            label="Berreman leaky branch")
    ax.axvline(1.0, color=INK2, lw=1.0, ls="--")
    ax.axvline(config.N_GLASS, color=INK2, lw=1.0, ls=":")
    ax.text(1.02, 1688, "air light line", rotation=90, fontsize=8.5,
            color=INK2, va="top")
    ax.text(config.N_GLASS + 0.02, 1688, "glass light line", rotation=90,
            fontsize=8.5, color=INK2, va="top")
    ax.axhline(lam_ze, color=C_VIOLET, lw=1.0, ls="--")
    ax.text(6.2, lam_ze - 8, r"Re $\varepsilon_{\rm ITO}=0$", color=C_VIOLET,
            fontsize=9)
    ax.plot([u_E.real], [wl_E], "o", ms=9, mfc="none", mec=C_MAGENTA, mew=2,
            label="target point")
    ax.set_xlabel(r"Re$(K)/k_0$")
    ax.set_ylabel("Wavelength (nm)")
    ax.set_xlim(0, 13)
    ax.set_title("real-$\\omega$, complex-$K$ branches")
    ax.legend(loc="lower right", fontsize=9)

    ax = axs[1]
    ax.plot(enz["u"].imag, enz["wl"], color=C_BLUE)
    ax.plot(berreman["u"].imag, berreman["wl"], color=C_ORANGE)
    ax.plot([u_E.imag], [wl_E], "o", ms=9, mfc="none", mec=C_MAGENTA, mew=2)
    ax.axhline(lam_ze, color=C_VIOLET, lw=1.0, ls="--")
    ax.axvline(0.0, color=INK2, lw=0.8)
    ax.set_xlabel(r"Im$(K)/k_0$")
    ax.set_title("in-plane damping\n(Im $K<0$ with Re $K>0$: overdamped/backward)")

    ax = axs[2]
    ax.plot(cplx_omega["u_K"], cplx_omega["lam_re"], color=C_AQUA,
            label=r"complex-$\omega$ pole (Drude fit)")
    ax.axhline(lam_ze, color=C_VIOLET, lw=1.0, ls="--")
    ax.axvline(1.0, color=INK2, lw=1.0, ls="--")
    ax.axvline(config.N_GLASS, color=INK2, lw=1.0, ls=":")
    ax.set_xlabel(r"$K/k_0(\lambda)$ (real)")
    ax.set_ylabel(r"Re $\lambda_{\rm pole}$ (nm)")
    ax.set_title("cross-check: flat ENZ-mode branch\n(real $K$, complex $\\omega$)")
    ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(f"TM mode dispersion, air / {config.D_ITO_NM:.0f} nm ITO / glass "
                 f"(n = {config.N_GLASS})", y=1.03)
    _save(fig, "enz_dispersion.png")


# ---------------------------------------------------------------- Figure 3
def fig3_field_1d(mode: ModeField, wl_E, norm_scale):
    d = mode.d
    La, Lg = mode.decay_lengths_nm()
    zmax = d + config.Z_PAD_FACTOR * La
    zmin = -config.Z_PAD_FACTOR * Lg
    z = np.linspace(zmin, zmax, 4001)
    Hy = mode.Hy(z) * norm_scale
    Ex = mode.Ex(z) * norm_scale
    Ez = mode.Ez(z) * norm_scale

    fig, axs = plt.subplots(2, 3, figsize=(12.6, 7.0), sharex=True)
    panels = [
        (np.abs(Hy), r"$|H_y(z)|$", C_AQUA, False),
        (np.abs(Ex), r"$|E_x(z)|$", C_ORANGE, False),
        (np.abs(Ez), r"$|E_z(z)|$", C_BLUE, False),
        (np.abs(Ez) ** 2, r"$|E_z(z)|^2$", C_BLUE, False),
        (Ez.real, r"Re $E_z(z)$", C_VIOLET, True),
        (np.angle(Ez), r"arg $E_z(z)$ (rad)", C_MAGENTA, True),
    ]
    for ax, (y, lab, col, signed) in zip(axs.ravel(), panels):
        ax.axvspan(0, d, color="#dbe7f7", zorder=0, label="ITO")
        ax.plot(z, y, color=col)
        ax.axvline(0, color=INK2, lw=0.9)
        ax.axvline(d, color=INK2, lw=0.9)
        if signed:
            ax.axhline(0, color=INK2, lw=0.6)
        ax.set_ylabel(lab)
    for ax in axs[1]:
        ax.set_xlabel("z (nm)")
    axs[0, 0].text(d + 3, np.abs(Hy).max() * 0.92, "air", color=INK2, fontsize=9)
    axs[0, 0].text(zmin + 3, np.abs(Hy).max() * 0.92, "glass", color=INK2, fontsize=9)
    fig.suptitle(
        rf"Target ENZ mode at $\lambda$ = {wl_E:.0f} nm  "
        rf"($K/k_0$ = {mode.K/mode.k0:.3f}; normalization "
        rf"$\int_{{\rm ITO}}|E_z|^2 dz = 1$; air/glass 1/e decay "
        rf"{La:.0f} / {Lg:.0f} nm)", y=1.0)
    _save(fig, "enz_field_1d.png")


# ---------------------------------------------------------------- Figure 4
def fig4_field_xz(mode: ModeField, wl_E, norm_scale):
    d = mode.d
    La, Lg = mode.decay_lengths_nm()
    # x-decaying representative of the (K, -K) pair: K_dec = -K
    K_dec = -mode.K
    lam_par = 2 * np.pi / abs(K_dec.real)
    L_x = 1.0 / abs(K_dec.imag)
    # The mode is overdamped in-plane (L_x << lam_par), so the x-window is set
    # by the decay scale; a full modal wavelength would span ~exp(13) in
    # amplitude and render the map unreadable.  Honest choice: 0..6 L_x.
    x = np.linspace(0.0, 6.0 * L_x, 481)
    z = np.linspace(-2.5 * Lg, d + 2.5 * La, 401)
    X, Z = np.meshgrid(x, z)
    Ez1d = mode.Ez(z) * norm_scale
    Ex1d = mode.Ex(z) * norm_scale
    prop = np.exp(1j * K_dec * X)
    EzXZ = Ez1d[:, None] * prop
    ExXZ = Ex1d[:, None] * prop

    fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.6), sharey=True)
    vmax = np.nanpercentile(np.abs(EzXZ.real), 99.5)
    im0 = axs[0].pcolormesh(X, Z, EzXZ.real, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                            shading="auto", rasterized=True)
    axs[0].set_title(r"Re $E_z(x,z)$   (x-decaying representative $-K$)")
    fig.colorbar(im0, ax=axs[0], label=r"Re $E_z$ (nm$^{-1/2}$)")
    I = np.abs(EzXZ) ** 2
    im1 = axs[1].pcolormesh(X, Z, I, cmap="Blues", vmin=0,
                            vmax=np.nanpercentile(I, 99.5), shading="auto",
                            rasterized=True)
    axs[1].set_title(r"$|E_z(x,z)|^2$")
    fig.colorbar(im1, ax=axs[1], label=r"$|E_z|^2$ (nm$^{-1}$)")

    # vector overlay on the intensity panel: arrows normalized per-position to
    # show the local (Ex, Ez) direction (amplitude already shown by the map)
    xs = x[::40]
    zs = z[::28]
    Xq, Zq = np.meshgrid(xs, zs)
    Eq = (Ex1d[::28, None] * np.exp(1j * K_dec * Xq))
    Ezq = (Ez1d[::28, None] * np.exp(1j * K_dec * Xq))
    mag = np.hypot(np.abs(Eq), np.abs(Ezq))
    mag[mag == 0] = 1.0
    axs[1].quiver(Xq, Zq, Eq.real / mag, Ezq.real / mag, color=INK,
                  scale=32, width=0.0035, alpha=0.6)

    for ax in axs:
        ax.axhline(0, color=INK, lw=1.0)
        ax.axhline(d, color=INK, lw=1.0)
        ax.set_xlabel("x (nm)")
        ax.grid(False)
    axs[0].set_ylabel("z (nm)")
    axs[0].text(x[6], d + 6, "air", fontsize=9)
    axs[0].text(x[6], -14, "glass", fontsize=9)
    fig.suptitle(rf"Target ENZ mode field, $\lambda$ = {wl_E:.0f} nm; "
                 rf"$\lambda_\parallel = 2\pi/|{{\rm Re}}K| \approx$ {lam_par:.0f} nm, "
                 rf"in-plane 1/e decay $1/|{{\rm Im}}K| \approx$ {L_x:.0f} nm "
                 "(overdamped: x-window spans the decay scale)", y=1.04)
    _save(fig, "enz_field_xz.png")


# ---------------------------------------------------------------- Figure 5
def fig5_diagnostics(rows):
    """rows: list of (label, value) strings."""
    fig, ax = plt.subplots(figsize=(9.6, 0.34 * len(rows) + 0.9))
    ax.axis("off")
    tbl = ax.table(cellText=[[a, b] for a, b in rows],
                   colLabels=["quantity", "value"],
                   colWidths=[0.55, 0.45], loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#e3e2df")
        if r == 0:
            cell.set_text_props(color=SURF, fontweight="bold")
            cell.set_facecolor(C_BLUE)
        elif r % 2 == 0:
            cell.set_facecolor("#f3f2ef")
    ax.set_title("Target ENZ mode: diagnostics summary", pad=14)
    _save(fig, "enz_diagnostics.png")
