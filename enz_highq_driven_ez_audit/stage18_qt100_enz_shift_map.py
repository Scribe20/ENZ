"""Stage 18: transfer-function-style map T(delta, lambda) for P925_Qt100
(frozen geometry, with ITO), lambda = 1410-1450 nm.

x axis = fractional plasma-frequency reduction of the ITO,
    delta = 1 - omega_p^2 / omega_p0^2,
the effect of intraband hot-electron excitation on a Drude ENZ film
(effective-mass increase lowers omega_p and red-shifts the ENZ crossing).
The ITO permittivity is the Phase-1 Drude fit of the measured CSV
(eps_inf = 3.89498, omega_p0 = 2.658573 rad/fs, gamma = 0.232230 rad/fs,
max fit error 1.9e-3); delta = 0 is cross-checked against the CSV-based
spectrum of Stage 17.  gamma is held fixed (assumption, stated).

NOT an incident-energy axis: no validated nonlinear ITO model (Delta omega_p
per nJ, eps(T_e)) exists in the repository, and none is invented here.
"""
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as cm
import forward_multi as fm

torch.set_num_threads(4)
OUT = cm.OUT / "stage18_qt100_enz_shift_map"
OUT.mkdir(parents=True, exist_ok=True)
GEO_FILE = cm.OUT / "stage16/P925_Qt100/rho_hard_binary.npy"
P, H = 925.0, 240.0
EPS_INF, WP0, GAMMA = 3.89498, 2.658573, 0.232230      # rad/fs (Phase-1 Drude fit)
C = cm.C_NM_FS
rho = torch.as_tensor(np.load(GEO_FILE), dtype=cm.GEO)


def eps_drude(lam, delta):
    w = 2 * np.pi * C / lam
    wp2 = WP0 ** 2 * (1 - delta)
    return complex(EPS_INF - wp2 / (w ** 2 + 1j * GAMMA * w))


def enz_crossing(delta):
    wp2 = WP0 ** 2 * (1 - delta)
    # Re eps = 0: eps_inf - wp2 (w^2)/(w^4 + g^2 w^2) = 0 -> w^2 = wp2/eps_inf - g^2
    w2 = wp2 / EPS_INF - GAMMA ** 2
    return 2 * np.pi * C / np.sqrt(w2)


lams = np.arange(1410.0, 1450.001, 0.5)
deltas = np.linspace(0.0, 0.25, 21)
T = np.zeros((len(deltas), len(lams))); R = np.zeros_like(T); A = np.zeros_like(T)
log = open(OUT / "stage18.log", "a")
for i, d in enumerate(deltas):
    for j, lam in enumerate(lams):
        with torch.no_grad():
            sim = fm.build_sim(rho, P, H, lam=float(lam), order=cm.ORDER, eps_ito_val=eps_drude(float(lam), d))
            Rr, Tt = fm.rt_all_orders(sim)
        R[i, j], T[i, j] = float(Rr), float(Tt); A[i, j] = 1 - R[i, j] - T[i, j]
    msg = f"delta={d:.4f} ENZ crossing={enz_crossing(d):.1f} nm: T(1433.5)={T[i, np.argmin(abs(lams-1433.5))]:.3f} A={A[i, np.argmin(abs(lams-1433.5))]:.3f}"
    print(msg, flush=True); log.write(msg + "\n"); log.flush()
    np.savez(OUT / "map.npz", lams=lams, deltas=deltas, T=T, R=R, A=A, enz_crossing=[enz_crossing(x) for x in deltas])

# delta = 0 cross-check against the CSV-based Stage-17 spectrum
s17 = pd.read_csv(cm.OUT / "stage17_qt100_ito_control/spectrum_with_ITO.csv")
s17 = s17[(s17["lam"] >= 1410) & (s17["lam"] <= 1450)]
T17 = np.interp(lams, s17["lam"].values, s17["T"].values)
xchk = float(np.max(np.abs(T[0] - T17)))
log.write(f"delta=0 vs CSV-based Stage-17 spectrum: max|dT| = {xchk:.4f}\n"); print(f"[check] max|dT| Drude vs CSV = {xchk:.4f}")
pd.DataFrame(T, index=[f"delta={d:.4f}" for d in deltas], columns=[f"{l:.1f}" for l in lams]).to_csv(OUT / "T_map.csv")
pd.DataFrame(A, index=[f"delta={d:.4f}" for d in deltas], columns=[f"{l:.1f}" for l in lams]).to_csv(OUT / "A_map.csv")
pd.DataFrame(R, index=[f"delta={d:.4f}" for d in deltas], columns=[f"{l:.1f}" for l in lams]).to_csv(OUT / "R_map.csv")

enz = np.array([enz_crossing(x) for x in deltas])
for key, M, cmap in (("T", T, "viridis"), ("A", A, "magma"), ("R", R, "cividis")):
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    im = ax.imshow(M.T, origin="lower", aspect="auto", cmap=cmap, extent=[deltas[0], deltas[-1], lams[0], lams[-1]], vmin=0, vmax=max(0.9, M.max()))
    ax.axhline(cm.LAMBDA_E, ls="--", c="w", lw=1); ax.text(0.005, cm.LAMBDA_E + 0.6, "lambda_E = 1433.5 nm", color="w", fontsize=8)
    ax.axhline(1433.985, ls=":", c="tab:red", lw=1); ax.text(0.15, 1434.6, "linear-regime pole 1434.0 nm", color="tab:red", fontsize=8)
    ax.set_xlabel(r"ITO plasma-frequency reduction  $\delta = 1-\omega_p^2/\omega_{p0}^2$  (NOT pulse energy)")
    ax.set_ylabel(r"$\lambda_p$ (nm)")
    sec = ax.secondary_xaxis("top", functions=(lambda d: np.interp(d, deltas, enz), lambda l: np.interp(l, enz, deltas)))
    sec.set_xlabel("ENZ crossing wavelength of the ITO (nm)")
    ax.set_title(f"({key}) P925_Qt100 with ITO: {key}(delta, lambda), normal incidence, lab-x, order [7,7]")
    fig.colorbar(im, ax=ax, label=key)
    fig.savefig(OUT / f"P925_Qt100_{key}_vs_ENZshift_map.png", dpi=160, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
for lp in (1415.0, 1425.0, 1433.5, 1440.0, 1447.0):
    j = int(np.argmin(np.abs(lams - lp)))
    ax.plot(deltas, T[:, j], marker="o", ms=3, label=f"lambda_p = {lams[j]:.1f} nm")
ax.set_xlabel(r"$\delta = 1-\omega_p^2/\omega_{p0}^2$"); ax.set_ylabel("T"); ax.grid(alpha=.3); ax.legend(fontsize=8)
ax.set_title("T vs ENZ-shift parameter at fixed probe wavelengths (horizontal cuts)")
fig.savefig(OUT / "P925_Qt100_T_cuts.png", dpi=160, bbox_inches="tight"); plt.close(fig)
print("[stage18] done"); log.write("[stage18] done\n"); log.close()
