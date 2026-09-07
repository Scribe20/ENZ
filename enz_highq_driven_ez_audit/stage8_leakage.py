"""Stage 8: controlled leakage tuning of the padded QNM parent.
Perturbation families (amplitude alpha), all clipped to the original 13-px
air ring so the positive padding is preserved; connectivity and minimum
feature are reported:
    shear   : x-displacement of the boundary proportional to (y - yc)      [px]
    local   : localized edge displacement (Gaussian in y) on the +x side   [px]
    notch   : rectangular air notch of width alpha, depth 3 alpha at the +x edge [px]
    scale   : global isotropic scale (1 + alpha) via the signed distance    [-]
    height  : h = 140 (1 + alpha)                                            [-]
    period  : P = 850 (1 + alpha)                                            [-]
For each geometry: driven F_Ez, A, eta_z at lambda_E; the parent branch is
tracked with s in {1, 0.25, 0.03, 0} -> Q_rad, Q_nr, ratio, lambda_pole,
eta_ENZ,z (lossless auxiliary).
"""
import numpy as np
import torch
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as cm
import poles as pl

torch.set_num_threads(4)
NAME = "padded QNM"
rho0, P0, h0 = cm.load_candidate(NAME)
B0 = rho0.numpy() > 0.5
M = np.load(cm.ROOT / "enz_padding_sideexperiment/outputs/geometries/design_mask.npy") > 0.5
br = cm.jload(cm.OUT / "stage2_branches.json")[NAME]
parent = min(br, key=lambda b: abs(b["rows"][0]["lambda_pole"] - cm.LAMBDA_E))
omega0 = complex(parent["rows"][0]["omega_re"], parent["rows"][0]["omega_im"])
S_LEV = (1.0, 0.25, 0.03, 0.0)


def sdf(b):
    return ndimage.distance_transform_edt(b) - ndimage.distance_transform_edt(~b)


def deform(b, dx_field, dy_field):
    """Warp the SDF by a displacement field (px) and rethreshold."""
    phi = sdf(b)
    ii, jj = np.meshgrid(np.arange(128), np.arange(128), indexing="ij")
    warped = ndimage.map_coordinates(phi, [ii - dx_field, jj - dy_field], order=1, mode="nearest")
    return (warped > 0) & M


ii, jj = np.meshgrid(np.arange(128), np.arange(128), indexing="ij")
yc = xc = 63.5


def fam_shear(a):
    return deform(B0, a * (jj - yc) / 64.0, np.zeros_like(ii, float)), P0, h0


def fam_local(a):
    w = np.exp(-((jj - yc) ** 2) / (2 * 12.0 ** 2)) * (ii > xc)
    return deform(B0, a * w, np.zeros_like(ii, float)), P0, h0


def fam_notch(a):
    b = B0.copy()
    cols = np.where(B0.any(0))[0]; xr = cols.max()
    d = int(round(3 * a)); wd = max(int(round(a)), 1)
    b[xr - d:xr + 1, int(yc) - wd // 2: int(yc) + wd // 2 + 1] = False
    return b & M, P0, h0


def fam_scale(a):
    phi = sdf(B0)
    warped = ndimage.zoom(phi, 1 + a, order=1)
    n = warped.shape[0]; o = (n - 128) // 2
    if n >= 128:
        w = warped[o:o + 128, o:o + 128]
    else:
        w = np.full((128, 128), -50.0); w[-o:-o + n, -o:-o + n] = warped
    return (w > 0) & M, P0, h0


def fam_height(a):
    return B0, P0, h0 * (1 + a)


def fam_period(a):
    return B0, P0 * (1 + a), h0


FAMILIES = {"shear": (fam_shear, [0.5, 1.0, 2.0, 4.0]),
            "local": (fam_local, [1.0, 2.0, 4.0, 6.0]),
            "notch": (fam_notch, [1.0, 2.0, 3.0, 4.0]),
            "scale": (fam_scale, [-0.04, -0.02, 0.02, 0.04]),
            "height": (fam_height, [-0.06, -0.03, 0.03, 0.06]),
            "period": (fam_period, [-0.04, -0.02, 0.02, 0.04])}


def geometry_checks(b):
    lab, n = ndimage.label(b)
    dx = P0 / 128
    def min_scale(mask):
        for r in range(1, 12):
            st = np.ones((2 * r + 1, 2 * r + 1), bool)
            if ndimage.binary_opening(mask, st).sum() < 0.98 * mask.sum():
                return (2 * r - 1) * dx
        return 23 * dx
    return dict(n_components=int(n), padding_ok=bool(not (b & ~M).any()),
                min_feature_nm=min_scale(b), fill=float(b.mean()))


def evaluate(b, P, h, tag):
    rho = torch.as_tensor(b.astype(float), dtype=cm.GEO)
    m = cm.driven_metrics(rho, P, h, n_xy=64)
    rows = pl.track_branch(rho, P, h, omega0, s_levels=S_LEV, refine_high_q=True, tag=tag)
    fit = pl.fit_gamma(rows)
    en = (cm.energy_participation(rho, P, h, fit["lossless_pole"]["lambda_nm"], 0.0)
          if fit.get("lossless_pole") else None)
    ovl0 = pl.overlap(pl.ez_map(rho0, P0, h0, parent["rows"][0]["lambda_pole"], 1.0),
                      pl.ez_map(rho, P, h, rows[0]["lambda_pole"], 1.0)) if rows else None
    rec = dict(tag=tag, P=P, h=h, F_Ez=m["F_Ez"], A=m["A_rt"], eta_z=m["eta_z"],
               lambda_pole=rows[0]["lambda_pole"] if rows else None,
               Q_loaded=rows[0]["Q"] if rows else None, overlap_with_parent=ovl0,
               Q_rad=fit.get("Q_rad"), Q_nr=fit.get("Q_nr"), gamma_ratio=fit.get("gamma_ratio"),
               linearity_resid=fit.get("linearity_resid"),
               eta_ENZ_z_modal=(en["eta_ENZ_z"] if en else None), eta_z_modal=(en["eta_z"] if en else None),
               checks=geometry_checks(b), rows=rows)
    print(f"  {tag:14s} F_Ez={m['F_Ez']:.3f} A={m['A_rt']:.3f} pole={rec['lambda_pole']} Q_l={rec['Q_loaded']} "
          f"Q_rad={rec['Q_rad']} ratio={rec['gamma_ratio']} etaENZ={rec['eta_ENZ_z_modal']} ovl={ovl0} {rec['checks']}", flush=True)
    return rec


res = {"parent": evaluate(B0, P0, h0, "parent a=0")}
for fam, (fn, alphas) in FAMILIES.items():
    res[fam] = []
    for a in alphas:
        b, P, h = fn(a)
        res[fam].append(dict(alpha=a, **evaluate(b, P, h, f"{fam} a={a:g}")))
    cm.jdump(res, cm.OUT / "stage8_leakage.json")
np.savez(cm.OUT / "stage8_geometries.npz",
         **{f"{fam}_{r['alpha']:g}".replace("-", "m").replace(".", "p"): FAMILIES[fam][0](r["alpha"])[0]
            for fam, rr in res.items() if fam != "parent" for r in rr})
fig, axs = plt.subplots(1, 4, figsize=(20, 4.5))
for fam, rr in res.items():
    if fam == "parent":
        continue
    a = [r["alpha"] for r in rr]
    axs[0].plot(a, [r["Q_rad"] or np.nan for r in rr], marker="o", label=fam)
    axs[1].plot(a, [r["F_Ez"] for r in rr], marker="o", label=fam)
    axs[2].plot(a, [r["A"] for r in rr], marker="o", label=fam)
    axs[3].plot(a, [r["eta_ENZ_z_modal"] or np.nan for r in rr], marker="o", label=fam)
p = res["parent"]
for ax, v, yl in zip(axs, (p["Q_rad"], p["F_Ez"], p["A"], p["eta_ENZ_z_modal"]),
                     ("Q_rad", "F_Ez(lambda_E) driven", "A_ITO(lambda_E)", "eta_ENZ,z modal (lossless)")):
    ax.axhline(v or np.nan, c="k", ls="--", lw=.8, label="parent"); ax.set_xlabel("alpha (px or fraction)"); ax.set_ylabel(yl); ax.grid(alpha=.3)
axs[0].set_yscale("log"); axs[0].legend(fontsize=7)
fig.suptitle("Fig. 7 - controlled leakage tuning of the padded QNM parent")
fig.savefig(cm.FIG / "fig7_leakage_tuning.png", dpi=150, bbox_inches="tight")
print("[stage8] done")
