"""Freeze the FDTDX validation candidates: copy the exact hard-binary masks selected by the TORCWA campaign,
record full provenance (source path, SHA256 of the file, of the uint8 array and of the float64 array used by
Stage B, P, h, d_ITO, pad, fill, materials, lambda_op, polarization, incidence) into manifest.json, and draw
every geometry from the authoritative binary (correct x/y orientation, cell boundary, pad ring, equal aspect).

    python register_finalists.py --tags <tag> [<tag> ...] --roles "<role>" ["<role>" ...]
Roles are free text (e.g. "primary: strongest positive dT", "control: highest eta_Dz parent").
"""
import argparse, hashlib, json, shutil, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as cm                       # noqa: E402
import candidates as cd                   # noqa: E402
import config                             # noqa: E402

MANIFEST = HERE / "manifest.json"


def geometry_png(rho, P, h, pad_frac, tag, role, path, ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(rho.T, origin="lower", extent=[0, P, 0, P], cmap="gray_r", interpolation="nearest", vmin=0, vmax=1)
    ax.plot([0, P, P, 0, 0], [0, 0, P, P, 0], "k-", lw=1.0)
    if pad_frac > 0:
        p = pad_frac * P
        ax.plot([p, P - p, P - p, p, p], [p, p, P - p, P - p, p], "r--", lw=0.8, label=f"hard-air pad {pad_frac*100:.0f} % of P")
        ax.legend(fontsize=7, loc="upper right")
    ax.set_aspect("equal"); ax.set_xlabel("x [nm]"); ax.set_ylabel("y [nm]")
    ax.set_title(f"{tag}\n{role[:70] + ('...' if len(role) > 70 else '')}\nP = {P:.0f} nm, h = {h:.1f} nm, fill {rho.mean():.3f}, {rho.shape[0]}x{rho.shape[1]} px of {P/rho.shape[0]:.2f} nm; black = a-Si:H", fontsize=8)
    if own:
        fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--roles", nargs="*", default=None)
    ap.add_argument("--extra-lams", type=float, nargs="*", default=[])
    a = ap.parse_args()
    roles = a.roles or ["" for _ in a.tags]
    man = json.load(open(MANIFEST)) if MANIFEST.exists() else dict(candidates={})
    man["provenance"] = cm.provenance()
    man["stack"] = "air / frozen freeform a-Si:H (h, rho) / ITO 23 nm / soda-lime glass; normal incidence from air, lab-frame x polarization, exp(-i w t)"
    for tag, role in zip(a.tags, roles):
        c = cd.by_tag(tag); rho = cd.get_rho(c)
        src = Path(c["rho_path"]) if c.get("rho_path") not in (None, "NONE") else None
        gdir = HERE / tag / "geometry"; gdir.mkdir(parents=True, exist_ok=True)
        r8 = (rho > 0.5).astype(np.uint8)
        np.save(gdir / "rho_hard_binary.npy", r8)
        B = cm.jload(cm.OUT / "candidates" / tag / "stageB.json") if (cm.OUT / "candidates" / tag / "stageB.json").exists() else {}
        pts = B.get("points", {})
        lam_op = pts.get("lambda_op_pos", {}).get("lam_nm"); lam_E = B.get("lam_E", 1302.282)
        pole = pts.get("lambda_op_pos", {}).get("loaded_pole_cold") or {}
        lams = [x for x in (lam_op, lam_E, pole.get("lambda_nm"), pts.get("lambda_op_neg", {}).get("lam_nm")) if x] + list(a.extra_lams)
        lams = sorted({round(float(x), 3) for x in lams})
        entry = dict(role=role, kind=c["kind"], source_rho=(str(src) if src else "generated: " + json.dumps(c.get("geom"))),
                     sha256_source_file=(hashlib.sha256(src.read_bytes()).hexdigest() if src else None),
                     sha256_uint8_array=hashlib.sha256(np.ascontiguousarray(r8).tobytes()).hexdigest(),
                     sha256_float64_array_stageB=cm.sha256_array(rho), sha256_copy_file=hashlib.sha256((gdir / "rho_hard_binary.npy").read_bytes()).hexdigest(),
                     shape=list(r8.shape), dtype="uint8", P_nm=float(c["P"]), h_nm=float(c["h"]), d_ito_nm=float(config.D_ITO_NM),
                     pad_frac=float(c.get("pad") or 0.0), pad_nm=float((c.get("pad") or 0.0) * c["P"]), pixel_nm=float(c["P"] / r8.shape[0]),
                     fill_fraction=float(r8.mean()), n_material_pixels=int(r8.sum()),
                     array_convention="rho[i, j]: i -> x (axis 0), j -> y (axis 1), cell-centred pixels of size P/128; 1 = a-Si:H, 0 = air (pad included)",
                     field_lams_nm=lams, lambda_op_pos_nm=lam_op, lambda_E_nm=lam_E, loaded_pole_cold=pole,
                     polarization="lab-frame x (E along x), normal incidence from air (theta = phi = 0), propagation -z (towards the glass)",
                     torcwa_stageB=dict(file=str(cm.OUT / "candidates" / tag / "stageB.json"), order=B.get("order"),
                                        at_lambda_op_pos={k: pts.get("lambda_op_pos", {}).get(k) for k in ("lam_nm", "T0", "R0", "A0", "Fz0", "Ftot0", "eta_z0", "mean_Ez2_0", "dT")},
                                        at_lambda_E={k: pts.get("lambda_E", {}).get(k) for k in ("lam_nm", "T0", "R0", "A0", "Fz0", "Ftot0", "eta_z0", "mean_Ez2_0", "dT")}),
                     torcwa_stageA=str(cm.OUT / "candidates" / tag / "stageA.json"), certification=c.get("cert_path"), Q_target=c.get("Q_target"),
                     cert_summary={k: (c.get("cert") or {}).get(k) for k in ("Q_r", "lambda_r", "eta_Dz_lamE", "eta_Dz_P_lamE")})
        man["candidates"][tag] = entry
        geometry_png(r8, c["P"], c["h"], entry["pad_frac"], tag, role, gdir / f"geometry_{tag}.png")
        print(f"registered {tag}: sha256(file)={entry['sha256_source_file'] and entry['sha256_source_file'][:16]} uint8={entry['sha256_uint8_array'][:16]} field lams {lams}")
    tags = list(man["candidates"])
    fig, axs = plt.subplots(1, len(tags), figsize=(5.0 * len(tags), 5.4))
    for ax, tag in zip(np.atleast_1d(axs), tags):
        e = man["candidates"][tag]
        geometry_png(np.load(HERE / tag / "geometry" / "rho_hard_binary.npy"), e["P_nm"], e["h_nm"], e["pad_frac"], tag, e["role"], None, ax=ax)
    fig.suptitle("FDTDX validation candidates: frozen hard-binary geometries (from the authoritative rho_hard_binary.npy)", fontsize=10)
    fig.tight_layout(); fig.savefig(HERE / "geometry_all_candidates.png", dpi=150); plt.close(fig)
    cm.jdump(man, MANIFEST)
    print(f"manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
