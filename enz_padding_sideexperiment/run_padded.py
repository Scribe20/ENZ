"""Side experiment: 85-nm hard lateral AIR PADDING around the freeform
a-Si meta-atom - does boundary isolation improve free-space -> ENZ-QNM
coupling?

Reuses the validated enz_inverse_design machinery (imported, not copied);
the ONLY change vs the authoritative unpadded 850-nm baseline (seed 333,
order [7,7], 128x128, 150 iters, bidir +-G10 QNM target at 1433.488 nm)
is a hard lateral design mask.  Vertical stack untouched (no spacer).

Mask realization on the 128-pixel grid (dx = 850/128 = 6.640625 nm):
pixel centers x_i = (i+0.5)dx are ACTIVE iff  p_air <= x_i <= P - p_air
with p_air = 85 nm -> active indices i in [13, 114] (102 px = 677.34 nm);
realized air ring: 13 px = 86.33 nm per side (documented, not silently
rounded).

Run:  python run_padded.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / "enz_inverse_design"
sys.path.insert(0, str(PKG))

import config                      # noqa: E402  (enz_inverse_design config)
import optimize_enz_overlap as opt  # noqa: E402

P_AIR_NM = 85.0


def build_mask(nx, ny, px_nm, py_nm, p_air_nm=P_AIR_NM):
    dx, dy = px_nm / nx, py_nm / ny
    xc = (np.arange(nx) + 0.5) * dx
    yc = (np.arange(ny) + 0.5) * dy
    mx = (xc >= p_air_nm) & (xc <= px_nm - p_air_nm)
    my = (yc >= p_air_nm) & (yc <= py_nm - p_air_nm)
    M = (mx[:, None] & my[None, :]).astype(float)
    # document the exact realized padding
    ix = np.where(mx)[0]
    realized_pad_lo = xc[ix[0]] - dx / 2
    realized_pad_hi = px_nm - (xc[ix[-1]] + dx / 2)
    info = {
        "requested_padding_nm": p_air_nm,
        "dx_nm": dx,
        "active_index_range": [int(ix[0]), int(ix[-1])],
        "active_pixels": int(len(ix)),
        "active_width_nm": float(len(ix) * dx),
        "realized_padding_nm_per_side": [float(realized_pad_lo),
                                         float(realized_pad_hi)],
        "pad_pixels_per_side": [int(ix[0]), int(nx - 1 - ix[-1])],
    }
    return M, info


def main():
    out = HERE / "outputs"
    for sub in ("histories", "geometries", "fields", "figures"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    M, info = build_mask(config.NX_DESIGN, config.NY_DESIGN,
                         config.PX_NM, config.PY_NM)
    np.save(out / "geometries" / "design_mask.npy", M)
    with open(out / "geometries" / "mask_info.json", "w") as f:
        json.dump(info, f, indent=1)
    print("[mask]", json.dumps(info))

    # identical schedule/seed/etc. - only the mask differs from the baseline
    F0, F1 = opt.main(design_mask=M, out_root=out)

    # hard-binary check: mask must hold exactly
    rho = np.load(out / "geometries" / "rho_proj_final.npy")
    leak = float(np.abs(rho * (1 - M)).max())
    print(f"[mask] max material density inside air ring after projection: "
          f"{leak:.2e} (must be 0)")
    rho_hard = (rho > 0.5).astype(float)
    assert np.abs(rho_hard * (1 - M)).max() == 0.0
    np.save(out / "geometries" / "rho_hard_binary.npy", rho_hard)
    print(f"[binarize] fill fraction (cell) = {rho_hard.mean():.4f}, "
          f"(active region) = {rho_hard.sum()/M.sum():.4f}")
    return F0, F1


if __name__ == "__main__":
    main()
