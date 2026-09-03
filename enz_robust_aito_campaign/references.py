"""Reference structures for the robust-A_ITO campaign (all frozen, read
only from the earlier campaigns; all live on the 850-nm / 140-nm cell)."""

from pathlib import Path

import numpy as np
import torch

import forward_multi as fm

ROOT = Path(__file__).resolve().parent.parent
P_REF, H_REF = 850.0, 140.0

FILES = {
    "unpadded QNM winner":
        ROOT / "enz_inverse_design/outputs/geometries/rho_hard_binary.npy",
    "padded QNM winner":
        ROOT / "enz_padding_sideexperiment/outputs/geometries/rho_hard_binary.npy",
    "padded F_ENZ winner":
        ROOT / "enz_direct_enz_excitation/outputs/geometries/rho_hard_binary.npy",
}


def edr_cuboid(nx=128, P=P_REF, lx=560.0, ly=500.0):
    xg = (np.arange(nx) + 0.5) / nx * P
    X, Y = np.meshgrid(xg, xg, indexing="ij")
    return ((np.abs(X - P / 2) < lx / 2) & (np.abs(Y - P / 2) < ly / 2)
            ).astype(float)


def load_all():
    """Ordered dict name -> (rho tensor or None, P, h, source)."""
    refs = {"bare ITO": (None, P_REF, H_REF, "air/ITO(23)/glass, no a-Si"),
            "EDR cuboid": (torch.as_tensor(edr_cuboid(), dtype=fm.GEO_DTYPE),
                           P_REF, H_REF, "560x500x140 nm cuboid, 850 cell")}
    for name, f in FILES.items():
        a = np.load(f)
        assert a.shape == (128, 128) and set(np.unique(a)) <= {0.0, 1.0}, f
        refs[name] = (torch.as_tensor(a, dtype=fm.GEO_DTYPE), P_REF, H_REF,
                      str(f.relative_to(ROOT)))
    return refs
