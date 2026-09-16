"""Shared helpers of the Q-Dz ACTIVATION extension (paths, provenance, I/O).

This package is a SIBLING of qdz_parent/: it never modifies qdz_parent, forward.py, materials.py,
analysis.py or nonlinear_activation_best/.  Every physical routine is imported from those packages.

Import order matters only for the sys.path set-up below (the repository uses plain modules + sys.path,
not installed packages); every module of this package starts with `from common import *`-style imports
of what it needs, so the path set-up happens exactly once.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent                  # .../qdz_activation
PKG = HERE.parent                                       # .../enz_jz_inverse_design
QDZ = PKG / "qdz_parent"
NONLIN = PKG / "nonlinear_activation_best"
OUT = HERE / "outputs"
for p in (str(PKG), str(QDZ), str(NONLIN)):
    if p not in sys.path:
        sys.path.insert(0, p)

C_NM_FS = 299.792458


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PKG, capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:
        return None


def git_dirty():
    try:
        r = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PKG,
                           capture_output=True, text=True, timeout=10)
        return bool(r.stdout.strip())
    except Exception:
        return None


def sha256_array(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jdump(obj, path):
    """JSON dump with numpy / complex / Path support (complex -> [re, im])."""
    def d(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, complex):
            return [o.real, o.imag]
        if isinstance(o, Path):
            return str(o)
        return str(o)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=d)


def jload(path):
    with open(path) as f:
        return json.load(f)


def omega_of(lam_nm):
    return 2 * np.pi * C_NM_FS / np.asarray(lam_nm, dtype=float)      # rad/fs


def lam_of(omega):
    return 2 * np.pi * C_NM_FS / np.asarray(omega, dtype=float)


def provenance(**extra):
    """Metadata block written into every result file of this package."""
    import config                                   # noqa: E402  (JZ package)
    return dict(package="qdz_activation", git_commit=git_commit(), git_dirty=git_dirty(),
                d_ito_nm=float(config.D_ITO_NM), nx=int(config.NX),
                sim_dtype=str(config.SIM_DTYPE), geo_dtype=str(config.GEO_DTYPE),
                materials=dict(ito=str(config.ITO_FILE.name), asi=str(config.ASI_FILE.name),
                               glass=str(config.GLASS_FILE.name),
                               sha256={k: sha256_file(v) for k, v in
                                       (("ito", config.ITO_FILE), ("asi", config.ASI_FILE), ("glass", config.GLASS_FILE))}),
                **extra)
