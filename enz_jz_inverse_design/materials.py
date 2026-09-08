"""Authoritative materials for the JZ campaign, parsed DIRECTLY from the
supplied files (materials_data/, sha256 in PROVENANCE.md).

    ITO      ITO_nk.csv   (Drude-Lorentz SE fit, 310-1675 nm, 1-nm step;
                          columns wavelength_nm, energy_eV, n, k, eps1, eps2)
    a-Si:H   aSi_H_measured_Postech.txt (246-999 nm ellipsometry;
                          1000-1400 nm Sellmeier extrapolation of the source
                          workbook, k = 0 there  -> PROVENANCE FLAG)
    glass    SiO2_substrate_measured.txt (soda-lime float glass, air side,
                          191-1689 nm; identical to the *_comsol.txt file)

Rules: cubic-spline interpolation strictly INSIDE the tabulated range; any
request outside raises (no silent clamping / extrapolation).  eps = (n+ik)^2
with the exp(-j w t) convention of TORCWA (Im eps > 0 = loss).
"""

import json
import hashlib
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

import config


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class NKTable:
    def __init__(self, wl, n, k, name, source, notes=""):
        o = np.argsort(wl)
        self.wl, self.n, self.k = np.asarray(wl)[o], np.asarray(n)[o], np.asarray(k)[o]
        self.name, self.source, self.notes = name, str(source), notes
        self.lo, self.hi = float(self.wl[0]), float(self.wl[-1])
        self._n = CubicSpline(self.wl, self.n)
        self._k = CubicSpline(self.wl, self.k)

    def _check(self, lam):
        lam = float(lam)
        if not (self.lo <= lam <= self.hi):
            raise ValueError(f"{self.name}: lambda = {lam} nm outside the supplied "
                             f"range [{self.lo}, {self.hi}] nm - refusing to extrapolate")
        return lam

    def nk(self, lam):
        lam = self._check(lam)
        return complex(float(self._n(lam)), float(self._k(lam)))

    def eps(self, lam):
        return self.nk(lam) ** 2


# ---------------------------------------------------------------------------
# parsers (no pandas: plain text, explicit columns)
# ---------------------------------------------------------------------------
def load_ito(path=config.ITO_FILE):
    header, rows, comments = None, [], []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                comments.append(s)
                continue
            if header is None and s.lower().startswith("wavelength"):
                header = [c.strip() for c in s.split(",")]
                continue
            rows.append([float(v) for v in s.split(",")])
    a = np.array(rows)
    col = {h: i for i, h in enumerate(header)}
    tab = NKTable(a[:, col["wavelength_nm"]], a[:, col["n"]], a[:, col["k"]],
                  "ITO", path, "; ".join(comments))
    tab.eps1_col = a[:, col["eps1"]]
    tab.eps2_col = a[:, col["eps2"]]
    tab.header = header
    tab.comments = comments
    return tab


def load_two_col_nk(path, name, notes=""):
    comments = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("#") or s.startswith("%"):
                comments.append(s)
    a = np.loadtxt(path, comments=("#", "%"))
    return NKTable(a[:, 0], a[:, 1], a[:, 2], name, path,
                   notes + " | " + "; ".join(comments))


_CACHE = {}
K_ZERO_TOL = 1e-12


def ito():
    if "ito" not in _CACHE:
        _CACHE["ito"] = load_ito()
    return _CACHE["ito"]


def asi():
    if "asi" not in _CACHE:
        _CACHE["asi"] = load_two_col_nk(
            config.ASI_FILE, "a-Si:H",
            "rows >= 1000 nm are a Sellmeier extrapolation of the source "
            "workbook (k = 0), NOT ellipsometry")
    return _CACHE["asi"]


def glass():
    if "glass" not in _CACHE:
        _CACHE["glass"] = load_two_col_nk(config.GLASS_FILE, "soda-lime glass")
    return _CACHE["glass"]


# ---------------------------------------------------------------------------
# permittivities used by the forward model
# ---------------------------------------------------------------------------
def eps_ito(lam, loss_scale=1.0):
    e = ito().eps(lam)
    return complex(e.real, loss_scale * e.imag)


def eps_asi(lam):
    return asi().eps(lam)


def n_glass(lam):
    """Real substrate index.  The semi-infinite output medium must be
    lossless for R/T to be defined; the supplied glass file has k = 0 above
    ~351 nm, which is checked here (no silent zeroing)."""
    nk = glass().nk(lam)
    # the tabulated k is exactly 0 above ~351 nm; the cubic spline returns
    # a ~1e-289 numerical residue there, so 'zero' means |k| < K_ZERO_TOL
    if abs(nk.imag) > K_ZERO_TOL:
        raise ValueError(f"glass k = {nk.imag:g} at {lam} nm is not zero; a lossy "
                         "semi-infinite substrate is outside this model")
    return nk.real


# ---------------------------------------------------------------------------
# ENZ zero crossing (interpolated, not hard-coded)
# ---------------------------------------------------------------------------
def ito_zero_crossing(lo=1000.0, hi=1600.0):
    t = ito()
    f = lambda l: (t.nk(l) ** 2).real
    grid = np.arange(max(lo, t.lo), min(hi, t.hi) + 1e-9, 1.0)
    vals = np.array([f(l) for l in grid])
    s = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    if len(s) != 1:
        raise RuntimeError(f"expected exactly one Re(eps)=0 crossing in [{lo},{hi}], found {len(s)}")
    lam_ze = brentq(f, grid[s[0]], grid[s[0] + 1], xtol=1e-9)
    # cross-check with the supplied eps1 column (linear interpolation)
    e1 = CubicSpline(t.wl, t.eps1_col)
    lam_ze_col = brentq(lambda l: float(e1(l)), grid[s[0]], grid[s[0] + 1], xtol=1e-9)
    return float(lam_ze), float(lam_ze_col)


def audit(out_json=None, verbose=True):
    """Stage-0 material audit: parse, verify (n+ik)^2 == eps columns, find
    lambda_ZE, report n,k,eps at lambda_ZE for all three materials."""
    t = ito()
    eps_nk = (t.n + 1j * t.k) ** 2
    err = np.abs(eps_nk - (t.eps1_col + 1j * t.eps2_col))
    lam_ze, lam_ze_col = ito_zero_crossing()
    e_ito = t.eps(lam_ze)
    nk_ito = t.nk(lam_ze)
    a, g = asi(), glass()
    nk_a, nk_g = a.nk(lam_ze), g.nk(lam_ze)
    ng = n_glass(lam_ze)
    rep = dict(
        files={n: dict(path=str(p), sha256=_sha(p)) for n, p in
               (("ITO", config.ITO_FILE), ("aSi", config.ASI_FILE), ("glass", config.GLASS_FILE))},
        ito=dict(range_nm=[t.lo, t.hi], n_rows=int(len(t.wl)), header=t.header,
                 comments=t.comments,
                 max_abs_err_eps_vs_nk2=float(err.max()),
                 max_rel_err_eps_vs_nk2=float((err / np.abs(eps_nk)).max()),
                 lambda_ZE_nm=lam_ze, lambda_ZE_from_eps1_column_nm=lam_ze_col,
                 n=nk_ito.real, k=nk_ito.imag, eps1=e_ito.real, eps2=e_ito.imag,
                 eps_at=[dict(lam=l, eps=[t.eps(l).real, t.eps(l).imag]) for l in
                         (1250.0, 1280.0, 1300.0, 1302.0, 1305.0, 1310.0, 1350.0, 1400.0)]),
        asi=dict(range_nm=[a.lo, a.hi], n_rows=int(len(a.wl)), n=nk_a.real, k=nk_a.imag,
                 eps=[a.eps(lam_ze).real, a.eps(lam_ze).imag],
                 provenance=a.notes, k_last_nonzero_nm=float(a.wl[a.k > 0].max()),
                 measured_rows_nm="246-999 (ellipsometry); 1000-1400 workbook extrapolation"),
        glass=dict(range_nm=[g.lo, g.hi], n_rows=int(len(g.wl)), n=nk_g.real, k=nk_g.imag,
                   n_used=ng, k_last_nonzero_nm=float(g.wl[g.k > 0].max()),
                   identical_to_comsol_file=bool(np.array_equal(
                       np.loadtxt(config.MAT_DIR / "soda_lime_float_glass_airside_nk_nm_comsol.txt", comments="%"),
                       np.loadtxt(config.GLASS_FILE, comments="#")))),
        diffraction=dict(P_first_order_glass_nm=lam_ze / ng, P_first_order_air_nm=lam_ze,
                         note="normal incidence: (+-1,0) propagates in glass for P > lambda/n_glass"),
    )
    if verbose:
        print(json.dumps({k: v for k, v in rep.items() if k != "files"}, indent=1))
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(rep, f, indent=1)
    return rep


if __name__ == "__main__":
    audit(config.OUT / "stage0" / "materials_audit.json")
