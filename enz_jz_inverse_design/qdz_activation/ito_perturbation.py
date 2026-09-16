"""The ITO permittivity PERTURBATION layer of the activation extension (configurable, logged).

    eps0(lambda) = physical cold state      = materials.eps_ito(lambda)          (supplied ITO_nk.csv)
    eps1(lambda) = perturbed state          = eps0(lambda) + delta_eps(lambda)

NATIVE MODEL (default, kind = "hot_electron_Te").  The repository's only nonlinear ITO model is the
Lane-B hot-electron model of nonlinear_activation_best/ito_nonlinear.py (branch
claude/jz-1302-nonlinear-activation, merged into this lineage for reuse):

    eps(lambda, Te) = eps_csv(lambda) + [eps_D(lambda, Te) - eps_D(lambda, 300 K)]
    eps_D(lambda, Te) = eps_inf - wp^2(Te) / (w^2 + i gamma w),      gamma constant in Te

with wp^2(Te) the Drude weight of a Kane band (m0* = 0.3964 m_e, C = 0.4191 eV^-1, Liu 2014 /
Alam 2016; N calibrated so that wp(300 K) equals the Drude fit of the supplied cold data).  It reduces
EXACTLY to the supplied ITO_nk.csv at 300 K (verified there and re-verified in tests/sanity_tests.py).
Its status is PROVISIONAL (NONLINEAR_MODEL_AUDIT.md: the band parameters are literature values for
other films; no nonlinear measurement of this ITO exists).  It is used here ONLY as the direction and
scale of a small permittivity change, never for an intensity axis.

WHAT IS ROBUST AND WHAT IS NOT.  For any Drude-weight change dwp2 = wp^2(Te) - wp0^2 < 0,

    delta_eps(lambda) = -dwp2 / (w^2 + i gamma w) = |dwp2| (w^2 - i gamma w) / (w^4 + gamma^2 w^2)

so the DIRECTION of the perturbation in the complex plane, arg(delta_eps) = -atan(gamma / w)
(= -7.3 deg at lambda_E for the fitted gamma = 0.1847 rad/fs), is fixed by the Drude fit of the
SUPPLIED cold data alone (eps_inf, wp0, gamma over 1150-1450 nm) - i.e. by a Lane-A quantity - while
the MAGNITUDE per kelvin (|dwp2|(Te)) depends on the Lane-B band model.  Consequently the normalized
sensitivity S_T = dT / |delta_eps| is robust to the provisional model (to first order in the
perturbation), whereas dT per kelvin or per pJ is provisional.  The kind = "drude_weight" mode exposes
this directly through the single scalar delta = 1 - wp^2/wp0^2 (the repository's Stage-18 precedent).

OTHER KINDS (diagnostics; every result file records which kind was used):
    "drude_weight"    delta_eps = delta * wp0^2 / (w^2 + i gamma w)      (delta = 1 - wp^2/wp0^2)
    "real_only"       Re of the hot_electron_Te delta only  (mechanism decomposition, as in the
    "imag_only"       Im of the hot_electron_Te delta only   nonlinear lane's mechanism_decomp.py)
    "custom"          a user-supplied complex delta_eps (constant in lambda) - FLAGGED NON-PHYSICAL,
                      allowed only for direction-sensitivity checks, never for selection.
"""
import numpy as np

import common as cm                                    # noqa: F401  (sys.path set-up)
import materials as mat                                # noqa: E402  (JZ package: supplied cold data)
import ito_nonlinear as nl                             # noqa: E402  (native Lane-B model, merged branch)

KINDS = ("hot_electron_Te", "drude_weight", "real_only", "imag_only", "custom")
_MODEL = {}


def hot_model():
    """The single shared ITOHot instance (Drude fit + Kane-band calibration happen once)."""
    if "m" not in _MODEL:
        _MODEL["m"] = nl.ITOHot()
    return _MODEL["m"]


class ITOPerturbation:
    def __init__(self, kind="hot_electron_Te", Te=1000.0, delta=None, custom_delta_eps=None):
        if kind not in KINDS:
            raise ValueError(f"unknown perturbation kind {kind}; choose from {KINDS}")
        self.kind, self.Te = kind, float(Te)
        self.model = hot_model()
        self.eps_inf, self.wp0, self.gamma = self.model.eps_inf, self.model.wp0, self.model.gamma
        if kind in ("hot_electron_Te", "real_only", "imag_only"):
            st = self.model.state(self.Te)
            self.delta = 1.0 - st["wp2_ratio"]                  # fractional Drude-weight loss
            self.state = st
        elif kind == "drude_weight":
            if delta is None:
                raise ValueError("kind='drude_weight' needs delta = 1 - wp^2/wp0^2")
            self.delta = float(delta); self.state = None
        else:
            if custom_delta_eps is None:
                raise ValueError("kind='custom' needs custom_delta_eps")
            self.delta = None; self.state = None
            self.custom = complex(custom_delta_eps)

    # ---- the two material states -------------------------------------------------
    def eps0(self, lam):
        return complex(mat.eps_ito(float(lam)))

    def delta_eps(self, lam):
        w = 2 * np.pi * cm.C_NM_FS / float(lam)
        if self.kind == "custom":
            return self.custom
        d = self.delta * self.wp0 ** 2 / (w ** 2 + 1j * self.gamma * w)
        if self.kind == "hot_electron_Te":
            # identical to ito_nonlinear.ITOHot.eps(lam, Te) - eps_ito(lam); asserted in the tests
            return complex(d)
        if self.kind == "real_only":
            return complex(d.real, 0.0)
        if self.kind == "imag_only":
            return complex(0.0, d.imag)
        return complex(d)                                # drude_weight

    def eps1(self, lam):
        return self.eps0(lam) + self.delta_eps(lam)

    def norm(self, lam):
        """||eps1 - eps0|| used to normalize dT (complex modulus)."""
        return abs(self.delta_eps(lam))

    # ---- provenance ----------------------------------------------------------------
    def describe(self, lams=()):
        lam_E, _ = mat.ito_zero_crossing()
        d = dict(kind=self.kind, Te_K=(self.Te if self.kind in ("hot_electron_Te", "real_only", "imag_only") else None),
                 delta_drude_weight=(None if self.delta is None else float(self.delta)),
                 drude_fit_supplied=dict(eps_inf=float(self.eps_inf), wp0_rad_fs=float(self.wp0),
                                         gamma_rad_fs=float(self.gamma), **{k: v for k, v in self.model.fit.items()
                                                                            if k in ("rms_resid", "max_abs_resid", "range_nm")}),
                 band_model=dict(m0_star_over_me=nl.M0_STAR / nl.me, C_eV_inv=nl.C_NP, N_m3=float(self.model.N),
                                 EF_300K_eV=float(self.model.EF), status="PROVISIONAL Lane B (literature band, N calibrated to the supplied cold fit)"),
                 direction_deg_at_lambda_E=float(np.degrees(np.angle(self.delta_eps(lam_E)))) if self.kind != "custom" else float(np.degrees(np.angle(self.custom))),
                 physical=(self.kind != "custom"), lambda_E_nm=float(lam_E),
                 lambda_ENZ_perturbed_nm=(self._enz_crossing() if self.kind in ("hot_electron_Te", "drude_weight") else None),
                 eps_at=[])
        if self.state:
            d["native_state"] = {k: float(v) for k, v in self.state.items()}
        for lam in [lam_E] + [float(x) for x in lams]:
            e0, e1 = self.eps0(lam), self.eps1(lam)
            d["eps_at"].append(dict(lam_nm=float(lam), eps0=[e0.real, e0.imag], eps1=[e1.real, e1.imag],
                                    delta_eps=[(e1 - e0).real, (e1 - e0).imag], abs_delta=abs(e1 - e0),
                                    rel_to_abs_eps0=abs(e1 - e0) / max(abs(e0), 1e-12)))
        return d

    def _enz_crossing(self, lo=1150.0, hi=1670.0):
        from scipy.optimize import brentq
        f = lambda l: self.eps1(l).real
        grid = np.arange(lo, hi, 2.0)
        v = np.array([f(l) for l in grid])
        s = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
        return float(brentq(f, grid[s[0]], grid[s[0] + 1], xtol=1e-6)) if len(s) else float("nan")


def from_args(kind, Te, delta=None, custom=None):
    if kind == "custom":
        re, im = [float(x) for x in str(custom).split(",")]
        return ITOPerturbation("custom", custom_delta_eps=complex(re, im))
    return ITOPerturbation(kind, Te=Te, delta=delta)
