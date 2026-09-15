# -*- coding: utf-8 -*-
"""_te_fluence_map.py -- TTM-based fluence <-> electron-temperature map for the TK-BIC / ENZ-ITO metasurface.

Purpose
-------
Label the Te axis of the RCWA / FDTD T(lam, Te) family with a *physically honest* incident laser
fluence.  Instead of the quasi-static ladder (int C_e dTe = A F / t_ITO, no e-ph loss) the electron
temperature Te(t) is integrated through the 280 fs pulse with a two-temperature model (TTM) in which
the absorption A(lam_pump, Te(t)) is taken self-consistently from the spectral family:

    C_e(Te) dTe/dt = S(t) - g (Te - T_l)        S(t) = A(lam_pump, Te) * F * I_norm(t) / t_ITO
    C_L    dT_l/dt =        g (Te - T_l)        I_norm: Gaussian, FWHM 280 fs, int I dt = 1

g models:  (a) g = C_e(Te)/tau_ep, tau_ep in {300, 450, 600, 1000} fs   (b) constant g = 1.8e17 W m^-3 K^-1.
C_e models (J m^-3 K^-1, ported from scratchpad/c3_ttm_check.py): "sommerfeld" (gamma_e Te, the c3 default),
"min" (min(gamma_e Te, 1.5 n kB)), "fd" (numerical Fermi-Dirac, parabolic band), "kane" (FD, Kane C=0.4191).

Units inside the module are SI (J m^-3 K^-1, W m^-3 K^-1, J m^-2, s, K); the public interface uses
fluence in mJ/cm^2, pulse energy in nJ, energy density u in J/cm^3, time in fs where stated.

Pulse energy conversion: E[nJ] = F[mJ/cm^2] * (8 um)^2 = 0.64 * F.  (E_sat 0.464 nJ ~ 0.72 mJ/cm^2.)

Public API
----------
load_family(path=None, kind="rcwa", corr=None, te_interp="log")    -> Family
HeatCapacity(kind)                                                   -> C_e(Te), u_e(Te), Te_of_u(u)
integrate_ttm(F_mJcm2, Afun, Ce, g_model, ...)                        -> dict (batched RK4, vectorised over F)
run_map(family_path, kind, lam_pump, F_grid, tau_list, g_const, two_zone=True, ...) -> dict
regression_check(fam, ...)                                           -> (text, rows)
main()                                                               -> CLI

CLI
---
python _te_fluence_map.py --family rcwa|fdtd [--path FILE] --lam-pump 1292 --out DIR
        [--ce sommerfeld|min|fd|kane] [--te-interp log|linear] [--tau 300,450,600,1000]
        [--g-const 1.8e17] [--nF 40 --F-min 0.01 --F-max 5] [--dt-fs 1.0] [--no-two-zone] [--no-regression]

npz keys written by run_map / CLI (tkbic_te_fluence_map.npz)
-------------------------------------------------------------
meta_json            : str  JSON of settings (family path/kind, lam_pump, corr, Ce kind, t_ITO, C_L, F_P, ...)
models               : str[nM]   model labels ("tau300fs", ..., "gconst1.8e+17")
tau_list_fs          : float[nTau]
g_const_Wm3K         : float
F_grid_mJcm2         : float[nF]     incident fluence grid
E_grid_nJ            : float[nF]     pulse energy (F * 0.64)
lam_pump_nm          : float
lam_probe_nm         : float[nP]
Te_pk[nM,nF]         : peak electron temperature (K)
t_pk_fs[nM,nF]       : time of Te peak relative to pulse centre (fs)
Te_avgI[nM,nF]       : intensity-weighted mean Te over the pulse (K)
Te_1ps[nM,nF]        : Te at +1 ps (K)
Tl_end[nM,nF]        : lattice temperature at +3 ps (K)
u_abs_Jcm3[nM,nF]    : absorbed energy density (J/cm^3)
A_eff[nM,nF]         : u_abs * t_ITO / F  (pulse-effective absorption)
Te_qs[nF]            : quasi-static ladder Te with A(lam_pump, 300 K) (K)
u_qs_Jcm3[nF]        : quasi-static absorbed energy density (J/cm^3)
eta_ret[nM,nF]       : Te_pk / Te_qs
eta_u[nM,nF]         : u_abs / u_qs
above_table[nM,nF]   : bool, trajectory exceeded the family's max Te (A held at last value)
dT_probe_pk[nM,nP,nF]  : T(lam_probe, Te_pk) - T(lam_probe, 300)          (probe at peak)
dT_probe_avg[nM,nP,nF] : int I T(lam_probe, Te(t)) dt / int I dt - T(lam_probe, 300)   (probe = pump replica, zero delay)
T_avg_pump[nM,nF]      : single-beam self-action, int I T(lam_pump, Te(t)) dt / int I dt (uniform)
T_avg_pump_2zone[nM,nF]: same, two-zone hot/cold area-weighted average (NaN if two_zone=False)
Te_pk_hot[nM,nF], Te_pk_cold[nM,nF] : two-zone peak temperatures
dT_probe_pk_2zone[nM,nP,nF]         : two-zone area-weighted pump-probe signal at peak
inv_Te_levels[nL]      : family Te levels used in the inverse map
inv_F_pk[nM,nL]        : fluence (mJ/cm^2) with Te_pk == level (NaN if outside 0.005-20 mJ/cm^2)
inv_E_pk[nM,nL]        : same as pulse energy (nJ)
inv_F_avg[nM,nL]       : fluence with <Te>_I == level
inv_E_avg[nM,nL]
inv_F_qs[nL]           : quasi-static ladder fluence for the level
family_Te, family_lam, family_T, family_A : the loaded (corrected) family
traj_t_fs, traj_Te[nM,nt], traj_Tl[nM,nt]  : trajectories at F = F_SAT (0.72 mJ/cm^2)
regression_text, checks_text, table_text, observables_text, brentq_text : str summaries (also in the .txt)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy.constants import hbar, e as E_CHG, m_e, epsilon_0, k as KB
from scipy.optimize import brentq

# ----------------------------------------------------------------------------- constants
KB_EV = 8.617333262e-5
P_NM = 588.0
D_NM = 405.63439375180764
F_P = np.pi * (D_NM / 2) ** 2 / P_NM ** 2          # pillar area fraction (0.3737)
T_ITO_M = 23e-9                                    # ITO thickness (m)
C_L = 2.6e6                                        # lattice heat capacity J m^-3 K^-1 (2.6 J cm^-3 K^-1)
FWHM = 280e-15                                     # pulse FWHM (s)
SIG = FWHM / (2 * np.sqrt(2 * np.log(2)))
SPOT_CM2 = (8e-4) ** 2                             # (8 um)^2 in cm^2
NJ_PER_MJCM2 = 1e-3 * SPOT_CM2 * 1e9               # 0.64 nJ per mJ/cm^2
F_SAT_MJ = 0.72                                    # E_sat fluence (mJ/cm^2)
E_SAT_NJ = 0.464                                   # quoted E_sat (nJ)
E_MAX_NJ = 700.0                                   # laser maximum pulse energy (nJ)
A_CORR_RCWA = 0.7619                               # S-matrix -> field-integral absorption renormalisation (c3 legacy)
#   NOTE 2026-09-04: 0.7619 = A_res 0.405 / torcwa raw 0.5317 (resonant-only / total) -- a 12.5 % over-correction.
#   Lumerical FDTD Te family (mesh 4, 9 Te, cubic-spline dip estimates; Phase-2 verified) gives, at each
#   method's own dip, (1-T-R)_FDTD/(1-T-R)_RCWA = 0.873-0.879, Te-FLAT 300-8000 K (= _rcwa_fast.A_CORR 0.8708);
#   on the A_ito (ITO flux difference) basis 0.855-0.862 -- the TTM-relevant one. (An earlier '0.844 at
#   8000 K' came from an np.interp/parabola dip estimator and is refuted.) At FIXED 1292 nm the ratio falls
#   0.858 -> 0.767 (8000 K) -- resonance-flank sensitivity, not S-matrix bias; no scalar corrects that.
#   Kept for reproducibility of the RCWA-fed map; the FDTD-fed map (A_ito, no correction) is the reference.
#   Pass corr=A_CORR_RCWA_FDTD to run_map/load_family for the corrected RCWA map.
A_CORR_RCWA_FDTD = 0.870                           # (1-T-R) basis; use 0.861 for the A_ito basis
G_CONST_DEFAULT = 1.8e17                           # W m^-3 K^-1
TAU_DEFAULT_FS = (300.0, 450.0, 600.0, 1000.0)
HOT_SCALE, COLD_SCALE = 1.74, 0.44                 # two-zone local absorption factors (ASSUM, from |E|^2 field integral)
U_8000_LABEL_JCM3 = 167.0                          # Sommerfeld u_e(8000 K) reference label
F_LIT_RANGE = (0.2, 40.0)                          # literature ENZ-ITO fluence range (mJ/cm^2)
HWP_EV, M_RATIO = 1.776, 0.35                      # Drude plasma energy / effective mass ratio (c3 values)

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # package root (patched for share)
DEFAULT_RCWA = os.path.join(_PKG, "cache", "tkbic_angnl_th0.npz")
DEFAULT_FDTD = os.path.join(_PKG, "cache", "tkbic_lumerical_Te_family.npz")
DEFAULT_OUT = os.path.join(_PKG, "figures")

# electron gas (parabolic band) -------------------------------------------------------------------------
_wp = HWP_EV * E_CHG / hbar
N_E_M3 = _wp ** 2 * epsilon_0 * M_RATIO * m_e / E_CHG ** 2            # 8.006e26 m^-3
E_F_J = (hbar ** 2 / (2 * M_RATIO * m_e)) * (3 * np.pi ** 2 * N_E_M3) ** (2 / 3)
E_F_EV = E_F_J / E_CHG                                                # 0.898 eV
T_F = E_F_EV / KB_EV                                                  # 10425 K
GAM_E = np.pi ** 2 * N_E_M3 * KB ** 2 / (2 * E_F_J)                   # J m^-3 K^-2 (5.233)
C_ND = 1.5 * N_E_M3 * KB                                              # J m^-3 K^-1 (non-degenerate)


def pulse_I(t):
    """Normalised Gaussian intensity envelope (1/s), int I dt = 1, FWHM 280 fs."""
    return np.exp(-t ** 2 / (2 * SIG ** 2)) / (SIG * np.sqrt(2 * np.pi))


def F_to_nJ(F_mJcm2):
    return np.asarray(F_mJcm2, float) * NJ_PER_MJCM2


def nJ_to_F(E_nJ):
    return np.asarray(E_nJ, float) / NJ_PER_MJCM2


# ----------------------------------------------------------------------------- family loader
class Family:
    """T(Te, lam), A(Te, lam) tables + interpolators.

    Te interpolation: log-linear (te_interp="log", default) or linear ("linear", c3-compatible).
    Te < Te[0] is clamped; Te > Te[-1] holds the last table value and is flagged (Family.above_table).
    """

    def __init__(self, lam, Te, T, A, kind, path, corr, te_interp="log", R=None):
        self.lam = np.asarray(lam, float)
        self.Te = np.asarray(Te, float)
        self.T = np.asarray(T, float)
        self.A = np.asarray(A, float)
        self.R = None if R is None else np.asarray(R, float)
        self.kind, self.path, self.corr, self.te_interp = kind, path, corr, te_interp
        self._x = np.log(self.Te) if te_interp == "log" else self.Te.copy()

    # helpers ---------------------------------------------------------------------------------
    def _xf(self, Te):
        Tc = np.clip(np.asarray(Te, float), self.Te[0], self.Te[-1])
        return np.log(Tc) if self.te_interp == "log" else Tc

    def above_table(self, Te):
        return np.asarray(Te, float) > self.Te[-1]

    def column(self, arr, lam0, mode="fixed"):
        """Column of a (nTe, nlam) table at wavelength lam0 (linear in lam); mode="dip" tracks argmin T per Te."""
        if mode == "dip":
            return np.array([arr[i, np.argmin(self.T[i])] for i in range(len(self.Te))])
        return np.array([np.interp(lam0, self.lam, arr[i]) for i in range(len(self.Te))])

    def fn_of_Te(self, arr, lam0, mode="fixed", floor=None):
        col = self.column(arr, lam0, mode)
        x = self._x

        def f(Te):
            out = np.interp(self._xf(Te), x, col)
            return out if floor is None else np.maximum(out, floor)
        f.col = col
        return f

    def T_fn(self, lam0, mode="fixed"):
        return self.fn_of_Te(self.T, lam0, mode)

    def A_fn(self, lam0, mode="fixed", floor=0.03):
        return self.fn_of_Te(self.A, lam0, mode, floor)

    def T2(self, lam, Te):
        """Generic interpolation T(lam, Te) (scalar lam, vector Te)."""
        return self.T_fn(float(lam))(Te)

    def A2(self, lam, Te):
        return self.A_fn(float(lam), floor=None)(Te)

    def dip_lam(self):
        return np.array([self.lam[np.argmin(self.T[i])] for i in range(len(self.Te))])


def _orient(arr, nTe, nlam):
    arr = np.asarray(arr, float)
    if arr.shape == (nTe, nlam):
        return arr
    if arr.shape == (nlam, nTe):
        return arr.T
    raise ValueError(f"table shape {arr.shape} incompatible with nTe={nTe}, nlam={nlam}")


def load_family(path=None, kind="rcwa", corr=None, te_interp="log"):
    """Load the Te spectral family.

    kind="rcwa": tkbic_angnl_th0.npz (lam, Te, Tp, Rp); A = (1 - Tp - Rp) * corr  (corr default 0.7619)
    kind="fdtd": tkbic_lumerical_Te_family.npz (Te, lam_nm, T, R, A, A_ito); absorption = A_ito, corr default 1.0
    """
    if kind == "rcwa":
        path = path or DEFAULT_RCWA
        corr = A_CORR_RCWA if corr is None else float(corr)
        d = np.load(path)
        lam, Te = d["lam"], d["Te"]
        Tp = _orient(d["Tp"], len(Te), len(lam))
        Rp = _orient(d["Rp"], len(Te), len(lam))
        A = (1.0 - Tp - Rp) * corr
        return Family(lam, Te, Tp, A, kind, path, corr, te_interp, R=Rp)
    if kind == "fdtd":
        path = path or DEFAULT_FDTD
        corr = 1.0 if corr is None else float(corr)
        d = np.load(path, allow_pickle=True)
        Te = np.asarray(d["Te"], float)
        lam = np.asarray(d["lam_nm"], float)
        T = _orient(d["T"], len(Te), len(lam))
        R = _orient(d["R"], len(Te), len(lam)) if "R" in d.files else None
        if "A_ito" in d.files:
            A = _orient(d["A_ito"], len(Te), len(lam))
        else:
            print("[load_family] WARNING: A_ito missing in FDTD family; falling back to key 'A'")
            A = _orient(d["A"], len(Te), len(lam))
        return Family(lam, Te, T, A * corr, kind, path, corr, te_interp, R=R)
    raise ValueError(kind)


# ----------------------------------------------------------------------------- electron heat capacity
_FD_CACHE = {}


def _fd_tables(kane_C=None, m0_ratio=M_RATIO, Tmax=40000.0, nT=300):
    """Free-electron (or Kane) gas at fixed n (port of c3 fd_tables, SI units).

    Returns Ts [K], u(Te)-u(0) [J/m^3], C_e [J/m^3/K], mu [eV], E_F0 [eV]."""
    key = (kane_C, m0_ratio, Tmax, nT)
    if key in _FD_CACHE:
        return _FD_CACHE[key]
    kF = (3 * np.pi ** 2 * N_E_M3) ** (1 / 3)
    Ek = hbar ** 2 * kF ** 2 / (2 * m0_ratio * m_e) / E_CHG          # eV
    if kane_C is None:
        EF0 = Ek

        def g(E):
            return np.sqrt(E)
    else:
        EF0 = (-1 + np.sqrt(1 + 4 * kane_C * Ek)) / (2 * kane_C)

        def g(E):
            return np.sqrt(E * (1 + kane_C * E)) * (1 + 2 * kane_C * E)
    E = np.linspace(0, 60.0, 120001)
    gE = g(E)
    n0 = np.trapz(gE * (E < EF0), E)
    gE = gE / n0 * N_E_M3                                           # states m^-3 eV^-1
    U0 = np.trapz(E * gE * (E < EF0), E)

    def n_of_mu(mu, T):
        x = np.clip((E - mu) / (KB_EV * T), -600, 600)
        f = 1 / (1 + np.exp(x))
        return np.trapz(gE * f, E), f
    Ts = np.linspace(300, Tmax, nT)
    mus, Us = [], []
    for T in Ts:
        lo, hi = -40.0, 10.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            nn, _ = n_of_mu(mid, T)
            if nn < N_E_M3:
                lo = mid
            else:
                hi = mid
        mu = 0.5 * (lo + hi)
        _, f = n_of_mu(mu, T)
        mus.append(mu)
        Us.append(np.trapz(E * gE * f, E))
    mus, Us = np.array(mus), np.array(Us)
    u = (Us - U0) * E_CHG                                             # J/m^3 above T=0
    Ce = np.gradient(u, Ts)
    out = (Ts, u, Ce, mus, EF0)
    _FD_CACHE[key] = out
    return out


class HeatCapacity:
    """Electron heat capacity C_e(Te) [J m^-3 K^-1] with u_e(Te) = int_300^Te C_e dT and its inverse.

    kinds: "sommerfeld" (gamma_e Te; c3 default), "min", "fd", "kane".
    """

    def __init__(self, kind="sommerfeld"):
        self.kind = kind
        if kind == "sommerfeld":
            self._tab = None
        elif kind == "min":
            Ts = np.linspace(1.0, 60000.0, 60000)
            Ce = np.minimum(GAM_E * Ts, C_ND)
            self._tab = (Ts, Ce)
        elif kind in ("fd", "kane"):
            Ts, u, Ce, _, _ = _fd_tables(None if kind == "fd" else 0.4191)
            self._tab = (Ts, Ce)
        else:
            raise ValueError(kind)
        if self._tab is not None:
            Ts, Ce = self._tab
            u = np.concatenate([[0.0], np.cumsum(0.5 * (Ce[1:] + Ce[:-1]) * np.diff(Ts))])
            u -= np.interp(300.0, Ts, u)
            self._u = (Ts, u)

    def __call__(self, Te):
        Te = np.maximum(np.asarray(Te, float), 1.0)
        if self._tab is None:
            return GAM_E * Te
        Ts, Ce = self._tab
        return np.interp(Te, Ts, Ce)

    def u(self, Te):
        """Electron energy density above 300 K (J/m^3)."""
        Te = np.asarray(Te, float)
        if self._tab is None:
            return 0.5 * GAM_E * (Te ** 2 - 300.0 ** 2)
        Ts, u = self._u
        return np.interp(Te, Ts, u)

    def Te_of_u(self, u):
        """Quasi-static ladder: Te such that int_300^Te C_e dT = u (u in J/m^3)."""
        u = np.asarray(u, float)
        if self._tab is None:
            return np.sqrt(300.0 ** 2 + 2 * u / GAM_E)
        Ts, ut = self._u
        return np.interp(u, ut, Ts)


# ----------------------------------------------------------------------------- TTM integrator
def integrate_ttm(F_mJcm2, Afun, Ce, g_model, probe_fns=None, scale=1.0, t0=-1e-12, t1=3e-12,
                  dt_fs=1.0, t_ito=T_ITO_M, C_L_=C_L, T0=300.0, store_traj=False, traj_every=10,
                  Te_table_max=None):
    """Batched fixed-step RK4 integration of the TTM for a vector of fluences.

    F_mJcm2  : scalar or array of incident fluences (mJ/cm^2)
    Afun     : vectorised A(Te) (dimensionless absorption of the ITO film at lam_pump)
    Ce       : HeatCapacity (J m^-3 K^-1)
    g_model  : ("tau", tau_ep_s) -> g = C_e(Te)/tau_ep ;  ("const", g_Wm3K)
    probe_fns: dict name -> vectorised f(Te); intensity-weighted averages int I f(Te) dt are returned
    scale    : local absorption multiplier (two-zone hot spot approximation)
    Returns dict with Te_pk, t_pk (s), Te_avgI, Te_1ps, Tl_end, u_abs (J/m^3), A_eff, avg[name], above_table,
    and optional trajectories.
    """
    F = np.atleast_1d(np.asarray(F_mJcm2, float)) * 10.0            # mJ/cm^2 -> J/m^2
    N = F.size
    probe_fns = probe_fns or {}
    dt = dt_fs * 1e-15
    nstep = int(round((t1 - t0) / dt))
    kind, gpar = g_model
    Te = np.full(N, T0)
    Tl = np.full(N, T0)
    U = np.zeros(N)
    Te_pk = np.full(N, T0)
    t_pk = np.full(N, t0)
    Te_max_seen = np.full(N, T0)
    sw = 0.0
    swTe = np.zeros(N)
    swP = {k: np.zeros(N) for k in probe_fns}
    Te_1ps = np.full(N, np.nan)
    i_1ps = int(round((1e-12 - t0) / dt))
    traj = [] if store_traj else None

    def rhs(t, Te_, Tl_):
        S = scale * Afun(Te_) * F * pulse_I(t) / t_ito
        c = Ce(Te_)
        g = c / gpar if kind == "tau" else gpar
        q = g * (Te_ - Tl_)
        return (S - q) / c, q / C_L_, S

    t = t0
    for i in range(nstep + 1):
        # accumulate intensity-weighted quantities at the step start (rectangle rule, dt <= 1 fs)
        w = pulse_I(t) * dt
        if w > 1e-30:
            sw += w
            swTe += w * Te
            for k, fn in probe_fns.items():
                swP[k] += w * fn(Te)
        m = Te > Te_pk
        if np.any(m):
            Te_pk = np.where(m, Te, Te_pk)
            t_pk = np.where(m, t, t_pk)
        Te_max_seen = np.maximum(Te_max_seen, Te)
        if i == i_1ps:
            Te_1ps = Te.copy()
        if store_traj and i % traj_every == 0:
            traj.append((t, Te.copy(), Tl.copy()))
        if i == nstep:
            break
        k1 = rhs(t, Te, Tl)
        k2 = rhs(t + 0.5 * dt, Te + 0.5 * dt * k1[0], Tl + 0.5 * dt * k1[1])
        k3 = rhs(t + 0.5 * dt, Te + 0.5 * dt * k2[0], Tl + 0.5 * dt * k2[1])
        k4 = rhs(t + dt, Te + dt * k3[0], Tl + dt * k3[1])
        Te = Te + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        Tl = Tl + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        U = U + dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        Te = np.maximum(Te, 1.0)
        t = t0 + (i + 1) * dt
    out = dict(F=F / 10.0, Te_pk=Te_pk, t_pk=t_pk, Te_avgI=swTe / sw, Te_1ps=Te_1ps, Tl_end=Tl,
               u_abs=U, A_eff=U * t_ito / F, avg={k: v / sw for k, v in swP.items()},
               Te_end=Te)
    if Te_table_max is not None:
        out["above_table"] = Te_max_seen > Te_table_max
    if store_traj:
        out["traj_t"] = np.array([a[0] for a in traj])
        out["traj_Te"] = np.array([a[1] for a in traj]).T
        out["traj_Tl"] = np.array([a[2] for a in traj]).T
    return out


def model_list(tau_list_fs=TAU_DEFAULT_FS, g_const=G_CONST_DEFAULT):
    """[(label, ("tau", tau_s)), ..., (label, ("const", g))]"""
    models = [(f"tau{int(round(t))}fs", ("tau", t * 1e-15)) for t in tau_list_fs]
    if g_const is not None and g_const > 0:
        models.append((f"gconst{g_const:.1e}", ("const", float(g_const))))
    return models


# ----------------------------------------------------------------------------- inverse map
def invert_batched(targets, metric, Afun, Ce, gm, Fmin=0.005, Fmax=20.0, iters=16, dt_fs=1.0):
    """Vectorised bisection in log F for Te_pk(F) == target (metric="pk") or <Te>_I(F) == target ("avg").

    Returns F (mJ/cm^2); NaN where the target is not bracketed in [Fmin, Fmax]."""
    targets = np.atleast_1d(np.asarray(targets, float))
    key = "Te_pk" if metric == "pk" else "Te_avgI"
    lo = np.full(targets.size, np.log(Fmin))
    hi = np.full(targets.size, np.log(Fmax))
    vlo = integrate_ttm(np.exp(lo), Afun, Ce, gm, dt_fs=dt_fs)[key]
    vhi = integrate_ttm(np.exp(hi), Afun, Ce, gm, dt_fs=dt_fs)[key]
    ok = (vlo <= targets) & (vhi >= targets)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        v = integrate_ttm(np.exp(mid), Afun, Ce, gm, dt_fs=dt_fs)[key]
        below = v < targets
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    F = np.exp(0.5 * (lo + hi))
    return np.where(ok, F, np.nan)


def invert_brentq(target, metric, Afun, Ce, gm, Fmin=0.005, Fmax=20.0, dt_fs=1.0, xtol=1e-4):
    """Scalar brentq inversion (used to cross-check the batched bisection)."""
    key = "Te_pk" if metric == "pk" else "Te_avgI"

    def f(logF):
        return float(integrate_ttm(np.exp(logF), Afun, Ce, gm, dt_fs=dt_fs)[key][0]) - target
    a, b = np.log(Fmin), np.log(Fmax)
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        return np.nan
    return float(np.exp(brentq(f, a, b, xtol=xtol)))


# ----------------------------------------------------------------------------- regression vs c3
C3_ROWS = [
    # (F mJ/cm2, A mode, tau_fs or None(const g), Ce kind, Te_pk_ref, ratio_ref(Te_qs/Te_pk), Te_avgI_ref, u_abs_ref J/cm3, dTl_ref)
    (0.72, "dip", 450, "sommerfeld", 4576, 1.52, 3608, 114.6, 44.1),
    (0.72, "fixed", 450, "sommerfeld", 4536, 1.54, 3599, 113.0, 43.4),
    (0.25, "dip", 450, "sommerfeld", 2847, 1.44, 2218, 42.8, 16.5),
    (1.00, "dip", 1000, "sommerfeld", 6083, 1.35, 4639, 147.6, 56.7),
    (1.90, "dip", 300, "sommerfeld", 6192, 1.83, 5049, 271.8, 104.4),
    (0.72, "dip", None, "sommerfeld", 2428, None, 1862, None, 47.8),
    (0.72, "dip", 450, "fd", 5905, None, 4372, None, 41.6),
]


def regression_check(fam, lam_pump=1292.0, dt_fs=1.0, g_const=G_CONST_DEFAULT, te_interps=("linear", "log")):
    """Reproduce c3_ttm_check.py rows (LSODA, linear Te interpolation, A floor 0.03, A(Te) from the
    S-matrix table x 0.7619).  Returns (text, rows).  Tolerance +-2 % on Te_pk and Te_qs/Te_pk."""
    lines = []
    rows = []
    lines.append("=== Regression vs scratchpad/c3_ttm_check.py (c3: LSODA rtol 1e-8, linear Te interp; here: RK4 dt=%.2f fs) ===" % dt_fs)
    lines.append("A(Te): S-matrix (1-Tp-Rp) x %.4f, floor 0.03; 'dip' = A at argmin T per Te (c3 A(Te)dip), 'fixed' = A at %.0f nm (c3 A(Te)1292)" % (fam.corr, lam_pump))
    hdr = (f"{'interp':7s} {'F':>5s} {'Amode':>5s} {'tau':>6s} {'Ce':>10s} | {'Te_pk':>7s} {'c3':>6s} {'dev%':>6s} | "
           f"{'Tqs/Tpk':>7s} {'c3':>5s} {'dev%':>6s} | {'<Te>_I':>7s} {'c3':>6s} | {'u_abs':>6s} {'c3':>6s} | {'dTl':>5s} {'c3':>5s} | pass")
    lines.append(hdr)
    ok_all = True
    for interp in te_interps:
        f2 = Family(fam.lam, fam.Te, fam.T, fam.A, fam.kind, fam.path, fam.corr, interp, fam.R)
        for (F, amode, tau, ck, Tpk_ref, ratio_ref, Tavg_ref, u_ref, dTl_ref) in C3_ROWS:
            Ce = HeatCapacity(ck)
            Afun = f2.A_fn(lam_pump, mode=amode, floor=0.03)
            gm = ("tau", tau * 1e-15) if tau is not None else ("const", g_const)
            r = integrate_ttm(F, Afun, Ce, gm, dt_fs=dt_fs)
            A0 = 0.40509                                   # c3 A_RES for the ladder
            u_qs = A0 * (F * 10.0) / T_ITO_M
            Te_qs = float(HeatCapacity("sommerfeld").Te_of_u(u_qs))
            Tpk = float(r["Te_pk"][0])
            ratio = Te_qs / Tpk
            dev_T = (Tpk / Tpk_ref - 1) * 100
            dev_r = (ratio / ratio_ref - 1) * 100 if ratio_ref else np.nan
            u_abs = float(r["u_abs"][0]) * 1e-6
            dTl = float(r["Tl_end"][0]) - 300
            passed = abs(dev_T) <= 2.0 and (ratio_ref is None or abs(dev_r) <= 2.0)
            if interp == "linear":
                ok_all &= passed
            rows.append(dict(interp=interp, F=F, Amode=amode, tau=tau, Ce=ck, Te_pk=Tpk, Te_pk_ref=Tpk_ref,
                             ratio=ratio, ratio_ref=ratio_ref, Te_avgI=float(r["Te_avgI"][0]), Te_avg_ref=Tavg_ref,
                             u_abs=u_abs, u_ref=u_ref, dTl=dTl, dTl_ref=dTl_ref, passed=passed))
            lines.append(f"{interp:7s} {F:5.2f} {amode:>5s} {('%dfs' % tau) if tau else 'gconst':>6s} {ck:>10s} | "
                         f"{Tpk:7.0f} {Tpk_ref:6.0f} {dev_T:+6.2f} | {ratio:7.2f} {(ratio_ref or np.nan):5.2f} {dev_r:+6.2f} | "
                         f"{r['Te_avgI'][0]:7.0f} {Tavg_ref:6.0f} | {u_abs:6.1f} {(u_ref or np.nan):6.1f} | {dTl:5.1f} {dTl_ref:5.1f} | "
                         f"{'OK' if passed else 'FAIL'}")
    lines.append(f"Regression (linear-interp rows, +-2 %): {'PASS' if ok_all else 'FAIL'}")
    return "\n".join(lines), rows


# ----------------------------------------------------------------------------- main map
def run_map(family_path=None, kind="rcwa", lam_pump=1292.0, F_grid=None, tau_list=TAU_DEFAULT_FS,
            g_const=G_CONST_DEFAULT, two_zone=True, ce_kind="sommerfeld", te_interp="log", corr=None,
            lam_probe=(1288.0, 1292.0, 1300.0), inv_levels=(600, 1000, 1500, 2000, 3000, 4500, 6000, 8000),
            dt_fs=1.0, A_floor=0.03, do_regression=True, brentq_check=True, verbose=True):
    """Full fluence <-> Te map.  Returns dict (keys documented in the module docstring)."""
    t_start = time.time()
    log = print if verbose else (lambda *a, **k: None)
    fam = load_family(family_path, kind, corr, te_interp)
    Ce = HeatCapacity(ce_kind)
    if F_grid is None:
        F_grid = np.logspace(np.log10(0.01), np.log10(5.0), 40)
    F_grid = np.asarray(F_grid, float)
    models = model_list(tau_list, g_const)
    labels = [m[0] for m in models]
    nM, nF, nP = len(models), F_grid.size, len(lam_probe)
    Afun = fam.A_fn(lam_pump, floor=A_floor)
    T_pump = fam.T_fn(lam_pump)
    T_probe = {L: fam.T_fn(L) for L in lam_probe}
    probes = {"pump": T_pump}
    probes.update({f"probe{L:.0f}": T_probe[L] for L in lam_probe})
    A0 = float(Afun(300.0))
    T0_probe = np.array([float(T_probe[L](300.0)) for L in lam_probe])
    log(f"[run_map] family={kind} {os.path.basename(fam.path)}  lam_pump={lam_pump} nm  A(300K)={A0:.4f}  "
        f"corr={fam.corr}  Ce={ce_kind}  te_interp={te_interp}  dt={dt_fs} fs")
    log("[run_map] A(lam_pump, Te) column: " + " ".join(f"{T:.0f}K:{a:.3f}" for T, a in zip(fam.Te, Afun.col)))

    # quasi-static ladder --------------------------------------------------------------------------
    u_qs = A0 * (F_grid * 10.0) / T_ITO_M                        # J/m^3
    Te_qs = Ce.Te_of_u(u_qs)

    # forward map on the F grid --------------------------------------------------------------------
    R = {}
    for k in ("Te_pk", "t_pk", "Te_avgI", "Te_1ps", "Tl_end", "u_abs", "A_eff", "T_avg_pump",
              "T_avg_pump_2zone", "Te_pk_hot", "Te_pk_cold"):
        R[k] = np.full((nM, nF), np.nan)
    R["above_table"] = np.zeros((nM, nF), bool)
    R["dT_probe_pk"] = np.full((nM, nP, nF), np.nan)
    R["dT_probe_avg"] = np.full((nM, nP, nF), np.nan)
    R["dT_probe_pk_2zone"] = np.full((nM, nP, nF), np.nan)
    traj = {}
    for im, (lab, gm) in enumerate(models):
        t1 = time.time()
        r = integrate_ttm(F_grid, Afun, Ce, gm, probe_fns=probes, dt_fs=dt_fs, Te_table_max=fam.Te[-1])
        for k in ("Te_pk", "t_pk", "Te_avgI", "Te_1ps", "Tl_end", "u_abs", "A_eff", "above_table"):
            R[k][im] = r[k]
        R["T_avg_pump"][im] = r["avg"]["pump"]
        for ip, L in enumerate(lam_probe):
            R["dT_probe_pk"][im, ip] = T_probe[L](r["Te_pk"]) - T0_probe[ip]
            R["dT_probe_avg"][im, ip] = r["avg"][f"probe{L:.0f}"] - T0_probe[ip]
        if two_zone:
            rh = integrate_ttm(F_grid, Afun, Ce, gm, probe_fns=probes, scale=HOT_SCALE, dt_fs=dt_fs)
            rc = integrate_ttm(F_grid, Afun, Ce, gm, probe_fns=probes, scale=COLD_SCALE, dt_fs=dt_fs)
            R["Te_pk_hot"][im], R["Te_pk_cold"][im] = rh["Te_pk"], rc["Te_pk"]
            R["T_avg_pump_2zone"][im] = F_P * rh["avg"]["pump"] + (1 - F_P) * rc["avg"]["pump"]
            for ip, L in enumerate(lam_probe):
                R["dT_probe_pk_2zone"][im, ip] = (F_P * T_probe[L](rh["Te_pk"]) + (1 - F_P) * T_probe[L](rc["Te_pk"])
                                                  - T0_probe[ip])
        # trajectory at F_sat
        rt = integrate_ttm(F_SAT_MJ, Afun, Ce, gm, dt_fs=dt_fs, store_traj=True, traj_every=5)
        traj[lab] = rt
        log(f"[run_map] {lab:14s} forward grid done ({time.time() - t1:.1f} s): Te_pk@F_sat={rt['Te_pk'][0]:.0f} K")
    R["eta_ret"] = R["Te_pk"] / Te_qs[None, :]
    R["eta_u"] = R["u_abs"] / u_qs[None, :]

    # inverse map ----------------------------------------------------------------------------------
    levels = np.asarray(inv_levels, float)
    nL = levels.size
    inv_F_pk = np.full((nM, nL), np.nan)
    inv_F_avg = np.full((nM, nL), np.nan)
    for im, (lab, gm) in enumerate(models):
        t1 = time.time()
        Fp = invert_batched(levels, "pk", Afun, Ce, gm, dt_fs=dt_fs)
        Fa = invert_batched(levels, "avg", Afun, Ce, gm, dt_fs=dt_fs)
        inv_F_pk[im], inv_F_avg[im] = Fp, Fa
        log(f"[run_map] {lab:14s} inverse map done ({time.time() - t1:.1f} s): F_pk=" +
            " ".join("nan" if np.isnan(v) else f"{v:.3f}" for v in Fp))
    inv_F_qs = Ce.u(levels) * T_ITO_M / A0 / 10.0                  # mJ/cm^2
    brentq_text = ""
    if brentq_check:
        # cross-check two entries with scalar brentq (spec: brentq on the 0.005-20 mJ/cm^2 log bracket)
        imc = min(1, nM - 1)
        lab, gm = models[imc]
        chk = []
        for lev in (4500.0, 2000.0):
            il = int(np.argmin(abs(levels - lev)))
            Fb = invert_brentq(levels[il], "pk", Afun, Ce, gm, dt_fs=dt_fs)
            chk.append(f"  {lab} Te_pk={levels[il]:.0f} K: bisection F={inv_F_pk[imc, il]:.4f}  brentq F={Fb:.4f} mJ/cm^2")
        brentq_text = "brentq cross-check of the batched log-bisection:\n" + "\n".join(chk)
        log(brentq_text)

    # texts ---------------------------------------------------------------------------------------
    if do_regression and kind == "rcwa":
        reg_text, reg_rows = regression_check(fam, lam_pump, dt_fs, g_const)
        log(reg_text)
    elif do_regression:
        reg_text, reg_rows = regression_check(load_family(None, "rcwa", None, te_interp), lam_pump, dt_fs, g_const)
        reg_text = "(regression run on the default RCWA family; the c3 reference rows are RCWA-based)\n" + reg_text
        log(reg_text)
    else:
        reg_text, reg_rows = "(regression skipped)", []
    table_text = _inverse_table_text(labels, levels, inv_F_pk, inv_F_avg, inv_F_qs, Ce, A0)
    checks_text = _physics_checks_text(labels, F_grid, R, Te_qs, u_qs, levels, inv_F_pk, Ce, fam, A0)
    obs_text = _observables_text(labels, F_grid, R, lam_probe, T0_probe, lam_pump, float(T_pump(300.0)), two_zone)
    log(table_text)
    log(checks_text)
    log(obs_text)

    meta = dict(family_path=fam.path, family_kind=kind, corr=fam.corr, lam_pump_nm=lam_pump, Ce_kind=ce_kind,
                te_interp=te_interp, t_ITO_m=T_ITO_M, C_L_Jm3K=C_L, F_P=F_P, hot_scale=HOT_SCALE,
                cold_scale=COLD_SCALE, two_zone=two_zone, dt_fs=dt_fs, A_floor=A_floor, A0=A0,
                gamma_e_Jm3K2=GAM_E, n_e_m3=N_E_M3, E_F_eV=E_F_EV, T_F_K=T_F, nJ_per_mJcm2=NJ_PER_MJCM2,
                F_sat_mJcm2=F_SAT_MJ, E_sat_nJ=E_SAT_NJ, E_max_nJ=E_MAX_NJ, fwhm_fs=FWHM * 1e15,
                inverse_method="batched log-bisection (16 iters, 0.005-20 mJ/cm^2) + brentq cross-check",
                runtime_s=time.time() - t_start)
    out = dict(meta=meta, models=labels, tau_list_fs=np.asarray(tau_list, float), g_const=g_const,
               F_grid=F_grid, E_grid_nJ=F_to_nJ(F_grid), lam_pump=lam_pump, lam_probe=np.asarray(lam_probe, float),
               Te_qs=Te_qs, u_qs_Jcm3=u_qs * 1e-6, inv_Te_levels=levels, inv_F_pk=inv_F_pk,
               inv_E_pk=F_to_nJ(inv_F_pk), inv_F_avg=inv_F_avg, inv_E_avg=F_to_nJ(inv_F_avg), inv_F_qs=inv_F_qs,
               family=fam, traj=traj, regression_text=reg_text, regression_rows=reg_rows, checks_text=checks_text,
               table_text=table_text, observables_text=obs_text, brentq_text=brentq_text, T0_probe=T0_probe,
               T0_pump=float(T_pump(300.0)))
    out.update(R)
    out["u_abs_Jcm3"] = R["u_abs"] * 1e-6
    return out


def _inverse_table_text(labels, levels, inv_F_pk, inv_F_avg, inv_F_qs, Ce, A0):
    L = ["=== Inverse map: family Te level -> incident fluence F (mJ/cm^2) [pulse energy nJ = 0.64 F] ===",
         f"Te_pk == level (peak) and <Te>_I == level (intensity-weighted mean); 'qs' = quasi-static ladder with A0={A0:.4f}",
         "--- Te_pk == level ---",
         "level(K) | u_e(level) J/cm3 | F_qs   |" + "".join(f" {lab:>16s}" for lab in labels)]
    for il, lev in enumerate(levels):
        row = f"{lev:8.0f} | {Ce.u(lev) * 1e-6:16.1f} | {inv_F_qs[il]:6.3f} |"
        for im in range(len(labels)):
            v = inv_F_pk[im, il]
            row += f" {'n/a':>16s}" if np.isnan(v) else f" {v:7.3f}({F_to_nJ(v):5.2f}nJ)"
        L.append(row)
    L.append("--- <Te>_I == level ---")
    L.append("level(K) |" + "".join(f" {lab:>16s}" for lab in labels))
    for il, lev in enumerate(levels):
        row = f"{lev:8.0f} |"
        for im in range(len(labels)):
            v = inv_F_avg[im, il]
            row += f" {'n/a':>16s}" if np.isnan(v) else f" {v:7.3f}({F_to_nJ(v):5.2f}nJ)"
        L.append(row)
    L.append(f"E_sat = {E_SAT_NJ} nJ <-> {F_SAT_MJ} mJ/cm^2 ; laser max {E_MAX_NJ:.0f} nJ <-> {nJ_to_F(E_MAX_NJ):.0f} mJ/cm^2 ; "
             "n/a = not reachable within 0.005-20 mJ/cm^2")
    return "\n".join(L)


def _physics_checks_text(labels, F_grid, R, Te_qs, u_qs, levels, inv_F_pk, Ce, fam, A0):
    L = ["=== Physics sanity checks (auto) ==="]
    iS = int(np.argmin(abs(F_grid - F_SAT_MJ)))
    L.append(f"F grid point nearest E_sat: F={F_grid[iS]:.3f} mJ/cm^2 ({F_to_nJ(F_grid[iS]):.3f} nJ); "
             f"ladder Te_qs={Te_qs[iS]:.0f} K, u_qs={u_qs[iS] * 1e-6:.1f} J/cm3")
    L.append(f"T_F = {T_F:.0f} K (E_F {E_F_EV:.3f} eV, n_e {N_E_M3 * 1e-6:.3e} cm^-3); u_e(8000 K, {Ce.kind}) = "
             f"{Ce.u(8000.0) * 1e-6:.1f} J/cm3 (label ref ~{U_8000_LABEL_JCM3:.0f})")
    for im, lab in enumerate(labels):
        Tpk, u, dTl = R["Te_pk"][im, iS], R["u_abs"][im, iS] * 1e-6, R["u_abs"][im, iS] / C_L
        Tpk_max, u_max = R["Te_pk"][im, -1], R["u_abs"][im, -1] * 1e-6
        flag_TF = "OK (< T_F)" if Tpk <= T_F else "EXCEEDS T_F -> Sommerfeld C_e invalid there"
        n_above = int(R["above_table"][im].sum())
        L.append(f"[{lab}] @F_sat: Te_pk={Tpk:.0f} K {flag_TF}; <Te>_I={R['Te_avgI'][im, iS]:.0f} K; u_abs={u:.1f} J/cm3 "
                 f"(A_eff={R['A_eff'][im, iS]:.3f} vs A0={A0:.3f}); dT_l={dTl:.1f} K (T_l end {R['Tl_end'][im, iS]:.0f} K); "
                 f"eta_ret=Te_pk/Te_qs={R['eta_ret'][im, iS]:.2f} (Te_qs/Te_pk={1 / R['eta_ret'][im, iS]:.2f}), eta_u={R['eta_u'][im, iS]:.2f}")
        L.append(f"          @F_max={F_grid[-1]:.2f}: Te_pk={Tpk_max:.0f} K ({'<' if Tpk_max <= T_F else '>'} T_F); u_abs={u_max:.1f} J/cm3; "
                 f"dT_l={R['u_abs'][im, -1] / C_L:.0f} K; {n_above}/{F_grid.size} grid points exceed the table max Te={fam.Te[-1]:.0f} K (A held)")
        F8 = inv_F_pk[im, int(np.argmin(abs(levels - 8000)))]
        hi_lev = levels >= 3000
        F_lo, F_hi = np.nanmin(inv_F_pk[im, hi_lev]), np.nanmax(inv_F_pk[im, hi_lev])
        in_lit = F_LIT_RANGE[0] <= F_lo and F_hi <= F_LIT_RANGE[1]
        L.append(f"          F for Te_pk=8000 K: {'n/a' if np.isnan(F8) else f'{F8:.2f} mJ/cm^2 ({F_to_nJ(F8):.2f} nJ)'}"
                 f"{'' if np.isnan(F8) else f' = {F8 / F_SAT_MJ:.1f} x F_sat'}; "
                 f"F for the 3000-8000 K labels {F_lo:.2f}-{F_hi:.2f} mJ/cm^2 -> {'inside' if in_lit else 'partly outside'} "
                 f"literature ENZ-ITO range {F_LIT_RANGE[0]}-{F_LIT_RANGE[1]} mJ/cm^2")
    L.append("ASSUM: lattice-damage criterion not quantified here; ITO crystallisation / thermal damage is usually associated with "
             "several-hundred-K lattice rises (single-shot dT_l above is per pulse, before ns-scale cooling into the glass); "
             "kHz accumulation is negligible (c3: <1 K steady state).")
    L.append("ASSUM: two-zone hot-spot factors x1.74 (under pillar, area fraction %.3f) / x0.44 (trench) are field-integral proxies; "
             "area-weighted mean = %.3f (7 %% energy deficit accepted, no renormalisation); transmission is evaluated as an area-weighted "
             "mixture of T(lam, Te_hot) and T(lam, Te_cold), which ignores the collective nature of the resonance."
             % (F_P, F_P * HOT_SCALE + (1 - F_P) * COLD_SCALE))
    L.append("ASSUM: Te > table max (8000 K) holds the last A and T values; C_e Sommerfeld is only valid for Te << T_F.")
    return "\n".join(L)


def _observables_text(labels, F_grid, R, lam_probe, T0_probe, lam_pump, T0_pump, two_zone):
    L = ["=== Observables at selected fluences (pump-probe dT at peak / time-averaged; single-beam T_avg) ==="]
    picks = [0.1, 0.25, 0.5, F_SAT_MJ, 1.0, 2.0, 5.0]
    L.append(f"T(300 K): pump {lam_pump:.0f} nm = {T0_pump:.4f}; probes " +
             ", ".join(f"{L_:.0f}={t:.4f}" for L_, t in zip(lam_probe, T0_probe)))
    for im, lab in enumerate(labels):
        L.append(f"[{lab}]  F(mJ/cm2)  E(nJ) | Te_pk  <Te>_I | " + " ".join(f"dTpk{L_:.0f}" for L_ in lam_probe) + " | " +
                 " ".join(f"dTavg{L_:.0f}" for L_ in lam_probe) + " | T_avg(pump) uniform" + ("  2-zone  diff" if two_zone else ""))
        for Fp in picks:
            i = int(np.argmin(abs(F_grid - Fp)))
            if abs(F_grid[i] / Fp - 1) > 0.08:
                continue
            row = f"   {F_grid[i]:8.3f} {F_to_nJ(F_grid[i]):6.3f} | {R['Te_pk'][im, i]:5.0f} {R['Te_avgI'][im, i]:6.0f} | "
            row += " ".join(f"{R['dT_probe_pk'][im, ip, i]:+8.4f}" for ip in range(len(lam_probe))) + " | "
            row += " ".join(f"{R['dT_probe_avg'][im, ip, i]:+9.4f}" for ip in range(len(lam_probe))) + " | "
            row += f"{R['T_avg_pump'][im, i]:8.4f}"
            if two_zone:
                row += f"            {R['T_avg_pump_2zone'][im, i]:7.4f} {R['T_avg_pump_2zone'][im, i] - R['T_avg_pump'][im, i]:+7.4f}"
            L.append(row)
    return "\n".join(L)


# ----------------------------------------------------------------------------- saving / plotting
def save_npz(res, path):
    fam = res["family"]
    models = res["models"]
    trj = res["traj"]
    t_ref = trj[models[0]]["traj_t"]
    d = dict(meta_json=json.dumps(res["meta"]), models=np.array(models), tau_list_fs=res["tau_list_fs"],
             g_const_Wm3K=float(res["g_const"] or 0.0), F_grid_mJcm2=res["F_grid"], E_grid_nJ=res["E_grid_nJ"],
             lam_pump_nm=float(res["lam_pump"]), lam_probe_nm=res["lam_probe"],
             Te_pk=res["Te_pk"], t_pk_fs=res["t_pk"] * 1e15, Te_avgI=res["Te_avgI"], Te_1ps=res["Te_1ps"],
             Tl_end=res["Tl_end"], u_abs_Jcm3=res["u_abs_Jcm3"], A_eff=res["A_eff"], Te_qs=res["Te_qs"],
             u_qs_Jcm3=res["u_qs_Jcm3"], eta_ret=res["eta_ret"], eta_u=res["eta_u"], above_table=res["above_table"],
             dT_probe_pk=res["dT_probe_pk"], dT_probe_avg=res["dT_probe_avg"], T_avg_pump=res["T_avg_pump"],
             T_avg_pump_2zone=res["T_avg_pump_2zone"], Te_pk_hot=res["Te_pk_hot"], Te_pk_cold=res["Te_pk_cold"],
             dT_probe_pk_2zone=res["dT_probe_pk_2zone"], inv_Te_levels=res["inv_Te_levels"], inv_F_pk=res["inv_F_pk"],
             inv_E_pk=res["inv_E_pk"], inv_F_avg=res["inv_F_avg"], inv_E_avg=res["inv_E_avg"], inv_F_qs=res["inv_F_qs"],
             family_Te=fam.Te, family_lam=fam.lam, family_T=fam.T, family_A=fam.A,
             traj_t_fs=t_ref * 1e15, traj_Te=np.array([trj[m]["traj_Te"][0] for m in models]),
             traj_Tl=np.array([trj[m]["traj_Tl"][0] for m in models]),
             regression_text=res["regression_text"], checks_text=res["checks_text"], table_text=res["table_text"],
             observables_text=res["observables_text"], brentq_text=res["brentq_text"])
    np.savez(path, **d)


def save_txt(res, path):
    m = res["meta"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("TK-BIC / ENZ-ITO 23 nm: TTM fluence <-> Te map (_te_fluence_map.py)\n")
        f.write(f"family={m['family_kind']} ({m['family_path']}), corr={m['corr']}, lam_pump={m['lam_pump_nm']} nm, "
                f"C_e={m['Ce_kind']}, Te interp={m['te_interp']}, RK4 dt={m['dt_fs']} fs, window -1..+3 ps\n")
        f.write(f"t_ITO=23 nm, C_L={C_L:.1e} J/m3/K, pulse FWHM 280 fs, spot (8 um)^2 -> E[nJ]=0.64 F[mJ/cm2]; "
                f"gamma_e={GAM_E:.3f} J/m3/K2, n_e={N_E_M3 * 1e-6:.3e} cm^-3, E_F={E_F_EV:.3f} eV, T_F={T_F:.0f} K\n")
        f.write(f"runtime {m['runtime_s']:.0f} s\n\n")
        f.write(res["regression_text"] + "\n\n")
        f.write(res["brentq_text"] + "\n\n")
        f.write(res["table_text"] + "\n\n")
        f.write(res["checks_text"] + "\n\n")
        f.write(res["observables_text"] + "\n")


def plot_map(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    F = res["F_grid"]
    models = res["models"]
    tau_fs = res["tau_list_fs"]
    nT = len(tau_fs)
    itau = list(range(nT))
    ig = nT if len(models) > nT else None
    jm = int(np.argmin(abs(tau_fs - 450)))
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    grey, ink = "#7a7975", "#0b0b0b"
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 9.8))
    fig.subplots_adjust(hspace=0.45, wspace=0.28, top=0.9, bottom=0.07, left=0.07, right=0.98)
    lam_pump = res["lam_pump"]

    def top_axis(ax):
        sec = ax.secondary_xaxis("top", functions=(lambda x: x * NJ_PER_MJCM2, lambda x: x / NJ_PER_MJCM2))
        sec.set_xlabel("pulse energy (nJ), (8 um)$^2$ spot", fontsize=8.5)
        sec.tick_params(labelsize=8)
        return sec

    def esat(ax):
        ax.axvline(F_SAT_MJ, color=grey, lw=1, ls="-.")

    # (a) Te_pk & <Te>_I vs F ----------------------------------------------------------------
    ax = axs[0, 0]
    Tpk_band = res["Te_pk"][itau]
    Tav_band = res["Te_avgI"][itau]
    ax.fill_between(F, Tpk_band.min(0), Tpk_band.max(0), color=pal[0], alpha=0.22, lw=0,
                    label=f"$T_e^{{pk}}$, $\\tau_{{ep}}$ {tau_fs.min():.0f}-{tau_fs.max():.0f} fs")
    for j in itau:
        ax.plot(F, res["Te_pk"][j], color=pal[0], lw=1, alpha=0.8)
    ax.fill_between(F, Tav_band.min(0), Tav_band.max(0), color=pal[2], alpha=0.22, lw=0,
                    label=r"$\langle T_e\rangle_I$, same band")
    for j in itau:
        ax.plot(F, res["Te_avgI"][j], color=pal[2], lw=1, alpha=0.8)
    if ig is not None:
        ax.plot(F, res["Te_pk"][ig], color=pal[1], lw=1.8, ls=":", label=f"$T_e^{{pk}}$, const g={res['g_const']:.1e}")
        ax.plot(F, res["Te_avgI"][ig], color=pal[1], lw=1.2, ls=":", alpha=0.7)
    ax.plot(F, res["Te_qs"], color=grey, lw=1.5, ls="--", label="quasi-static ladder (no e-ph)")
    ax.axhline(T_F, color=grey, lw=0.8, ls=":")
    ax.text(F[0] * 1.05, T_F * 1.02, f"$T_F$ = {T_F:.0f} K", fontsize=8, color=grey)
    esat(ax)
    ax.text(F_SAT_MJ * 1.05, 350, "$E_{sat}$", fontsize=8, color=grey)
    for lev in res["inv_Te_levels"]:
        ax.axhline(lev, color=grey, lw=0.4, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("incident fluence F (mJ/cm$^2$)")
    ax.set_ylabel("electron temperature (K)")
    ax.set_ylim(0, max(12000, np.nanmax(res["Te_qs"]) * 1.02))
    ax.set_title(f"(a) TTM peak / mean $T_e$ vs fluence, $\\lambda_{{pump}}$ = {lam_pump:.0f} nm", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    top_axis(ax)

    # (b) inverse map ----------------------------------------------------------------------
    ax = axs[0, 1]
    levels = res["inv_Te_levels"]
    Fpk = res["inv_F_pk"]
    Fav = res["inv_F_avg"]
    y = np.arange(len(levels))
    for il, lev in enumerate(levels):
        a, b = np.nanmin(Fpk[itau, il]), np.nanmax(Fpk[itau, il])
        if np.isfinite(a):
            ax.plot([a, b], [y[il], y[il]], color=pal[0], lw=6, solid_capstyle="butt", alpha=0.85,
                    label="$T_e^{pk}$ = level, $\\tau_{ep}$ band" if il == 0 else None)
            ax.text(b * 1.12, y[il] + 0.05, f"{a:.2f}-{b:.2f}", fontsize=7.5, color=ink, va="center")
        a2, b2 = np.nanmin(Fav[itau, il]), np.nanmax(Fav[itau, il])
        if np.isfinite(a2):
            ax.plot([a2, b2], [y[il] - 0.22, y[il] - 0.22], color=pal[2], lw=3.5, alpha=0.85,
                    label=r"$\langle T_e\rangle_I$ = level" if il == 0 else None)
        if ig is not None and np.isfinite(Fpk[ig, il]):
            ax.plot(Fpk[ig, il], y[il] + 0.22, marker="D", ms=5, color=pal[1], ls="none",
                    label="$T_e^{pk}$, const g" if il == 0 else None)
    ax.plot(res["inv_F_qs"], y, marker="|", ms=12, color=grey, ls="none", mew=1.5, label="quasi-static ladder")
    esat(ax)
    ax.axvline(nJ_to_F(E_MAX_NJ), color=grey, lw=0.8, ls=":")
    ax.text(F_SAT_MJ * 1.05, len(levels) - 0.6, "$E_{sat}$ 0.464 nJ", fontsize=8, color=grey)
    ax.text(nJ_to_F(E_MAX_NJ) * 0.9, 0.2, "laser max 700 nJ", fontsize=7.5, color=grey, rotation=90, ha="right", va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{lev:.0f} K" for lev in levels])
    ax.set_xscale("log")
    ax.set_xlim(0.004, 2000)
    ax.set_ylim(-0.7, len(levels) - 0.3)
    ax.set_xlabel("incident fluence F (mJ/cm$^2$)")
    ax.set_ylabel("family $T_e$ level")
    ax.set_title("(b) inverse map: spectral-family $T_e$ label -> fluence", fontsize=10, loc="left")
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    top_axis(ax)

    # (c) pump-probe dT --------------------------------------------------------------------
    ax = axs[1, 0]
    for ip, L_ in enumerate(res["lam_probe"]):
        band = res["dT_probe_pk"][itau, ip]
        ax.fill_between(F, band.min(0), band.max(0), color=pal[ip], alpha=0.25, lw=0)
        ax.plot(F, res["dT_probe_pk"][jm, ip], color=pal[ip], lw=1.8, label=f"probe {L_:.0f} nm, at peak")
        ax.plot(F, res["dT_probe_avg"][jm, ip], color=pal[ip], lw=1.2, ls="--", label=f"probe {L_:.0f} nm, time-avg")
    ax.axhline(0, color=grey, lw=0.6)
    esat(ax)
    ax.set_xscale("log")
    ax.set_xlabel("pump fluence F (mJ/cm$^2$)")
    ax.set_ylabel("$\\Delta T$ = T($\\lambda_{probe}$, $T_e$) - T(300 K)")
    ax.set_title(f"(c) pump-probe transmission change (band: $\\tau_{{ep}}$; line: {tau_fs[jm]:.0f} fs)", fontsize=10, loc="left")
    ax.legend(fontsize=7.5, loc="best", frameon=False, ncol=1)
    top_axis(ax)

    # (d) single-beam T_avg ----------------------------------------------------------------
    ax = axs[1, 1]
    band = res["T_avg_pump"][itau]
    ax.fill_between(F, band.min(0), band.max(0), color=pal[0], alpha=0.25, lw=0, label="uniform, $\\tau_{ep}$ band")
    ax.plot(F, res["T_avg_pump"][jm], color=pal[0], lw=1.8)
    if np.isfinite(res["T_avg_pump_2zone"]).any():
        band2 = res["T_avg_pump_2zone"][itau]
        ax.fill_between(F, band2.min(0), band2.max(0), color=pal[1], alpha=0.25, lw=0, label="two-zone hot/cold (ASSUM)")
        ax.plot(F, res["T_avg_pump_2zone"][jm], color=pal[1], lw=1.8)
    if ig is not None:
        ax.plot(F, res["T_avg_pump"][ig], color=pal[0], lw=1.4, ls=":", label="uniform, const g")
    ax.axhline(res["T0_pump"], color=grey, lw=0.8, ls="--")
    ax.text(F[0] * 1.05, res["T0_pump"] + 0.005, f"T(300 K) = {res['T0_pump']:.3f}", fontsize=8, color=grey)
    esat(ax)
    ax.set_xscale("log")
    ax.set_xlabel("incident fluence F (mJ/cm$^2$)")
    ax.set_ylabel(f"pulse-averaged transmission T({lam_pump:.0f} nm)")
    ax.set_title("(d) single-beam self-action $T_{avg}$ (A and T self-consistent on $T_e(t)$)", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="best", frameon=False)
    top_axis(ax)

    m = res["meta"]
    fig.suptitle(f"TK-BIC ENZ-ITO 23 nm: TTM fluence <-> Te map  [{m['family_kind'].upper()} family, corr {m['corr']}, "
                 f"C_e {m['Ce_kind']}, 280 fs, Te interp {m['te_interp']}]", fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="TTM fluence <-> Te map for the TK-BIC ENZ-ITO metasurface")
    ap.add_argument("--family", choices=["rcwa", "fdtd"], default="rcwa")
    ap.add_argument("--path", default=None, help="family npz (default per --family)")
    ap.add_argument("--lam-pump", type=float, default=1292.0)
    ap.add_argument("--lam-probe", default="1288,1292,1300")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory (or file prefix without extension)")
    ap.add_argument("--name", default="tkbic_te_fluence_map")
    ap.add_argument("--ce", choices=["sommerfeld", "min", "fd", "kane"], default="sommerfeld")
    ap.add_argument("--te-interp", choices=["log", "linear"], default="log")
    ap.add_argument("--corr", type=float, default=None, help="absorption correction (rcwa default 0.7619, fdtd 1.0)")
    ap.add_argument("--tau", default="300,450,600,1000", help="tau_ep list in fs")
    ap.add_argument("--g-const", type=float, default=G_CONST_DEFAULT, help="constant g (W/m^3/K); 0 disables")
    ap.add_argument("--nF", type=int, default=40)
    ap.add_argument("--F-min", type=float, default=0.01)
    ap.add_argument("--F-max", type=float, default=5.0)
    ap.add_argument("--dt-fs", type=float, default=1.0)
    ap.add_argument("--no-two-zone", action="store_true")
    ap.add_argument("--no-regression", action="store_true")
    ap.add_argument("--no-brentq-check", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)

    F_grid = np.logspace(np.log10(a.F_min), np.log10(a.F_max), a.nF)
    tau = [float(x) for x in a.tau.split(",") if x.strip()]
    probes = [float(x) for x in a.lam_probe.split(",") if x.strip()]
    res = run_map(a.path, a.family, a.lam_pump, F_grid, tau, a.g_const if a.g_const > 0 else None,
                  two_zone=not a.no_two_zone, ce_kind=a.ce, te_interp=a.te_interp, corr=a.corr,
                  lam_probe=tuple(probes), dt_fs=a.dt_fs, do_regression=not a.no_regression,
                  brentq_check=not a.no_brentq_check)
    if os.path.isdir(a.out) or a.out.endswith(("/", "\\")):
        os.makedirs(a.out, exist_ok=True)
        prefix = os.path.join(a.out, a.name)
    else:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        prefix = a.out
    save_npz(res, prefix + ".npz")
    save_txt(res, prefix + ".txt")
    if not a.no_plot:
        plot_map(res, prefix + ".png")
    print(f"[saved] {prefix}.npz / .txt" + ("" if a.no_plot else " / .png"))
    return res


if __name__ == "__main__":
    main()
