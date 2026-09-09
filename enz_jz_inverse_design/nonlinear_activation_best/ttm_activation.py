"""Two-temperature-model (TTM) + 150-fs pulse coupling for the FROZEN best design,
using the RCWA lookup table T/R/A/Fx/Fy/Fz/Ftot(lambda, Te) of build_lookup.py and the
Lane-B electron thermodynamics of ito_nonlinear.py.

    dU_e/dt = A(lambda_op, T_e) I(t) / d_ITO - G (T_e - T_l),   C_l dT_l/dt = G (T_e - T_l)
    I(t) = I_peak exp(-4 ln2 t^2 / tau^2),  tau = 150 fs,  T_e = T_e(U_e) from the Kane-band U(T_e)

Quasi-static optics (the RCWA response at the instantaneous T_e), heating by the TOTAL
ITO absorption A = Fx+Fy+Fz = 1-R-T, pulse-averaged observables
    <Q> = int Q(T_e(t)) I(t) dt / int I(t) dt   (transmitted / reflected / absorbed energy fractions).

    python ttm_activation.py [--lookup outputs/nonlinear_best/lookup_order7.npz] [--G 1.3e17] [--tag base]
"""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ito_nonlinear as nl          # noqa: E402

OUT = HERE / "outputs" / "nonlinear_best"
D_ITO = 23e-9                        # m
TAU = 150e-15                        # s (FWHM)
P_NM = 825.0
TE_MAX_MODEL, TE_TRUST = 8000.0, 6000.0
KEYS = ("T", "R", "A", "Fz", "Ftot", "Fx", "Fy", "mean_Ez2")


def pulse_shape(t):
    return np.exp(-4 * np.log(2) * t ** 2 / TAU ** 2)


def e_cell_from_I(I_peak_Wcm2):
    """Incident pulse energy per unit cell [J] for uniform illumination of one P x P cell."""
    return I_peak_Wcm2 * 1e4 * (P_NM * 1e-9) ** 2 * TAU * np.sqrt(np.pi / (4 * np.log(2)))


class Lookup:
    def __init__(self, path=None, *, Te=None, lam=None, tab=None, eps=None, order=0):
        if path is not None:
            z = np.load(path)
            self.Te, self.lam = z["Te"], z["lam"]
            self.tab = {k: z[k] for k in KEYS if k in z}
            self.eps = z["eps"]
            self.order = int(z["order"][0])
        else:                                   # synthetic table (e.g. a fitted TCMT evaluated on the same grid)
            self.Te, self.lam, self.tab, self.eps, self.order = np.asarray(Te, float), np.asarray(lam, float), dict(tab), eps, order
        for k in KEYS:
            self.tab.setdefault(k, np.zeros((len(self.Te), len(self.lam))))
        self._col = {}                                                                   # per-(key, j) splines (fast path)
        if len(self.Te) >= 2:
            self.spl = {k: CubicSpline(self.Te, self.tab[k], axis=0) for k in KEYS}    # Q(Te, all lambda)
            self.eps_spl_re = CubicSpline(self.Te, self.eps.real, axis=0)
            self.eps_spl_im = CubicSpline(self.Te, self.eps.imag, axis=0)
        else:                                                                            # single-Te table (cold scans)
            self.spl = {k: (lambda Te, k=k: np.broadcast_to(self.tab[k][0], np.shape(Te) + self.tab[k][0].shape)) for k in KEYS}
            self.eps_spl_re = lambda Te: np.broadcast_to(self.eps[0].real, np.shape(Te) + self.eps[0].shape)
            self.eps_spl_im = lambda Te: np.broadcast_to(self.eps[0].imag, np.shape(Te) + self.eps[0].shape)

    def at(self, key, Te, j):
        """Q(Te) at lambda index j, vectorized in Te (clipped to the table range: Te above the
        last tabulated value is NOT extrapolated; such states are flagged by the caller)."""
        s = self._col.get((key, j))
        if s is None:
            if len(self.Te) >= 2:
                s = self._col[(key, j)] = CubicSpline(self.Te, self.tab[key][:, j])
            else:
                s = self._col[(key, j)] = (lambda Te, v=float(self.tab[key][0, j]): np.full(np.shape(Te), v))
        return s(np.clip(Te, self.Te[0], self.Te[-1]))

    def col(self, key, Te):
        return self.spl[key](np.clip(Te, self.Te[0], self.Te[-1]))


class Thermo:
    """U(Te) table and its inverse for the Kane band (Lane B)."""
    def __init__(self, model, Te_max=12000.0):
        self.model = model
        Tg = np.concatenate([np.linspace(300, 2000, 60), np.linspace(2050, Te_max, 120)])
        U = np.array([model.state(T)["U_J_m3"] for T in Tg])
        self.Tg, self.Ug = Tg, U
        self.U_of_T = interp1d(Tg, U, kind="cubic")
        self.T_of_U = interp1d(U, Tg, kind="cubic", bounds_error=False, fill_value=(300.0, Te_max))
        self.U300 = float(U[0])
        self.Ce = interp1d(Tg, np.gradient(U, Tg), kind="cubic")


def integrate(lk, th, j, I_peaks_Wcm2, G, Cl, dt=0.5e-15, t_span=(-3 * TAU, 8 * TAU), traces=False):
    """Vectorized over I_peaks for one operating wavelength index j.  RK4 in (U_e, T_l).
    Returns pulse-averaged observables, peak Te, energy-balance residual, optional traces."""
    I0 = np.asarray(I_peaks_Wcm2, float) * 1e4                      # W/m^2
    t = np.arange(t_span[0], t_span[1] + dt / 2, dt)
    n = len(I0)
    U = np.full(n, th.U300); Tl = np.full(n, 300.0)
    Te = np.full(n, 300.0)
    wI = pulse_shape(t)
    acc = {k: np.zeros(n) for k in ("T", "R", "A", "Fz", "Fx", "Fy", "Ftot")}
    absorbed = np.zeros(n)                                             # int A I dt  [J/m^2]
    Te_peak = np.zeros(n)
    tr = {"t": t, "Te": [], "Tl": [], "T": [], "A": [], "I": []} if traces else None

    def rhs(U_, Tl_, I_):
        Te_ = np.asarray(th.T_of_U(U_))
        A_ = lk.at("A", Te_, j)
        dU = A_ * I_ / D_ITO - G * (Te_ - Tl_)
        dTl = G * (Te_ - Tl_) / Cl
        return dU, dTl, Te_, A_
    for i, ti in enumerate(t):
        I_now = I0 * wI[i]
        k1 = rhs(U, Tl, I_now)
        Te = k1[2]; A_now = k1[3]
        # accumulate pulse-averaged observables at the current state (left Riemann; dt small)
        for k in acc:
            acc[k] += lk.at(k, Te, j) * I_now * dt
        absorbed += A_now * I_now * dt
        Te_peak = np.maximum(Te_peak, Te)
        if traces:
            tr["Te"].append(Te.copy()); tr["Tl"].append(Tl.copy()); tr["T"].append(lk.at("T", Te, j)); tr["A"].append(A_now.copy()); tr["I"].append(I_now.copy())
        I_mid = I0 * pulse_shape(ti + dt / 2); I_next = I0 * pulse_shape(ti + dt)
        k2 = rhs(U + 0.5 * dt * k1[0], Tl + 0.5 * dt * k1[1], I_mid)
        k3 = rhs(U + 0.5 * dt * k2[0], Tl + 0.5 * dt * k2[1], I_mid)
        k4 = rhs(U + dt * k3[0], Tl + dt * k3[1], I_next)
        U = U + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        Tl = Tl + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    fluence = I0 * TAU * np.sqrt(np.pi / (4 * np.log(2)))              # J/m^2 (analytic)
    fl_num = np.sum(I0[:, None] * wI[None, :], axis=1) * dt
    out = {k: acc[k] / fl_num for k in acc}
    # TTM energy balance: d/dt (U_e + C_l T_l) d_ITO = A I  ->  stored+transferred energy == absorbed fluence
    stored = (U - th.U300 + Cl * (Tl - 300.0)) * D_ITO
    out.update(Te_peak=Te_peak, Te_end=np.asarray(th.T_of_U(U)), Tl_end=Tl, absorbed_J_m2=absorbed, stored_J_m2=stored,
               energy_resid=(stored - absorbed) / np.maximum(absorbed, 1e-30), fluence_J_m2=fluence, fluence_num_J_m2=fl_num)
    if traces:
        for k in ("Te", "Tl", "T", "A", "I"):
            tr[k] = np.array(tr[k])
        out["traces"] = tr
    return out


def calibrate_G(th, tau_ep=150e-15, Te_ref=2000.0):
    return float(th.Ce(Te_ref) / tau_ep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookup", default=str(OUT / "lookup_order7.npz"))
    ap.add_argument("--G", type=float, default=None, help="W m^-3 K^-1 (default: calibrated, Ce(2000 K)/150 fs)")
    ap.add_argument("--G-scale", type=float, default=1.0)
    ap.add_argument("--Cl", type=float, default=2.4e6)
    ap.add_argument("--dt", type=float, default=0.5e-15)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--nI", type=int, default=61)
    ap.add_argument("--Imin", type=float, default=1e5)
    ap.add_argument("--Imax", type=float, default=1e11)
    a = ap.parse_args()
    t0 = time.time()
    lk = Lookup(a.lookup)
    model = nl.ITOHot()
    th = Thermo(model)
    G = (a.G if a.G else calibrate_G(th)) * a.G_scale
    I_peaks = np.logspace(np.log10(a.Imin), np.log10(a.Imax), a.nI)
    E_cell = e_cell_from_I(I_peaks)
    lams = lk.lam
    res = {k: np.zeros((len(lams), len(I_peaks))) for k in ("T", "R", "A", "Fz", "Fx", "Fy", "Ftot", "Te_peak", "Te_end", "Tl_end", "energy_resid", "absorbed_J_m2")}
    log = open(OUT / f"ttm_{a.tag}.log", "w")
    print(f"[ttm {a.tag}] G = {G:.3e} W/m3/K (tau_ep(2000 K) = {th.Ce(2000)/G*1e15:.0f} fs), Cl = {a.Cl:.2e}, dt = {a.dt*1e15:.2f} fs, {len(lams)} lambdas x {len(I_peaks)} intensities", flush=True)
    for j, lam in enumerate(lams):
        o = integrate(lk, th, j, I_peaks, G, a.Cl, dt=a.dt)
        for k in res:
            res[k][j] = o[k]
        if j % 10 == 0:
            msg = f"  lam {lam:.0f}: T_cold={o['T'][0]:.4f} T(1e9)={o['T'][np.argmin(abs(I_peaks-1e9))]:.4f} T(1e10)={o['T'][np.argmin(abs(I_peaks-1e10))]:.4f} Te_peak(1e10)={o['Te_peak'][np.argmin(abs(I_peaks-1e10))]:.0f} K  max|energy resid|={np.abs(o['energy_resid']).max():.1e}  {time.time()-t0:.0f}s"
            print(msg, flush=True); log.write(msg + "\n")
    valid_model = res["Te_peak"] <= TE_MAX_MODEL
    valid_trust = res["Te_peak"] <= TE_TRUST
    # eps at peak Te (at each lambda_op)
    d_eps_re = np.zeros_like(res["T"]); d_eps_im = np.zeros_like(res["T"])
    for j in range(len(lams)):
        e_re = lk.eps_spl_re(np.clip(res["Te_peak"][j], 300, TE_MAX_MODEL))[:, j]
        e_im = lk.eps_spl_im(np.clip(res["Te_peak"][j], 300, TE_MAX_MODEL))[:, j]
        d_eps_re[j] = e_re - lk.eps[0, j].real; d_eps_im[j] = e_im - lk.eps[0, j].imag
    meta = dict(tag=a.tag, G_W_m3_K=G, tau_ep_2000K_fs=float(th.Ce(2000) / G * 1e15), Cl_J_m3_K=a.Cl, dt_fs=a.dt * 1e15,
                tau_fwhm_fs=TAU * 1e15, P_nm=P_NM, d_ITO_nm=D_ITO * 1e9, lookup=str(a.lookup), order=int(np.load(a.lookup)["order"][0]),
                Te_max_model=TE_MAX_MODEL, Te_trust=TE_TRUST, wall_s=time.time() - t0,
                E_cell_check="E_cell = I_peak P^2 tau sqrt(pi/(4 ln2)); e.g. 1e9 W/cm2 -> %.3e J" % e_cell_from_I(1e9))
    for k, v in res.items():
        np.savez_compressed(OUT / f"heatmap_{k}_{a.tag}.npz", lam=lams, I_peak_Wcm2=I_peaks, E_cell_J=E_cell, value=v,
                            valid_model=valid_model, valid_trust=valid_trust, **{kk: vv for kk, vv in meta.items() if isinstance(vv, (int, float, str))})
    np.savez_compressed(OUT / f"heatmap_deps_{a.tag}.npz", lam=lams, I_peak_Wcm2=I_peaks, E_cell_J=E_cell, d_eps_re=d_eps_re, d_eps_im=d_eps_im,
                        valid_model=valid_model, valid_trust=valid_trust)
    if a.tag == "base":
        for k in ("T", "R", "A", "Fz"):
            np.savez_compressed(OUT / f"heatmap_{k}.npz", lam=lams, I_peak_Wcm2=I_peaks, E_cell_J=E_cell, value=res[k], Te_peak=res["Te_peak"],
                                valid_model=valid_model, valid_trust=valid_trust, G_W_m3_K=G)
    with open(OUT / f"ttm_meta_{a.tag}.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[ttm {a.tag}] done in {time.time()-t0:.0f} s; max |energy resid| = {np.abs(res['energy_resid']).max():.2e}; "
          f"Te_peak range {res['Te_peak'].min():.0f}-{res['Te_peak'].max():.0f} K; valid(<=8000K) fraction {valid_model.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
