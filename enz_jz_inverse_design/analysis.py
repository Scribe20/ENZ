"""Post-hoc certification and physics analysis for the JZ F_z campaign.

Reuses the repository's validated methodologies (function-level ports with
attribution; the historical packages are not imported so that this package
depends only on the NEW materials):
  * spectra          F_z, F_x, F_y, A, R, T, r_xx, t_xx vs wavelength with the
                     supplied dispersion of ALL materials at every wavelength
  * poles            AAA rational fit of r(omega) and t(omega) on the real axis,
                     poles accepted only if found in BOTH r and t, damped, with a
                     significant Lorentzian peak contribution
                     (enz_absorption_campaign/pole_rt.py, enz_highq_driven_ez_audit/poles.py)
  * loss scaling     gamma(s) = gamma_rad + s*gamma_nr on the field-overlap-tracked
                     branch, s = Im(eps_ITO) scale in {1, .5, .25, .1, .03, 0}
                     -> Q_rad, Q_nr, gamma_rad/gamma_nr (critical coupling test)
  * mode identity    same geometry with ITO removed / lossless ITO
  * fabrication      fill, min feature / min gap (morphological opening), connected
                     components, boundary contact, erosion/dilation, h and P sensitivity
  * convergence      F_z vs Fourier order and z-sampling for the hard-binary design
"""

import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from scipy.interpolate import AAA

import config
import forward as fwd
import materials as mat

C_NM_FS = 299.792458          # nm/fs
GEO = config.GEO_DTYPE

# a-Si:H data end at 1400 nm and ITO at 1675 nm; the supplied glass covers
# 191-1689 nm.  Spectra therefore stop at 1400 nm (no extrapolation).
LAM_MIN, LAM_MAX = 1000.0, 1400.0


def lam_grid(lo=1100.0, hi=LAM_MAX, step=2.0, avoid=None):
    g = np.arange(lo, hi + 1e-9, step)
    if avoid is not None:                      # never hit eps_ITO = 0 exactly (lossless runs)
        g = g[np.abs(g - avoid) > 0.05]
    return g


# ---------------------------------------------------------------------------
# spectra
# ---------------------------------------------------------------------------
def spectrum(rho, P, h, lams, order=config.ORDER_FULL, *, with_ito=True, s=1.0,
             n_z=config.Z_SAMPLES_ITO, log=None, want_r_t=True):
    rows = []
    for i, lam in enumerate(lams):
        with torch.no_grad():
            sim = fwd.build_sim(rho, P, h, float(lam), order, with_ito=with_ito, ito_loss_scale=s)
            R, T = fwd.rt_all_orders(sim)
            row = dict(lam=float(lam), R=float(R), T=float(T), A=float(1 - R - T))
            if with_ito:
                d = fwd.loss_components(sim, n_z=n_z)
                row.update(Fz=float(d["Fz"]), Fx=float(d["Fx"]), Fy=float(d["Fy"]), Ftot=float(d["Ftot"]),
                           mean_Ez2=float(d["mean_Ez2"]))
            else:
                # no ITO: the field at the same height is the a-Si/glass interface
                row.update(Fz=np.nan, Fx=np.nan, Fy=np.nan, Ftot=np.nan, mean_Ez2=np.nan)
            if want_r_t:
                r, t = fwd.specular_rt_amplitudes(sim)
                row.update(r_re=r.real, r_im=r.imag, t_re=t.real, t_im=t.imag)
        rows.append(row)
        if log and (i % 20 == 0):
            log(f"    lam {lam:.0f}: A={row['A']:.4f} Fz={row['Fz']:.4f} R={row['R']:.3f} T={row['T']:.3f}")
    return rows


def spec_arrays(rows):
    lam = np.array([r["lam"] for r in rows])
    out = {k: np.array([r[k] for r in rows]) for k in rows[0] if k not in ("lam",)}
    out["lam"] = lam
    if "r_re" in rows[0]:
        out["r"] = out["r_re"] + 1j * out["r_im"]
        out["t"] = out["t_re"] + 1j * out["t_im"]
    return out


# ---------------------------------------------------------------------------
# poles (port of enz_highq_driven_ez_audit/poles.py acceptance rules)
# ---------------------------------------------------------------------------
RT_TOL, DAMP_MIN, PEAK_FRAC = 0.02, 5e-4, 0.03


def omega_of(lam):
    return 2 * np.pi * C_NM_FS / np.asarray(lam)        # rad/fs


def _aaa_poles(lams, vals):
    fit = AAA(omega_of(lams), vals)
    return [(complex(q), abs(res)) for q, res in zip(fit.poles(), fit.residues()) if q.imag < -DAMP_MIN]


def significant_poles(lams, r, t, window=None):
    """r/t-agreeing, damped, significant poles (lambda, Q, gamma, residues)."""
    pr, pt = _aaa_poles(lams, r), _aaa_poles(lams, t)
    sr, st = float(np.max(np.abs(r))), float(np.max(np.abs(t)))
    out = []
    for q, a in pr:
        if not pt:
            continue
        m = min(pt, key=lambda p: abs(p[0] - q))
        if abs(m[0] - q) / abs(q) >= RT_TOL:
            continue
        peak_r, peak_t = a / abs(q.imag) / sr, m[1] / abs(m[0].imag) / st
        if peak_r < PEAK_FRAC or peak_t < PEAK_FRAC:
            continue
        qa = 0.5 * (q + m[0])
        lam = 2 * np.pi * C_NM_FS / qa.real
        if window and not (window[0] <= lam <= window[1]):
            continue
        out.append(dict(omega_re=qa.real, omega_im=qa.imag, lambda_nm=float(lam),
                        Q=float(abs(qa.real / (2 * qa.imag))), gamma=float(abs(qa.imag)),
                        fwhm_nm=float(lam / abs(qa.real / (2 * qa.imag))),
                        res_r=float(a), res_t=float(m[1]), peak_r=float(peak_r), peak_t=float(peak_t),
                        rt_rel_diff=float(abs(m[0] - q) / abs(q))))
    return sorted(out, key=lambda p: p["lambda_nm"])


def ez_map(rho, P, h, lam, s=1.0, order=config.ORDER_FULL, n=48):
    with torch.no_grad():
        sim = fwd.build_sim(rho, P, h, lam, order, ito_loss_scale=s)
        x, y = fwd.cell_axes(P, n)
        E, _ = sim.field_xy(fwd.ITO_LAYER, x, y, config.D_ITO_NM / 2)
    return E[2].numpy()


def overlap(a, b):
    return float(abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


S_LEVELS = (1.0, 0.5, 0.25, 0.1, 0.03, 0.0)
OVL_MIN = 0.6


def track_branch(rho, P, h, omega0, lams, s_levels=S_LEVELS, order=config.ORDER_FULL, log=print, tag="",
                 spectra_cache=None):
    """Follow one pole branch through the ITO loss levels with field-overlap
    continuity (enz_highq_driven_ez_audit/poles.track_branch)."""
    rows, prev, prev_map = [], None, None
    for s in s_levels:
        if spectra_cache is not None and s in spectra_cache:
            sp = spectra_cache[s]
        else:
            sp = spec_arrays(spectrum(rho, P, h, lams, order, s=s))
            if spectra_cache is not None:
                spectra_cache[s] = sp
        cands = significant_poles(sp["lam"], sp["r"], sp["t"])
        ref = complex(prev["omega_re"], prev["omega_im"]) if prev is not None else omega0
        if not cands:
            log(f"  [{tag}] s={s:.2f}: no significant pole"); continue
        by_dist = sorted(cands, key=lambda p: abs(complex(p["omega_re"], p["omega_im"]) - ref))
        near = [p for p in by_dist if abs(complex(p["omega_re"], p["omega_im"]) - ref) / abs(ref) < 0.06] or by_dist[:1]
        chosen = near[0]
        alts = {}
        if prev_map is not None and len(near) > 1:
            for p in near[:3]:
                p["_map"] = ez_map(rho, P, h, p["lambda_nm"], s, order)
                alts[f"{p['lambda_nm']:.1f}"] = overlap(prev_map, p["_map"])
            chosen = max(near[:3], key=lambda p: overlap(prev_map, p["_map"]))
        cmap = chosen.get("_map") if "_map" in chosen else ez_map(rho, P, h, chosen["lambda_nm"], s, order)
        ovl = overlap(prev_map, cmap) if prev_map is not None else 1.0
        jump = bool(ovl < OVL_MIN or (len(near) > 1 and near[0] is not chosen))
        row = {k: v for k, v in chosen.items() if not k.startswith("_")}
        row.update(s=s, overlap_prev=ovl, alternatives=alts, jump_flag=jump, n_candidates_near=len(near))
        rows.append(row)
        log(f"  [{tag}] s={s:.2f}: pole {chosen['lambda_nm']:8.2f} nm Q={chosen['Q']:8.2f} gamma={chosen['gamma']:.5f} "
            f"ovl={ovl:.3f}{' JUMP?' if jump else ''}")
        prev, prev_map = chosen, cmap
    return rows


def fit_gamma(rows):
    """gamma(s) = gamma_rad + s gamma_nr on the non-jumping rows -> Q_rad, Q_nr."""
    use = [r for r in rows if not r.get("jump_flag")] or rows
    if len(use) < 3:
        return dict(error="insufficient tracked levels", n=len(use))
    S = np.array([r["s"] for r in use]); G = np.array([r["gamma"] for r in use])
    slope, icpt = np.polyfit(S, G, 1)
    resid = float(np.max(np.abs(G - (slope * S + icpt))) / G.max())
    w0 = use[0]["omega_re"]
    g_rad, g_nr = float(icpt), float(slope)
    lossless = next((r for r in rows if r["s"] == 0.0 and not r.get("jump_flag")), None)
    return dict(gamma_rad=g_rad, gamma_nr=g_nr,
                Q_rad=(w0 / (2 * g_rad) if g_rad > 0 else float("inf")),
                Q_nr=(w0 / (2 * g_nr) if g_nr > 0 else float("inf")),
                Q_loaded=rows[0]["Q"], lambda_loaded=rows[0]["lambda_nm"],
                gamma_ratio=(g_rad / g_nr if g_nr > 0 else float("inf")),
                linearity_resid=resid, n_levels=len(use),
                lossless_pole=(dict(lambda_nm=lossless["lambda_nm"], Q=lossless["Q"]) if lossless else None),
                pole_shift_nm=(rows[0]["lambda_nm"] - lossless["lambda_nm"]) if lossless else None,
                critical_coupling_metric=abs(np.log(g_rad / g_nr)) if (g_nr > 0 and g_rad > 0) else None)


# ---------------------------------------------------------------------------
# fabrication / locality metrics on the hard-binary design
# ---------------------------------------------------------------------------
def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def min_feature_px(b, max_r=12, tol=0.02):
    """Largest disk radius r such that opening with the disk removes < tol
    of the material (min feature ~ 2r+1 px).  Same for the air gap on ~b."""
    def probe(img):
        base = img.sum()
        r_ok = 0
        for r in range(1, max_r + 1):
            op = ndimage.binary_opening(img, structure=_disk(r))
            if (base - op.sum()) / max(base, 1) > tol:
                break
            r_ok = r
        return r_ok
    return probe(b), probe(~b)


def fab_metrics(rho_hard, P, M):
    b = np.asarray(rho_hard) > 0.5
    nx = b.shape[0]
    dx = P / nx
    lab, ncomp = ndimage.label(b)                       # pad ring isolates the atom: no wrap needed
    sizes = ndimage.sum(b, lab, range(1, ncomp + 1)) if ncomp else []
    ring = (np.asarray(M) < 0.5)
    active = ~ring
    # boundary contact: material on the outermost active row/column (touching the air ring)
    er = ndimage.binary_erosion(active, iterations=1)
    edge = active & ~er
    rf, rg = min_feature_px(b, tol=0.02)
    inner_air = (~b) & active
    labA, nair = ndimage.label(inner_air)
    return dict(fill_cell=float(b.mean()), fill_active=float(b.sum() / active.sum()),
                n_components=int(ncomp), component_sizes_px=[int(s) for s in sorted(sizes, reverse=True)],
                largest_component_frac=(float(max(sizes) / b.sum()) if ncomp else 0.0),
                boundary_contact_px=int((b & edge).sum()), material_in_ring_px=int((b & ring).sum()),
                min_feature_nm_est=float((2 * rf + 1) * dx), min_air_gap_nm_est=float((2 * rg + 1) * dx),
                dx_nm=float(dx), n_enclosed_air_regions=int(nair))


def morph(rho_hard, px):
    b = np.asarray(rho_hard) > 0.5
    if px > 0:
        b = ndimage.binary_dilation(b, structure=_disk(px))
    elif px < 0:
        b = ndimage.binary_erosion(b, structure=_disk(-px))
    return b.astype(float)


def evaluate_hard(rho, P, h, lam, order, n_z=config.Z_SAMPLES_ITO, want_maps=False):
    r = torch.as_tensor(np.asarray(rho), dtype=GEO)
    return fwd.to_floats(fwd.evaluate(r, P, h, lam, order, n_z=n_z, want_maps=want_maps))


def sensitivity(rho_hard, P, h, lam, M, order=config.ORDER_FULL):
    """Erosion/dilation (+-1, +-2 px), height (+-5%, +-10 nm), period (+-2%)."""
    Mt = np.asarray(M)
    base = evaluate_hard(rho_hard, P, h, lam, order)
    out = dict(base=base, morph={}, height={}, period={})
    for px in (-2, -1, 1, 2):
        rr = morph(rho_hard, px) * Mt
        out["morph"][f"{px:+d}px ({px*P/128:+.1f} nm)"] = evaluate_hard(rr, P, h, lam, order)
    for dh in (-0.10 * h, -10.0, 10.0, 0.10 * h):
        out["height"][f"h{dh:+.1f}"] = evaluate_hard(rho_hard, P, h + dh, lam, order)
    for fP in (0.98, 1.02):
        out["period"][f"P x{fP}"] = evaluate_hard(rho_hard, P * fP, h, lam, order)
    return out


def convergence_table(rho_hard, P, h, lam, orders=([5, 5], [7, 7], [9, 9], [11, 11]), n_zs=(7, 15, 31)):
    rows = []
    for od in orders:
        with torch.no_grad():
            r = torch.as_tensor(np.asarray(rho_hard), dtype=GEO)
            sim = fwd.build_sim(r, P, h, lam, od)
            R, T = fwd.rt_all_orders(sim)
            row = dict(order=od[0], R=float(R), T=float(T), A=float(1 - R - T))
            for n_z in n_zs:
                d = fwd.loss_components(sim, n_z=n_z)
                row[f"Fz_nz{n_z}"] = float(d["Fz"]); row[f"Ftot_nz{n_z}"] = float(d["Ftot"])
                row[f"resid_nz{n_z}"] = float(d["Ftot"]) - row["A"]
            E = fwd.ito_fields(sim, 128, 7)
            I2 = (E.abs() ** 2)
            row["max_Ez2"] = float(I2[:, 2].max()); row["mean_Ez2"] = float(I2[:, 2].mean())
            row["Fx_nz7"] = float(fwd.loss_components(sim, n_z=7)["Fx"])
            row["Fy_nz7"] = float(fwd.loss_components(sim, n_z=7)["Fy"])
        rows.append(row)
    return rows


def field_maps(rho_hard, P, h, lam, order=config.ORDER_FULL, n_z=config.Z_SAMPLES_ITO, nx=128):
    """Real-space ITO fields (n_z,3,nx,nx) + per-component loss-density maps
    (z-integrated (w/2) Im eps |E_i|^2 / P_inc per unit area) + a-Si mid-plane field."""
    with torch.no_grad():
        r = torch.as_tensor(np.asarray(rho_hard), dtype=GEO)
        sim = fwd.build_sim(r, P, h, lam, order)
        E = fwd.ito_fields(sim, nx, n_z).numpy()
        x, y = fwd.cell_axes(P, nx)
        Esi, Hsi = sim.field_xy(0, x, y, h / 2)
        Esi = np.stack([c.numpy() for c in Esi]); Hsi = np.stack([c.numpy() for c in Hsi])
        # xz cut through the cell centre (y = P/2): air 300 nm above, a-Si, ITO, glass 300 nm below
        z = torch.linspace(-300.0, h + config.D_ITO_NM + 300.0, 241, dtype=GEO)
        Exz, Hxz = sim.field_xz(x, z, P / 2)
        Exz = np.stack([c.numpy() for c in Exz]); Hxz = np.stack([c.numpy() for c in Hxz])
    dz = config.D_ITO_NM / n_z
    pref = 0.5 * (2 * np.pi / lam) * mat.eps_ito(lam).imag / fwd.p_inc_cell(P)
    dens = pref * (np.abs(E) ** 2).sum(0) * dz            # (3, nx, nx): int_z loss density / P_inc  [1/nm^2]
    return dict(E_ito=E, loss_density_xyz=dens, E_si_mid=Esi, H_si_mid=Hsi, E_xz=Exz, H_xz=Hxz,
                z_xz=z.numpy(), x=x.numpy())


def jdump(obj, path):
    def d(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, complex):
            return [o.real, o.imag]
        return str(o)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=d)
