"""Post-processing of the FDTDX runs (fx_sim.py): R/T/A spectra, normalized complex fields, field maps, z-resolved
field energies, Cartesian multipoles of the a-Si:H polarization current, ITO volume loss and Jz, mesh comparison.

    /opt/venv-fdtdx/bin/python fx_post.py spectra            # all designs found (prod + noito) -> per-design CSV/NPZ, plots, overlays
    /opt/venv-fdtdx/bin/python fx_post.py fields             # per design: normalized fields, maps, U(z), Ez slices, multipoles, ITO loss
    /opt/venv-fdtdx/bin/python fx_post.py convergence        # final3 prod (5 ITO cells) vs prod_ito10 (10 cells)
    /opt/venv-fdtdx/bin/python fx_post.py all
Conventions: FDTDX phasors P(w) = sum_n F(t_n) e^{+i w t_n} (pulse mode, stride-weighted) => complex amplitude in the
exp(-i w t) time convention (same as TORCWA).  FDTDX H is eta0-normalized (H_fdtdx = eta0 H_SI).
"""
import json, sys, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fx_sim import eps_model, MODELS, PROV                     # noqa: E402

C0 = 299792458.0; EPS0 = 8.8541878128e-12; MU0 = 4e-7 * np.pi; ETA0 = np.sqrt(MU0 / EPS0)
DESIGNS = ["final3", "final1", "final0", "final2"]
LAM_ZE = MODELS["lambda_ZE_data_nm"]
import os
REF_TAG = os.environ.get("FX_REF_TAG", "ref64")
COMP = HERE / "comparison"; COMP.mkdir(exist_ok=True)


# ----------------------------------------------------------------------------- loading
def load_run(design, tag):
    d = HERE / ("reference" if design == "reference" else design) / tag
    if not (d / "phasors.npz").exists():
        return None
    z = np.load(d / "phasors.npz"); meta = json.load(open(d / "run_meta.json"))
    return dict(z=z, meta=meta, dir=d)


def plane_flux(ph, dxy):
    """ph: (Nlam, 4, nx, ny, 1) with components Ex, Ey, Hx, Hy -> S_z(lam) = sum 1/2 Re(Ex Hy* - Ey Hx*) dA  [FDTDX units]."""
    Ex, Ey, Hx, Hy = ph[:, 0, :, :, 0], ph[:, 1, :, :, 0], ph[:, 2, :, :, 0], ph[:, 3, :, :, 0]
    return 0.5 * np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx)).sum(axis=(1, 2)) * dxy ** 2


def spectra_of(run, ref):
    lam = run["z"]["lam_spec_m"] * 1e9
    dxy = run["meta"]["dxy_m"]
    P_inc = -plane_flux(ref["z"]["T_plane/phasor"], dxy)                    # incident (-z) power spectrum, positive
    leak = plane_flux(ref["z"]["R_plane/phasor"], dxy)                      # TFSF backward leakage at the R plane (reference)
    S_T = plane_flux(run["z"]["T_plane/phasor"], dxy)
    S_R = plane_flux(run["z"]["R_plane/phasor"], dxy)
    T = -S_T / P_inc; R = (S_R - leak) / P_inc; A = 1 - R - T
    return dict(lam=lam, R=R, T=T, A=A, P_inc=P_inc, leak_over_Pinc=leak / P_inc)


def at_lam(lam, y, l0):
    return float(np.interp(l0, lam, y))


# ----------------------------------------------------------------------------- spectra
def do_spectra():
    ref = load_run("reference", REF_TAG); assert ref is not None, "reference run missing"
    table = {}
    overl = {}
    for design in DESIGNS:
        for tag in ("prod", "noito"):
            run = load_run(design, tag)
            if run is None:
                continue
            s = spectra_of(run, ref)
            out = run["dir"]
            with open(out / f"spectra_{tag}.csv", "w", newline="") as f:
                w = csv.writer(f); w.writerow(["lambda_nm", "R", "T", "A"])
                for i in range(len(s["lam"])):
                    w.writerow([f"{s['lam'][i]:.2f}", f"{s['R'][i]:.6f}", f"{s['T'][i]:.6f}", f"{s['A'][i]:.6f}"])
            np.savez(out / f"spectra_{tag}.npz", lam_nm=s["lam"], R=s["R"], T=s["T"], A=s["A"], P_inc=s["P_inc"], leak_over_Pinc=s["leak_over_Pinc"])
            jA = int(np.argmax(s["A"]))
            table[(design, tag)] = dict(R_ZE=at_lam(s["lam"], s["R"], LAM_ZE), T_ZE=at_lam(s["lam"], s["T"], LAM_ZE), A_ZE=at_lam(s["lam"], s["A"], LAM_ZE),
                                        lam_Amax=float(s["lam"][jA]), A_max=float(s["A"][jA]), lam_Tmin=float(s["lam"][int(np.argmin(s["T"]))]), T_min=float(s["T"].min()),
                                        lam_Rmax=float(s["lam"][int(np.argmax(s["R"]))]), R_max=float(s["R"].max()), max_leak_over_Pinc=float(np.abs(s["leak_over_Pinc"]).max()),
                                        n_steps=run["meta"]["n_steps_run"], time_fs=run["meta"]["time_s"] * 1e15)
            overl[(design, tag)] = s
            fig, ax = plt.subplots(figsize=(7, 4.2))
            ax.plot(s["lam"], s["R"], label="R"); ax.plot(s["lam"], s["T"], label="T"); ax.plot(s["lam"], s["A"], label="A = 1 − R − T")
            ax.axvline(LAM_ZE, color="k", ls=":", lw=0.8, label=f"λ_ZE = {LAM_ZE:.1f} nm"); ax.set_xlabel("λ [nm]"); ax.set_ylabel("R, T, A"); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=8)
            ax.set_title(f"{design} {'with ITO' if tag == 'prod' else 'without ITO'} — FDTDX ({run['meta']['nxy']}² × {run['meta']['idx']['n_z']} cells, {run['meta']['n_ito']} ITO cells, {run['meta']['time_s']*1e15:.0f} fs)", fontsize=9)
            fig.tight_layout(); fig.savefig(out / f"spectra_{tag}.png", dpi=150); plt.close(fig)
    # overlays
    for tag, keys, fname in (("prod", ("A",), "overlay_A_withITO.png"), ("prod", ("R", "T"), "overlay_RT_withITO.png"), ("noito", ("R",), "overlay_R_noITO.png"), ("noito", ("R", "T"), "overlay_RT_noITO.png")):
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        for design in DESIGNS:
            if (design, tag) in overl:
                s = overl[(design, tag)]
                for k, ls in zip(keys, ("-", "--")):
                    ax.plot(s["lam"], s[k], ls=ls, label=f"{design} {k}")
        ax.axvline(LAM_ZE, color="k", ls=":", lw=0.8); ax.set_xlabel("λ [nm]"); ax.set_ylabel(" / ".join(keys)); ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
        ax.set_title(f"FDTDX {'with' if tag == 'prod' else 'without'} ITO — {' and '.join(keys)}", fontsize=10); fig.tight_layout(); fig.savefig(COMP / fname, dpi=150); plt.close(fig)
    for design in DESIGNS:
        if (design, "prod") in overl and (design, "noito") in overl:
            a, b = overl[(design, "prod")], overl[(design, "noito")]
            fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
            axs[0].plot(a["lam"], a["R"], "b-", label="R with ITO"); axs[0].plot(b["lam"], b["R"], "b--", label="R without ITO")
            axs[0].plot(a["lam"], a["T"], "r-", label="T with ITO"); axs[0].plot(b["lam"], b["T"], "r--", label="T without ITO")
            axs[1].plot(a["lam"], a["A"], "k-", label="A with ITO"); axs[1].plot(b["lam"], b["A"], "k--", label="A without ITO (numerical residual; all media lossless)")
            for ax in axs:
                ax.axvline(LAM_ZE, color="k", ls=":", lw=0.8); ax.set_xlabel("λ [nm]"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
            axs[0].set_ylabel("R, T"); axs[1].set_ylabel("A"); fig.suptitle(f"{design}: with vs without ITO (FDTDX)", fontsize=10); fig.tight_layout()
            fig.savefig(HERE / design / "compare_with_without_ITO.png", dpi=150); plt.close(fig)
    with open(COMP / "spectra_table.csv", "w", newline="") as f:
        keys = ["design", "tag"] + list(next(iter(table.values())).keys())
        w = csv.writer(f); w.writerow(keys)
        for (d, t), v in table.items():
            w.writerow([d, t] + [v[k] for k in keys[2:]])
    json.dump({f"{d}/{t}": v for (d, t), v in table.items()}, open(COMP / "spectra_table.json", "w"), indent=1)
    for (d, t), v in table.items():
        print(f"{d:7s} {t:6s}: R_ZE={v['R_ZE']:.4f} T_ZE={v['T_ZE']:.4f} A_ZE={v['A_ZE']:.4f} | A_max={v['A_max']:.4f} @ {v['lam_Amax']:.0f} nm | T_min={v['T_min']:.4f} @ {v['lam_Tmin']:.0f} | R_max={v['R_max']:.4f} @ {v['lam_Rmax']:.0f} | leak/Pinc<={v['max_leak_over_Pinc']:.1e}")
    return table


# ----------------------------------------------------------------------------- fields, U(z), multipoles, ITO loss
def geometry(run):
    m = run["meta"]; idx = m["idx"]; ze = np.array(m["z_edges_m"]); zc = 0.5 * (ze[:-1] + ze[1:]); dz = np.diff(ze)
    dxy = m["dxy_m"]; n = m["nxy"]; xc = (np.arange(n) + 0.5) * dxy
    return dict(ze=ze, zc=zc, dz=dz, dxy=dxy, xc=xc, idx=idx, P=m["P_m"], h=m["h_m"], n_ito=m["n_ito"], no_ito=m["no_ito"])


def normalized_fields(run, ref):
    """E / E_inc at each field wavelength: E_inc = reference Ex phasor at the ITO mid-plane position (uniform plane, air)."""
    lamf = run["z"]["lam_field_m"] * 1e9
    lamr = ref["z"]["lam_field_m"] * 1e9
    assert np.allclose(lamf, lamr), (lamf, lamr)
    E0 = ref["z"]["inc_field_plane/phasor"][:, 0, :, :, 0].mean(axis=(1, 2))          # (Nlam,) complex Ex incident amplitude
    E0_unif = ref["z"]["inc_field_plane/phasor"][:, 0, :, :, 0].std(axis=(1, 2)) / np.abs(E0)
    vol = run["z"]["vol_fields/phasor"] / E0[:, None, None, None, None]
    xz = run["z"]["xz_plane/phasor"] / E0[:, None, None, None, None]
    yz = run["z"]["yz_plane/phasor"] / E0[:, None, None, None, None]
    return dict(lam=lamf, vol=vol, xz=xz[:, :, :, 0, :], yz=yz[:, :, 0, :, :], E0=E0, E0_nonuniformity=E0_unif)


def multipoles(E, xc, yc, zc_loc, dz_loc, dxy, fill3, lam_nm, eps_asi):
    """Cartesian current multipoles of P = eps0 (eps-1) E in the a-Si:H (E normalized to E_inc = 1 V/m).
    J = -i w P (exp(-i w t)); origin at the given local coordinate frame (x,y centred, z centred on the a-Si slab)."""
    w = 2 * np.pi * C0 / (lam_nm * 1e-9); k = w / C0
    X, Y, Z = np.meshgrid(xc, yc, zc_loc, indexing="ij")
    dV = (dxy ** 2 * dz_loc)[None, None, :] * fill3                                   # a-Si volume in each cell
    Jx, Jy, Jz = (-1j * w * EPS0 * (eps_asi - 1.0) * E[c] for c in range(3))
    J = np.stack([Jx, Jy, Jz]); r = np.stack([X, Y, Z])
    def integ(f):
        return (f * dV).sum(axis=(-3, -2, -1))
    p = 1j / w * np.array([integ(J[a]) for a in range(3)])
    rxJ = np.cross(r, J, axis=0)
    m = 0.5 * np.array([integ(rxJ[a]) for a in range(3)])
    rdotJ = (r * J).sum(axis=0); r2 = (r ** 2).sum(axis=0)
    T = 1.0 / (10 * C0) * np.array([integ(rdotJ * r[a] - 2 * r2 * J[a]) for a in range(3)])
    Qe = np.zeros((3, 3), complex); Qm = np.zeros((3, 3), complex)
    for a in range(3):
        for b in range(3):
            Qe[a, b] = 1j / w * integ(3 * (r[b] * J[a] + r[a] * J[b]) - 2 * rdotJ * (1.0 if a == b else 0.0))
            Qm[a, b] = 1.0 / 3.0 * integ(rxJ[a] * r[b] + rxJ[b] * r[a])
    p_eff = p + 1j * k * T
    P_p = MU0 * w ** 4 * np.sum(np.abs(p) ** 2) / (12 * np.pi * C0)
    P_peff = MU0 * w ** 4 * np.sum(np.abs(p_eff) ** 2) / (12 * np.pi * C0)
    P_T = MU0 * w ** 4 * k ** 2 * np.sum(np.abs(T) ** 2) / (12 * np.pi * C0)
    P_m = MU0 * w ** 4 * np.sum(np.abs(m) ** 2) / (12 * np.pi * C0 ** 3)
    P_Qe = MU0 * w ** 6 * np.sum(np.abs(Qe) ** 2) / (1440 * np.pi * C0 ** 3)
    P_Qm = MU0 * w ** 6 * np.sum(np.abs(Qm) ** 2) / (160 * np.pi * C0 ** 5)
    tot = P_peff + P_m + P_Qe + P_Qm
    return dict(lambda_nm=float(lam_nm), omega_rad_s=float(w), k0_m=float(k), eps_aSiH=[float(eps_asi.real), float(eps_asi.imag)],
                p_Cm=[[float(v.real), float(v.imag)] for v in p], m_Am2=[[float(v.real), float(v.imag)] for v in m], T_Cm2=[[float(v.real), float(v.imag)] for v in T],
                Qe_Cm2=[[[float(v.real), float(v.imag)] for v in row] for row in Qe], Qm_Am3=[[[float(v.real), float(v.imag)] for v in row] for row in Qm],
                p_eff_Cm=[[float(v.real), float(v.imag)] for v in p_eff],
                P_ED_eff_W=float(P_peff), P_ED_p_only_W=float(P_p), P_TD_diag_W=float(P_T), P_MD_W=float(P_m), P_EQ_W=float(P_Qe), P_MQ_W=float(P_Qm), P_sum_eff_W=float(tot),
                frac_ED_eff=float(P_peff / tot), frac_MD=float(P_m / tot), frac_EQ=float(P_Qe / tot), frac_MQ=float(P_Qm / tot), TD_diag_over_sum=float(P_T / tot), ED_p_only_over_sum=float(P_p / tot),
                formulas="p=(i/w)∫J; m=½∫r×J; T=(1/10c)∫[(r·J)r−2r²J]; Qe=(i/w)∫[3(r_b J_a+r_a J_b)−2(r·J)δ]; Qm=⅓∫[(r×J)_a r_b+(r×J)_b r_a]; P_p=μ0w⁴|p|²/(12πc); P_m=μ0w⁴|m|²/(12πc³); P_Qe=μ0w⁶Σ|Qe|²/(1440πc³); P_Qm=μ0w⁶Σ|Qm|²/(160πc⁵); p_eff=p+ikT (vacuum long-wavelength Cartesian forms; fractions normalize P_ED_eff+P_MD+P_EQ+P_MQ, TD not added separately)",
                origin="x=y=P/2, z = a-Si:H mid-height; E normalized to E_inc = 1 V/m; J = −iω ε0 (ε_aSiH − 1) E on the a-Si:H cells (partial interface cells weighted by their exact fill fraction)")


def do_fields(design, tag="prod", lam_targets=None):
    ref = load_run("reference", REF_TAG); run = load_run(design, tag)
    if run is None:
        print(f"[fields] {design}/{tag} missing"); return None
    g = geometry(run); F = normalized_fields(run, ref); out = run["dir"]; idx = g["idx"]
    lam = F["lam"]; n = run["meta"]["nxy"]
    fill = __import__("fx_sim").area_fraction(np.load(HERE / design / "rho_hard_binary.npy"), n)
    zv0, zv1 = idx["z_vol0"], idx["z_vol1"]; zc_v = g["zc"][zv0:zv1]; dz_v = g["dz"][zv0:zv1]
    a0, a1 = idx["z_asi0"] - zv0, idx["z_asi1"] - zv0                 # a-Si range inside the volume block
    i0, i1 = idx["z_ito0"] - zv0, idx["z_asi0"] - zv0                 # ITO range inside the volume block
    zmid = i0 + g["n_ito"] // 2
    # ---- save normalized complex fields (all recorded wavelengths)
    np.savez_compressed(out / "fields_normalized.npz", lam_nm=lam, E_over_Einc=F["vol"].astype(np.complex64), x_m=g["xc"], y_m=g["xc"], z_m=zc_v, dz_m=dz_v,
                        z_asi_range_idx=[a0, a1], z_ito_range_idx=[i0, i1], z_ito_mid_idx=zmid, xz_plane=F["xz"].astype(np.complex64), yz_plane=F["yz"].astype(np.complex64),
                        z_plane_m=g["zc"][idx["z_plane0"]:idx["z_plane1"]], E_inc_phasor=F["E0"], E_inc_nonuniformity=F["E0_nonuniformity"],
                        note="E_over_Einc[l, c, ix, iy, iz], c = (Ex, Ey, Ez), complex amplitude in exp(-i w t) convention, normalized by the incident Ex phasor at the ITO mid-plane position (reference run)")
    # ---- per-wavelength diagnostics
    res = {}
    eps_ito = eps_model("ITO", lam * 1e-9); eps_a = eps_model("aSiH", lam * 1e-9)
    for il, l in enumerate(lam):
        E = F["vol"][il]
        I2 = np.abs(E) ** 2                                                  # (3, nx, ny, nz)
        # ITO loss (FDTD analogue of F_x, F_y, F_z): (w/c) Im(eps) ∫|E_c|² dV / P²   (E normalized; incident power = P²/(2 eta0)·|E_inc|²)
        w = 2 * np.pi * C0 / (l * 1e-9)
        if not g["no_ito"]:
            dV = g["dxy"] ** 2 * dz_v[i0:i1]
            Fc = [(w / C0) * eps_ito[il].imag * np.sum(I2[c, :, :, i0:i1] * dV[None, None, :]) / g["P"] ** 2 for c in range(3)]
        else:
            Fc = [0.0, 0.0, 0.0]
        Ez2_mid = I2[2, :, :, zmid]
        # U(z) in the a-Si:H
        U = [np.sum(I2[c, :, :, a0:a1], axis=(0, 1)) * g["dxy"] ** 2 for c in range(3)]
        res[f"{l:.3f}"] = dict(lambda_nm=float(l), Fx=Fc[0], Fy=Fc[1], Fz=Fc[2], Ftot=float(sum(Fc)), eta_z=float(Fc[2] / sum(Fc)) if sum(Fc) > 0 else None,
                               mean_Ez2_ito_mid=float(Ez2_mid.mean()), max_Ez2_ito_mid=float(Ez2_mid.max()), mean_E2_ito_mid=float(I2[:, :, :, zmid].sum(0).mean()),
                               mean_Ez2_ito_vol=float(I2[2, :, :, i0:i1].mean()), max_Ez2_ito_vol=float(I2[2, :, :, i0:i1].max()),
                               eps_ITO_model=[float(eps_ito[il].real), float(eps_ito[il].imag)])
        # ---- U(z)
        zs = zc_v[a0:a1] - g["ze"][idx["z_asi0"]]
        with open(out / f"Uz_{l:.1f}nm.csv", "w", newline="") as f:
            wr = csv.writer(f); wr.writerow(["z_from_aSi_bottom_nm", "z_over_h", "Ux", "Uy", "Uz", "Utot", "note: U = ∫∫|E_c/E_inc|² dx dy [m²] at the cell centre"])
            for k in range(len(zs)):
                wr.writerow([f"{zs[k]*1e9:.3f}", f"{zs[k]/g['h']:.4f}", f"{U[0][k]:.6e}", f"{U[1][k]:.6e}", f"{U[2][k]:.6e}", f"{(U[0][k]+U[1][k]+U[2][k]):.6e}"])
        z10 = np.arange(5e-9, g["h"], 10e-9)
        np.savez(out / f"Uz_{l:.1f}nm.npz", z_m=zs, z_over_h=zs / g["h"], Ux=U[0], Uy=U[1], Uz=U[2], Utot=U[0] + U[1] + U[2],
                 z10_m=z10, Ux_10nm=np.interp(z10, zs, U[0]), Uy_10nm=np.interp(z10, zs, U[1]), Uz_10nm=np.interp(z10, zs, U[2]), Utot_10nm=np.interp(z10, zs, U[0] + U[1] + U[2]))
        np.savez_compressed(out / f"Ez2_slices_aSi_{l:.1f}nm.npz", z_from_aSi_bottom_m=zs, z_over_h=zs / g["h"], Ez2_over_Einc2=I2[2, :, :, a0:a1].astype(np.float32), x_m=g["xc"], y_m=g["xc"],
                            note="|Ez/E_inc|² on every a-Si:H z cell (graded 4.6 nm near the ITO, 12.9 nm in the bulk)")
        # ---- multipoles (a-Si:H region)
        fill3 = np.repeat(fill[:, :, None], a1 - a0, axis=2)
        mp = multipoles(E[:, :, :, a0:a1], g["xc"] - g["P"] / 2, g["xc"] - g["P"] / 2, zs - g["h"] / 2, dz_v[a0:a1], g["dxy"], fill3, l, eps_a[il])
        res[f"{l:.3f}"]["multipoles"] = mp
        # ---- ITO Jz map (optional): J_z = -i w eps0 (eps_ITO - 1) Ez (total polarization current density of the dispersive ITO)
        if not g["no_ito"]:
            Jz = -1j * w * EPS0 * (eps_ito[il] - 1.0) * E[2, :, :, zmid]
            np.savez_compressed(out / f"Jz_ITO_{l:.1f}nm.npz", Jz_A_m2_per_Vm=Jz.astype(np.complex64), x_m=g["xc"], y_m=g["xc"], z_m=zc_v[zmid], eps_ITO=eps_ito[il],
                                note="J_z = -i w eps0 (eps_ITO(w) - 1) E_z at the ITO mid-plane, E normalized to E_inc = 1 V/m")
    json.dump(res, open(out / "field_diagnostics.json", "w"), indent=1)
    # ---- figures: maps at lambda_ZE and the closest recorded wavelength to the A-max
    sp = json.load(open(COMP / "spectra_table.json")).get(f"{design}/{tag}", {})
    targets = [LAM_ZE]
    if sp:
        lA = sp["lam_Amax"]; jn = int(np.argmin(abs(lam - lA))); targets.append(float(lam[jn]))
        res["_lam_Amax_spectrum"] = lA; res["_lam_field_nearest_to_Amax"] = float(lam[jn]); res["_mismatch_nm"] = float(abs(lam[jn] - lA))
    if lam_targets:
        targets += list(lam_targets)
    for lt in dict.fromkeys(targets):
        il = int(np.argmin(abs(lam - lt))); E = F["vol"][il]; I2 = np.abs(E) ** 2
        Exz = np.abs(F["xz"][il]) ** 2; Eyz = np.abs(F["yz"][il]) ** 2; zp = g["zc"][idx["z_plane0"]:idx["z_plane1"]] * 1e9
        fig, axs = plt.subplots(3, 4, figsize=(19, 12))
        titles = ["|Ex/E_inc|²", "|Ey/E_inc|²", "|Ez/E_inc|²", "|E/E_inc|²"]
        for c in range(4):
            m_xy = I2[c, :, :, zmid] if c < 3 else I2[:, :, :, zmid].sum(0)
            m_xz = Exz[c] if c < 3 else Exz.sum(0); m_yz = Eyz[c] if c < 3 else Eyz.sum(0)
            im = axs[0, c].imshow(m_xy.T, origin="lower", extent=[0, g["P"] * 1e9, 0, g["P"] * 1e9], cmap="inferno"); plt.colorbar(im, ax=axs[0, c], pad=0.01)
            axs[0, c].set_title(f"{titles[c]} — xy at ITO mid-plane" if not g["no_ito"] else f"{titles[c]} — xy at z of the (removed) ITO mid-plane", fontsize=8); axs[0, c].set_xlabel("x [nm]"); axs[0, c].set_ylabel("y [nm]")
            im = axs[1, c].imshow(m_xz.T, origin="lower", extent=[0, g["P"] * 1e9, zp[0], zp[-1]], cmap="inferno", aspect="auto"); plt.colorbar(im, ax=axs[1, c], pad=0.01)
            axs[1, c].set_title(f"{titles[c]} — xz at y = P/2", fontsize=8); axs[1, c].set_xlabel("x [nm]"); axs[1, c].set_ylabel("z [nm]")
            im = axs[2, c].imshow(m_yz.T, origin="lower", extent=[0, g["P"] * 1e9, zp[0], zp[-1]], cmap="inferno", aspect="auto"); plt.colorbar(im, ax=axs[2, c], pad=0.01)
            axs[2, c].set_title(f"{titles[c]} — yz at x = P/2", fontsize=8); axs[2, c].set_xlabel("y [nm]"); axs[2, c].set_ylabel("z [nm]")
            for ax in (axs[1, c], axs[2, c]):
                for zl in (g["ze"][idx["z_ito0"]], g["ze"][idx["z_asi0"]], g["ze"][idx["z_asi1"]]):
                    ax.axhline(zl * 1e9, color="w", lw=0.5, ls="--")
        fig.suptitle(f"{design} {'with' if not g['no_ito'] else 'without'} ITO — FDTDX fields at λ = {lam[il]:.2f} nm (normalized to the incident field)", fontsize=11)
        fig.tight_layout(); fig.savefig(out / f"fields_{lam[il]:.1f}nm.png", dpi=130); plt.close(fig)
    # ---- U(z) figure
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for il, l in enumerate(lam):
        if abs(l - LAM_ZE) > 0.5:
            continue
        d = np.load(out / f"Uz_{l:.1f}nm.npz")
        for k, ls in (("Ux", "-"), ("Uy", "--"), ("Uz", "-."), ("Utot", ":")):
            ax.plot(d["z_m"] * 1e9, d[k], ls=ls, label=f"{k} @ {l:.1f} nm")
    ax.set_xlabel("z from the a-Si:H bottom [nm]"); ax.set_ylabel("U_c(z) = ∫∫|E_c/E_inc|² dx dy [m²]"); ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_title(f"{design} ({tag})", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "Uz_profiles.png", dpi=150); plt.close(fig)
    json.dump(res, open(out / "field_diagnostics.json", "w"), indent=1)
    r = res[f"{LAM_ZE:.3f}"]
    print(f"[fields] {design}/{tag} @ λ_ZE: Fx={r['Fx']:.4f} Fy={r['Fy']:.4f} Fz={r['Fz']:.4f} Ftot={r['Ftot']:.4f} | <|Ez/Einc|²>_ITO-mid={r['mean_Ez2_ito_mid']:.2f} max={r['max_Ez2_ito_mid']:.1f} | "
          f"ED_eff={r['multipoles']['frac_ED_eff']:.3f} MD={r['multipoles']['frac_MD']:.3f} EQ={r['multipoles']['frac_EQ']:.3f} MQ={r['multipoles']['frac_MQ']:.3f} TD/sum={r['multipoles']['TD_diag_over_sum']:.3f} | E0 nonuniformity {F['E0_nonuniformity'].max():.1e}")
    return res


def do_overlays_U():
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    for design in DESIGNS:
        p = HERE / design / "prod" / f"Uz_{LAM_ZE:.1f}nm.npz"
        if p.exists():
            d = np.load(p); axs[0].plot(d["z_over_h"], d["Uz"], label=design); axs[1].plot(d["z_over_h"], d["Utot"], label=design)
    axs[0].set_ylabel("U_z(z) [m²]"); axs[1].set_ylabel("U_tot(z) [m²]")
    for ax in axs:
        ax.set_xlabel("z / h"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"FDTDX field energy through the a-Si:H height at λ_ZE = {LAM_ZE:.1f} nm (E normalized to E_inc)", fontsize=10); fig.tight_layout(); fig.savefig(COMP / "overlay_Uz_Utot_vs_z_over_h.png", dpi=150); plt.close(fig)
    # cross-design ITO-plane |Ez|^2 with a common colour scale
    maps = {}
    for design in DESIGNS:
        p = HERE / design / "prod" / "fields_normalized.npz"
        if p.exists():
            z = np.load(p); il = int(np.argmin(abs(z["lam_nm"] - LAM_ZE))); maps[design] = np.abs(z["E_over_Einc"][il, 2, :, :, int(z["z_ito_mid_idx"])]) ** 2
    if maps:
        vmax = max(m.max() for m in maps.values())
        fig, axs = plt.subplots(1, len(maps), figsize=(4.6 * len(maps), 4.4))
        for ax, (d, m) in zip(np.atleast_1d(axs), maps.items()):
            im = ax.imshow(m.T, origin="lower", extent=[0, 825, 0, 825], cmap="inferno", vmin=0, vmax=vmax); ax.set_title(f"{d}: |Ez/E_inc|², ITO mid-plane, max {m.max():.1f}, mean {m.mean():.2f}", fontsize=8); ax.set_xlabel("x [nm]"); ax.set_ylabel("y [nm]")
        plt.colorbar(im, ax=list(np.atleast_1d(axs)), pad=0.01, shrink=0.9)
        fig.suptitle(f"ITO-plane |Ez/E_inc|² at λ_ZE, common colour scale (0 … {vmax:.1f})", fontsize=10); fig.savefig(COMP / "overlay_ITO_plane_Ez2_common_scale.png", dpi=150); plt.close(fig)


def do_convergence():
    ref = load_run("reference", REF_TAG); a = load_run("final3", "prod"); b = load_run("final3", "prod_ito10")
    if a is None or b is None:
        print("[convergence] runs missing"); return
    sa, sb = spectra_of(a, ref), spectra_of(b, ref)
    band = (sa["lam"] >= 1200) & (sa["lam"] <= 1400)
    da = json.load(open(a["dir"] / "field_diagnostics.json")) if (a["dir"] / "field_diagnostics.json").exists() else {}
    db = json.load(open(b["dir"] / "field_diagnostics.json")) if (b["dir"] / "field_diagnostics.json").exists() else {}
    key = f"{LAM_ZE:.3f}"
    out = dict(mesh_A=dict(n_ito=a["meta"]["n_ito"], dz_ito_nm=a["meta"]["dz_ito_m"] * 1e9, dt_as=a["meta"]["dt_s"] * 1e18, steps=a["meta"]["n_steps_run"], cells=a["meta"]["n_cells"]),
               mesh_B=dict(n_ito=b["meta"]["n_ito"], dz_ito_nm=b["meta"]["dz_ito_m"] * 1e9, dt_as=b["meta"]["dt_s"] * 1e18, steps=b["meta"]["n_steps_run"], cells=b["meta"]["n_cells"]),
               max_abs_dR=float(np.abs(sa["R"] - sb["R"])[band].max()), max_abs_dT=float(np.abs(sa["T"] - sb["T"])[band].max()), max_abs_dA=float(np.abs(sa["A"] - sb["A"])[band].max()),
               R_ZE=[at_lam(sa["lam"], sa["R"], LAM_ZE), at_lam(sb["lam"], sb["R"], LAM_ZE)], T_ZE=[at_lam(sa["lam"], sa["T"], LAM_ZE), at_lam(sb["lam"], sb["T"], LAM_ZE)], A_ZE=[at_lam(sa["lam"], sa["A"], LAM_ZE), at_lam(sb["lam"], sb["A"], LAM_ZE)],
               lam_Amax=[float(sa["lam"][np.argmax(sa["A"])]), float(sb["lam"][np.argmax(sb["A"])])])
    if key in da and key in db:
        for k in ("Fz", "Ftot", "mean_Ez2_ito_mid", "max_Ez2_ito_mid", "mean_Ez2_ito_vol"):
            out[k] = [da[key][k], db[key][k]]
        out["multipole_fracs_A"] = {k: da[key]["multipoles"][k] for k in ("frac_ED_eff", "frac_MD", "frac_EQ", "frac_MQ", "TD_diag_over_sum")}
        out["multipole_fracs_B"] = {k: db[key]["multipoles"][k] for k in ("frac_ED_eff", "frac_MD", "frac_EQ", "frac_MQ", "TD_diag_over_sum")}
    json.dump(out, open(COMP / "mesh_convergence_final3.json", "w"), indent=1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for k, c in (("R", "b"), ("T", "r"), ("A", "k")):
        axs[0].plot(sa["lam"], sa[k], c + "-", label=f"{k} (5 ITO cells, 4.6 nm)"); axs[0].plot(sb["lam"], sb[k], c + "--", label=f"{k} (10 ITO cells, 2.3 nm)")
        axs[1].plot(sa["lam"], sb[k] - sa[k], c + "-", label=f"Δ{k} (10 − 5 cells)")
    for ax in axs:
        ax.axvline(LAM_ZE, color="k", ls=":", lw=0.8); ax.set_xlabel("λ [nm]"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axs[0].set_ylabel("R, T, A"); axs[1].set_ylabel("difference"); fig.suptitle("final3 mesh check: ITO z-refinement (same in-plane grid)", fontsize=10); fig.tight_layout(); fig.savefig(COMP / "mesh_convergence_final3.png", dpi=150); plt.close(fig)
    print("[convergence]", json.dumps(out, indent=1))


def do_decay():
    """End-of-run field level relative to the peak from the two time-domain probes (decay check for the DFT)."""
    out = {}
    for d in DESIGNS:
        for t in ("prod", "prod_ito10", "noito"):
            run = load_run(d, t)
            if run is None:
                continue
            r = {}
            for k in ("probe_asi", "probe_ito"):
                f = run["z"][f"{k}/fields"]; a = np.abs(f).max(axis=1); pk = a.max()
                r[f"{k}_end_over_peak"] = float(a[-50:].max() / pk); r[f"{k}_peak_time_fs"] = float(np.argmax(a) * run["meta"]["dt_s"] * 1e15)
            out[f"{d}/{t}"] = r
    json.dump(out, open(COMP / "decay_check.json", "w"), indent=1)
    for k, v in out.items():
        print(f"[decay] {k}: a-Si {v['probe_asi_end_over_peak']:.1e} (peak at {v['probe_asi_peak_time_fs']:.0f} fs), ITO {v['probe_ito_end_over_peak']:.1e}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("decay", "all"):
        do_decay()
    if what in ("spectra", "all"):
        do_spectra()
    if what in ("fields", "all"):
        for d in DESIGNS:
            for t in ("prod", "noito", "prod_ito10"):
                if (HERE / d / t / "phasors.npz").exists():
                    do_fields(d, t)
        do_overlays_U()
    if what in ("convergence", "all"):
        do_convergence()
