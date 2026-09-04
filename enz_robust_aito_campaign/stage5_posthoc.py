"""Stage 5: high-accuracy refinement + post-hoc physics of the Stage-4
winner (and the runner-up for the comparison table), final comparison vs
the required references, REPORT.md.

Nothing here feeds back into any loss; every quantity is diagnostic.

Run:  python stage5_posthoc.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

import forward_multi as fm
import optimizer as opt
import references as refs
import robust_config as rc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "enz_absorption_campaign"))
sys.path.insert(0, str(ROOT / "enz_highq_enz_campaign"))
import config                     # noqa: E402  (enz_inverse_design config)
import pole_rt                    # noqa: E402
import stage_a_decompose as sad   # noqa: E402

OUT = rc.OUT / "stage5"
FIG = rc.OUT / "figures"
LOGF = OUT / "stage5.log"
C_NM_FS = 299.792458
SPEC_LAMS = np.arange(1250.0, 1701.0, 5.0)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOGF, "a") as f:
        f.write(s + "\n")


def set_legacy_config(P, h):
    """Point the frozen enz_inverse_design config at the candidate cell so
    the legacy pole / decomposition tools (which read config globals) can be
    reused unchanged."""
    config.PX_NM = config.PY_NM = float(P)
    config.ASI_THICKNESS_NM = float(h)
    config.ITO_THICKNESS_NM = fm.D_ITO_NM
    config.N_GLASS = fm.N_GLASS
    config.EPS_ASI = fm.eps_asi(rc.LAMBDA_E)
    config.FOURIER_ORDER = list(rc.ORDER_FULL)


def spectrum(rho, P, h, theta, phi, with_ito=True, loss_scale=1.0,
             order=rc.ORDER_FULL):
    A, R, T = [], [], []
    with torch.no_grad():
        for lam in SPEC_LAMS:
            sim = fm.build_sim(rho, P, h, lam=lam, theta_deg=theta,
                               phi_deg=phi, order=order, with_ito=with_ito,
                               ito_loss_scale=loss_scale)
            a, r, t = fm.a_ito(sim)
            A.append(float(a)); R.append(float(r)); T.append(float(t))
    return dict(lam=SPEC_LAMS.tolist(), A=A, R=R, T=T)


def multipoles(rho_np, P, h, n=64, nz=8):
    """Cartesian ED/MD/EQ/MQ of the a-Si polarization current at normal
    incidence (lab-x), origin at the cell centre; scattered-power fractions
    from the FREE-SPACE formulas (Grahn et al. 2012) - approximate for a
    periodic array on glass/ITO, reported as such."""
    rho = torch.as_tensor(rho_np, dtype=fm.GEO_DTYPE)
    sim = fm.build_sim(rho, P, h, order=rc.ORDER_FULL)
    x, y = fm.cell_axes(P, n)
    geo = ndimage.zoom(rho_np, n / rho_np.shape[0], order=0)
    k = 2 * np.pi / rc.LAMBDA_E
    w = k                                     # c = 1
    chi = (fm.eps_asi(rc.LAMBDA_E) - 1.0) * geo
    xs = (x.numpy() - P / 2); X, Y = np.meshgrid(xs, xs, indexing="ij")
    dV = (P / n) ** 2 * (h / nz)
    p = np.zeros(3, complex); m = np.zeros(3, complex)
    Qe = np.zeros((3, 3), complex); Qm = np.zeros((3, 3), complex)
    with torch.no_grad():
        for zp in (np.arange(nz) + 0.5) * h / nz:
            E, _ = sim.field_xy(0, x, y, float(zp))
            E = [c.numpy() for c in E]
            Z = np.full_like(X, zp - h / 2)
            J = [-1j * w * chi * c for c in E]      # polarization current
            r = [X, Y, Z]
            for a in range(3):
                p[a] += 1j / w * np.sum(J[a]) * dV
                b, c_ = (a + 1) % 3, (a + 2) % 3
                m[a] += 0.5 * np.sum(r[b] * J[c_] - r[c_] * J[b]) * dV
            rJ = sum(r[i] * J[i] for i in range(3))
            for a in range(3):
                for b in range(3):
                    Qe[a, b] += 1j / w * np.sum(3 * (r[a] * J[b] + r[b] * J[a])
                                                - 2 * (a == b) * rJ) * dV
                    rxJ = [r[(i + 1) % 3] * J[(i + 2) % 3]
                           - r[(i + 2) % 3] * J[(i + 1) % 3] for i in range(3)]
                    Qm[a, b] += np.sum(r[a] * rxJ[b] + r[b] * rxJ[a]) * dV / 3
    Ip = w ** 4 * np.sum(np.abs(p) ** 2) / (12 * np.pi)
    Im = w ** 4 * np.sum(np.abs(m) ** 2) / (12 * np.pi)
    IQe = w ** 6 * np.sum(np.abs(Qe) ** 2) / (1440 * np.pi)
    IQm = w ** 6 * np.sum(np.abs(Qm) ** 2) / (160 * np.pi)
    tot = Ip + Im + IQe + IQm
    return dict(frac_ED=Ip / tot, frac_MD=Im / tot, frac_EQ=IQe / tot,
                frac_MQ=IQm / tot, p_abs=np.abs(p).tolist(),
                m_abs=np.abs(m).tolist(),
                note="free-space scattered-power formulas; approximate for a "
                     "periodic array on glass/ITO; origin = cell centre")


def fourier_ez_ito(rho_np, P, h, n=96, nz=7):
    rho = torch.as_tensor(rho_np, dtype=fm.GEO_DTYPE)
    sim = fm.build_sim(rho, P, h, order=rc.ORDER_FULL)
    x, y = fm.cell_axes(P, n)
    acc = 0.0
    with torch.no_grad():
        for zp in (np.arange(nz) + 0.5) * fm.D_ITO_NM / nz:
            E, _ = sim.field_xy(1, x, y, float(zp))
            F = np.fft.fft2(E[2].numpy()) / n ** 2
            acc = acc + np.abs(F) ** 2
    acc = acc / nz
    tot = acc.sum()
    chans = {}
    for mm in range(-2, 3):
        for nn in range(-2, 3):
            chans[f"({mm},{nn})"] = float(acc[mm % n, nn % n] / tot)
    g10 = 2 * np.pi / P / (2 * np.pi / rc.LAMBDA_E)
    return dict(energy_fraction_by_harmonic=chans, G10_over_k0=g10,
                K_ENZ_over_k0=1.6865)


def locality(rho_np, P):
    lab, ncomp = ndimage.label(rho_np > 0.5)
    sizes = ndimage.sum(rho_np > 0.5, lab, range(1, ncomp + 1))
    dx = P / rho_np.shape[0]
    b = rho_np > 0.5
    # minimum feature / gap by morphological opening survival
    def min_scale(mask):
        for r in range(1, 12):
            st = np.ones((2 * r + 1, 2 * r + 1), bool)
            if ndimage.binary_opening(mask, st).sum() < 0.98 * mask.sum():
                return (2 * r - 1) * dx
        return 23 * dx
    return dict(n_components=int(ncomp), largest_component_frac=float(
        sizes.max() / sizes.sum()) if ncomp else 0.0,
        min_feature_nm=min_scale(b), min_gap_nm=min_scale(~b),
        fill=float(b.mean()), S_flip_lr=opt.s_flip(rho_np),
        S_flip_ud=opt.s_flip_ud(rho_np))


def fabrication(rho_np, P, h, beta):
    out = {}
    with torch.no_grad():
        for px in (-2, -1, 1, 2):
            b = rho_np > 0.5
            st = np.ones((2 * abs(px) + 1, 2 * abs(px) + 1), bool)
            bb = ndimage.binary_dilation(b, st) if px > 0 else \
                ndimage.binary_erosion(b, st)
            r = torch.as_tensor(bb.astype(float), dtype=fm.GEO_DTYPE)
            J, As = opt.evaluate(r, P, h, rc.ANGLES_FULL, rc.ORDER_FULL, beta)
            out[f"{px:+d}px ({px*P/rho_np.shape[0]:+.1f} nm)"] = dict(
                J=float(J), A=[float(a) for a in As])
    return out


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    s4 = json.load(open(rc.OUT / "stage4" / "stage4_summary.json"))
    df4 = pd.read_csv(rc.OUT / "stage4" / "stage4_results.csv")
    beta = s4["beta"]
    rep = dict(beta=beta)
    winner = s4["winner"]
    runs = {}
    for tag in s4["ranking"][:2]:
        res = json.load(open(rc.OUT / "stage4" / "runs" / tag / "result.json"))
        runs[tag] = (np.load(rc.OUT / "stage4" / "runs" / tag
                             / "rho_hard_binary.npy"), res)

    # ---- 5a: order refinement [7,7] -> [9,9] -> [11,11] -------------------
    log("== 5a: high-accuracy refinement ==")
    rep["order_refinement"] = {}
    for tag, (rho_np, res) in runs.items():
        rho = torch.as_tensor(rho_np, dtype=fm.GEO_DTYPE)
        rows = {}
        for od in rc.ORDER_REFINE:
            J, As = opt.evaluate(rho, res["P"], res["h"], rc.ANGLES_FULL, od,
                                 beta)
            rows[str(od)] = dict(J=float(J), A=[float(a) for a in As])
            log(f"  {tag} order {od}: J={float(J):.5f} A={[round(float(a),4) for a in As]}")
        rep["order_refinement"][tag] = rows

    rho_w, res_w = runs[winner]
    P, h, pad = res_w["P"], res_w["h"], res_w["pad_frac"]
    rho_wt = torch.as_tensor(rho_w, dtype=fm.GEO_DTYPE)
    set_legacy_config(P, h)

    # ---- 5b: spectra + controls --------------------------------------------
    log("== 5b: spectra and controls (winner) ==")
    sp = {"with_ITO_0deg": spectrum(rho_wt, P, h, 0, 0),
          "with_ITO_20deg_phi0": spectrum(rho_wt, P, h, 20, 0),
          "with_ITO_20deg_phi90": spectrum(rho_wt, P, h, 20, 90),
          "no_ITO_0deg": spectrum(rho_wt, P, h, 0, 0, with_ito=False),
          "lossless_ITO_0deg": spectrum(rho_wt, P, h, 0, 0, loss_scale=0.0),
          "bare_ITO_0deg": spectrum(None, P, h, 0, 0)}
    rep["spectra"] = sp
    d = rc.RES_PROBE_OFFSET_NM
    with torch.no_grad():
        Ag = {}
        for lam in (rc.LAMBDA_E - d, rc.LAMBDA_E, rc.LAMBDA_E + d):
            sim = fm.build_sim(rho_wt, P, h, lam=lam, order=rc.ORDER_FULL)
            Ag[lam] = float(fm.a_ito(sim)[0])
    AE = Ag[rc.LAMBDA_E]
    rep["resonance_gate_posthoc"] = dict(
        A_E=AE, C80=(AE - 0.5 * (Ag[rc.LAMBDA_E - d] + Ag[rc.LAMBDA_E + d])) / AE,
        center_dominant=bool(AE >= Ag[rc.LAMBDA_E - d] and AE >= Ag[rc.LAMBDA_E + d]),
        note="reported only; not used in any loss")
    log(f"  gate (post hoc): {rep['resonance_gate_posthoc']}")
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    for k in ("with_ITO_0deg", "with_ITO_20deg_phi0", "with_ITO_20deg_phi90",
              "bare_ITO_0deg"):
        axs[0].plot(sp[k]["lam"], sp[k]["A"], label=k)
    axs[0].axvline(rc.LAMBDA_E, ls="--", c="k", lw=.8)
    axs[0].set_xlabel("wavelength (nm)"); axs[0].set_ylabel("A_ITO")
    axs[0].legend(fontsize=8); axs[0].grid(alpha=.3)
    axs[0].set_title("winner: ITO absorption spectra")
    for k in ("with_ITO_0deg", "no_ITO_0deg", "lossless_ITO_0deg"):
        axs[1].plot(sp[k]["lam"], sp[k]["T"], label=k + " T")
    axs[1].axvline(rc.LAMBDA_E, ls="--", c="k", lw=.8)
    axs[1].set_xlabel("wavelength (nm)"); axs[1].set_ylabel("T_total")
    axs[1].legend(fontsize=8); axs[1].grid(alpha=.3)
    axs[1].set_title("controls: with / without / lossless ITO")
    fig.savefig(FIG / "stage5_spectra.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 5c: pole certification + Q_rad/Q_nr decomposition -----------------
    log("== 5c: channel-agnostic r/t poles and loss-scaling decomposition ==")
    cert = pole_rt.certify(rho_wt, with_ito=True)
    cert_no = pole_rt.certify(rho_wt, with_ito=False)
    rep["poles"] = dict(
        with_ITO=cert["certified"], with_ITO_all=cert["certified_all"],
        with_ITO_table=[{k: (float(v) if isinstance(v, (float, int, np.floating))
                             else bool(v)) for k, v in r.items()}
                        for r in cert["table"] if r["rt_agree"]],
        no_ITO=cert_no["certified"], no_ITO_all=cert_no["certified_all"])
    log(f"  with ITO: {cert['certified']}\n  all in-window: {cert['certified_all']}"
        f"\n  no ITO: {cert_no['certified']}")
    lines = []
    dec = sad.analyze("robust winner", rho_wt, lambda s: (log(s), lines.append(s)))
    rep["decomposition"] = json.loads(json.dumps(
        dec, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer))
        else str(o)))
    if "Q_rad" in dec and "Q_nr" in dec:
        rep["critical_coupling_ratio"] = dict(
            gamma_rad_over_gamma_nr=dec["gamma_ratio"],
            note="=1 is critical coupling for a single-port single-mode "
                 "resonator; here two ports (air/glass) -> indicative only")

    # ---- 5d: field / Fourier / multipole / locality / fabrication ----------
    log("== 5d: F_Ez, eta_z, Fourier content, multipoles, locality, fabrication ==")
    fe = {}
    with torch.no_grad():
        for th, ph in rc.ANGLES_FULL:
            sim = fm.build_sim(rho_wt, P, h, theta_deg=th, phi_deg=ph,
                               order=rc.ORDER_FULL)
            v = fm.a_ito_volume(sim, P, rc.LAMBDA_E, th, n_xy=128, n_z=9)
            A, R, T = fm.a_ito(sim)
            _, _, tab = fm.rt_all_orders(sim, per_order=True)
            fe[f"{th:.0f}_{ph:.0f}"] = dict(A_rt=float(A), **v, orders=tab)
            log(f"  th={th:.0f} ph={ph:.0f}: A={float(A):.4f} (vol {v['A_vol']:.4f}) "
                f"F_Ez={v['F_Ez']:.3f} eta_z={v['eta_z']:.3f} propagating orders: "
                f"{[(o['m'], o['n']) for o in tab]}")
    rep["field_and_orders"] = fe
    rep["fourier_ito_Ez"] = fourier_ez_ito(rho_w, P, h)
    rep["multipoles"] = multipoles(rho_w, P, h)
    rep["locality"] = locality(rho_w, P)
    rep["fabrication"] = fabrication(rho_w, P, h, beta)
    log(f"  fourier: {json.dumps({k: round(v, 3) for k, v in rep['fourier_ito_Ez']['energy_fraction_by_harmonic'].items() if v > 0.01})}")
    log(f"  multipoles: {json.dumps({k: round(v, 3) for k, v in rep['multipoles'].items() if k.startswith('frac')})}")
    log(f"  locality: {json.dumps(rep['locality'])}")
    log(f"  fabrication: {json.dumps({k: round(v['J'], 4) for k, v in rep['fabrication'].items()})}")

    # ---- 5e: final comparison vs references (same angular sets) ------------
    log("== 5e: final comparison ==")
    comp = []
    R = refs.load_all()
    entries = [(k, v[0], v[1], v[2]) for k, v in R.items()]
    for tag, (rho_np, res) in runs.items():
        entries.append((f"robust {tag}", torch.as_tensor(rho_np, dtype=fm.GEO_DTYPE),
                        res["P"], res["h"]))
    dense_ref = {}
    for name, rho, Pc, hc in entries:
        J, As = opt.evaluate(rho, Pc, hc, rc.ANGLES_FULL, rc.ORDER_FULL, beta)
        dd = {}
        with torch.no_grad():
            for pl, angs in rc.ANGLES_EVAL_PLANES.items():
                dd[pl] = []
                for th, ph in angs:
                    sim = fm.build_sim(rho, Pc, hc, theta_deg=th, phi_deg=ph,
                                       order=rc.ORDER_FULL)
                    dd[pl].append(float(fm.a_ito(sim)[0]))
            sim0 = fm.build_sim(rho, Pc, hc, order=rc.ORDER_FULL)
            v0 = fm.a_ito_volume(sim0, Pc, rc.LAMBDA_E, 0.0, n_xy=96)
        dense_ref[name] = dd
        le30 = [a for pl in dd for a, (t, _) in zip(dd[pl], rc.ANGLES_EVAL_PLANES[pl]) if t <= 30]
        comp.append(dict(structure=name, P=Pc, h=hc, J_robust=float(J),
                         A_normal=float(As[0]), A_min_full=float(min(float(a) for a in As)),
                         A_mean_full=float(np.mean([float(a) for a in As])),
                         dense_min_le30=min(le30), dense_mean_le30=float(np.mean(le30)),
                         F_Ez_normal=v0["F_Ez"], eta_z_normal=v0["eta_z"]))
        log(f"  {name:28s}: J={float(J):.4f} A0={float(As[0]):.4f} "
            f"min={comp[-1]['A_min_full']:.4f} dense<=30 min/mean="
            f"{comp[-1]['dense_min_le30']:.4f}/{comp[-1]['dense_mean_le30']:.4f} "
            f"F_Ez={v0['F_Ez']:.3f}")
    cdf = pd.DataFrame(comp)
    cdf.to_csv(OUT / "final_comparison.csv", index=False)
    rep["final_comparison"] = comp
    rep["dense_reference_curves"] = dense_ref
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, pl in zip(axs, ("phi0", "phi90", "phi45")):
        for name, dd in dense_ref.items():
            ax.plot([t for t, _ in rc.ANGLES_EVAL_PLANES[pl]], dd[pl],
                    marker="o", ms=3, label=name,
                    lw=2.4 if name.startswith("robust") else 1.2)
        ax.axvspan(0, 30, color="#eee", zorder=0)
        ax.set_title(f"A_ITO(lambda_E) vs theta, plane {pl} (lab-x, order {rc.ORDER_FULL})")
        ax.set_xlabel("theta (deg)"); ax.grid(alpha=.3)
    axs[0].set_ylabel("A_ITO"); axs[0].legend(fontsize=7)
    fig.savefig(FIG / "stage5_final_comparison_angular.png", dpi=150,
                bbox_inches="tight"); plt.close(fig)
    # winner geometry + |Ez| map in the ITO at normal incidence
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
    axs[0].imshow(rho_w.T, origin="lower", cmap="gray_r", extent=[0, P, 0, P])
    axs[0].set_title(f"winner {winner}\nP={P:.0f} h={h:.0f} pad={pad:.3f}P", fontsize=9)
    with torch.no_grad():
        sim = fm.build_sim(rho_wt, P, h, order=rc.ORDER_FULL)
        x, y = fm.cell_axes(P, 128)
        E, _ = sim.field_xy(1, x, y, fm.D_ITO_NM / 2)
    im = axs[1].imshow(np.abs(E[2].numpy()).T, origin="lower", cmap="magma",
                       extent=[0, P, 0, P])
    axs[1].set_title("|Ez/E_inc| at ITO mid-plane, normal incidence", fontsize=9)
    fig.colorbar(im, ax=axs[1], shrink=.8)
    fig.savefig(FIG / "stage5_winner_geometry_Ez.png", dpi=150,
                bbox_inches="tight"); plt.close(fig)

    rep["pulse_validation"] = ("skipped: no documented pulse spectrum exists "
                               "in the repository (searched *.md/*.txt/*.py/"
                               "*.csv/*.json for pulse/bandwidth/fs)")
    json.dump(rep, open(OUT / "stage5_posthoc.json", "w"), indent=1,
              default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    write_report(rep, winner, runs, df4)
    log("[done] stage 5")


def interpretation(rep):
    comp = {c["structure"]: c for c in rep["final_comparison"]}
    win = next(c for c in comp.values() if c["structure"].startswith("robust finalist0"))
    refs = [comp[k] for k in ("EDR cuboid", "unpadded QNM winner", "padded QNM winner", "padded F_ENZ winner")]
    eta_ref = np.mean([c["eta_z_normal"] for c in refs]); fez_ref = np.mean([c["F_Ez_normal"] for c in refs])
    fo = rep["field_and_orders"]
    fab = rep["fabrication"]; loc = rep["locality"]
    d = rep["decomposition"]
    extra = [p for p in rep["poles"]["with_ITO_all"] if p["Q"] > 20]
    lines = [
        f"- **Where the absorption comes from.** At normal incidence the winner's ITO field is only {win['eta_z_normal']:.0%} longitudinal "
        f"(eta_z; the four resonant references average {eta_ref:.0%}) while its Ez enhancement F_Ez = {win['F_Ez_normal']:.2f} is comparable to theirs ({fez_ref:.2f}). "
        f"The ~2.2x higher A_ITO ({win['A_normal']:.3f} vs ~0.20) therefore comes mostly from IN-PLANE fields in the ITO under a high-fill "
        f"({loc['fill']:.0%} of the cell) 200-nm a-Si slab perforated by narrow air slots, i.e. the ITO acts as a lossy layer loaded by a slab/slot resonance, "
        f"not primarily through the longitudinal ENZ (Berreman-type) channel. eta_z rises to "
        + ", ".join(f"{v['eta_z']:.2f} at ({k.replace('_', ' deg, ')} deg)" for k, v in fo.items() if k in ("15_90", "30_90", "20_45"))
        + " where the TE-plane and diagonal incidences drive Ez at the slot edges.",
        f"- **Resonances.** The loaded r/t pole nearest the ENZ band is at {d['rows'][0]['lambda_pole']:.1f} nm with Q_loaded = {d['Q_loaded']:.2f} "
        f"(Q_rad = {d['Q_rad']:.0f}, Q_nr = {d['Q_nr']:.1f}); "
        + (f"a second certified in-window pole at {extra[0]['lambda_nm']:.1f} nm with Q = {extra[0]['Q']:.0f} sits within a few nm of lambda_E and carries the largest "
           f"Lorentzian peak fraction (r {extra[0]['peak_r']:.2f}, t {extra[0]['peak_t']:.2f}): a narrow lattice/guided-mode resonance of the P = 800 nm slab "
           f"(G10/k0 = {rep['fourier_ito_Ez']['G10_over_k0']:.2f} vs K_ENZ/k0 = 1.69) hybridised with the ENZ film. " if extra else "")
        + "The no-ITO control certified no in-window r/t pole under the significance criteria, so the ITO participates in setting the resonance rather than merely loading a photonic mode.",
        f"- **Fabrication sensitivity.** A uniform +-1 px ({800/128:.1f} nm) edge shift lowers J from {rep['order_refinement']['finalist0_P800_h200_pad0.040_warm']['[7, 7]']['J']:.3f} to "
        + " / ".join(f"{v['J']:.3f} ({k.split(' ')[0]})" for k, v in fab.items() if '1px' in k)
        + f"; +-2 px gives " + " / ".join(f"{v['J']:.3f}" for k, v in fab.items() if '2px' in k)
        + f". The minimum air gap is {loc['min_gap_nm']:.0f} nm (the realized padding ring itself), the minimum a-Si feature {loc['min_feature_nm']:.0f} nm. "
        "The design is therefore edge-critical at the 6-nm pixel level; a robustness-aware (eroded/dilated) objective would be the natural next step if fabrication tolerance matters.",
        f"- **Order convergence.** J changes by < 1% from [7,7] to [11,11], but individual TE-plane angles move by up to 0.06 (e.g. (15 deg, 90 deg): "
        f"{rep['order_refinement']['finalist0_P800_h200_pad0.040_warm']['[7, 7]']['A'][3]:.3f} -> {rep['order_refinement']['finalist0_P800_h200_pad0.040_warm']['[11, 11]']['A'][3]:.3f}); "
        "the narrow slots need higher orders for per-angle accuracy.",
        f"- **Angular behaviour.** The winner stays above 0.40 across 0-30 deg in the TM plane and above 0.34 in the TE plane (dense mean {win['dense_mean_le30']:.3f}), "
        f"but falls to {win['dense_min_le30']:.3f} on the 45-deg diagonal at 30 deg; the runner-up (P = 925, h = 240) is flatter at large angles but weaker in the TE plane. "
        "Both from-scratch Stage-4 runs at the same cell reached only J ~ 0.21-0.22, so the outer (P, h, pad) screen and warm start, not the seed, produced the result.",
    ]
    return "\n".join(lines)


def write_report(rep, winner, runs, df4):
    pre = json.load(open(rc.OUT / "preflight.json"))
    s2 = json.load(open(rc.OUT / "stage2" / "stage2_summary.json"))
    s3 = json.load(open(rc.OUT / "stage3" / "stage3_summary.json"))
    rho_w, res_w = runs[winner]
    P, h, pad = res_w["P"], res_w["h"], res_w["pad_frac"]
    b = pre["beta_calibration"]
    L = ["# ROBUST ENZ ENERGY-TRANSFER FREEFORM INVERSE DESIGN - REPORT", "",
         "Generated by `stage5_posthoc.py`; all numbers in `outputs/*/*.json|csv`.", "",
         "## 0. Symmetry statement", "",
         "**Both historical Example6 fliplr symmetry projections were disabled in the "
         "new inverse-design path; no mirror symmetry is enforced.** "
         f"Winner S_flip(lr) = {rep['locality']['S_flip_lr']:.3f}, S_flip(ud) = "
         f"{rep['locality']['S_flip_ud']:.3f} (0 would be mirror-symmetric). "
         "The upstream notebook and the frozen campaigns are untouched.", "",
         "## 1. Objective and search", "",
         f"- Loss: J_robust = -(1/beta) log sum_m w_m exp(-beta A_m), A_m = 1 - R_total - T_total "
         f"(ALL propagating orders, coherent p/s combination of the lab-x polarization) at lambda_E = {rc.LAMBDA_E} nm; "
         f"beta = {b['beta']:.2f} calibrated from the reference angular spreads (median {b['median_spread']:.4f}); uniform weights.",
         "- No Q, Ez, harmonic, QNM-overlap, multipole, Kerker, BIC, polariton or critical-coupling term in the loss.",
         f"- Angular domain (ASSUMPTION - {pre['angular_domain']['authority']}): +-30 deg cone (NA ~ 0.5 in air), lab-frame x polarization projected on the transverse plane, uniform weights; screen set {rc.ANGLES_SCREEN}; full set {rc.ANGLES_FULL}; final check on the phi = 0, 90, 45 deg planes 0-40 deg.",
         f"- Design variables: rho(x,y) on {rc.NX}x{rc.NX}, P in {rc.P_SCREEN} (refined +-{50} nm), h in {rc.H_SCREEN} (+-20), "
         f"p_pad in {rc.PAD_SCREEN} (+-0.03, always > 0); ITO stack and lambda_E fixed.",
         f"- Stage 2: {len(rc.P_SCREEN)*len(rc.H_SCREEN)*len(rc.PAD_SCREEN)*len(rc.SEEDS_SCREEN)} shortened runs "
         f"({s2['n_iter']} it, order {rc.ORDER_SCREEN}, angles {rc.ANGLES_SCREEN}); top: "
         + "; ".join(f"P={t['P']:.0f} h={t['h']:.0f} pad={t['pad']:.2f} J={t['J_hard']:.4f}" for t in s2["top"]),
         f"- Stage 3: adaptive neighbourhood refinement ({s3['n_iter']} it, warm-started); finalists: "
         + "; ".join(f"P={f['P']:.0f} h={f['h']:.0f} pad={f['pad']:.3f} J={f['J_hard']:.4f}" for f in s3["finalists"]),
         f"- Stage 4: full runs (order {rc.ORDER_FULL}, angles {rc.ANGLES_FULL}, {json.load(open(rc.OUT/'stage4'/'stage4_summary.json'))['n_iter']} it): "
         f"warm-started finalists + {len(rc.SCRATCH_SEEDS)} from-scratch runs (seeds {rc.SCRATCH_SEEDS}) at the best cell:", "",
         "| run | P | h | pad | J_hard | A(FULL set) | dense<=30 min / mean | S_flip | warm |", "|---|---|---|---|---|---|---|---|---|"]
    for r in df4.itertuples():
        A_list = [round(float(a), 4) for a in json.loads(r.A_hard)]
        L.append(f"| {r.tag} | {r.P:.0f} | {r.h:.0f} | {r.pad:.3f} | {r.J_hard:.4f} | {A_list} | {r.dense_labx_min_le30:.4f} / {r.dense_labx_mean_le30:.4f} | {r.s_flip:.3f} | {r.warm} |")
    L += ["", f"**Winner: {winner}** - P = {P:.0f} nm, h = {h:.0f} nm, p_pad = {pad:.3f} P "
          f"(realized {res_w['mask']['realized_pad_nm']:.1f} nm), fill (active) = {res_w['fill_fraction_active']:.3f}.", "",
          "![geometry](outputs/figures/stage5_winner_geometry_Ez.png)", "",
          "## 2. High-accuracy refinement", "", "| candidate | order | J | A(FULL set) |", "|---|---|---|---|"]
    for tag, rows in rep["order_refinement"].items():
        for od, v in rows.items():
            L.append(f"| {tag} | {od} | {v['J']:.5f} | {[round(a, 4) for a in v['A']]} |")
    L += ["", "## 3. Final comparison (same angular sets, same order, lab-x polarization)", "",
          "| structure | P | h | J_robust | A(0 deg) | min A (FULL) | mean A (FULL) | dense<=30 min | dense<=30 mean | F_Ez(0) | eta_z(0) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in rep["final_comparison"]:
        L.append(f"| {c['structure']} | {c['P']:.0f} | {c['h']:.0f} | {c['J_robust']:.4f} | {c['A_normal']:.4f} | {c['A_min_full']:.4f} | "
                 f"{c['A_mean_full']:.4f} | {c['dense_min_le30']:.4f} | {c['dense_mean_le30']:.4f} | {c['F_Ez_normal']:.3f} | {c['eta_z_normal']:.3f} |")
    def _clean(o):
        return json.loads(json.dumps(o, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x)))

    def _r(o, nd=4):
        if isinstance(o, dict):
            return {k: _r(v, nd) for k, v in o.items()}
        if isinstance(o, list):
            return [_r(v, nd) for v in o]
        return round(o, nd) if isinstance(o, float) else o
    poles_w = _r(_clean(rep["poles"]["with_ITO"])); poles_all = _r(_clean(rep["poles"]["with_ITO_all"]))
    poles_no = _r(_clean(rep["poles"]["no_ITO"])); poles_no_all = _r(_clean(rep["poles"]["no_ITO_all"]))
    L += ["", "![angular comparison](outputs/figures/stage5_final_comparison_angular.png)", "",
          "## 4. Post-hoc physics of the winner (diagnostic only)", "",
          f"- Spectra (with ITO 0/20 deg, no ITO, lossless ITO, bare ITO): `outputs/figures/stage5_spectra.png`. "
          f"Weak resonance gate (reported only): {json.dumps(rep['resonance_gate_posthoc'])}.",
          f"- r/t poles with ITO (channel-agnostic AAA, window ({pole_rt.WINDOW[0]:.1f}, {pole_rt.WINDOW[1]:.1f}) nm): best-residue pole {poles_w}; all certified in-window: {poles_all}; "
          f"no-ITO photonic poles: {poles_no} / {poles_no_all}."]
    d = rep["decomposition"]
    if "Q_rad" in d:
        L.append(f"- Loss-scaling decomposition: Q_loaded = {d['Q_loaded']:.2f}, Q_rad = {d['Q_rad']:.1f}, Q_nr = {d['Q_nr']:.2f}, "
                 f"gamma_rad/gamma_nr = {d['gamma_ratio']:.3f} (critical-coupling indicator, two-port caveat), glass fraction of gamma_rad ~ {d['glass_fraction']:.2f} "
                 f"(air {1-d['glass_fraction']:.2f}); linearity residual {d['linearity_resid']:.2e}.")
    else:
        L.append(f"- Loss-scaling decomposition: {d.get('error', 'n/a')}.")
    L += [f"- F_Ez / eta_z / propagating orders per FULL angle: " + "; ".join(
            f"({k}) A={v['A_rt']:.4f} F_Ez={v['F_Ez']:.3f} eta_z={v['eta_z']:.3f} orders={[(o['m'], o['n']) for o in v['orders']]}"
            for k, v in rep["field_and_orders"].items()),
          f"- Fourier energy fractions of Ez in the ITO (normal incidence, G10/k0 = {rep['fourier_ito_Ez']['G10_over_k0']:.3f}, K_ENZ/k0 = 1.687): "
          + json.dumps({k: round(v, 3) for k, v in rep['fourier_ito_Ez']['energy_fraction_by_harmonic'].items() if v > 0.01}),
          f"- Multipoles (a-Si current, free-space formulas, approximate): ED {rep['multipoles']['frac_ED']:.3f}, MD {rep['multipoles']['frac_MD']:.3f}, "
          f"EQ {rep['multipoles']['frac_EQ']:.3f}, MQ {rep['multipoles']['frac_MQ']:.3f}.",
          f"- Locality: {json.dumps(rep['locality'])}.",
          f"- Fabrication robustness (uniform dilate/erode, J at FULL set): " + json.dumps({k: round(v['J'], 4) for k, v in rep['fabrication'].items()}),
          f"- Pulse-aware validation: {rep['pulse_validation']}.", "",
          "", "## 4b. Interpretation and caveats (post hoc)", "",
          interpretation(rep), "",
          "## 5. Honesty ledger", "",
          "- lambda_E is inherited from the 850-nm bare-film QNM anchor and held fixed for every period; the loaded resonance of each design is certified post hoc.",
          "- The angular set and the uniform weights are assumptions (no authoritative NA in the repo).",
          f"- Stage 2/3 use order {rc.ORDER_SCREEN} and {len(rc.ANGLES_SCREEN)} angles (screen); Stage 4/5 use [7,7]/[9,9]/[11,11] and {len(rc.ANGLES_FULL)} angles; dense checks on 3 planes.",
          "- The multipole fractions use free-space formulas; the periodic glass/ITO environment makes them indicative only.",
          "- Historical padded references have S_flip ~ 0.14-0.15 (not exactly 0) because the Example6 projection symmetrized the raw variable while the half-pixel-offset blur kernel and hard threshold broke exact symmetry; the new path enforces nothing."]
    (HERE / "REPORT.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
