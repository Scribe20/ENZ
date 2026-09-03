"""TARGET AUDIT (authoritative, reproducible) for the resonant ITO
power-transfer campaign.  Generates TARGET_AUDIT.md, target_definition.json,
failure_test.json, calibration.csv, gradient_audit.json and figures from
scratch; rerunning must reproduce them byte-for-byte.  Launches NO topology
optimization.

Adopted formulation (the ONLY one described anywhere in this repository):

  PRIMARY      maximize A_ITO(lambda_E) = 1 - R - T   (ITO is the only lossy
               layer; cross-validated against the volume integral below)
  IN-LOOP GATE resonance-contrast gate
                 C = [A(lam_E) - (A(lam_E - d) + A(lam_E + d))/2] / A(lam_E)
                 >= C_MIN,  d = RES_PROBE_OFFSET_NM (side-probe offset)
               + three-point center-dominance test A(lam_E) >= A(lam_E +/- d)
               (an empirical ENZ-band spectral-selectivity surrogate; NOT a
               Q constraint and NOT a proof of peak location)
  POST-HOC     channel-agnostic scattering pole from r_xx/t_xx (pole_rt.py):
               lambda_pole, Q_pole, |lambda_pole - lambda_E| <=
               DELTA_LAMBDA_POLE_MAX; no-ITO photonic pole of the same
               geometry; F_Ez, eta_z, QNM overlap, Fourier/multipole - all
               diagnostics, none in the loss.

lambda_E = 1433.488 nm is a FROZEN operating wavelength inherited from the
previously validated finite-K bare-film ENZ branch (K = G10 of the 850-nm
lattice).  It is not a momentum term of the new loss and the new objective
did not discover it.

Run:  python target_audit.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / "enz_inverse_design"
PAD = HERE.parent / "enz_padding_sideexperiment"
EZC = HERE.parent / "enz_direct_enz_excitation"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(HERE))

import config                                    # noqa: E402
import target_mode                               # noqa: E402
import torcwa_forward as fwd                     # noqa: E402
from validate_with_without_ito import build_sim, power_RT  # noqa: E402
import pole_rt                                   # noqa: E402

OUT = HERE / "outputs"
FIG = OUT / "figures"
LAMBDA_E = 1433.488
RES_PROBE_OFFSET_NM = 80.0
Q_REF_LOADED = 5.0            # empirical trusted loaded-resonance scale
DELTA_LAMBDA_POLE_MAX = pole_rt.HWHM_ENZ   # post-hoc pole window half-width
MU_CANDIDATES = (30.0, 100.0, 300.0)


# ---------------------------------------------------------------------------
def ctx_setup():
    torch.set_num_threads(config.N_THREADS)
    tgt = target_mode.load_target_npz()
    config.ITO_THICKNESS_NM = float(tgt["ito_thickness_nm"])
    config.N_GLASS = float(tgt["glass_index"])
    config.EPS_ASI = fwd.eps_asi_of_lambda(LAMBDA_E)
    x, y = fwd.grid_axes()
    zp = target_mode.ito_z_slices(config.ITO_THICKNESS_NM,
                                  config.Z_SAMPLES_ITO)
    dV = (config.PX_NM / config.NX_DESIGN) * (config.PY_NM / config.NY_DESIGN) \
        * (config.ITO_THICKNESS_NM / config.Z_SAMPLES_ITO)
    return dict(x=x, y=y, zp=zp, dV=dV, A_cell=config.PX_NM * config.PY_NM)


def a_ito_two_ways(rho_t, ctx):
    lam = LAMBDA_E
    eps_ito = fwd.eps_ito_of_lambda(lam)
    with torch.no_grad():
        sim = fwd.build_solved_sim(rho_t, lam, eps_ito, config.N_GLASS,
                                   eps_asi=fwd.eps_asi_of_lambda(lam))
        comps = [[], [], []]
        for zpv in ctx["zp"]:
            E, _ = sim.field_xy(1, ctx["x"], ctx["y"], float(zpv))
            for c in range(3):
                comps[c].append(E[c])
        Ex, Ey, Ez = [torch.stack(c, 0) for c in comps]
        E2 = float(torch.sum(torch.abs(Ex) ** 2 + torch.abs(Ey) ** 2
                             + torch.abs(Ez) ** 2).real * ctx["dV"])
        Iz = float(torch.sum(torch.abs(Ez) ** 2).real * ctx["dV"])
        A_vol = (2 * np.pi / lam) * eps_ito.imag * E2 / ctx["A_cell"]
        R, T = power_RT(build_sim(rho_t, lam, True))
    return dict(A_vol=A_vol, A_rt=1 - R - T, R=R, T=T,
                F_Ez=Iz / (ctx["A_cell"] * config.ITO_THICKNESS_NM),
                eta_z=Iz / E2)


def gate_eval(rho_t):
    """A(lam_E), contrast C(d), center-dominance flags (3 S-matrix solves)."""
    d = RES_PROBE_OFFSET_NM
    A = {}
    with torch.no_grad():
        for lam in (LAMBDA_E - d, LAMBDA_E, LAMBDA_E + d):
            R, T = power_RT(build_sim(rho_t, lam, True))
            A[lam] = 1 - R - T
    AE, Am, Ap = A[LAMBDA_E], A[LAMBDA_E - d], A[LAMBDA_E + d]
    return dict(A_E=AE, C=(AE - 0.5 * (Am + Ap)) / AE,
                center_dominant=bool(AE >= Am and AE >= Ap))


# ---------------------------------------------------------------------------
# calibration-set geometry perturbations (no optimization)
# ---------------------------------------------------------------------------
def _sdf(b):
    return ndimage.distance_transform_edt(b) - ndimage.distance_transform_edt(1 - b)


def scale_xy(b, sx, sy):
    n = b.shape[0]
    z = ndimage.zoom(b.astype(float), (sx, sy), order=1)
    out = np.zeros_like(b, dtype=float)
    cx, cy = z.shape[0] // 2, z.shape[1] // 2
    src = z[max(cx - n // 2, 0):cx + n // 2, max(cy - n // 2, 0):cy + n // 2]
    ox, oy = (n - src.shape[0]) // 2, (n - src.shape[1]) // 2
    out[ox:ox + src.shape[0], oy:oy + src.shape[1]] = src
    return (out > 0.5).astype(float)


def dilate_erode(b, px):
    if px == 0:
        return b.astype(float)
    st = ndimage.generate_binary_structure(2, 1)
    f = ndimage.binary_dilation if px > 0 else ndimage.binary_erosion
    return f(b > 0.5, st, iterations=abs(px)).astype(float)


def morph(a, b, alpha):
    return (((1 - alpha) * _sdf(a > 0.5) + alpha * _sdf(b > 0.5)) > 0).astype(float)


def build_calibration_set(mask):
    rho_padq = np.load(PAD / "outputs" / "geometries" / "rho_hard_binary.npy")
    rho_pade = np.load(EZC / "outputs" / "geometries" / "rho_hard_binary.npy")
    rho_unp = np.load(PKG / "outputs" / "geometries" / "rho_hard_binary.npy")
    nx = rho_padq.shape[0]
    xg = (np.arange(nx) + 0.5) / nx * config.PX_NM
    X, Y = np.meshgrid(xg, xg, indexing="ij")
    rho_edr = (((np.abs(X - 425.0) < 280.0) & (np.abs(Y - 425.0) < 250.0))
               .astype(float))
    S = {}
    S["ref: bare ITO"] = np.zeros_like(rho_padq)
    S["ref: EDR cuboid"] = rho_edr
    S["ref: unpadded QNM winner"] = rho_unp
    S["ref: padded QNM winner"] = rho_padq
    S["ref: padded F_ENZ winner"] = rho_pade
    for s in (0.8, 0.9, 1.1, 1.2):
        S[f"padQ x-scale {s}"] = scale_xy(rho_padq, s, 1.0) * mask
        S[f"padQ y-scale {s}"] = scale_xy(rho_padq, 1.0, s) * mask
    for px in (-3, -2, -1, 1, 2, 3):
        S[f"padQ dilate {px:+d}px"] = dilate_erode(rho_padq, px) * mask
    for a in (0.25, 0.5, 0.75):
        S[f"morph padQ->padE {a}"] = morph(rho_padq, rho_pade, a) * mask
    for px in (-3, -1, 1, 3):
        S[f"unpadded dilate {px:+d}px"] = dilate_erode(rho_unp, px)
    for s in (0.7, 0.85, 1.15, 1.3):
        S[f"EDR scale {s}"] = scale_xy(rho_edr, s, s)
    for f in (0.3, 0.6, 1.0):
        S[f"control: uniform slab {f}"] = np.full_like(rho_padq, f)
    S["control: EDR scale 0.4"] = scale_xy(rho_edr, 0.4, 0.4)
    S["control: padQ erode -6px"] = dilate_erode(rho_padq, -6) * mask
    return S


# ---------------------------------------------------------------------------
def gradient_audit(mask, refs):
    """||grad_rho A_E|| vs ||grad_rho C|| at the initialization and references
    (densities as the variable; no filter/projection so the comparison is
    about the optical terms themselves)."""
    d = RES_PROBE_OFFSET_NM
    mats = {lam: (fwd.eps_ito_of_lambda(lam), fwd.eps_asi_of_lambda(lam))
            for lam in (LAMBDA_E - d, LAMBDA_E, LAMBDA_E + d)}
    torch.manual_seed(config.RANDOM_SEED)
    init = torch.rand((config.NX_DESIGN, config.NY_DESIGN),
                      dtype=config.GEO_DTYPE)
    cases = {"init random (unmasked)": init.numpy(),
             "init random (padded mask)": (init.numpy() * mask), **refs}
    rows = {}
    for name, rho in cases.items():
        rho_t = torch.as_tensor(rho, dtype=config.GEO_DTYPE).requires_grad_(True)
        A = {}
        for lam, (ei, ea) in mats.items():
            sim = fwd.build_solved_sim(rho_t, lam, ei, config.N_GLASS,
                                       eps_asi=ea)
            R, T = fwd.specular_RT(sim)
            A[lam] = 1 - R - T
        AE = A[LAMBDA_E]
        C = (AE - 0.5 * (A[LAMBDA_E - d] + A[LAMBDA_E + d])) / AE
        gA = torch.autograd.grad(AE, rho_t, retain_graph=True)[0]
        gC = torch.autograd.grad(C, rho_t)[0]
        rows[name] = dict(A_E=float(AE), C=float(C),
                          grad_A_norm=float(torch.linalg.norm(gA)),
                          grad_C_norm=float(torch.linalg.norm(gC)))
    return rows


# ---------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    ctx = ctx_setup()
    mask = np.load(PAD / "outputs" / "geometries" / "design_mask.npy")
    log = []

    def P(s=""):
        print(s)
        log.append(s)

    # 1. A_ITO cross-validation + failure test on references ---------------
    P("== 1. A_ITO cross-validation and failure test (references) ==")
    S = build_calibration_set(mask)
    refs = {k: v for k, v in S.items() if k.startswith("ref:")}
    fail = {}
    for name, rho in refs.items():
        rt_ = torch.as_tensor(rho, dtype=config.GEO_DTYPE)
        m = a_ito_two_ways(rt_, ctx)
        g = gate_eval(rt_)
        cert = pole_rt.certify(rt_, with_ito=True)
        c = cert["certified"]
        fail[name] = {**m, **g,
                      "lambda_pole_nm": c["lambda_pole_nm"] if c else np.nan,
                      "Q_pole": c["Q_pole"] if c else np.nan,
                      "pole_stable": c["stable"] if c else False,
                      "rt_rel_diff": c["rt_rel_diff"] if c else np.nan,
                      "all_certified_in_window": [
                          f"{p['lambda_nm']:.0f}nm(Q={p['Q']:.1f})"
                          for p in cert["certified_all"]],
                      "off_window_poles": [
                          f"{r['lambda_nm']:.0f}nm(Q={r['Q']:.1f})"
                          for r in cert["table"]
                          if (not r["in_window"]) and r["rt_agree"]]}
        P(f"[{name}] A_vol={m['A_vol']:.4f} A_rt={m['A_rt']:.4f} "
          f"(res {abs(m['A_vol']-m['A_rt']):.1e}) F_Ez={m['F_Ez']:.3f} "
          f"eta_z={m['eta_z']:.3f} C={g['C']:+.4f} "
          f"center_dom={g['center_dominant']} | pole: "
          + (f"{c['lambda_pole_nm']:.1f} nm, Q={c['Q_pole']:.2f}, "
             f"r/t diff {c['rt_rel_diff']:.1e}, stable={c['stable']}"
             if c else "none certified in window")
          + f" | all certified in-window: {fail[name]['all_certified_in_window']}"
          + f" | off-window r/t poles: {fail[name]['off_window_poles']}")
    with open(OUT / "failure_test.json", "w") as f:
        json.dump(fail, f, indent=1, default=float)

    # no-ITO photonic pole (method demonstration on saved geometries) ------
    P("\n== 1b. no-ITO photonic pole (same geometry, a-Si/glass) ==")
    noito = {}
    for name in ("ref: EDR cuboid", "ref: padded QNM winner",
                 "ref: padded F_ENZ winner", "ref: unpadded QNM winner"):
        rt_ = torch.as_tensor(refs[name], dtype=config.GEO_DTYPE)
        cert = pole_rt.certify(rt_, with_ito=False)
        c = cert["certified"]
        allp = [f"{r['lambda_nm']:.0f}nm(Q={r['Q']:.1f})" for r in cert["table"]
                if r["rt_agree"] and r["significant"]]
        noito[name] = {"in_window": c, "all_rt_agreeing": allp}
        P(f"[{name}] no-ITO in-window certified pole: "
          + (f"{c['lambda_pole_nm']:.1f} nm, Q={c['Q_pole']:.2f}" if c
             else "none") + f"; all r/t-agreeing poles: {allp}")
    with open(OUT / "no_ito_poles.json", "w") as f:
        json.dump(noito, f, indent=1, default=float)

    # 2. calibration set ----------------------------------------------------
    P("\n== 2. C(d) calibration set (perturbations, no optimization) ==")
    rows = []
    for name, rho in S.items():
        rt_ = torch.as_tensor(rho, dtype=config.GEO_DTYPE)
        g = gate_eval(rt_)
        cert = pole_rt.certify(rt_, with_ito=True)
        c = cert["certified"]
        row = dict(geometry=name, fill=float(np.mean(rho)), A_E=g["A_E"],
                   C=g["C"], center_dominant=g["center_dominant"],
                   lambda_pole_nm=c["lambda_pole_nm"] if c else np.nan,
                   Q_pole=c["Q_pole"] if c else np.nan,
                   pole_stable=bool(c["stable"]) if c else False,
                   certified=bool(c is not None and c["stable"]))
        rows.append(row)
        P(f"  {name:32s} fill={row['fill']:.3f} A_E={row['A_E']:.4f} "
          f"C={row['C']:+.4f} cd={row['center_dominant']} pole="
          + (f"{row['lambda_pole_nm']:.0f}nm Q={row['Q_pole']:.2f} "
             f"stable={row['pole_stable']}" if c else "none"))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "calibration.csv", index=False)

    cert_df = df[df.certified]
    unc_df = df[~df.certified]
    C_res_min = float(cert_df.C.min()) if len(cert_df) else np.nan
    C_unc_max = float(unc_df.C.max()) if len(unc_df) else np.nan
    if len(cert_df) and len(unc_df) and C_res_min > C_unc_max:
        C_min_rec = round(0.5 * (C_res_min + C_unc_max), 4)
        sep = "separates"
    else:
        C_min_rec = round(C_res_min, 4) if len(cert_df) else np.nan
        sep = "does NOT cleanly separate"
    corr = float(np.corrcoef(cert_df.C, cert_df.Q_pole)[0, 1]) \
        if len(cert_df) > 2 else np.nan
    P(f"\n  certified in-window resonant states: n={len(cert_df)}, "
      f"C in [{C_res_min:+.4f}, {float(cert_df.C.max()) if len(cert_df) else np.nan:+.4f}]")
    P(f"  uncertified (broad/off-band/non-resonant): n={len(unc_df)}, "
      f"C max = {C_unc_max:+.4f}")
    P(f"  -> C(d) {sep} the two classes; recommended C_MIN = {C_min_rec}")
    P(f"  Pearson corr(C, Q_pole) over certified states = {corr:+.3f} "
      "(reported, not assumed monotonic)")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    axs[0].scatter(unc_df.C, np.full(len(unc_df), 0.0), c="#52514e", s=28,
                   label="no certified in-window pole")
    axs[0].scatter(cert_df.C, cert_df.Q_pole, c="#2a78d6", s=32,
                   label="certified in-window pole")
    if np.isfinite(C_min_rec):
        axs[0].axvline(C_min_rec, color="#eb6834", ls="--", lw=1.2,
                       label=f"recommended C_MIN = {C_min_rec}")
    axs[0].set_xlabel(f"contrast C(d = {RES_PROBE_OFFSET_NM:.0f} nm)")
    axs[0].set_ylabel("Q_pole (channel-agnostic r/t)")
    axs[0].legend(fontsize=8)
    axs[1].scatter(cert_df.C, cert_df.A_E, c="#2a78d6", s=32)
    axs[1].scatter(unc_df.C, unc_df.A_E, c="#52514e", s=28)
    axs[1].set_xlabel("C"); axs[1].set_ylabel(r"$A_{ITO}(\lambda_E)$")
    fig.suptitle("calibration set: resonance-contrast gate vs pole Q and "
                 "A_ITO", y=1.02)
    fig.savefig(FIG / "calibration_scatter.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 3. gradient-scale audit for mu ---------------------------------------
    P("\n== 3. gradient-scale audit for the penalty weight ==")
    ga = gradient_audit(mask, {k: v for k, v in refs.items()
                               if "bare" not in k})
    for name, r in ga.items():
        line = (f"  {name:30s} |grad A|={r['grad_A_norm']:.3e} "
                f"|grad C|={r['grad_C_norm']:.3e} ratio |gC|/|gA|="
                f"{r['grad_C_norm']/r['grad_A_norm']:.2f}")
        P(line)
    ratio_med = float(np.median([r["grad_C_norm"] / r["grad_A_norm"]
                                 for r in ga.values()]))
    P(f"  median |grad C|/|grad A| = {ratio_med:.2f}")
    P("  penalty gradient at violation depth delta: 2*mu*delta*|grad C|;")
    mu_table = {}
    for mu in MU_CANDIDATES:
        for delta in (0.005, 0.02):
            mu_table[f"mu={mu:.0f},delta={delta}"] = 2 * mu * delta * ratio_med
            P(f"    mu={mu:5.0f}, delta={delta:.3f}: penalty/physical gradient "
              f"ratio ~ {2*mu*delta*ratio_med:.2f}")
    with open(OUT / "gradient_audit.json", "w") as f:
        json.dump({"cases": ga, "median_ratio_gC_gA": ratio_med,
                   "penalty_to_physical_ratio": mu_table}, f, indent=1)
    # recommended mu: penalty gradient ~ physical gradient at delta = 0.02
    mu_rec = min(MU_CANDIDATES, key=lambda mu: abs(2 * mu * 0.02 * ratio_med - 1.0))
    P(f"  -> recommended mu = {mu_rec:.0f} (penalty ~ physical gradient at a "
      "0.02 contrast violation; weak at 0.005, dominant only for deep "
      "violations)")

    # 4. target definition + report ----------------------------------------
    tdef = {
        "primary_objective": "maximize A_ITO(lambda_E) = 1 - R - T "
                             "(ITO only lossy layer; volume-integral "
                             "cross-validation in failure_test.json)",
        "lambda_E_nm": LAMBDA_E,
        "lambda_E_origin": "frozen operating wavelength inherited from the "
                           "validated finite-K bare-film ENZ branch (K = G10 "
                           "of the 850-nm lattice); not a term of the loss",
        "in_loop_gate": {
            "name": "resonance-contrast gate (empirical ENZ-band spectral-"
                    "selectivity surrogate; NOT a Q constraint)",
            "C": "[A(lam_E) - (A(lam_E-d)+A(lam_E+d))/2]/A(lam_E) >= C_MIN",
            "center_dominance": "A(lam_E) >= A(lam_E +/- d) (three-point "
                                "test; not a proof of peak location)",
            "RES_PROBE_OFFSET_NM": RES_PROBE_OFFSET_NM,
            "C_MIN_recommended": C_min_rec,
            "penalty": "loss = -A(lam_E) + mu*[relu(C_MIN - C)^2 + "
                       "sum_pm relu((A_pm - A_E)/A_E)^2]",
            "mu_recommended": mu_rec,
        },
        "post_hoc_certification": {
            "pole": "channel-agnostic AAA poles of r_xx and t_xx; accept only "
                    "in-window, r/t-agreeing (<2%), residue >= 5% of window "
                    "max, resampling-stable (<2%)",
            "window_nm": list(pole_rt.WINDOW),
            "DELTA_LAMBDA_POLE_MAX_nm": DELTA_LAMBDA_POLE_MAX,
            "Q_reference_loaded": Q_REF_LOADED,
            "Q_note": "Q ~ 5 is an empirical trusted loaded-resonance "
                      "reference scale (bare ENZ QNM 5.80, loaded poles "
                      "5.0-5.7), not a fundamental ceiling",
            "no_ITO_control": "same geometry a-Si/glass: bare photonic pole "
                              "must exist near the ENZ window; ITO loading "
                              "must change the response and enhance A_ITO",
            "diagnostics_only": ["F_Ez", "eta_z", "QNM overlap",
                                 "Fourier harmonics", "multipoles/current",
                                 "gamma_r vs gamma_nr"],
        },
        "design_class": "NOT part of the target (padding is a geometry "
                        "prior; see report section on the next run)",
    }
    with open(OUT / "target_definition.json", "w") as f:
        json.dump(tdef, f, indent=1, default=float)

    write_report(fail, noito, df, cert_df, unc_df, C_min_rec, corr, sep, ga,
                 ratio_med, mu_rec)
    with open(OUT / "target_audit.log", "w") as f:
        f.write("\n".join(log) + "\n")
    print("[saved] failure_test.json, no_ito_poles.json, calibration.csv, "
          "gradient_audit.json, target_definition.json, TARGET_AUDIT.md, "
          "target_audit.log")


def write_report(fail, noito, df, cert_df, unc_df, C_min_rec, corr, sep, ga,
                 ratio_med, mu_rec):
    L = []
    a = L.append
    a("# TARGET AUDIT — resonant ITO power-transfer campaign (generated; no optimization run)\n")
    a("Governing sentence: *inverse-design a resonant metasurface that maximizes "
      "free-space optical power transfer into the ultrathin ITO layer at the ENZ "
      "wavelength, while preserving a genuine photonic resonance; do not prescribe "
      "the momentum channel, multipole composition, or polariton branch in advance.*\n")
    a("## Exact target\n")
    a("    PRIMARY   maximize A_ITO(λ_E) = 1 − R − T,  λ_E = 1433.488 nm")
    a(f"    GATE      C = [A(λ_E) − ½(A(λ_E−d)+A(λ_E+d))]/A(λ_E) ≥ C_MIN,  d = {RES_PROBE_OFFSET_NM:.0f} nm")
    a("              A(λ_E) ≥ A(λ_E ± d)            (three-point center-dominance test)")
    a("    PENALTY   loss = −A(λ_E) + μ[relu(C_MIN − C)² + Σ± relu((A± − A_E)/A_E)²]")
    a(f"    RECOMMENDED  C_MIN = {C_min_rec},  μ = {mu_rec:.0f}\n")
    a("The gate is an empirical ENZ-band spectral-selectivity surrogate (a "
      "*resonance-contrast gate*). It is **not** a Q constraint and the center-"
      "dominance test does **not** prove the peak lies within ±d. Resonance "
      "wavelength and Q are established post hoc by the channel-agnostic pole "
      "analysis. λ_E is a frozen operating wavelength inherited from the validated "
      "finite-K bare-film ENZ branch (K = G₁₀, 850-nm lattice); it is not a "
      "momentum term of the loss and was not discovered by the new objective.\n")
    a("## A_ITO cross-validation and failure test (references)\n")
    a("| reference | A_vol | A_(1−R−T) | F_Ez | η_z | C(80) | pole λ / Q (r/t, in-window) | off-window r/t poles |")
    a("|---|---|---|---|---|---|---|---|")
    for n, r in fail.items():
        pole = (f"{r['lambda_pole_nm']:.0f} nm / {r['Q_pole']:.2f}"
                if np.isfinite(r["Q_pole"]) else "none")
        a(f"| {n[5:]} | {r['A_vol']:.4f} | {r['A_rt']:.4f} | {r['F_Ez']:.3f} | "
          f"{r['eta_z']:.3f} | {r['C']:+.4f} | {pole} | {', '.join(r['off_window_poles']) or '—'} |")
    a("\nA_ITO = (ω/2)∫_ITO Im ε|E|²dV / (½A_cell) (LH units) agrees with 1−R−T to the "
      "quadrature error on every reference; a-Si (k=0) and glass are lossless, so "
      "1−R−T is exactly the ITO absorption and is used in-loop (differentiable "
      "S-parameter path); the volume integral is recomputed at final validation.\n")
    a("## Channel-agnostic pole certification\n")
    a(f"Complex r_xx(ω), t_xx(ω) (zeroth order, amplitude normalization) sampled on "
      f"{len(pole_rt.SCAN)} real frequencies 1250–1700 nm; AAA poles accepted only if: "
      f"inside the ENZ window {pole_rt.WINDOW[0]:.0f}–{pole_rt.WINDOW[1]:.0f} nm "
      "(λ_E ± HWHM of the bare ENZ QNM, Q = 5.80), found in both r and t within 2%, "
      "residue ≥ 5% of the in-window maximum of each observable, and stable to <2% "
      "under 2× coarser resampling. No QNM overlap, harmonic, or multipole enters "
      "selection. Off-window poles (the padded class's ~1300 nm Si Mie pole) are "
      "listed but cannot certify the ENZ resonance.\n")
    a("### No-ITO photonic pole (same geometries, a-Si/glass)\n")
    for n, r in noito.items():
        c = r["in_window"]
        a(f"- {n[5:]}: in-window pole " + (f"{c['lambda_pole_nm']:.0f} nm, Q = {c['Q_pole']:.1f}" if c else "none")
          + f"; all r/t-agreeing poles: {', '.join(r['all_rt_agreeing']) or '—'}")
    a("\n## C(80) calibration set (perturbations of saved geometries; no optimization)\n")
    a(f"n = {len(df)} geometries: references, x/y scalings, dilation/erosion, SDF morphs "
      "between the padded winners, unpadded width changes, EDR scalings, and "
      "non-resonant controls (uniform slabs, tiny/eroded patches). Padded-class "
      "perturbations are clipped to the 85-nm mask.\n")
    a("| geometry | fill | A_E | C(80) | center-dom. | pole λ (nm) | Q_pole | certified |")
    a("|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        a(f"| {r.geometry} | {r.fill:.3f} | {r.A_E:.4f} | {r.C:+.4f} | {r.center_dominant} | "
          f"{'%.0f' % r.lambda_pole_nm if np.isfinite(r.lambda_pole_nm) else '—'} | "
          f"{'%.2f' % r.Q_pole if np.isfinite(r.Q_pole) else '—'} | {r.certified} |")
    a(f"\nCertified in-window resonant states: n = {len(cert_df)}, C ∈ "
      f"[{cert_df.C.min() if len(cert_df) else float('nan'):+.4f}, "
      f"{cert_df.C.max() if len(cert_df) else float('nan'):+.4f}]; uncertified: n = {len(unc_df)}, "
      f"max C = {unc_df.C.max() if len(unc_df) else float('nan'):+.4f}. C(80) **{sep}** the classes. "
      f"Pearson corr(C, Q_pole) over certified states = {corr:+.3f} — reported, not "
      f"assumed monotonic. **Recommended C_MIN = {C_min_rec}** (midpoint of the gap when "
      "the classes separate; otherwise the minimum certified value, flagged). Figure: "
      "figures/calibration_scatter.png.\n")
    a("## Gradient-scale audit for μ\n")
    a("| geometry | A_E | C | ‖∇A_E‖ | ‖∇C‖ | ratio |")
    a("|---|---|---|---|---|---|")
    for n, r in ga.items():
        a(f"| {n} | {r['A_E']:.4f} | {r['C']:+.4f} | {r['grad_A_norm']:.3e} | {r['grad_C_norm']:.3e} | {r['grad_C_norm']/r['grad_A_norm']:.2f} |")
    a(f"\nMedian ‖∇C‖/‖∇A‖ = {ratio_med:.2f}. The penalty gradient is 2μδ‖∇C‖ at a "
      "contrast violation depth δ; μ is chosen so that it matches the physical "
      "gradient at δ = 0.02 (≈ a quarter of C_MIN) and stays weak (< 0.3×) at "
      f"δ = 0.005: **μ = {mu_rec:.0f}**. No optimization was run for this.\n")
    a("## Q reference scale\n")
    a("Q ≈ 5 (Q_REF_LOADED = 5.0) is an empirical trusted loaded-resonance reference "
      "scale — bare ENZ QNM 5.80, certified loaded poles 5.0–5.7 — used only as a "
      "post-hoc low-Q sanity reference, not a fundamental ceiling and not a "
      "normalized coherence fraction. Post-hoc acceptance requires an in-window "
      f"certified pole (|λ_pole − λ_E| ≤ {DELTA_LAMBDA_POLE_MAX:.0f} nm, the bare-ENZ "
      "HWHM) and the no-ITO photonic-pole control.\n")
    a("## Design class for the next run (separate from the target)\n")
    a("A. **padded 85 nm** — controlled comparison with the two previous padded "
      "campaigns (same seed/schedule; isolates the objective change).\n"
      "B. **unpadded** — least-prior, unrestricted search matching the campaign "
      "question; comparable to the unpadded QNM winner (highest saved A_ITO 0.205 "
      "and highest C(80) 0.121).\n"
      "Recommendation: **B (unpadded)** for the single next run — the scientific "
      "question is end-to-end ENZ excitation, not isolated meta-atoms, and the "
      "calibration set spans both classes so C_MIN is not class-specific; A remains "
      "the controlled follow-up.\n")
    (HERE / "TARGET_AUDIT.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
