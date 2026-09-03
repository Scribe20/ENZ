"""TARGET AUDIT for the resonant ITO power-transfer campaign.

Governing sentence: "Inverse-design a resonant metasurface that maximizes
free-space optical power transfer into the ultrathin ITO layer at the ENZ
wavelength, while preserving a genuine photonic resonance; do not prescribe
the momentum channel, multipole composition, or polariton branch in
advance."

This script LAUNCHES NO OPTIMIZATION.  It:
  1. defines and cross-validates A_ITO (volume integral vs 1-R-T identity),
  2. runs the Section-14 failure test on all saved candidates
     (A_ITO, F_Ez, eta_z, lambda_res, Q - spectral and AAA-pole),
  3. derives Q_min and Delta_lambda_allowed from trusted references,
  4. prints the Section-13 A-L audit for user sign-off.

Run:  python target_audit.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import AAA

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / "enz_inverse_design"
PAD = HERE.parent / "enz_padding_sideexperiment"
EZC = HERE.parent / "enz_direct_enz_excitation"
sys.path.insert(0, str(PKG))

import config                        # noqa: E402
import target_mode                   # noqa: E402
import torcwa_forward as fwd         # noqa: E402
from validate_with_without_ito import build_sim, power_RT  # noqa: E402

C_NM_FS = 299.792458
OUT = HERE / "outputs"


def ctx_setup():
    torch.set_num_threads(config.N_THREADS)
    tgt = target_mode.load_target_npz()
    lam = float(tgt["wavelength_nm"])
    config.ITO_THICKNESS_NM = float(tgt["ito_thickness_nm"])
    config.N_GLASS = float(tgt["glass_index"])
    config.EPS_ASI = fwd.eps_asi_of_lambda(lam)
    x, y = fwd.grid_axes()
    zp = target_mode.ito_z_slices(config.ITO_THICKNESS_NM,
                                  config.Z_SAMPLES_ITO)
    dV = (config.PX_NM / config.NX_DESIGN) * (config.PY_NM / config.NY_DESIGN) \
        * (config.ITO_THICKNESS_NM / config.Z_SAMPLES_ITO)
    return dict(tgt=tgt, lam=lam, x=x, y=y, zp=zp, dV=dV,
                A_cell=config.PX_NM * config.PY_NM)


def e_all_in_ito(sim, ctx):
    comps = [[], [], []]
    for zpv in ctx["zp"]:
        E, _ = sim.field_xy(1, ctx["x"], ctx["y"], float(zpv))
        for c in range(3):
            comps[c].append(E[c])
    return [torch.stack(c, 0) for c in comps]


def a_ito_two_ways(rho_t, lam, ctx):
    """A_ITO via (a) the volume integral (omega/2)Im(eps)|E|^2 dV / P_inc
    (Lorentz-Heaviside: eps0 = 1, P_inc = 0.5*A_cell for |E_inc| = 1) and
    (b) the closure identity 1-R-T (exact here because a-Si k = 0 above the
    gap and glass is lossless, so ITO is the only dissipative layer)."""
    eps_ito = fwd.eps_ito_of_lambda(lam)
    eps_asi = fwd.eps_asi_of_lambda(lam)
    with torch.no_grad():
        sim = fwd.build_solved_sim(rho_t, lam, eps_ito, config.N_GLASS,
                                   eps_asi=eps_asi)
        Ex, Ey, Ez = e_all_in_ito(sim, ctx)
        E2 = float(torch.sum(torch.abs(Ex) ** 2 + torch.abs(Ey) ** 2
                             + torch.abs(Ez) ** 2).real * ctx["dV"])
        Iz = float(torch.sum(torch.abs(Ez) ** 2).real * ctx["dV"])
        omega = 2 * np.pi / lam
        A_vol = omega * eps_ito.imag * E2 / ctx["A_cell"]   # /(0.5A)*(w/2)
        R, T = power_RT(build_sim(rho_t, lam, True))
        A_rt = 1 - R - T
        v_ito = ctx["A_cell"] * config.ITO_THICKNESS_NM
        return dict(A_vol=A_vol, A_rt=A_rt, R=R, T=T,
                    F_Ez=Iz / v_ito, eta_z=Iz / E2)


def a_spectrum(rho_t, lams):
    A = []
    with torch.no_grad():
        for lam in lams:
            R, T = power_RT(build_sim(rho_t, lam, True))
            A.append(1 - R - T)
    return np.array(A)


def spectral_resonance(lams, A):
    """Peak wavelength, FWHM, and Q_spec = lambda/FWHM from A(lambda)."""
    i = int(np.argmax(A))
    lam_res, A_pk = float(lams[i]), float(A[i])
    base = float(np.min(A))
    half = base + (A_pk - base) / 2
    above = A >= half
    li = np.where(above)[0]
    lo_trunc = li[0] == 0
    hi_trunc = li[-1] == len(A) - 1
    fwhm = float(lams[li[-1]] - lams[li[0]])
    q = lam_res / fwhm if fwhm > 0 else np.inf
    return dict(lambda_res=lam_res, A_peak=A_pk, fwhm_nm=fwhm, Q_spec=q,
                truncated=bool(lo_trunc or hi_trunc))


def pole_q(rho_t, ctx, tag):
    """AAA pole of the complex driven a+(omega) response (validated method:
    residue ranking + minimum damping cut; bare-slab pole recovered to
    ~1e-6 with the same machinery)."""
    import objective as obj
    lam_grid = np.arange(1340.0, 1701.0, 8.0)
    Tp, dV = target_mode.build_target_field(
        ctx["tgt"], ctx["x"].cpu().numpy(),
        np.asarray(ctx["x"].cpu().numpy()), ctx["zp"], "+x")
    vals = []
    with torch.no_grad():
        for lam in lam_grid:
            eps_ito = fwd.eps_ito_of_lambda(lam)
            eps_asi = fwd.eps_asi_of_lambda(lam)
            sim_ref = fwd.build_solved_sim(None, lam, eps_ito, config.N_GLASS)
            Ez_ref = fwd.ez_in_ito(sim_ref, ctx["x"], ctx["x"], ctx["zp"])
            sim = fwd.build_solved_sim(rho_t, lam, eps_ito, config.N_GLASS,
                                       eps_asi=eps_asi)
            Ez = fwd.ez_in_ito(sim, ctx["x"], ctx["x"], ctx["zp"])
            vals.append(complex(obj.overlap_amplitude(Tp, Ez - Ez_ref, dV)))
    oms = 2 * np.pi * C_NM_FS / lam_grid
    fit = AAA(oms, np.array(vals))
    cand = [(q, r) for q, r in zip(fit.poles(), fit.residues())
            if q.imag < -0.005 and 1.05 < q.real < 1.45]
    cand.sort(key=lambda t: -abs(t[1]))
    if not cand:
        return dict(lambda_pole=np.nan, Q_pole=np.nan)
    q = cand[0][0]
    return dict(lambda_pole=2 * np.pi * C_NM_FS / q.real,
                Q_pole=abs(q.real / (2 * q.imag)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = ctx_setup()
    lam_E = ctx["lam"]
    eps_at = fwd.eps_ito_of_lambda(lam_E)

    candidates = {
        "bare ITO (no metasurface)": None,
        "EDR-like cuboid 560x500": "edr",
        "unpadded QNM winner": PKG / "outputs" / "geometries"
                               / "rho_hard_binary.npy",
        "padded QNM winner": PAD / "outputs" / "geometries"
                             / "rho_hard_binary.npy",
        "padded F_ENZ winner (Ez-objective)": EZC / "outputs" / "geometries"
                                              / "rho_hard_binary.npy",
    }

    nx, ny = config.NX_DESIGN, config.NY_DESIGN
    xg = (np.arange(nx) + 0.5) / nx * config.PX_NM
    yg = (np.arange(ny) + 0.5) / ny * config.PY_NM
    X, Y = np.meshgrid(xg, yg, indexing="ij")
    rho_edr = (((np.abs(X - config.PX_NM / 2) < 280.0)
                & (np.abs(Y - config.PY_NM / 2) < 250.0)).astype(float))

    lams = np.arange(1300.0, 1652.0, 4.0)
    rows = {}
    for name, src in candidates.items():
        if src is None:
            rho_t = torch.zeros((nx, ny), dtype=config.GEO_DTYPE)
        elif src == "edr":
            rho_t = torch.as_tensor(rho_edr, dtype=config.GEO_DTYPE)
        else:
            rho_t = torch.as_tensor(np.load(src), dtype=config.GEO_DTYPE)
        m = a_ito_two_ways(rho_t, lam_E, ctx)
        print(f"[{name}] A_ITO(volume) = {m['A_vol']:.4f}, "
              f"A_ITO(1-R-T) = {m['A_rt']:.4f} "
              f"(identity residual {abs(m['A_vol']-m['A_rt']):.1e}); "
              f"F_Ez = {m['F_Ez']:.3f}, eta_z = {m['eta_z']:.3f}")
        A = a_spectrum(rho_t, lams)
        sr = spectral_resonance(lams, A)
        print(f"    A(lambda): peak {sr['A_peak']:.4f} at "
              f"{sr['lambda_res']:.0f} nm, FWHM = {sr['fwhm_nm']:.0f} nm, "
              f"Q_spec = {sr['Q_spec']:.2f}"
              + (" [FWHM truncated by scan range]" if sr["truncated"] else ""))
        rows[name] = {**{k: v for k, v in m.items()}, **sr,
                      "A_lam": A.tolist()}

    # AAA pole Q for the two padded winners (driven-pole definition)
    for name in ("padded QNM winner", "padded F_ENZ winner (Ez-objective)"):
        src = candidates[name]
        rho_t = torch.as_tensor(np.load(src), dtype=config.GEO_DTYPE)
        pq = pole_q(rho_t, ctx, name)
        rows[name].update(pq)
        print(f"[{name}] AAA pole: lambda = {pq['lambda_pole']:.1f} nm, "
              f"Q_pole = {pq['Q_pole']:.2f}")

    with open(OUT / "failure_test.json", "w") as f:
        json.dump(rows, f, indent=1, default=float)
    np.save(OUT / "audit_lams.npy", lams)
    print(f"[saved] {OUT/'failure_test.json'}")

    # ------------------------------------------------------------------
    # Q_min and Delta_lambda derivation (trusted references)
    # ------------------------------------------------------------------
    Q_ENZ_BARE = 5.80        # validated bare-film QNM at K=G10(850)
    Q_HYBRID_770 = 5.04      # validated AAA hybrid pole of the 770 design
    Q_MIN = 5.0
    W_NM = lam_E / Q_MIN
    DL_ALLOWED = W_NM / 2
    print("\n[Q_min derivation]")
    print(f"  trusted resonant references: bare ENZ QNM at G10(850) Q = "
          f"{Q_ENZ_BARE} (solve_periodic_target, |D|=1.6e-15); loaded "
          f"hybrid ENZ pole Q = {Q_HYBRID_770} (AAA, stability 1.5e-5); "
          "EDR-like cuboid Mie resonance Q = 39 (bare) / 52 (loaded).")
    print(f"  Q_min = {Q_MIN}: an ENZ-coupled state cannot be sharper than "
          "the material-limited ENZ resonance by much (loaded ceiling ~5-6),"
          " so requiring Q >= 5 demands the full coherence of the ENZ "
          "resonance itself without forbidding ENZ loading (a Mie-only "
          "Q~39 criterion would exclude every ENZ-loaded state).")
    print(f"  Delta_lambda_allowed = lambda_E/(2*Q_min) = {DL_ALLOWED:.0f} nm"
          " (= the resonance's own HWHM at Q_min: a resonance detuned "
          "beyond its half width delivers < half its peak at lambda_E; "
          "note the objective A_ITO(lambda_E) itself already penalizes "
          "detuning, so this constraint is a guard, not a driver).")

    with open(OUT / "target_definition.json", "w") as f:
        json.dump({
            "objective": "maximize A_ITO(lambda_E) = 1-R-T (identity: ITO "
                         "is the only lossy layer; cross-validated against "
                         "the volume integral above)",
            "lambda_E_nm": lam_E,
            "eps_ito_at_lambda_E": [eps_at.real, eps_at.imag],
            "Q_min": Q_MIN, "W_nm": W_NM,
            "Delta_lambda_allowed_nm": DL_ALLOWED,
            "constraint_surrogate": "A(lambda_E +/- W/2) <= A(lambda_E)/2 "
                                    "(each side), differentiable, 2 extra "
                                    "solves/iteration; equivalent to "
                                    "spectral FWHM <= W i.e. Q_spec >= "
                                    "Q_min for a single-peaked A(lambda)",
            "penalty_form": "loss = -A(lam_E) + mu * sum_pm relu(A(lam_E "
                            "+/- W/2)/A(lam_E) - 0.5)^2, mu = 10 "
                            "(normalized, dimensionless; sensitivity mu in "
                            "{3,10,30} to be checked on the final candidate"
                            " only, not tuned to shape the geometry)",
            "final_Q_verification": "AAA pole of driven response + fine "
                                    "A(lambda) FWHM (surrogate is in-loop "
                                    "only)",
        }, f, indent=1)
    print(f"[saved] {OUT/'target_definition.json'}")


if __name__ == "__main__":
    main()
