"""Sanity tests of the activation extension.  Every new metric has at least one check; the results are
written to outputs/tests/sanity_tests.json with pass/fail, tolerances and the numbers.

    python tests/sanity_tests.py [--threads 4] [--quick]

Tests
  T1  perturbation layer reduces EXACTLY to the supplied ITO_nk.csv at 300 K, and equals the native
      ito_nonlinear.ITOHot.eps(lambda, Te) for Te = 1000 K (no second material model)
  T2  eps1 = eps0  ->  dT, dR, dA, dFz vanish to numerical precision (freeform design, real ITO)
  T3  energy / identity: R + T + A = 1 exactly by construction, |F_tot - A| within the repository's
      validated tolerance (config.TOL identity_abs) for cold AND perturbed states; lossless parent
      R + T = 1 to config.TOL energy_abs
  T4  sign convention of "positive transmission activation" on the PLANAR air / ITO / glass film,
      checked against an INDEPENDENT transfer-matrix (Fresnel) calculation written here: T_TMM equals
      T_RCWA (both states) and dT has the same sign; at normal incidence E_z = 0 in the planar ITO so the
      change is a pure Im(eps) / Re(eps) tangential effect (analytic control of the sign)
  T5  qdz_parent baseline reproducibility: parent_metrics on a certified candidate at the certification
      order reproduces certification.json (eta_Dz, eta_Dz_P, R, T) and Q_r / lambda_r are read from the
      exact pole, never from a proxy; an ill-posed proxy is refused by the selection code
  T6  total vs specular transmission: in the single-order window (lambda > n_glass P) the per-order
      table has exactly one propagating order in each medium and T_total == T_00; below it they differ
  T7  cold and perturbed states use identical geometry, wavelength, order, polarization, incidence and
      z-quadrature (structural check on the two sims) and the ITO layer index is the same
  T8  angle hook: theta = 0 through the generic path equals the default path; theta = 3 deg runs and
      obeys R + T + A = 1 with the oblique Rayleigh set (no ensemble is run)
  T9  Rayleigh wavelengths: dispersive solver vs the fixed-index analysis.rayleigh_wavelengths agree to
      < 0.5 nm for P = 825 and 588 nm
  T10 background-set logic: samples inside +-k FWHM of every pole are excluded, the cap is recorded, and
      an over-broad pole makes the background 'undefined' instead of silently returning a number
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as cm                       # noqa: E402
import config                             # noqa: E402
import forward as fwd                     # noqa: E402
import materials as mat                   # noqa: E402
import analysis as an                     # noqa: E402
import parent_fwd as pf                   # noqa: E402
import candidates as cd                   # noqa: E402
import ito_perturbation as ip             # noqa: E402
import ito_nonlinear as nl                # noqa: E402
import loaded_sensitivity as ls           # noqa: E402
import parent_background as pb            # noqa: E402
import rayleigh as ry                     # noqa: E402

GEO = config.GEO_DTYPE
RESULTS = {}


def record(name, passed, **vals):
    RESULTS[name] = dict(passed=bool(passed), **vals)
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: " + ", ".join(f"{k}={v}" for k, v in vals.items() if k != "note"), flush=True)


# ---------------------------------------------------------------------------------------------
def tmm_planar(lam, eps_ito, d_ito, n_glass):
    """Independent transfer-matrix T, R of air / ITO(d) / glass at normal incidence (exp(-i w t))."""
    k0 = 2 * np.pi / lam
    n1, n2, n3 = 1.0, np.sqrt(eps_ito + 0j), n_glass
    # characteristic matrix of the film
    kz = k0 * n2
    M = np.array([[np.cos(kz * d_ito), -1j * np.sin(kz * d_ito) / n2], [-1j * n2 * np.sin(kz * d_ito), np.cos(kz * d_ito)]])
    B, C = M @ np.array([1.0, n3])
    r = (n1 * B - C) / (n1 * B + C)
    t = 2 * n1 / (n1 * B + C)
    return float(abs(t) ** 2 * n3 / n1), float(abs(r) ** 2)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(); fwd.set_threads(a.threads)
    t0 = time.time()
    lam_E = pf.lambda_E()
    c = cd.by_tag("fr_Q200_fresh4242_h700"); rho = cd.get_rho(c); rho_t = torch.as_tensor(rho, dtype=GEO)
    P, h = c["P"], c["h"]
    order = [7, 7]

    # ---- T1: material layer -------------------------------------------------------------------
    p300 = ip.ITOPerturbation("hot_electron_Te", Te=300.0)
    lams = np.arange(1150.0, 1400.1, 5.0)
    dev300 = max(abs(p300.eps1(l) - mat.eps_ito(l)) for l in lams)
    p1000 = ip.ITOPerturbation("hot_electron_Te", Te=1000.0); m = ip.hot_model()
    dev1000 = max(abs(p1000.eps1(l) - m.eps(l, 1000.0)) for l in lams)
    record("T1_material_layer_reduces_to_supplied_csv_and_native_model", dev300 < 1e-12 and dev1000 < 1e-12,
           max_dev_300K=dev300, max_dev_vs_ITOHot_1000K=dev1000, delta_drude_weight_1000K=p1000.delta,
           eps0_lamE=p1000.eps0(lam_E), eps1_lamE=p1000.eps1(lam_E), direction_deg=p1000.describe()["direction_deg_at_lambda_E"])

    # ---- T2: dT = 0 at eps1 = eps0 -------------------------------------------------------------
    r0 = ls.evaluate_pair(rho_t, P, h, lam_E, order, p300)
    record("T2_dT_vanishes_for_identical_states", abs(r0["dT"]) < 1e-13 and abs(r0["dR"]) < 1e-13 and abs(r0["Fz1"] - r0["Fz0"]) < 1e-13,
           dT=r0["dT"], dR=r0["dR"], dA=r0["dA"], dFz=r0["Fz1"] - r0["Fz0"], tol=1e-13)

    # ---- T3: energy / identity ----------------------------------------------------------------
    r1 = ls.evaluate_pair(rho_t, P, h, lam_E, order, p1000)
    closure0 = r1["R0"] + r1["T0"] + r1["A0"] - 1.0; closure1 = r1["R1"] + r1["T1"] + r1["A1"] - 1.0
    with torch.no_grad():
        sim = fwd.build_sim(rho_t, P, h, lam_E, order, with_ito=False); R, T = fwd.rt_all_orders(sim)
    par_resid = float(R + T - 1.0)
    ok = abs(closure0) < 1e-12 and abs(closure1) < 1e-12 and abs(r1["resid0"]) < config.TOL["identity_abs"] and abs(r1["resid1"]) < config.TOL["identity_abs"] and abs(par_resid) < config.TOL["energy_abs"]
    record("T3_energy_conservation_and_absorption_identity", ok, closure_cold=closure0, closure_hot=closure1,
           Ftot_minus_A_cold=r1["resid0"], Ftot_minus_A_hot=r1["resid1"], tol_identity=config.TOL["identity_abs"],
           parent_R_plus_T_minus_1=par_resid, tol_energy=config.TOL["energy_abs"])

    # ---- T4: sign convention on the planar film vs independent TMM -------------------------------
    rows = []
    ok = True
    for lam in (1280.0, lam_E, 1340.0):
        rr = ls.evaluate_pair(None, 825.0, 100.0, lam, [3, 3], p1000)
        ng = mat.n_glass(lam)
        T0m, R0m = tmm_planar(lam, p1000.eps0(lam), config.D_ITO_NM, ng)
        T1m, R1m = tmm_planar(lam, p1000.eps1(lam), config.D_ITO_NM, ng)
        dTm = T1m - T0m
        same_sign = np.sign(dTm) == np.sign(rr["dT"])
        ok &= same_sign and abs(T0m - rr["T0"]) < 1e-9 and abs(T1m - rr["T1"]) < 1e-9 and abs(R0m - rr["R0"]) < 1e-9
        rows.append(dict(lam=lam, T0_rcwa=rr["T0"], T0_tmm=T0m, T1_rcwa=rr["T1"], T1_tmm=T1m, dT_rcwa=rr["dT"], dT_tmm=dTm,
                         R0_rcwa=rr["R0"], R0_tmm=R0m, Ez2_in_ito=rr["mean_Ez2_0"], same_sign=bool(same_sign)))
    record("T4_sign_convention_planar_film_vs_independent_TMM", ok, rows=rows,
           note="dT > 0 means the perturbed state transmits MORE; planar film at normal incidence has Ez = 0 in the ITO")

    # ---- T5: baseline reproducibility + proxy refusal -------------------------------------------
    cert = cm.jload(c["cert_path"])
    byo = {tuple(x["order"]): x for x in cert["metrics"]["lambda_E"]["by_order"]}
    diffs = {}
    for od in ([5, 5], [7, 7]) if a.quick else ([5, 5], [7, 7], [9, 9]):
        f = pf.to_floats(pf.parent_metrics(rho_t, P, h, lam_E, od, n_z=9))
        ref = byo[tuple(od)]
        diffs[str(od)] = {k: abs(f[k] - ref[k]) / max(abs(ref[k]), 1e-30) for k in ("eta_Dz", "eta_Dz_P", "R", "T", "U_mid")}
    worst = max(v for d in diffs.values() for v in d.values())
    # exact pole, not proxy: the registry exposes Q_r from the pole and the proxy separately with its well_posed flag
    reg = c["cert"]
    proxy_refused = (reg["Q_r"] == cert["pole_nearest_lambda_E"]["Q"]) and ("Q_proxy_well_posed" in reg)
    bad = cd.by_tag("fr_Q100_final3_h640")["cert"]          # certified with an ill-posed proxy (Q_proxy 7.9e12)
    record("T5_qdz_parent_baseline_reproduced_and_proxy_not_used", worst < 1e-9 and proxy_refused and bad["Q_proxy_well_posed"] is False and bad["Q_r"] < 1e4,
           max_rel_diff_vs_certification=worst, per_order=diffs, Q_r_from_exact_pole=reg["Q_r"], Q_proxy=reg["Q_proxy"],
           ill_posed_example=dict(tag="fr_Q100_final3_h640", Q_proxy=bad["Q_proxy"], well_posed=bad["Q_proxy_well_posed"], Q_r_exact=bad["Q_r"]))

    # ---- T6: total vs specular -------------------------------------------------------------------
    lamR = ry.rayleigh_wavelengths(825.0, lo=900, hi=1400)[-1]["lam"]
    above = pb.per_order_table(rho_t, P, h, lamR + 5.0, order, with_ito=True)
    at_E = pb.per_order_table(rho_t, P, h, lam_E, order, with_ito=True)
    below = pb.per_order_table(rho_t, P, h, lamR - 5.0, order, with_ito=True)
    ok = above["n_propagating_orders"] == 1 and at_E["n_propagating_orders"] == 1 and abs(above["total_minus_specular_T"]) < 1e-14 \
        and abs(at_E["total_minus_specular_T"]) < 1e-14 and below["n_propagating_orders"] > 1 and below["total_minus_specular_T"] > 0
    record("T6_total_equals_specular_in_single_order_window", ok, lambda_R_glass=lamR,
           above=dict(lam=above["lam"], n_orders=above["n_propagating_orders"], T_tot=above["T_tot"], T_00=above["T_00"]),
           at_lamE=dict(lam=at_E["lam"], n_orders=at_E["n_propagating_orders"], T_tot=at_E["T_tot"], T_00=at_E["T_00"]),
           below=dict(lam=below["lam"], n_orders=below["n_propagating_orders"], T_tot=below["T_tot"], T_00=below["T_00"], diff=below["total_minus_specular_T"]))

    # ---- T7: identical settings for the two states ------------------------------------------------
    with torch.no_grad():
        s0 = fwd.build_sim(rho_t, P, h, lam_E, order, eps_ito=p1000.eps0(lam_E))
        s1 = fwd.build_sim(rho_t, P, h, lam_E, order, eps_ito=p1000.eps1(lam_E))
    same = all(getattr(s0, k) == getattr(s1, k) for k in ("_P", "_h", "_lam", "_d_ito", "_amp_ps", "_theta_deg", "_phi_deg", "_ito_layer", "_eps_asi", "_n_glass", "order_N"))
    same &= bool(torch.equal(s0.eps_conv[0], s1.eps_conv[0])) and (s0._eps_ito != s1._eps_ito) and list(s0.order) == list(s1.order)
    record("T7_cold_and_perturbed_sims_identical_except_eps_ITO", same, eps_cold=s0._eps_ito, eps_hot=s1._eps_ito,
           ito_layer=s0._ito_layer, thickness_layers=[float(x) for x in s0.thickness])

    # ---- T8: angle hook ----------------------------------------------------------------------------
    rd = ls.evaluate_pair(rho_t, P, h, lam_E, order, p1000)
    rg = ls.evaluate_pair(rho_t, P, h, lam_E, order, p1000, theta_deg=0.0, phi_deg=0.0, pol="labx")
    ro = ls.evaluate_pair(rho_t, P, h, lam_E, order, p1000, theta_deg=3.0, phi_deg=0.0, pol="labx")
    ray_o = ry.rayleigh_wavelengths(825.0, lo=900, hi=1400, theta_deg=3.0)
    ok = abs(rd["dT"] - rg["dT"]) < 1e-14 and abs(ro["R0"] + ro["T0"] + ro["A0"] - 1) < 1e-12 and abs(ro["resid0"]) < 5 * config.TOL["identity_abs"] and len(ray_o) > 1
    record("T8_angle_hook", ok, dT_default=rd["dT"], dT_theta0=rg["dT"], theta3=dict(T0=ro["T0"], dT=ro["dT"], A0=ro["A0"], resid=ro["resid0"], n_orders=ro["n_orders0"]),
           rayleigh_theta3=[(r["m"], r["n"], r["medium"], round(r["lam"], 2)) for r in ray_o])

    # ---- T9: Rayleigh --------------------------------------------------------------------------------
    ok, rows = True, []
    for Pp in (825.0, 588.0):
        d = ry.rayleigh_wavelengths(Pp, lo=600.0, hi=1400.0)
        f = an.rayleigh_wavelengths(Pp, float(mat.n_glass(1300.0)), lo=600.0, hi=1400.0)
        for x in d:
            near = min(f, key=lambda y: abs(y["lam"] - x["lam"]))
            in_window = x["lam"] >= 1000.0          # the fixed index is n_glass(1300 nm): only meaningful near the window
            ok &= (abs(near["lam"] - x["lam"]) < 0.5) if in_window else True
            rows.append(dict(P=Pp, m=x["m"], n=x["n"], medium=x["medium"], dispersive=x["lam"], fixed_index=near["lam"], compared=in_window))
    record("T9_rayleigh_dispersive_vs_fixed_index", ok, rows=rows)

    # ---- T10: background-set logic ---------------------------------------------------------------------
    lams_s = np.arange(1160.0, 1400.1, 2.0); T_s = np.full_like(lams_s, 0.9); R_s = 1 - T_s
    poles = [dict(lam=1300.0, Q=100.0, fwhm=13.0), dict(lam=1206.0, Q=20.0, fwhm=58.0)]
    bg = pb.background_set(lams_s, T_s, R_s, poles, lam_E, 1251.2, D_op=50.0, m_R=10.0, k_excl=3.0)
    hw_used = min(3 * 13.0, bg["parameters"]["hw_max_nm"])
    inside = any(abs(l - 1300.0) <= hw_used for l in bg["lambdas_nm"])
    bg2 = pb.background_set(lams_s, T_s, R_s, [dict(lam=1300.0, Q=8.0, fwhm=160.0)], lam_E, 1251.2, D_op=50.0, m_R=10.0, k_excl=3.0, hw_max=1e9)
    record("T10_background_set_logic", (not inside) and bg["defined"] and bg["exclusions"][1]["cap_active"] and (not bg2["defined"]) and bg2["T_bg_median"] is None,
           n_points=bg["n_points"], window=bg["window"], cap_active_on_broad_pole=bg["exclusions"][1]["cap_active"],
           undefined_when_overbroad=(not bg2["defined"]), note_overbroad=bg2["note"])

    n_pass = sum(r["passed"] for r in RESULTS.values())
    out = dict(n_tests=len(RESULTS), n_passed=n_pass, all_passed=(n_pass == len(RESULTS)), wall_s=time.time() - t0,
               candidate_used=c["tag"], order=order, results=RESULTS, provenance=cm.provenance())
    cm.jdump(out, cm.OUT / "tests" / "sanity_tests.json")
    print(f"\n{n_pass}/{len(RESULTS)} passed ({time.time()-t0:.0f} s) -> outputs/tests/sanity_tests.json")
    return 0 if out["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
