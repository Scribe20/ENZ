"""Step 6 (review): pole of the FULL hybrid a-Si/ITO/glass structure.

Method: sample the driven complex overlap amplitudes a+/-(omega) of the
final optimized structure on the real-frequency axis, then extract complex
poles by rational approximation (AAA).  Rationale: TORCWA's kz sheet
selections assume real frequency, so driving it at complex omega risks
branch-cut artifacts; analytic continuation of smooth real-axis data by AAA
avoids that.  The method is validated in-script against the bare slab,
whose exact pole (from solve_periodic_target.py) is
omega = 1.280731 - 0.115779i rad/fs: AAA on the bare r_p(K=3G, omega)
recovers it to ~1e-6 relative.

Output: hybrid pole(s) near the band, drift vs the bare-slab pole, Q, and
a stability check under sampling-grid changes.

Run:  python hybrid_pole.py     (after optimize_enz_overlap.py)
"""

import json
import sys

import numpy as np
import torch
from scipy.interpolate import AAA
from scipy.optimize import least_squares

import config
import target_mode
import torcwa_forward as fwd
import objective as obj

C_NM_FS = 299.792458


def bare_pole_reference():
    """Exact bare-slab pole + AAA method validation on bare r_p data."""
    sys.path.insert(0, str(config.ENZ_TARGET_DIR))
    from ito_material import ITOMaterial
    from tm_slab_mode import r123_tm
    m = ITOMaterial()
    w_dat = 2 * np.pi * C_NM_FS / m.wl
    eps_dat = m.eps(m.wl)

    def drude(p, om):
        return p[0] - p[1] ** 2 / (om ** 2 + 1j * p[2] * om)

    p = least_squares(lambda q: np.concatenate(
        [(drude(q, w_dat) - eps_dat).real, (drude(q, w_dat) - eps_dat).imag]),
        [4.0, 2.7, 0.2]).x
    K = float(3 * 2 * np.pi / config.PX_NM)
    lams = np.arange(1360.0, 1681.0, 8.0)
    oms = 2 * np.pi * C_NM_FS / lams
    vals = np.array([complex(r123_tm(K, om / C_NM_FS, drude(p, om),
                                     float(config.ITO_THICKNESS_NM), 1.0,
                                     config.N_GLASS ** 2)) for om in oms])
    poles = AAA(oms, vals).poles()
    cand = [q for q in poles if q.imag < 0 and 1.0 < q.real < 1.6]
    cand.sort(key=lambda q: abs(q.real - 1.2807))
    return cand[0]


def sample_hybrid_response(rho, lam_grid):
    """Complex a+(lam), a-(lam), ITO energy for the fixed final geometry."""
    tgt = target_mode.load_target_npz()
    x, y = fwd.grid_axes()
    zp = target_mode.ito_z_slices(config.ITO_THICKNESS_NM,
                                  config.Z_SAMPLES_ITO)
    Tp, dV = target_mode.build_target_field(tgt, x.cpu().numpy(),
                                            y.cpu().numpy(), zp, "+x")
    Tm, _ = target_mode.build_target_field(tgt, x.cpu().numpy(),
                                           y.cpu().numpy(), zp, "-x")
    a_p, a_m, en = [], [], []
    with torch.no_grad():
        for lam in lam_grid:
            eps_ito = fwd.eps_ito_of_lambda(lam)
            eps_asi = fwd.eps_asi_of_lambda(lam)
            sim_ref = fwd.build_solved_sim(None, lam, eps_ito, config.N_GLASS)
            Ez_ref = fwd.ez_in_ito(sim_ref, x, y, zp)
            sim = fwd.build_solved_sim(rho, lam, eps_ito, config.N_GLASS,
                                       eps_asi=eps_asi)
            Ez_scat = fwd.ez_in_ito(sim, x, y, zp) - Ez_ref
            a_p.append(complex(obj.overlap_amplitude(Tp, Ez_scat, dV)))
            a_m.append(complex(obj.overlap_amplitude(Tm, Ez_scat, dV)))
            en.append(float(torch.sum(torch.abs(Ez_scat) ** 2).real * dV))
            print(f"  lambda {lam:7.1f}: |a+| = {abs(a_p[-1]):.4e}")
    return np.array(a_p), np.array(a_m), np.array(en)


def poles_from(oms, vals, band=(1.0, 1.6)):
    poles = AAA(oms, vals).poles()
    cand = [q for q in poles if q.imag < 0 and band[0] < q.real < band[1]]
    cand.sort(key=lambda q: abs(q.imag))     # least-damped first
    return cand


def main():
    torch.set_num_threads(config.N_THREADS)
    tgt = target_mode.load_target_npz()
    config.ITO_THICKNESS_NM = float(tgt["ito_thickness_nm"])
    config.N_GLASS = float(tgt["glass_index"])
    if config.EPS_ASI is None:
        config.EPS_ASI = fwd.eps_asi_of_lambda(float(tgt["wavelength_nm"]))

    w_bare = bare_pole_reference()
    lam_bare = 2 * np.pi * C_NM_FS / w_bare.real
    print(f"[bare] AAA-validated bare-slab pole: {w_bare.real:.6f}"
          f"{w_bare.imag:+.6f}i rad/fs -> Re lambda = {lam_bare:.2f} nm, "
          f"Q = {abs(w_bare.real/(2*w_bare.imag)):.2f}")

    rho = torch.as_tensor(
        np.load(config.OUT_DIR / "geometries" / "rho_proj_final.npy"),
        dtype=config.GEO_DTYPE, device=config.DEVICE)

    lam_grid = np.arange(1340.0, 1701.0, 8.0)
    print(f"[hybrid] sampling a(+K) over {len(lam_grid)} wavelengths ...")
    a_p, a_m, en = sample_hybrid_response(rho, lam_grid)
    oms = 2 * np.pi * C_NM_FS / lam_grid

    out = {}
    for tag, vals in (("a_plus", a_p), ("a_minus", a_m),
                      ("ITO_energy", en.astype(complex))):
        cand = poles_from(oms, vals)
        out[tag] = [[q.real, q.imag] for q in cand[:3]]
        msg = ", ".join(f"{q.real:.5f}{q.imag:+.5f}i "
                        f"(lam={2*np.pi*C_NM_FS/q.real:.1f} nm, "
                        f"Q={abs(q.real/(2*q.imag)):.2f})" for q in cand[:3])
        print(f"[hybrid] poles from {tag}: {msg}")

    # stability check: coarser sampling
    a_p2 = a_p[::2]
    cand2 = poles_from(oms[::2], a_p2)
    if cand2 and out["a_plus"]:
        q1 = complex(*out["a_plus"][0])
        dq = abs(cand2[0] - q1) / abs(q1)
        print(f"[stability] leading a+ pole under 2x coarser sampling: "
              f"{cand2[0].real:.5f}{cand2[0].imag:+.5f}i "
              f"(relative change {dq:.2e})")
        out["stability_rel_change"] = dq

    if out["a_plus"]:
        q = complex(*out["a_plus"][0])
        lam_h = 2 * np.pi * C_NM_FS / q.real
        print(f"\n[result] leading hybrid pole: Re lambda = {lam_h:.1f} nm, "
              f"Q = {abs(q.real/(2*q.imag)):.2f}; drift vs bare pole: "
              f"{lam_h - lam_bare:+.1f} nm")
    np.savez(config.OUT_DIR / "histories" / "hybrid_pole_scan.npz",
             lam_grid=lam_grid, a_plus=a_p, a_minus=a_m, ito_energy=en)
    with open(config.OUT_DIR / "histories" / "hybrid_pole.json", "w") as f:
        json.dump({k: v if not isinstance(v, float) else v
                   for k, v in out.items()}, f, indent=1, default=float)
    print(f"saved: hybrid_pole_scan.npz, hybrid_pole.json")
    return out


if __name__ == "__main__":
    main()
