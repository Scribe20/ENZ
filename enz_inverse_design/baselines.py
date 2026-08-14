"""Baseline comparison for the freeform result (review Step 5).

The freeform optimum is essentially a 3-period 1D binary grating, so the
honest baselines are:
  1. uniform (unpatterned) a-Si slabs of several fill fractions,
  2. ideal 3-period binary gratings (primitive period p/3 = 256.7 nm) over a
     duty-cycle sweep, same thickness and materials,
  3. the freeform-optimized design.

For each geometry the same forward model and metrics are evaluated:
  F (bidir overlap FoM), eta_pm (fraction of ITO Ez_scat energy in the
  +-K target subspace), B_ITO (ITO int|Ez|^2 dV / P_inc).

Run:  python baselines.py     (after optimize_enz_overlap.py)
"""

import json
from pathlib import Path

import numpy as np
import torch

import config
import target_mode
import torcwa_forward as fwd
import objective as obj


def evaluate(rho, ctx):
    with torch.no_grad():
        sim = fwd.build_solved_sim(rho, ctx["lam"], ctx["eps_ito"],
                                   config.N_GLASS)
        Ez_full = fwd.ez_in_ito(sim, ctx["x"], ctx["y"], ctx["zp"])
        Ez_scat = Ez_full - ctx["Ez_ref"]
        F, d = obj.enz_objective(ctx["Tp"], Ez_scat, ctx["dV"], ctx["p_inc"],
                                 target_minus=ctx["Tm"], direction="bidir")
        I_scat = float(torch.sum(torch.abs(Ez_scat) ** 2).real * ctx["dV"])
        eta = (float(torch.abs(d["a_plus"]) ** 2)
               + float(torch.abs(d["a_minus"]) ** 2)) / I_scat if I_scat else 0.0
        ito2 = float(torch.sum(torch.abs(Ez_full) ** 2).real * ctx["dV"])
        R, T = fwd.specular_RT(sim)
        return {"F": float(F), "eta_pm": eta, "B_ITO": ito2 / ctx["p_inc"],
                "R": float(R), "T": float(T), "A": float(1 - R - T)}


def main():
    torch.set_num_threads(config.N_THREADS)
    tgt = target_mode.load_target_npz()
    lam = float(tgt["wavelength_nm"])
    config.ITO_THICKNESS_NM = float(tgt["ito_thickness_nm"])
    config.N_GLASS = float(tgt["glass_index"])
    if config.EPS_ASI is None:
        config.EPS_ASI = fwd.eps_asi_of_lambda(lam)
    eps_ito = complex(float(tgt["eps_ito_real"]), float(tgt["eps_ito_imag"]))

    x, y = fwd.grid_axes()
    zp = target_mode.ito_z_slices(config.ITO_THICKNESS_NM,
                                  config.Z_SAMPLES_ITO)
    Tp, dV = target_mode.build_target_field(tgt, x.cpu().numpy(),
                                            y.cpu().numpy(), zp, "+x")
    Tm, _ = target_mode.build_target_field(tgt, x.cpu().numpy(),
                                           y.cpu().numpy(), zp, "-x")
    with torch.no_grad():
        Ez_ref = fwd.ez_in_ito(
            fwd.build_solved_sim(None, lam, eps_ito, config.N_GLASS),
            x, y, zp).detach()
    ctx = dict(lam=lam, eps_ito=eps_ito, x=x, y=y, zp=zp, Tp=Tp, Tm=Tm,
               dV=dV, p_inc=fwd.p_inc_cell(), Ez_ref=Ez_ref)

    nx, ny = config.NX_DESIGN, config.NY_DESIGN
    xg = (np.arange(nx) + 0.5) / nx          # fractional x in the cell
    results = {}

    print(f"\n{'geometry':34s} {'F_bidir':>10s} {'eta_pm':>8s} "
          f"{'B_ITO':>10s} {'T':>7s} {'R':>7s} {'A':>7s}")

    def show(name, r):
        results[name] = r
        print(f"{name:34s} {r['F']:10.3e} {r['eta_pm']:8.3f} "
              f"{r['B_ITO']:10.3e} {r['T']:7.3f} {r['R']:7.3f} {r['A']:7.3f}")

    # 1. uniform slabs
    for fill in (0.25, 0.5, 0.75, 1.0):
        rho = torch.full((nx, ny), fill, dtype=config.GEO_DTYPE,
                         device=config.DEVICE)
        show(f"uniform a-Si slab, rho = {fill:.2f}", evaluate(rho, ctx))

    # 2. ideal 3-period binary gratings, duty sweep
    best_g = (None, -1.0)
    for duty in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        line = (np.mod(3.0 * xg, 1.0) < duty).astype(float)
        rho = torch.as_tensor(np.repeat(line[:, None], ny, axis=1),
                              dtype=config.GEO_DTYPE, device=config.DEVICE)
        r = evaluate(rho, ctx)
        show(f"3-period grating, duty = {duty:.1f}", r)
        if r["F"] > best_g[1]:
            best_g = (duty, r["F"])

    # 3. freeform result
    rho_f = np.load(config.OUT_DIR / "geometries" / "rho_proj_final.npy")
    rf = evaluate(torch.as_tensor(rho_f, dtype=config.GEO_DTYPE,
                                  device=config.DEVICE), ctx)
    show("freeform optimized", rf)

    print(f"\nbest simple grating: duty {best_g[0]:.1f}, F = {best_g[1]:.3e}")
    print(f"freeform / best-grating F ratio = {rf['F']/best_g[1]:.3f}")
    with open(config.OUT_DIR / "histories" / "baselines.json", "w") as f:
        json.dump(results, f, indent=1)
    print(f"saved: {config.OUT_DIR/'histories'/'baselines.json'}")
    return results


if __name__ == "__main__":
    main()
