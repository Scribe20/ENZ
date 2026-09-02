"""Direct-ENZ-excitation campaign.

One-sentence definition: inverse-design the boundary-isolated a-Si
metasurface to maximize the volume-averaged longitudinal electric-field
intensity <|Ez/E_inc|^2> inside the 23-nm ITO layer at the ENZ wavelength,
without prescribing any momentum channel, QNM, or Mie multipole.

Reuses the validated enz_inverse_design machinery via the new selectable
objective hook (config.OBJECTIVE = "ito_ez_volume"); geometry class,
symmetry, mask, seed, grid, order, and schedules are identical to the
authoritative padded-85nm QNM campaign - ONLY the optical objective
changes.

Run:  python run_campaign.py       (preflight + gradient check + baseline
                                    evaluation + full optimization)
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / "enz_inverse_design"
PAD = HERE.parent / "enz_padding_sideexperiment"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PAD))

import config                       # noqa: E402
config.OBJECTIVE = "ito_ez_volume"  # THE campaign switch (before other imports)

import target_mode                  # noqa: E402
import torcwa_forward as fwd        # noqa: E402
import objective as obj             # noqa: E402
import optimize_enz_overlap as opt  # noqa: E402
from run_padded import build_mask   # noqa: E402  (exact same mask realization)

OUT = HERE / "outputs"


def f_enz_of(rho_t, ctx):
    """F_ENZ = <|Ez/E_inc|^2>_ITO for a given design (|E_inc| = 1)."""
    with torch.no_grad():
        sim = fwd.build_solved_sim(rho_t, ctx["lam"], ctx["eps_ito"],
                                   config.N_GLASS)
        Ez = fwd.ez_in_ito(sim, ctx["x"], ctx["y"], ctx["zp"])
        v_ito = config.PX_NM * config.PY_NM * config.ITO_THICKNESS_NM
        return float(torch.sum(torch.abs(Ez) ** 2).real * ctx["dV"] / v_ito)


def preflight():
    torch.set_num_threads(config.N_THREADS)
    tgt = target_mode.load_target_npz()
    lam = float(tgt["wavelength_nm"])
    config.ITO_THICKNESS_NM = float(tgt["ito_thickness_nm"])
    config.N_GLASS = float(tgt["glass_index"])
    config.EPS_ASI = fwd.eps_asi_of_lambda(lam)
    M, info = build_mask(config.NX_DESIGN, config.NY_DESIGN,
                         config.PX_NM, config.PY_NM)

    x, y = fwd.grid_axes()
    zp = target_mode.ito_z_slices(config.ITO_THICKNESS_NM,
                                  config.Z_SAMPLES_ITO)
    dV = (config.PX_NM / config.NX_DESIGN) * (config.PY_NM / config.NY_DESIGN) \
        * (config.ITO_THICKNESS_NM / config.Z_SAMPLES_ITO)
    eps_ito = complex(float(tgt["eps_ito_real"]), float(tgt["eps_ito_imag"]))
    ctx = dict(lam=lam, eps_ito=eps_ito, x=x, y=y, zp=zp, dV=dV)

    checklist = {
        "authoritative_ENZ_wavelength_nm": lam,
        "ITO_dispersion_source": "enz_target CSV (recommended-physical), "
                                 f"eps({lam:.2f}) = {eps_ito}",
        "source": "TORCWA plane wave, forward, amplitude [1,0] (x-pol), "
                  "normal incidence, |E_inc| = 1 (LH units)",
        "grid": [config.NX_DESIGN, config.NY_DESIGN],
        "rcwa_order": config.FOURIER_ORDER,
        "seed": config.RANDOM_SEED,
        "filter_projection": f"Gauss r={config.FILTER_RADIUS_NM}nm, tanh "
                             f"beta ramp 1->{config.BETA_PROJ_MAX}, "
                             f"{config.N_ITER} iters, lr {config.LR_INITIAL} "
                             "cosine",
        "mask_realization": info,
        "y_mirror": config.MIRROR_SYMMETRY_Y,
        "x_symmetry": "free",
        "objective": config.OBJECTIVE,
        "F_ENZ_definition": "sum_ITO |Ez|^2 dV / (V_ITO * |E_inc|^2), "
                            "7 midpoint z-slices (interior quadrature), "
                            "all harmonics via real-space total field",
        "field_region": "full ITO volume of one unit cell, interfaces "
                        "excluded by midpoint rule",
        "baseline_identity": "enz_padding_sideexperiment rho_hard_binary "
                             "(padded-85nm QNM-target winner)",
    }
    print("[preflight]", json.dumps(checklist, indent=1, default=str))
    (OUT / "histories").mkdir(parents=True, exist_ok=True)
    with open(OUT / "histories" / "preflight.json", "w") as f:
        json.dump(checklist, f, indent=1, default=str)

    # sanity check of the objective on known geometries:
    # bare stack (rho=0): plane wave at normal incidence has Ez = 0
    zero = torch.zeros((config.NX_DESIGN, config.NY_DESIGN),
                       dtype=config.GEO_DTYPE)
    f0 = f_enz_of(zero, ctx)
    print(f"[verify] F_ENZ(empty cell) = {f0:.3e} (must be ~0: no Ez at "
          "normal incidence in planar stack)")
    assert f0 < 1e-20

    # baseline: old padded QNM winner under the NEW metric
    rho_base = np.load(PAD / "outputs" / "geometries" / "rho_hard_binary.npy")
    f_base = f_enz_of(torch.as_tensor(rho_base, dtype=config.GEO_DTYPE), ctx)
    print(f"[baseline] F_ENZ(old padded QNM winner) = {f_base:.4f} "
          "(cross-check vs B_ITO/(2*d) = 101.8/46 = 2.213)")

    # gradient sanity for the new objective (production size, one step)
    rho = torch.rand((config.NX_DESIGN, config.NY_DESIGN),
                     dtype=config.GEO_DTYPE) * torch.as_tensor(
                         M, dtype=config.GEO_DTYPE)
    rho.requires_grad_(True)
    sim = fwd.build_solved_sim(rho, lam, eps_ito, config.N_GLASS)
    Ez = fwd.ez_in_ito(sim, x, y, zp)
    v_ito = config.PX_NM * config.PY_NM * config.ITO_THICKNESS_NM
    F = torch.sum(Ez.real ** 2 + Ez.imag ** 2) * dV / v_ito
    F.backward()
    g = rho.grad
    print(f"[gradient] F_ENZ = {float(F):.4e}; grad: norm = "
          f"{float(torch.linalg.norm(g)):.3e}, max|g| = "
          f"{float(g.abs().max()):.3e}, finite = "
          f"{bool(torch.all(torch.isfinite(g)))}")
    assert float(torch.linalg.norm(g)) > 0
    return M, f_base


def main():
    for sub in ("histories", "geometries", "fields", "figures"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    M, f_base = preflight()
    with open(OUT / "histories" / "baseline_f_enz.json", "w") as f:
        json.dump({"F_ENZ_old_padded_qnm_winner": f_base}, f)

    # reset lazily-resolved config values so opt.main re-resolves identically
    config.WAVELENGTH_NM = None
    F0, F1 = opt.main(design_mask=M, out_root=OUT)

    rho = np.load(OUT / "geometries" / "rho_proj_final.npy")
    Mn = np.load(OUT / "geometries" / "design_mask.npy") \
        if (OUT / "geometries" / "design_mask.npy").exists() else M
    np.save(OUT / "geometries" / "design_mask.npy", M)
    leak = float(np.abs(rho * (1 - M)).max())
    rho_hard = (rho > 0.5).astype(float)
    assert np.abs(rho_hard * (1 - M)).max() == 0.0
    np.save(OUT / "geometries" / "rho_hard_binary.npy", rho_hard)
    print(f"[finalize] air-ring leakage = {leak:.1e}; fill (cell) = "
          f"{rho_hard.mean():.4f}; soft-final F = {F1:.4f}")
    return F0, F1


if __name__ == "__main__":
    main()
