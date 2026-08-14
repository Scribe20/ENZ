"""Gradient validation: autograd statistics + central finite differences.

Run:  python validate_gradient.py          (smoke-size configuration)
"""

import numpy as np
import torch

import config
config.apply_smoke()          # small, fast configuration for validation
config.N_ITER = 1

import target_mode            # noqa: E402
import torcwa_forward as fwd  # noqa: E402
import objective as obj       # noqa: E402
from optimize_enz_overlap import gaussian_kernel_fft, filter_rho, project_rho  # noqa: E402


def build_problem():
    torch.set_num_threads(config.N_THREADS)
    tgt = target_mode.load_target_npz()
    lam = float(tgt["wavelength_nm"])
    config.ITO_THICKNESS_NM = float(tgt["ito_thickness_nm"])
    config.N_GLASS = float(tgt["glass_index"])
    if config.EPS_ASI is None:
        config.EPS_ASI = fwd.eps_asi_of_lambda(lam)
    eps_ito = complex(float(tgt["eps_ito_real"]), float(tgt["eps_ito_imag"]))
    x_axis, y_axis = fwd.grid_axes()
    z_prop = target_mode.ito_z_slices(config.ITO_THICKNESS_NM,
                                      config.Z_SAMPLES_ITO)
    T_plus, dV = target_mode.build_target_field(
        tgt, x_axis.cpu().numpy(), y_axis.cpu().numpy(), z_prop, "+x")
    with torch.no_grad():
        sim_ref = fwd.build_solved_sim(None, lam, eps_ito, config.N_GLASS)
        Ez_ref = fwd.ez_in_ito(sim_ref, x_axis, y_axis, z_prop).detach()
    p_inc = fwd.p_inc_cell()

    nx, ny = config.NX_DESIGN, config.NY_DESIGN
    g_fft = gaussian_kernel_fft(nx, ny, config.PX_NM / nx, config.PY_NM / ny,
                                config.FILTER_RADIUS_NM)
    beta = 8.0    # fixed mid-schedule projection sharpness for the check

    def F_of_rho(rho):
        rho_proj = project_rho(filter_rho(rho, g_fft), beta)
        sim = fwd.build_solved_sim(rho_proj, lam, eps_ito, config.N_GLASS)
        Ez_scat = fwd.ez_in_ito(sim, x_axis, y_axis, z_prop) - Ez_ref
        F, _ = obj.enz_objective(T_plus, Ez_scat, dV, p_inc,
                                 direction="+x")
        return F

    return F_of_rho


def main():
    F_of_rho = build_problem()
    torch.manual_seed(config.RANDOM_SEED)
    nx, ny = config.NX_DESIGN, config.NY_DESIGN
    rho0 = torch.rand((nx, ny), dtype=config.GEO_DTYPE, device=config.DEVICE)

    # ---- autograd pass ------------------------------------------------
    rho = rho0.clone().requires_grad_(True)
    F = F_of_rho(rho)
    print(f"F_ENZ = {float(F):.6e}, requires_grad = {F.requires_grad}")
    assert F.requires_grad
    F.backward()
    g = rho.grad
    print("rho.grad stats: min = {:.3e}, max = {:.3e}, mean = {:.3e}, "
          "std = {:.3e}, norm = {:.3e}".format(
              float(g.min()), float(g.max()), float(g.mean()),
              float(g.std()), float(torch.linalg.norm(g))))
    assert torch.all(torch.isfinite(g)), "gradient contains NaN/Inf"
    assert float(torch.linalg.norm(g)) > 0, "gradient identically zero"

    # ---- central finite differences on representative pixels ----------
    rng = np.random.default_rng(config.RANDOM_SEED)
    # avoid near-zero-gradient pixels: sample among the top-quartile |grad|
    gflat = torch.abs(g).flatten()
    idx_pool = torch.nonzero(gflat > torch.quantile(gflat, 0.75)).flatten()
    picks = idx_pool[rng.integers(0, len(idx_pool), size=4)]
    delta = 1e-3
    print("\npixel (i,j)      autograd        finite-diff     rel.err")
    for lin in picks:
        i, j = int(lin) // ny, int(lin) % ny
        rp = rho0.clone(); rp[i, j] += delta
        rm = rho0.clone(); rm[i, j] -= delta
        with torch.no_grad():
            fd = (float(F_of_rho(rp)) - float(F_of_rho(rm))) / (2 * delta)
        ag = float(g[i, j])
        rel = abs(ag - fd) / max(abs(ag), abs(fd), 1e-30)
        print(f"({i:3d},{j:3d})   {ag:+.6e}   {fd:+.6e}   {rel:.3e}")


if __name__ == "__main__":
    main()
