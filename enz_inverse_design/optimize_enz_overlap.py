"""Phase-2 main script: freeform a-Si metasurface optimized for ENZ overlap.

Scaffold: the supplied TORCWA Example6 topology optimization (preserved
unmodified as original_pixel_inverse_design.ipynb).  This script keeps its
architecture — FFT Gaussian blur filter, tanh projection with beta ramp,
linear rho->eps mixing, hand-rolled Adam with clamping and y-mirror
symmetry — and replaces only the objective with the ENZ-mode overlap FoM.

Run:  python optimize_enz_overlap.py [--smoke]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import config


def gaussian_kernel_fft(nx, ny, dx, dy, radius_nm):
    """Example6's frequency-domain Gaussian blur kernel (unchanged scheme)."""
    xk = (torch.arange(nx, dtype=config.GEO_DTYPE, device=config.DEVICE)
          - (nx - 1) / 2) * dx
    yk = (torch.arange(ny, dtype=config.GEO_DTYPE, device=config.DEVICE)
          - (ny - 1) / 2) * dy
    xg, yg = torch.meshgrid(xk, yk, indexing="ij")
    g = torch.exp(-(xg ** 2 + yg ** 2) / radius_nm ** 2)
    g = g / torch.sum(g)
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(g)))


def filter_rho(rho, g_fft):
    rho_fft = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(rho)))
    return torch.real(torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(rho_fft * g_fft))))


def project_rho(rho_bar, beta):
    return 0.5 + torch.tanh(2.0 * beta * rho_bar - beta) / (2.0 * np.tanh(beta))


def binarization_metric(rho_proj):
    """0 for fully binary, 1 for uniformly gray (mean of 4*rho*(1-rho))."""
    return float(torch.mean(4.0 * rho_proj * (1.0 - rho_proj)))


def main(smoke=False, n_iter=None):
    if smoke:
        config.apply_smoke()
    if n_iter is not None:
        config.N_ITER = n_iter
    torch.set_num_threads(config.N_THREADS)

    import target_mode
    import torcwa_forward as fwd
    import objective as obj

    out = Path(config.OUT_DIR)
    for sub in ("histories", "geometries", "fields", "figures"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Target mode
    # ------------------------------------------------------------------
    tgt = target_mode.load_target_npz()
    lam = float(tgt["wavelength_nm"]) if config.WAVELENGTH_NM is None \
        else config.WAVELENGTH_NM
    config.WAVELENGTH_NM = lam
    if config.ITO_THICKNESS_NM is None:
        config.ITO_THICKNESS_NM = float(tgt["ito_thickness_nm"])
    if config.N_GLASS is None:
        config.N_GLASS = float(tgt["glass_index"])
    eps_ito = complex(float(tgt["eps_ito_real"]), float(tgt["eps_ito_imag"]))
    if config.EPS_ASI is None:
        config.EPS_ASI = fwd.eps_asi_of_lambda(lam)
    print(f"[setup] eps_aSi({lam:.1f} nm) = {config.EPS_ASI:.4f} "
          f"(n = {config.EPS_ASI**0.5:.4f}) from measured POSTECH n,k file")

    target_mode.momentum_diagnostic(tgt)

    x_axis, y_axis = fwd.grid_axes()
    z_prop = target_mode.ito_z_slices(config.ITO_THICKNESS_NM,
                                      config.Z_SAMPLES_ITO)
    T_plus, dV = target_mode.build_target_field(
        tgt, x_axis.cpu().numpy(), y_axis.cpu().numpy(), z_prop, "+x")
    T_minus, _ = target_mode.build_target_field(
        tgt, x_axis.cpu().numpy(), y_axis.cpu().numpy(), z_prop, "-x")
    assert not T_plus.requires_grad and not T_minus.requires_grad
    p_inc = fwd.p_inc_cell()
    print(f"[setup] P_inc_cell = {p_inc:.1f} (LH units, |E0|=1)")

    # ------------------------------------------------------------------
    # 2. Reference field (rho-independent; computed once, detached)
    # ------------------------------------------------------------------
    with torch.no_grad():
        sim_ref = fwd.build_solved_sim(None, lam, eps_ito, config.N_GLASS)
        Ez_ref = fwd.ez_in_ito(sim_ref, x_axis, y_axis, z_prop).detach()
        R_ref, T_ref = fwd.specular_RT(sim_ref)
    print(f"[reference] air/ITO/glass: R = {float(R_ref):.4f}, "
          f"T = {float(T_ref):.4f}, max|Ez_ref| = "
          f"{float(torch.max(torch.abs(Ez_ref))):.3e} "
          "(normal incidence: Ez_ref ~ 0 as expected)")

    # ------------------------------------------------------------------
    # 3. Initialization (Example6 scheme, reproducible)
    # ------------------------------------------------------------------
    nx, ny = config.NX_DESIGN, config.NY_DESIGN
    dx, dy = config.PX_NM / nx, config.PY_NM / ny
    g_fft = gaussian_kernel_fft(nx, ny, dx, dy, config.FILTER_RADIUS_NM)
    beta_sched = np.exp(np.arange(config.N_ITER) * np.log(config.BETA_PROJ_MAX)
                        / config.N_ITER)
    lr_sched = config.LR_INITIAL * 0.5 * (
        1 + np.cos(np.arange(config.N_ITER) * np.pi / config.N_ITER))

    torch.manual_seed(config.RANDOM_SEED)
    rho = torch.rand((nx, ny), dtype=config.GEO_DTYPE, device=config.DEVICE)
    if config.MIRROR_SYMMETRY_Y:
        rho = (rho + torch.fliplr(rho)) / 2
    rho = filter_rho(rho, g_fft)
    np.save(out / "geometries" / "rho_initial.npy", rho.cpu().numpy())

    momentum = torch.zeros_like(rho)
    velocity = torch.zeros_like(rho)

    def forward_F(rho_proj, want_diags=False):
        sim = fwd.build_solved_sim(rho_proj, lam, eps_ito, config.N_GLASS)
        Ez_full = fwd.ez_in_ito(sim, x_axis, y_axis, z_prop)
        Ez_scat = Ez_full - Ez_ref
        F, diags = obj.enz_objective(
            T_plus, Ez_scat, dV, p_inc, target_minus=T_minus,
            direction=config.TARGET_DIRECTION)
        if want_diags:
            with torch.no_grad():
                R, T = fwd.specular_RT(sim)
                # dimensionless review metrics:
                #   eta_pm: fraction of the ITO Ez_scat energy in the {T+,T-}
                #           subspace (targets are grid-orthonormal, checked)
                #   B_ITO:  ITO |Ez|^2 buildup per incident power
                I_scat = float(torch.sum(torch.abs(Ez_scat) ** 2).real * dV)
                eta = (float(torch.abs(diags["a_plus"]) ** 2)
                       + float(torch.abs(diags["a_minus"]) ** 2)) / I_scat \
                    if I_scat > 0 else 0.0
                ito_Ez2 = float(torch.sum(torch.abs(Ez_full) ** 2).real * dV)
                diags.update(R=float(R), T=float(T), ito_Ez2=ito_Ez2,
                             eta_pm=eta, B_ITO=ito_Ez2 / p_inc,
                             Ez_full=Ez_full.detach(),
                             Ez_scat=Ez_scat.detach())
        return F, diags

    hist = {k: [] for k in ("F", "absa_plus", "absa_minus", "R", "T",
                            "ito_Ez2", "eta_pm", "B_ITO", "grad_norm",
                            "binarization", "beta", "lr")}
    # orthogonality of the two targets on the overlap grid (must be ~0 for
    # the eta_pm metric to be a clean projection)
    t_cross = float(torch.abs(torch.sum(torch.conj(T_plus) * T_minus)) * dV)
    print(f"[setup] |<T+,T->| on grid = {t_cross:.2e} (orthogonal)")
    F_initial = None
    t0 = time.time()

    # ------------------------------------------------------------------
    # 4. Optimization loop (Example6 architecture)
    # ------------------------------------------------------------------
    for it in range(config.N_ITER):
        rho.requires_grad_(True)
        rho_bar = filter_rho(rho, g_fft)
        rho_proj = project_rho(rho_bar, beta_sched[it])

        want = (it % config.SAVE_EVERY == 0) or (it == config.N_ITER - 1)
        F, diags = forward_F(rho_proj, want_diags=want)
        loss = -torch.log(F + config.LOG_LOSS_EPS) if config.USE_LOG_LOSS \
            else -F
        loss.backward()

        with torch.no_grad():
            grad = rho.grad
            rho.grad = None
            gnorm = float(torch.linalg.norm(grad))

            if F_initial is None:
                F_initial = float(F)
                np.save(out / "fields" / "Ez_scat_initial.npy",
                        diags["Ez_scat"].cpu().numpy())
                np.save(out / "fields" / "Ez_full_initial.npy",
                        diags["Ez_full"].cpu().numpy())

            hist["F"].append(float(F))
            hist["absa_plus"].append(float(torch.abs(diags["a_plus"])))
            hist["absa_minus"].append(float(torch.abs(diags["a_minus"]))
                                      if "a_minus" in diags else np.nan)
            hist["R"].append(diags.get("R", np.nan))
            hist["T"].append(diags.get("T", np.nan))
            hist["ito_Ez2"].append(diags.get("ito_Ez2", np.nan))
            hist["eta_pm"].append(diags.get("eta_pm", np.nan))
            hist["B_ITO"].append(diags.get("B_ITO", np.nan))
            hist["grad_norm"].append(gnorm)
            hist["binarization"].append(binarization_metric(rho_proj))
            hist["beta"].append(float(beta_sched[it]))
            hist["lr"].append(float(lr_sched[it]))

            # ascent (maximize F): Example6 adds the gradient of the FoM;
            # here loss = -F so we SUBTRACT the loss gradient.
            momentum = config.ADAM_BETA1 * momentum \
                + (1 - config.ADAM_BETA1) * (-grad)
            velocity = config.ADAM_BETA2 * velocity \
                + (1 - config.ADAM_BETA2) * grad ** 2
            rho = rho.detach() + lr_sched[it] \
                * (momentum / (1 - config.ADAM_BETA1 ** (it + 1))) \
                / torch.sqrt(velocity / (1 - config.ADAM_BETA2 ** (it + 1))
                             + config.ADAM_EPS)
            rho[rho > 1] = 1
            rho[rho < 0] = 0
            if config.MIRROR_SYMMETRY_Y:
                rho = (rho + torch.fliplr(rho)) / 2

            if want:
                np.save(out / "geometries" / f"rho_proj_it{it:04d}.npy",
                        rho_proj.detach().cpu().numpy())
            print(f"it {it:4d}  F = {float(F):.6e}  |a+| = "
                  f"{hist['absa_plus'][-1]:.4e}  |grad| = {gnorm:.3e}  "
                  f"binar = {hist['binarization'][-1]:.3f}  "
                  f"t = {time.time()-t0:.0f}s")

    # ------------------------------------------------------------------
    # 5. Final evaluation and saving
    # ------------------------------------------------------------------
    with torch.no_grad():
        rho_bar = filter_rho(rho, g_fft)
        rho_proj = project_rho(rho_bar, beta_sched[-1])
        F_fin, diags_fin = forward_F(rho_proj, want_diags=True)
        F_final = float(F_fin)
        np.save(out / "geometries" / "rho_raw_final.npy", rho.cpu().numpy())
        np.save(out / "geometries" / "rho_filtered_final.npy",
                rho_bar.cpu().numpy())
        np.save(out / "geometries" / "rho_proj_final.npy",
                rho_proj.cpu().numpy())
        np.save(out / "fields" / "Ez_scat_final.npy",
                diags_fin["Ez_scat"].cpu().numpy())
        np.save(out / "fields" / "Ez_full_final.npy",
                diags_fin["Ez_full"].cpu().numpy())
        np.save(out / "fields" / "Ez_target_plus.npy", T_plus.cpu().numpy())
        np.save(out / "fields" / "Ez_reference.npy", Ez_ref.cpu().numpy())
        with open(out / "histories" / "history.json", "w") as f:
            json.dump({**hist, "F_initial": F_initial, "F_final": F_final,
                       "config": {k: str(getattr(config, k)) for k in
                                  ("WAVELENGTH_NM", "PX_NM", "PY_NM",
                                   "ASI_THICKNESS_NM", "ITO_THICKNESS_NM",
                                   "N_GLASS", "EPS_ASI", "FOURIER_ORDER",
                                   "NX_DESIGN", "NY_DESIGN", "Z_SAMPLES_ITO",
                                   "N_ITER", "RANDOM_SEED",
                                   "TARGET_DIRECTION")}}, f, indent=1)

    print("\n================ summary ================")
    print(f"Initial F_ENZ       = {F_initial:.6e}")
    print(f"Final F_ENZ         = {F_final:.6e}")
    print(f"Enhancement         = {F_final/F_initial:.2f}x "
          "(ratio vs a near-orthogonal random start; see eta_pm for the "
          "dimensionless picture)")
    print(f"Initial |a+|,|a-|   = {hist['absa_plus'][0]:.4e}, "
          f"{hist['absa_minus'][0]:.4e}")
    print(f"Final |a+|,|a-|     = {float(torch.abs(diags_fin['a_plus'])):.4e}, "
          f"{float(torch.abs(diags_fin['a_minus'])):.4e}")
    print(f"Initial eta_pm      = {hist['eta_pm'][0]:.4f}  "
          "(fraction of ITO Ez_scat energy in the +-K target subspace)")
    print(f"Final eta_pm        = {diags_fin['eta_pm']:.4f}")
    print(f"Initial B_ITO       = {hist['B_ITO'][0]:.4e}  "
          "(ITO int|Ez|^2 dV / P_inc, units nm)")
    print(f"Final B_ITO         = {diags_fin['B_ITO']:.4e}")
    print(f"Initial ITO |Ez|^2  = {hist['ito_Ez2'][0]:.4e}")
    print(f"Final ITO |Ez|^2    = {diags_fin['ito_Ez2']:.4e}")
    print(f"Initial T, R        = {hist['T'][0]:.4f}, {hist['R'][0]:.4f}")
    print(f"Final T, R          = {diags_fin['T']:.4f}, {diags_fin['R']:.4f}")
    print("NOTE: This first-stage optimization uses a frozen bare-structure "
          "ENZ target. Self-consistent ENZ-mode drift and hybrid-mode "
          "tracking are not yet included. F_ENZ is a target ENZ-mode overlap "
          "FoM; it is not a coupling rate g and does not by itself prove "
          "strong coupling.")
    return F_initial, F_final


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--iters", type=int, default=None)
    args = ap.parse_args()
    main(smoke=args.smoke, n_iter=args.iters)
