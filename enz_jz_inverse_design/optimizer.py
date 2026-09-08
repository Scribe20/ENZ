"""Topology optimizer for the JZ F_z campaign.

Architecture = the supplied TORCWA Example6 scheme as re-implemented and
validated in the repository (FFT Gaussian blur, tanh projection with a
geometric beta ramp, linear rho->eps mixing, hand-rolled Adam with cosine
learning rate and [0,1] clamp, hard lateral air-padding mask applied to the
raw variable BEFORE the periodic filter and again AFTER projection).

Deliberate differences from the historical Example6-derived path:
  * NO symmetry projection of any kind.  The two historical lines
        rho = (rho + torch.fliplr(rho)) / 2      (initialization)
        rho = (rho + torch.fliplr(rho)) / 2      (after every update)
    of enz_inverse_design/optimize_enz_overlap.py (MIRROR_SYMMETRY_Y) and of
    enz_inverse_design/original_pixel_inverse_design.ipynb cell 2 are ABSENT
    here; torch.fliplr/flipud appear only inside the asymmetry METRICS.
  * objective = F_z (longitudinal ITO absorbed-power fraction) only; every
    other quantity (F_x, F_y, F_tot, A = 1-R-T, R, T, the identity residual
    F_tot-(1-R-T), binarization, S_flip) is logged as a diagnostic.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

import config
import forward as fwd

GEO, DEV = config.GEO_DTYPE, config.DEVICE


# ---------------------------------------------------------------------------
# Example6 building blocks (unchanged scheme)
# ---------------------------------------------------------------------------
def gaussian_kernel_fft(nx, ny, dx, dy, radius_nm):
    xk = (torch.arange(nx, dtype=GEO, device=DEV) - (nx - 1) / 2) * dx
    yk = (torch.arange(ny, dtype=GEO, device=DEV) - (ny - 1) / 2) * dy
    xg, yg = torch.meshgrid(xk, yk, indexing="ij")
    g = torch.exp(-(xg ** 2 + yg ** 2) / radius_nm ** 2)
    g = g / torch.sum(g)
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(g)))


def filter_rho(rho, g_fft):
    rho_fft = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(rho)))
    return torch.real(torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(rho_fft * g_fft))))


def project_rho(rho_bar, beta):
    return 0.5 + torch.tanh(2.0 * beta * rho_bar - beta) / (2.0 * np.tanh(beta))


def binarization_metric(rho_proj):
    """0 for fully binary, 1 for uniformly gray."""
    return float(torch.mean(4.0 * rho_proj * (1.0 - rho_proj)))


# ---- asymmetry METRICS only (the only place fliplr/flipud are used) --------
def s_flip(rho):
    r = torch.as_tensor(rho, dtype=GEO)
    n = float(torch.linalg.norm(r))
    return float(torch.linalg.norm(r - torch.fliplr(r))) / n if n > 0 else 0.0


def s_flip_ud(rho):
    r = torch.as_tensor(rho, dtype=GEO)
    n = float(torch.linalg.norm(r))
    return float(torch.linalg.norm(r - torch.flipud(r))) / n if n > 0 else 0.0


def s_flip_centered(rho):
    r = torch.as_tensor(rho, dtype=GEO)
    r = r - r.mean()
    n = float(torch.linalg.norm(r))
    return float(torch.linalg.norm(r - torch.fliplr(r))) / n if n > 0 else 0.0


def s_inv(rho):
    """Inversion (C2) asymmetry ||rho - rot180(rho)|| / ||rho||."""
    r = torch.as_tensor(rho, dtype=GEO)
    n = float(torch.linalg.norm(r))
    return float(torch.linalg.norm(r - torch.rot90(r, 2))) / n if n > 0 else 0.0


def preprocess(rho_raw, M, g_fft, beta):
    """raw -> mask -> periodic filter -> projection -> mask."""
    rho_bar = filter_rho(rho_raw * M, g_fft)
    return project_rho(rho_bar, beta) * M, rho_bar


# ---------------------------------------------------------------------------
def optimize(P, h, pad_frac, seed, n_iter, order, out_dir, lam, *,
             nx=config.NX, n_z=config.Z_SAMPLES_ITO, rho_init=None,
             beta_proj_start=1.0, beta_proj_max=config.BETA_PROJ_MAX,
             lr0=config.LR_INITIAL, filter_nm=config.FILTER_RADIUS_NM,
             save_every=10, log=print, tag="", n_threads=None,
             eval_order_final=None):
    """One F_z topology-optimization run.  Writes result.json, histories and
    geometries (initial, snapshots, raw/proj/hard-binary finals) to out_dir.
    rho_init: optional (nx,nx) array used as the RAW variable (warm start)."""
    if n_threads is not None:
        fwd.set_threads(n_threads)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dx = P / nx
    g_fft = gaussian_kernel_fft(nx, nx, dx, dx, filter_nm)
    beta_sched = np.exp(np.linspace(np.log(beta_proj_start), np.log(beta_proj_max), n_iter))
    lr_sched = lr0 * 0.5 * (1 + np.cos(np.arange(n_iter) * np.pi / n_iter))
    M_np, mask_info = fwd.build_pad_mask(nx, P, pad_frac)
    M = torch.as_tensor(M_np, dtype=GEO, device=DEV)

    # ---- initialization: Example6 scheme, NO symmetry projection ----------
    if rho_init is None:
        torch.manual_seed(int(seed))
        rho = torch.rand((nx, nx), dtype=GEO, device=DEV)
        rho = filter_rho(rho * M, g_fft) * M
    else:
        rho = torch.as_tensor(np.asarray(rho_init), dtype=GEO, device=DEV) * M
    np.save(out / "rho_initial.npy", rho.cpu().numpy())
    s_init = dict(lr=s_flip(rho), ud=s_flip_ud(rho), c=s_flip_centered(rho), inv=s_inv(rho))
    momentum = torch.zeros_like(rho)
    velocity = torch.zeros_like(rho)

    keys = ("Fz", "Fx", "Fy", "Ftot", "A", "R", "T", "identity_resid", "mean_Ez2",
            "max_Ez2", "grad_norm", "binarization", "s_flip", "s_flipud", "beta_proj", "lr", "t")
    hist = {k: [] for k in keys}
    t0 = time.time()
    for it in range(n_iter):
        rho.requires_grad_(True)
        rho_proj, _ = preprocess(rho, M, g_fft, beta_sched[it])
        want = (it % save_every == 0) or (it == n_iter - 1)
        d = fwd.evaluate(rho_proj, P, h, lam, order, nx=nx, n_z=n_z, with_grad=True, want_maps=want)
        loss = -d["Fz"]
        loss.backward()
        with torch.no_grad():
            grad = rho.grad
            rho.grad = None
            gnorm = float(torch.linalg.norm(grad))
            f = fwd.to_floats(d)
            for k in ("Fz", "Fx", "Fy", "Ftot", "A", "R", "T", "mean_Ez2"):
                hist[k].append(f[k])
            hist["max_Ez2"].append(f.get("max_Ez2", float("nan")))
            hist["identity_resid"].append(f["Ftot"] - f["A"])
            hist["grad_norm"].append(gnorm)
            hist["binarization"].append(binarization_metric(rho_proj))
            hist["s_flip"].append(s_flip(rho_proj))
            hist["s_flipud"].append(s_flip_ud(rho_proj))
            hist["beta_proj"].append(float(beta_sched[it]))
            hist["lr"].append(float(lr_sched[it]))
            hist["t"].append(time.time() - t0)
            # Adam ascent on F_z (loss = -F_z -> subtract the loss gradient)
            momentum = config.ADAM_B1 * momentum + (1 - config.ADAM_B1) * (-grad)
            velocity = config.ADAM_B2 * velocity + (1 - config.ADAM_B2) * grad ** 2
            rho = rho.detach() + lr_sched[it] * (momentum / (1 - config.ADAM_B1 ** (it + 1))) \
                / torch.sqrt(velocity / (1 - config.ADAM_B2 ** (it + 1)) + config.ADAM_EPS)
            rho[rho > 1] = 1
            rho[rho < 0] = 0
            rho = rho * M                      # (no fliplr projection here)
            if want:
                np.save(out / f"rho_proj_it{it:04d}.npy", rho_proj.detach().cpu().numpy())
            log(f"[{tag}] it {it:3d} Fz={f['Fz']:.5f} Ftot={f['Ftot']:.5f} A={f['A']:.5f} "
                f"res={f['Ftot']-f['A']:+.1e} R={f['R']:.3f} T={f['T']:.3f} |g|={gnorm:.2e} "
                f"bin={hist['binarization'][-1]:.3f} Sf={hist['s_flip'][-1]:.3f} t={time.time()-t0:.0f}s")
        del d

    # ---- final: soft and hard-binary evaluation ----------------------------
    with torch.no_grad():
        rho_proj, rho_bar = preprocess(rho, M, g_fft, beta_sched[-1])
        rho_hard = (rho_proj > 0.5).to(GEO) * M
        eo = order if eval_order_final is None else eval_order_final
        d_soft = fwd.to_floats(fwd.evaluate(rho_proj, P, h, lam, eo, nx=nx, n_z=n_z, want_maps=True))
        d_hard = fwd.to_floats(fwd.evaluate(rho_hard, P, h, lam, eo, nx=nx, n_z=n_z, want_maps=True))
    np.save(out / "rho_raw_final.npy", rho.cpu().numpy())
    np.save(out / "rho_filtered_final.npy", rho_bar.cpu().numpy())
    np.save(out / "rho_proj_final.npy", rho_proj.cpu().numpy())
    np.save(out / "rho_hard_binary.npy", rho_hard.cpu().numpy())
    np.save(out / "design_mask.npy", M_np)
    leak_soft = float((rho_proj * (1 - M)).abs().max())
    leak_hard = float((rho_hard * (1 - M)).abs().max())
    res = dict(tag=tag, P=float(P), h=float(h), pad_frac=float(pad_frac), seed=int(seed),
               n_iter=int(n_iter), order=list(order), eval_order_final=list(eo), lam=float(lam),
               nx=nx, n_z=n_z, mask=mask_info, filter_nm=filter_nm, lr0=lr0,
               beta_proj=[beta_proj_start, beta_proj_max],
               soft=d_soft, hard=d_hard,
               Fz_soft=d_soft["Fz"], Fz_hard=d_hard["Fz"],
               fill_fraction=float(rho_hard.mean()),
               fill_fraction_active=float(rho_hard.sum() / M.sum()),
               s_flip_init=s_init,
               s_flip_final=dict(lr=s_flip(rho_hard), ud=s_flip_ud(rho_hard),
                                 c=s_flip_centered(rho_hard), inv=s_inv(rho_hard)),
               pad_leak=dict(soft=leak_soft, hard=leak_hard),
               warm_start=rho_init is not None, wall_s=time.time() - t0, history=hist)
    with open(out / "result.json", "w") as f:
        json.dump(res, f, indent=1)
    log(f"[{tag}] done: Fz_soft={d_soft['Fz']:.5f} Fz_hard={d_hard['Fz']:.5f} "
        f"A_hard={d_hard['A']:.4f} eta_z={d_hard['Fz']/max(d_hard['Ftot'],1e-30):.3f} "
        f"fill={res['fill_fraction']:.3f} S_flip={res['s_flip_final']['lr']:.3f} "
        f"leak={leak_soft:.1e}/{leak_hard:.1e} wall={res['wall_s']:.0f}s")
    return res
