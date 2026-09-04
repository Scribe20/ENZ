"""Topology optimizer for the robust-A_ITO campaign.

Architecture = the supplied Example6 / enz_inverse_design scheme (FFT
Gaussian blur, tanh projection with beta ramp, linear rho->eps mixing,
hand-rolled Adam with cosine lr, [0,1] clamp, hard lateral air-padding
mask) with ONE deliberate change:

    Both historical Example6 fliplr symmetry projections
        rho = (rho + torch.fliplr(rho)) / 2      (at initialization)
        rho = (rho + torch.fliplr(rho)) / 2      (after every update)
    are DISABLED in this path.  No mirror symmetry is enforced.

The upstream notebook and enz_inverse_design/optimize_enz_overlap.py are
untouched (MIRROR_SYMMETRY_Y there remains True for the frozen campaigns).

Objective: J_robust = -(1/beta) log sum_m w_m exp(-beta A_m), A_m the
all-orders ITO absorption at angle m (forward_multi.a_ito).
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

import forward_multi as fm
import robust_config as rc

GEO = fm.GEO_DTYPE
DEV = fm.DEVICE


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
    return torch.real(torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(rho_fft * g_fft))))


def project_rho(rho_bar, beta):
    return 0.5 + torch.tanh(2.0 * beta * rho_bar - beta) / (2.0 * np.tanh(beta))


def binarization_metric(rho_proj):
    return float(torch.mean(4.0 * rho_proj * (1.0 - rho_proj)))


def s_flip(rho):
    """Mirror-asymmetry measure S_flip = ||rho - fliplr(rho)||_2 / ||rho||_2."""
    r = torch.as_tensor(rho, dtype=GEO)
    n = float(torch.linalg.norm(r))
    return float(torch.linalg.norm(r - torch.fliplr(r))) / n if n > 0 else 0.0


def s_flip_centered(rho):
    """Same measure on rho - mean(rho): removes the symmetric DC part that
    dominates ||rho|| for gray-scale fields (a filtered random init has
    S_flip ~ 0.05 but S_flip_centered ~ 1)."""
    r = torch.as_tensor(rho, dtype=GEO)
    r = r - r.mean()
    n = float(torch.linalg.norm(r))
    return float(torch.linalg.norm(r - torch.fliplr(r))) / n if n > 0 else 0.0


def s_flip_ud(rho):
    r = torch.as_tensor(rho, dtype=GEO)
    n = float(torch.linalg.norm(r))
    return float(torch.linalg.norm(r - torch.flipud(r))) / n if n > 0 else 0.0


def build_pad_mask(nx, P, pad_frac):
    """Hard air ring: pixel centers x_i = (i+0.5) P/nx are active iff
    pad <= x_i <= P - pad, pad = pad_frac * P.  Returns mask + realized pad."""
    dx = P / nx
    xc = (np.arange(nx) + 0.5) * dx
    pad = pad_frac * P
    m = (xc >= pad) & (xc <= P - pad)
    M = (m[:, None] & m[None, :]).astype(float)
    ix = np.where(m)[0]
    info = dict(P_nm=P, pad_frac=pad_frac, requested_pad_nm=pad, dx_nm=dx,
                active_index_range=[int(ix[0]), int(ix[-1])],
                active_pixels=int(len(ix)),
                realized_pad_nm=float(xc[ix[0]] - dx / 2),
                realized_pad_frac=float((xc[ix[0]] - dx / 2) / P))
    return M, info


def smooth_min(A_list, weights, beta):
    A = torch.stack(list(A_list))
    w = torch.as_tensor(weights, dtype=A.dtype, device=A.device)
    return -(1.0 / beta) * torch.log(torch.sum(w * torch.exp(-beta * A)))


def weights_for(angles):
    return [1.0 / len(angles)] * len(angles)


# ---------------------------------------------------------------------------
def evaluate(rho_proj, P, h, angles, order, beta, weights=None,
             with_grad=False):
    """A_m at every angle + J_robust (tensor).  Fills only the graph when
    with_grad."""
    weights = weights or weights_for(angles)
    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        As = []
        for th, ph in angles:
            sim = fm.build_sim(rho_proj, P, h, theta_deg=th, phi_deg=ph,
                               order=order)
            A, _R, _T = fm.a_ito(sim)
            As.append(A)
        J = smooth_min(As, weights, beta)
    return J, As


def optimize(P, h, pad_frac, seed, n_iter, order, angles, beta, out_dir,
             nx=rc.NX, rho_init=None, beta_proj_start=1.0,
             beta_proj_max=rc.BETA_PROJ_MAX, lr0=rc.LR_INITIAL,
             filter_nm=rc.FILTER_RADIUS_NM, save_every=10, log=print,
             tag=""):
    """One topology-optimization run.  Returns a result dict (also written
    to out_dir/result.json)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    weights = weights_for(angles)
    dx = P / nx
    g_fft = gaussian_kernel_fft(nx, nx, dx, dx, filter_nm)
    beta_sched = np.exp(np.linspace(np.log(beta_proj_start),
                                    np.log(beta_proj_max), n_iter))
    lr_sched = lr0 * 0.5 * (1 + np.cos(np.arange(n_iter) * np.pi / n_iter))
    M_np, mask_info = build_pad_mask(nx, P, pad_frac)
    M = torch.as_tensor(M_np, dtype=GEO, device=DEV)

    # ---- initialization: Example6 scheme WITHOUT the fliplr projection ----
    if rho_init is None:
        torch.manual_seed(seed)
        rho = torch.rand((nx, nx), dtype=GEO, device=DEV)
        # (historical: rho = (rho + torch.fliplr(rho)) / 2  -- DISABLED)
        rho = filter_rho(rho * M, g_fft) * M
    else:
        rho = torch.as_tensor(np.asarray(rho_init), dtype=GEO, device=DEV) * M
    s_init = s_flip(rho)
    np.save(out / "rho_initial.npy", rho.cpu().numpy())
    momentum = torch.zeros_like(rho)
    velocity = torch.zeros_like(rho)

    hist = dict(J=[], A=[], A_min=[], A_mean=[], grad_norm=[],
                binarization=[], s_flip=[], beta_proj=[], lr=[])
    t0 = time.time()
    for it in range(n_iter):
        rho.requires_grad_(True)
        rho_bar = filter_rho(rho * M, g_fft)
        rho_proj = project_rho(rho_bar, beta_sched[it]) * M
        J, As = evaluate(rho_proj, P, h, angles, order, beta, weights,
                         with_grad=True)
        loss = -J
        loss.backward()
        with torch.no_grad():
            grad = rho.grad
            rho.grad = None
            gnorm = float(torch.linalg.norm(grad))
            A_np = [float(a) for a in As]
            hist["J"].append(float(J)); hist["A"].append(A_np)
            hist["A_min"].append(min(A_np)); hist["A_mean"].append(
                float(np.mean(A_np)))
            hist["grad_norm"].append(gnorm)
            hist["binarization"].append(binarization_metric(rho_proj))
            hist["s_flip"].append(s_flip(rho_proj))
            hist["beta_proj"].append(float(beta_sched[it]))
            hist["lr"].append(float(lr_sched[it]))
            # Adam ascent on J (loss = -J, so subtract the loss gradient)
            momentum = rc_b1 * momentum + (1 - rc_b1) * (-grad)
            velocity = rc_b2 * velocity + (1 - rc_b2) * grad ** 2
            rho = rho.detach() + lr_sched[it] \
                * (momentum / (1 - rc_b1 ** (it + 1))) \
                / torch.sqrt(velocity / (1 - rc_b2 ** (it + 1)) + 1e-8)
            rho[rho > 1] = 1
            rho[rho < 0] = 0
            # (historical: rho = (rho + torch.fliplr(rho)) / 2  -- DISABLED)
            rho = rho * M
            if it % save_every == 0 or it == n_iter - 1:
                np.save(out / f"rho_proj_it{it:04d}.npy",
                        rho_proj.detach().cpu().numpy())
            log(f"[{tag}] it {it:3d} J={float(J):.5f} A_min={min(A_np):.4f} "
                f"A_mean={np.mean(A_np):.4f} |g|={gnorm:.2e} "
                f"bin={hist['binarization'][-1]:.3f} "
                f"S_flip={hist['s_flip'][-1]:.3f} t={time.time()-t0:.0f}s")

    # ---- final: soft and hard-binary evaluation ----------------------------
    with torch.no_grad():
        rho_bar = filter_rho(rho * M, g_fft)
        rho_proj = project_rho(rho_bar, beta_sched[-1]) * M
        rho_hard = (rho_proj > 0.5).to(GEO) * M
        J_soft, A_soft = evaluate(rho_proj, P, h, angles, order, beta, weights)
        J_hard, A_hard = evaluate(rho_hard, P, h, angles, order, beta, weights)
    np.save(out / "rho_raw_final.npy", rho.cpu().numpy())
    np.save(out / "rho_proj_final.npy", rho_proj.cpu().numpy())
    np.save(out / "rho_hard_binary.npy", rho_hard.cpu().numpy())
    np.save(out / "design_mask.npy", M_np)
    res = dict(tag=tag, P=P, h=h, pad_frac=pad_frac, seed=seed, n_iter=n_iter,
               order=list(order), angles=[list(a) for a in angles],
               weights=weights, beta=beta, mask=mask_info,
               J_soft=float(J_soft), J_hard=float(J_hard),
               A_soft=[float(a) for a in A_soft],
               A_hard=[float(a) for a in A_hard],
               A_hard_min=float(min(float(a) for a in A_hard)),
               fill_fraction=float(rho_hard.mean()),
               fill_fraction_active=float(rho_hard.sum() / M.sum()),
               s_flip_init=s_init, s_flip_final=s_flip(rho_hard),
               s_flip_centered_init=s_flip_centered(
                   np.load(out / "rho_initial.npy")),
               s_flip_centered_final=s_flip_centered(rho_hard),
               s_flipud_final=s_flip_ud(rho_hard),
               warm_start=rho_init is not None,
               beta_proj_start=beta_proj_start, wall_s=time.time() - t0,
               history=hist)
    with open(out / "result.json", "w") as f:
        json.dump(res, f, indent=1)
    log(f"[{tag}] done: J_soft={float(J_soft):.5f} J_hard={float(J_hard):.5f} "
        f"A_hard={[round(float(a), 4) for a in A_hard]} "
        f"S_flip={res['s_flip_final']:.3f} wall={res['wall_s']:.0f}s")
    return res


rc_b1, rc_b2 = 0.9, 0.999
