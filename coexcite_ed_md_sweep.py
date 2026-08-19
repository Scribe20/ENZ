"""
coexcite_ed_md_sweep.py
=======================

TORCWA topology-optimization search for freeform (deliberately NON-symmetric)
periodic a-Si:H metasurfaces on a fused-silica-like substrate that co-excite

  * an in-plane electric-dipole-like response  (p_x  <->  E_x), and
  * an out-of-plane magnetic-dipole-like response (m_z  <->  H_z)

at the SAME target wavelength, under the SAME forward, normally incident,
x-polarized plane wave (Example6 illumination convention: incidence from the
substrate side, `direction='forward'`, `amplitude=[1., 0.]`).

Authoritative figure of merit (topology-search objective; never modified):

    F_co = 0.5 * [ log(S_ED + eps_F) + log(S_MD + eps_F) ],   eps_F = 1e-12

with two INDEPENDENT dimensionless modal-response proxies (kept separate,
never multiplied pointwise / never <|Ex||Hz|>):

    S_ED = < |Ex / E_inc|^2 >_Omega
    S_MD = < |Hz / H_inc|^2 >_Omega

Omega = the full unit-cell xy sampling plane inside the patterned Si layer,
averaged over three z slices at z/h = 0.25, 0.50, 0.75 (all three slices are
reconstructed from ONE electromagnetic solve via torcwa's `field_xy`, so the
extra cost is small).

--------------------------------------------------------------------------
FIELD-NORMALIZATION CONVENTION (verified by inspection of the installed
torcwa 0.1.4.2 source, which is byte-identical to GitHub main modulo CRLF):
--------------------------------------------------------------------------
* `torcwa.rcwa.__init__` docstring: "Lorentz-Heaviside units, Speed of
  light: 1, Time harmonics notation: exp(-j w t)".  In Lorentz-Heaviside
  units with c = 1 the vacuum impedance is Z0 = 1, so a plane wave in a
  medium of refractive index n satisfies |H| = n * |E| in code units.
* `rcwa.source_fourier` (called by `source_planewave(amplitude=[1,0],
  direction='forward')`) literally sets the (0,0)-order Ex Fourier amplitude
  of the incident wave in the INPUT layer (the substrate) to 1.  Hence
  E_inc = 1 exactly, in code units.
* Therefore H_inc (the magnitude of the incident magnetic FIELD VECTOR, not
  its z component) is  H_inc = n_sub * E_inc / Z0 = n_sub = sqrt(eps_sub).
  At normal incidence the incident H is purely transverse, so this is also
  the plane-wave transverse H amplitude.
* `rcwa.field_xy(layer_num=0, x, y, z_prop)` reconstructs internal-layer
  fields from the layer eigenmodes (E_eigvec/H_eigvec), the internal mode
  coefficients `self.C` (always assembled by `solve_global_smatrix` through
  `_RS_prod`), and the convolution matrices.  H is returned in the same
  Z0 = 1 units:  Hz_mn = mu_conv^-1 (Kx_norm Ey - Ky_norm Ex)  with
  K_norm = k/k0, which equals Z0*H_phys for exp(-jwt) convention.
=>  S_ED = <|Ex|^2> / 1^2      (torcwa units)
    S_MD = <|Hz|^2> / n_sub^2  (torcwa units)

The entire chain rho -> eps grid -> _material_conv (FFT + differentiable
gather) -> Eig (custom autograd with Lorentzian-broadened backward) ->
layer/global S-matrix -> C coefficients -> field_xy -> S_ED/S_MD -> F_co is
pure torch and remains connected to autograd (confirmed by the gradient
smoke test in --mode smoke).

--------------------------------------------------------------------------
NO IMPOSED GEOMETRIC SYMMETRY
--------------------------------------------------------------------------
Example6's mirror averaging (rho averaged with its left-right mirrored copy)
has been removed from BOTH the random initialization and the update loop.  No mirror,
inversion, C2/C4, diagonal, or rotational symmetry operation is applied to
rho anywhere in this file; `verify_no_symmetry_ops()` greps this source at
preflight to prove it.  Only the unit-cell periodicity of RCWA remains.

Material model: `Materials.aSiH` (torcwa example material, tabulated n,k for
192-999 nm with cubic interpolation).  NOTE: at the default target
wavelength 1332.5 nm the tabulation is CLAMPED to its 999-nm endpoint
(n = 2.98701, k = 0.008881  =>  eps ~ 8.9221 + 0.0531j).  This is what "the
installed a-Si model evaluated at the target wavelength" yields; it is
flagged at preflight and in the final report.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import torcwa            # noqa: E402
import Materials         # noqa: E402  (repo-local copy of torcwa example Materials.py)

# ---------------------------------------------------------------------------
# Configuration (single source of truth; CLI can override a few knobs)
# ---------------------------------------------------------------------------

TARGET_WAVELENGTH_NM = 1332.5          # midpoint of Figure-4 ED (1300) / MD (1365)
PERIODS_NM  = [750, 790, 830, 870, 900]
HEIGHTS_NM  = [100, 125, 150, 175, 200]
SEEDS       = [11, 29, 47]

SUBSTRATE_EPS = 1.46 ** 2              # fused-silica-like, non-dispersive
N_SUB = 1.46

DISCOVERY_DX_NM = 5.0                  # physical grid spacing for discovery
REFINE_DX_NM    = 2.5                  # physical grid spacing for refinement

DISCOVERY_ORDER = [7, 7]               # Fourier order for the 75-run discovery
REFINE_ORDER    = [9, 9]               # CPU-adapted (see BENCHMARK note below)
VERIFY_ORDERS   = [[9, 9], [11, 11], [13, 13]]
SPECTRA_ORDER   = [9, 9]

# BENCHMARK/adaptation note (CPU-only environment, 4 cores, no GPU):
# complex64 eig timings (1 thread): [7,7] 0.17 s, [9,9] 0.54 s, [11,11] 1.26 s,
# [13,13] 2.74 s; a full fwd+bwd iteration is ~6x-10x an eig. REFINE_ORDER
# [11,11] would cost ~2-3 h per candidate * 12 candidates; [9,9] keeps refine
# feasible while final verification still runs at [9,9]/[11,11]/[13,13].

BLUR_RADIUS_NM = 20.0                  # physical Gaussian blur radius (nm)
EPS_F = 1.0e-12                        # F_co log regularizer
Z_SLICE_FRACS = [0.25, 0.50, 0.75]     # z/h evaluation slices inside Si layer
EVAL_GRID_N = 64                       # unit-cell sampling grid (exact band-
                                       # limited average needs > 2*max_order+1 = 27)

DISCOVERY_ITERS = 150
REFINE_ITERS    = 550                  # ~700 total effective with discovery

GAR_INITIAL = 0.02                     # Adam ascent rate (Example6 value)
ADAM_BETA1  = 0.9
ADAM_BETA2  = 0.999
ADAM_EPS    = 1.0e-8
BETA_MAX_DISCOVERY = 1000.0            # tanh projection continuation (Example6)
BETA_START_REFINE  = 8.0               # refinement continuation restarts here
BETA_MAX_REFINE    = 2000.0

ALLOW_PROPAGATING_HIGHER_ORDERS = False

INC_ANG_RAD = 0.0                      # normal incidence
AZI_ANG_RAD = 0.0
SOURCE_AMPLITUDE = [1.0, 0.0]          # x-polarized E, forward (substrate-side)
SOURCE_DIRECTION = 'forward'

CHECKPOINT_EVERY = 25

RESULTS_ROOT = REPO_ROOT / 'results_ed_md_coexcitation'

GEO_DTYPE = torch.float32
SIM_DTYPE = torch.complex64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.backends.cuda.matmul.allow_tf32 = False   # matches Example6 (no-op on CPU)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def run_identifier(period_nm, height_nm, seed, lamb_nm=TARGET_WAVELENGTH_NM):
    lam_tag = ('%g' % lamb_nm).replace('.', 'p')
    return f"P{int(round(period_nm)):04d}_H{int(round(height_nm)):04d}_seed{seed:03d}_lam{lam_tag}"


class RunLogger:
    def __init__(self, logfile=None):
        self.logfile = logfile

    def __call__(self, msg):
        stamp = time.strftime('%H:%M:%S')
        line = f"[{stamp}] {msg}"
        print(line, flush=True)
        if self.logfile is not None:
            with open(self.logfile, 'a') as f:
                f.write(line + '\n')


def silicon_eps(lamb_nm):
    """a-Si:H permittivity from the installed torcwa example material model,
    evaluated at the actual target wavelength (NOT frozen at 532 nm).
    Beyond the 192-999 nm tabulation the model clamps to its endpoints."""
    lamb = torch.tensor(float(lamb_nm), dtype=GEO_DTYPE, device=DEVICE)
    nk = Materials.aSiH.apply(lamb)
    return nk ** 2


def diffraction_flags(period_nm, lamb_nm):
    """Return (n_sub*P/lambda, P/lambda) - propagating first-order thresholds
    on the substrate and air sides."""
    return N_SUB * period_nm / lamb_nm, period_nm / lamb_nm


# ---------------------------------------------------------------------------
# Symmetry-freedom verification (preflight requirement)
# ---------------------------------------------------------------------------

def verify_no_symmetry_ops():
    """Grep this source file for any symmetry projection applied to rho.
    Tokens are assembled by concatenation so this checker never matches
    itself."""
    forbidden = ['flip' + 'lr', 'flip' + 'ud', 'torch.' + 'flip', 'rot' + '90']
    src = Path(__file__).read_text()
    hits = []
    for tok in forbidden:
        for ln, line in enumerate(src.splitlines(), 1):
            if tok in line:
                hits.append((tok, ln, line.strip()))
    if hits:
        for tok, ln, line in hits:
            print(f"SYMMETRY-OP FOUND: '{tok}' at line {ln}: {line}")
        raise RuntimeError('Symmetry projection operations found in optimization source!')
    print('Verified: no mirror/rotational symmetry projection is applied to rho.')
    return True


# ---------------------------------------------------------------------------
# Geometry / filtering / projection
# ---------------------------------------------------------------------------

def configure_geometry(period_nm, dx_nm):
    """Square unit cell Lx = Ly = P; grid chosen from a PHYSICAL grid spacing."""
    n = int(round(period_nm / dx_nm))
    if n % 2 != 0:
        n += 1  # keep even for clean FFT grids
    return n, n


def build_filter(nx, ny, period_nm, blur_radius_nm):
    """FFT Gaussian blur kernel with a PHYSICAL radius in nm (recomputed for
    every period/grid combination). Follows Example6's construction."""
    dx, dy = period_nm / nx, period_nm / ny
    xk = (torch.arange(nx, dtype=GEO_DTYPE, device=DEVICE) - (nx - 1) / 2) * dx
    yk = (torch.arange(ny, dtype=GEO_DTYPE, device=DEVICE) - (ny - 1) / 2) * dy
    xg, yg = torch.meshgrid(xk, yk, indexing='ij')
    g = torch.exp(-(xg ** 2 + yg ** 2) / blur_radius_nm ** 2)
    g = g / torch.sum(g)
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(g)))


def initialize_rho(nx, ny, seed):
    """Genuinely random, NON-symmetrized initial density (deterministic
    per seed). NO mirror averaging - contrast with Example6."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.rand((nx, ny), dtype=GEO_DTYPE, device=DEVICE)


def filter_and_project(rho, g_fft, beta):
    """Fourier Gaussian blur followed by Example6's tanh projection."""
    rho_fft = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(rho)))
    rho_bar = torch.real(torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(rho_fft * g_fft))))
    rho_tilda = 0.5 + torch.tanh(2 * beta * rho_bar - beta) / (2 * math.tanh(beta))
    return rho_bar, rho_tilda


# ---------------------------------------------------------------------------
# Simulation + objective
# ---------------------------------------------------------------------------

def build_simulation(rho_tilda, period_nm, height_nm, lamb_nm, order, si_eps):
    """Solve one RCWA problem for the given projected density. Illumination
    follows Example6: input layer = substrate, output = air, forward source."""
    L = [float(period_nm), float(period_nm)]
    sim = torcwa.rcwa(freq=1.0 / lamb_nm, order=list(order), L=L,
                      dtype=SIM_DTYPE, device=DEVICE)
    sim.add_input_layer(eps=SUBSTRATE_EPS)
    sim.set_incident_angle(inc_ang=INC_ANG_RAD, azi_ang=AZI_ANG_RAD)
    layer_eps = rho_tilda * si_eps + (1.0 - rho_tilda) * 1.0
    sim.add_layer(thickness=float(height_nm), eps=layer_eps)
    sim.solve_global_smatrix()
    sim.source_planewave(amplitude=SOURCE_AMPLITUDE, direction=SOURCE_DIRECTION)
    return sim


def eval_axes(period_nm, n_eval):
    """Uniform unit-cell sampling axes. For a band-limited RCWA field the
    grid average equals the exact continuous unit-cell average whenever
    n_eval > 2*max_order (here 64 > 26 even at order [13,13])."""
    step = period_nm / n_eval
    ax = (torch.arange(n_eval, dtype=GEO_DTYPE, device=DEVICE) + 0.5) * step
    return ax, ax


def compute_fields(sim, period_nm, height_nm, n_eval=EVAL_GRID_N,
                   z_fracs=Z_SLICE_FRACS, keep_all=False):
    """Ex, Hz (and optionally all six components) on z slices inside layer 0.
    All slices reuse the single solve stored in `sim`."""
    x_ax, y_ax = eval_axes(period_nm, n_eval)
    Ex_slices, Hz_slices, all_fields = [], [], []
    for zf in z_fracs:
        E, H = sim.field_xy(0, x_ax, y_ax, z_prop=float(zf * height_nm))
        Ex_slices.append(E[0])
        Hz_slices.append(H[2])
        if keep_all:
            all_fields.append((E, H))
    return Ex_slices, Hz_slices, all_fields


def compute_mode_scores(Ex_slices, Hz_slices):
    """S_ED and S_MD as SEPARATE scalars (no pointwise products anywhere).
    Normalization: E_inc = 1, H_inc = n_sub (see module docstring)."""
    S_ED = torch.stack([torch.mean(torch.abs(Ex) ** 2) for Ex in Ex_slices]).mean()
    S_MD = torch.stack([torch.mean(torch.abs(Hz) ** 2) for Hz in Hz_slices]).mean() / (N_SUB ** 2)
    return S_ED, S_MD


def compute_fco(S_ED, S_MD, eps_f=EPS_F):
    """Authoritative FoM: logarithmic geometric mean (NOT a sum, NOT a raw
    product, NOT a spatial-overlap objective)."""
    return 0.5 * (torch.log(S_ED + eps_f) + torch.log(S_MD + eps_f))


def evaluate_density(rho_tilda, period_nm, height_nm, lamb_nm, order, si_eps):
    """One full differentiable evaluation: solve -> fields -> scores -> F_co."""
    sim = build_simulation(rho_tilda, period_nm, height_nm, lamb_nm, order, si_eps)
    Ex_s, Hz_s, _ = compute_fields(sim, period_nm, height_nm)
    S_ED, S_MD = compute_mode_scores(Ex_s, Hz_s)
    return compute_fco(S_ED, S_MD), S_ED, S_MD


# ---------------------------------------------------------------------------
# Single optimization run (discovery or refinement)
# ---------------------------------------------------------------------------

def optimize_single_run(period_nm, height_nm, seed, out_dir, *,
                        iters=DISCOVERY_ITERS, dx_nm=DISCOVERY_DX_NM,
                        order=DISCOVERY_ORDER, lamb_nm=TARGET_WAVELENGTH_NM,
                        beta_schedule=None, init_rho=None, stage='discovery'):
    """Adam gradient-ascent topology optimization of F_co with Example6-style
    filtering/projection/continuation, per-iteration history, checkpointing
    and resume. Returns a status string."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = RunLogger(out_dir / 'run.log')
    run_id = run_identifier(period_nm, height_nm, seed, lamb_nm)

    cfg_path = out_dir / 'config.json'
    if cfg_path.exists():
        st = json.loads(cfg_path.read_text()).get('status', '')
        if st == 'completed':
            log(f"{run_id}: already completed - skipping.")
            return 'completed'

    nx, ny = configure_geometry(period_nm, dx_nm)
    g_fft = build_filter(nx, ny, period_nm, BLUR_RADIUS_NM)
    si_eps = silicon_eps(lamb_nm)

    if beta_schedule is None:
        beta_schedule = np.exp(np.arange(iters) * math.log(BETA_MAX_DISCOVERY) / iters)
    gar = GAR_INITIAL * 0.5 * (1 + np.cos(np.arange(iters) * np.pi / iters))

    ckpt_path = out_dir / 'checkpoint.pt'
    start_it = 0
    history = []
    best = {'F_co': -np.inf, 'iteration': -1}
    if ckpt_path.exists():
        try:
            ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            rho = ck['rho'].to(DEVICE)
            momentum, velocity = ck['momentum'].to(DEVICE), ck['velocity'].to(DEVICE)
            start_it = ck['iteration'] + 1
            history = ck['history']
            best = ck['best']
            log(f"{run_id}: resuming from checkpoint at iteration {start_it}.")
        except Exception as e:
            log(f"{run_id}: checkpoint load failed ({e}) - restarting run.")
            ckpt_path.unlink(missing_ok=True)

    if start_it == 0:
        if init_rho is not None:
            rho = init_rho.to(device=DEVICE, dtype=GEO_DTYPE).clone()
            # deterministic seeding still applied for any downstream RNG use
            torch.manual_seed(seed)
            np.random.seed(seed)
        else:
            rho = initialize_rho(nx, ny, seed)
            # one initial blur like Example6 (but with NO mirror averaging)
            rho, _ = filter_and_project(rho, g_fft, 1.0)
            rho = rho.detach()
        momentum = torch.zeros_like(rho)
        velocity = torch.zeros_like(rho)

    config = {
        'run_id': run_id, 'stage': stage, 'period_nm': period_nm,
        'height_nm': height_nm, 'seed': seed, 'target_wavelength_nm': lamb_nm,
        'nx': nx, 'ny': ny, 'dx_nm': period_nm / nx,
        'fourier_order': list(order), 'iters': iters,
        'blur_radius_nm': BLUR_RADIUS_NM, 'eps_F': EPS_F,
        'z_slice_fracs': Z_SLICE_FRACS, 'eval_grid_n': EVAL_GRID_N,
        'gar_initial': GAR_INITIAL, 'adam_beta1': ADAM_BETA1,
        'adam_beta2': ADAM_BETA2, 'adam_eps': ADAM_EPS,
        'substrate_eps': SUBSTRATE_EPS,
        'silicon_eps': [float(torch.real(si_eps)), float(torch.imag(si_eps))],
        'inc_ang_rad': INC_ANG_RAD, 'azi_ang_rad': AZI_ANG_RAD,
        'source_amplitude': SOURCE_AMPLITUDE, 'source_direction': SOURCE_DIRECTION,
        'normalization': 'Lorentz-Heaviside Z0=1: E_inc=1, H_inc=n_sub=%.4f' % N_SUB,
        'symmetry_projection': 'NONE',
        'sim_dtype': str(SIM_DTYPE), 'geo_dtype': str(GEO_DTYPE),
        'torcwa_version': torcwa.__version__, 'torch_version': torch.__version__,
        'status': 'running',
    }
    cfg_path.write_text(json.dumps(config, indent=2))

    t0 = time.time()
    status = 'completed'
    rho_tilda_last = None
    for it in range(start_it, iters):
        beta_it = float(beta_schedule[it])
        rho.requires_grad_(True)
        _, rho_tilda = filter_and_project(rho, g_fft, beta_it)
        try:
            F_co, S_ED, S_MD = evaluate_density(
                rho_tilda, period_nm, height_nm, lamb_nm, order, si_eps)
        except Exception as e:
            log(f"{run_id}: solver exception at iter {it}: {e}")
            status = f'failed_solver_iter{it}'
            break

        if not torch.isfinite(F_co):
            log(f"{run_id}: NON-FINITE F_co at iter {it} - checkpointing and aborting run.")
            status = f'failed_nonfinite_iter{it}'
            break

        F_co.backward()

        with torch.no_grad():
            grad = rho.grad
            rho.grad = None
            if grad is None or not torch.all(torch.isfinite(grad)):
                log(f"{run_id}: NON-FINITE/absent gradient at iter {it} - aborting run.")
                status = f'failed_nonfinite_grad_iter{it}'
                break

            f_val, ed_val, md_val = float(F_co), float(S_ED), float(S_MD)
            balance = min(ed_val, md_val) / (max(ed_val, md_val) + 1e-12)
            history.append({
                'iteration': it, 'F_co': f_val, 'S_ED': ed_val, 'S_MD': md_val,
                'balance': balance, 'learning_rate': float(gar[it]),
                'projection_beta': beta_it,
                'elapsed_time_s': time.time() - t0,
            })
            if f_val > best['F_co']:
                best = {'F_co': f_val, 'S_ED': ed_val, 'S_MD': md_val,
                        'iteration': it}

            # Adam ascent (Example6 update, WITHOUT the mirror averaging)
            momentum = ADAM_BETA1 * momentum + (1 - ADAM_BETA1) * grad
            velocity = ADAM_BETA2 * velocity + (1 - ADAM_BETA2) * grad ** 2
            mhat = momentum / (1 - ADAM_BETA1 ** (it + 1))
            vhat = velocity / (1 - ADAM_BETA2 ** (it + 1))
            rho = rho.detach() + gar[it] * mhat / torch.sqrt(vhat + ADAM_EPS)
            rho.clamp_(0.0, 1.0)
            rho_tilda_last = rho_tilda.detach()

            if it % 10 == 0 or it == iters - 1:
                log(f"{run_id}: it {it:4d}/{iters}  F_co={f_val:+.4f}  "
                    f"S_ED={ed_val:.4f}  S_MD={md_val:.4f}  bal={balance:.3f}  "
                    f"beta={beta_it:7.1f}  lr={gar[it]:.5f}  "
                    f"t={time.time()-t0:7.1f}s")

            if (it + 1) % CHECKPOINT_EVERY == 0 or it == iters - 1:
                torch.save({'iteration': it, 'rho': rho.detach().cpu(),
                            'momentum': momentum.cpu(), 'velocity': velocity.cpu(),
                            'history': history, 'best': best}, ckpt_path)

    runtime_s = time.time() - t0

    # ------------------------------------------------------------------
    # Final evaluation + saving (everything below is post-gradient, so
    # detaching here cannot affect the optimization)
    # ------------------------------------------------------------------
    with torch.no_grad():
        beta_final = float(beta_schedule[-1])
        rho_final = rho.detach()
        _, rho_projected = filter_and_project(rho_final, g_fft, beta_final)
        rho_binary = (rho_projected >= 0.5).to(GEO_DTYPE)

        final_scores = {}
        try:
            for tag, dens in [('projected', rho_projected), ('binary', rho_binary)]:
                F_co, S_ED, S_MD = evaluate_density(
                    dens, period_nm, height_nm, lamb_nm, order, si_eps)
                final_scores[tag] = {'F_co': float(F_co), 'S_ED': float(S_ED),
                                     'S_MD': float(S_MD),
                                     'balance': float(min(S_ED, S_MD) / (max(S_ED, S_MD) + 1e-12))}
        except Exception as e:
            log(f"{run_id}: final evaluation failed: {e}")

        np.save(out_dir / 'rho_final.npy', rho_final.cpu().numpy())
        np.save(out_dir / 'rho_projected.npy', rho_projected.cpu().numpy())
        np.save(out_dir / 'rho_binary.npy', rho_binary.cpu().numpy())

        with open(out_dir / 'history.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['iteration', 'F_co', 'S_ED', 'S_MD',
                                              'balance', 'learning_rate',
                                              'projection_beta', 'elapsed_time_s'])
            w.writeheader()
            for row in history:
                w.writerow(row)

        # target-wavelength field maps of the PROJECTED final geometry
        try:
            sim = build_simulation(rho_projected, period_nm, height_nm,
                                   lamb_nm, order, si_eps)
            Ex_s, Hz_s, _ = compute_fields(sim, period_nm, height_nm)
            S_ED_f, S_MD_f = compute_mode_scores(Ex_s, Hz_s)
            np.savez_compressed(
                out_dir / 'fields_target.npz',
                Ex_slices=torch.stack(Ex_s).cpu().numpy(),
                Hz_slices=torch.stack(Hz_s).cpu().numpy(),
                z_slice_fracs=np.array(Z_SLICE_FRACS),
                x_axis=eval_axes(period_nm, EVAL_GRID_N)[0].cpu().numpy(),
                note='torcwa LH units: E_inc=1, H_inc=n_sub=%.4f; Hz stored unnormalized' % N_SUB,
            )
            plot_run_outputs(out_dir, rho_final, rho_projected, rho_binary,
                             Ex_s, Hz_s, history, config, final_scores)
        except Exception as e:
            log(f"{run_id}: field-map save failed: {e}")

        config['status'] = status
        config['runtime_s'] = runtime_s
        config['best'] = best
        config['final_scores'] = final_scores
        config['fill_fraction'] = float(rho_binary.mean())
        cfg_path.write_text(json.dumps(config, indent=2))

    log(f"{run_id}: {status} after {runtime_s:.0f}s  "
        f"final(projected)={final_scores.get('projected')}")
    return status


def plot_run_outputs(out_dir, rho_final, rho_projected, rho_binary,
                     Ex_slices, Hz_slices, history, config, final_scores):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    P, H = config['period_nm'], config['height_nm']
    seed, lam = config['seed'], config['target_wavelength_nm']
    fs = final_scores.get('projected', {})
    label = (f"P={P} nm, h={H} nm, seed={seed}, lam={lam} nm\n"
             f"S_ED={fs.get('S_ED', float('nan')):.3f}, "
             f"S_MD={fs.get('S_MD', float('nan')):.3f}, "
             f"F_co={fs.get('F_co', float('nan')):+.3f}")

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, dat, ttl in zip(axs,
                            [rho_final.cpu(), rho_projected.cpu(), rho_binary.cpu()],
                            ['raw density', 'projected (continuous)', 'binary (0.5 thresh)']):
        im = ax.imshow(dat.numpy().T, origin='lower', cmap='gray_r',
                       extent=[0, 1, 0, 1], vmin=0, vmax=1)
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel('x/P')
        ax.set_ylabel('y/P')
    fig.suptitle(label, fontsize=9)
    fig.colorbar(im, ax=axs, shrink=0.8)
    fig.savefig(out_dir / 'topology.png', dpi=130, bbox_inches='tight')
    plt.close(fig)

    if history:
        it = [h['iteration'] for h in history]
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.5))
        axs[0].plot(it, [h['F_co'] for h in history])
        axs[0].set_title('F_co')
        axs[1].semilogy(it, [h['S_ED'] for h in history], label='S_ED')
        axs[1].semilogy(it, [h['S_MD'] for h in history], label='S_MD')
        axs[1].legend()
        axs[1].set_title('S_ED / S_MD')
        axs[2].plot(it, [h['balance'] for h in history])
        axs[2].set_title('balance min/max')
        for a in axs:
            a.set_xlabel('iteration')
        fig.suptitle(label, fontsize=9)
        fig.savefig(out_dir / 'fom_history.png', dpi=130, bbox_inches='tight')
        plt.close(fig)

    mid = len(Ex_slices) // 2
    Ex2 = (torch.abs(Ex_slices[mid]) ** 2).cpu().numpy()
    Hz2 = (torch.abs(Hz_slices[mid]) ** 2 / N_SUB ** 2).cpu().numpy()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, dat, ttl in zip(axs, [Ex2, Hz2],
                            ['|Ex/E_inc|^2 (mid-plane)', '|Hz/H_inc|^2 (mid-plane)']):
        im = ax.imshow(dat.T, origin='lower', cmap='inferno', extent=[0, 1, 0, 1])
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel('x/P')
        ax.set_ylabel('y/P')
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(label, fontsize=9)
    fig.savefig(out_dir / 'fields_target.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def preflight(iters=DISCOVERY_ITERS, order=DISCOVERY_ORDER, dx_nm=DISCOVERY_DX_NM):
    verify_no_symmetry_ops()
    si_eps = silicon_eps(TARGET_WAVELENGTH_NM)
    print('=' * 76)
    print('PREFLIGHT - ED/MD co-excitation topology search')
    print('=' * 76)
    print(f"Target wavelength      : {TARGET_WAVELENGTH_NM} nm  <-- primary FoM wavelength")
    print(f"Period list            : {PERIODS_NM} nm")
    print(f"Height list            : {HEIGHTS_NM} nm")
    print(f"Seed list              : {SEEDS}")
    print(f"Substrate eps          : {SUBSTRATE_EPS:.4f} (n={N_SUB})")
    print(f"Silicon eps (aSiH)     : {complex(si_eps):.4f}  "
          f"[NOTE: aSiH table 192-999 nm CLAMPED to 999-nm endpoint at "
          f"{TARGET_WAVELENGTH_NM} nm]")
    print(f"Incident polarization  : x-polarized E (amplitude {SOURCE_AMPLITUDE}), "
          f"direction={SOURCE_DIRECTION} (from substrate)")
    print(f"Incident angle         : inc={INC_ANG_RAD} rad, azi={AZI_ANG_RAD} rad (normal incidence)")
    for P in PERIODS_NM:
        nsub_ratio, nair_ratio = diffraction_flags(P, TARGET_WAVELENGTH_NM)
        flag = ''
        if nsub_ratio >= 1.0:
            flag = '  <-- PROPAGATING substrate order!'
            if not ALLOW_PROPAGATING_HIGHER_ORDERS:
                flag += ' (excluded by ALLOW_PROPAGATING_HIGHER_ORDERS=False)'
        elif nsub_ratio > 0.97:
            flag = '  (near substrate diffraction threshold)'
        nx, ny = configure_geometry(P, dx_nm)
        print(f"  P={P:4d} nm: n_sub*P/lam={nsub_ratio:.4f}, P/lam={nair_ratio:.4f}"
              f"  grid {nx}x{ny} (dx={P/nx:.2f} nm){flag}")
    print(f"Fourier order          : discovery {order}, refine {REFINE_ORDER}, "
          f"verify {VERIFY_ORDERS}")
    print(f"Blur radius            : {BLUR_RADIUS_NM} nm (physical, per-grid kernel)")
    print(f"Iteration count        : discovery {iters}, refine {REFINE_ITERS}")
    print(f"Device                 : {DEVICE} "
          f"({'GPU ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU, no GPU available'})")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"GPU memory             : {free/2**30:.1f} / {total/2**30:.1f} GiB free")
    else:
        import shutil
        print(f"Host RAM (approx)      : "
              f"{os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')/2**30:.1f} GiB")
    print("Field normalization    : torcwa Lorentz-Heaviside units, Z0=1, exp(-jwt);")
    print("                         E_inc=1 (source_fourier sets (0,0)-order Ex=1 in substrate),")
    print(f"                         H_inc=n_sub*E_inc={N_SUB} (|H|=n|E| when Z0=1, c=1)")
    print(f"Probe/evaluation region: full unit-cell {EVAL_GRID_N}x{EVAL_GRID_N} plane, "
          f"z/h={Z_SLICE_FRACS} inside patterned layer (single solve, 3 slices)")
    print("Symmetry projection    : NONE (verified by source grep above)")
    print(f"Objective              : F_co = 0.5*[log(S_ED+{EPS_F}) + log(S_MD+{EPS_F})]")
    print('=' * 76, flush=True)


# ---------------------------------------------------------------------------
# Gradient smoke test + finite-difference check
# ---------------------------------------------------------------------------

def gradient_smoke_test():
    """Reduced-resolution differentiability test + finite-difference
    directional-derivative check, run in float64/complex128 for a clean
    numerical comparison."""
    global GEO_DTYPE, SIM_DTYPE
    geo_bak, sim_bak = GEO_DTYPE, SIM_DTYPE
    GEO_DTYPE, SIM_DTYPE = torch.float64, torch.complex128
    try:
        P, Hh, seed = 870.0, 150.0, 29
        order = [3, 3]
        nx, ny = configure_geometry(P, 15.0)   # coarse 58x58 grid
        g_fft = build_filter(nx, ny, P, BLUR_RADIUS_NM)
        si_eps = silicon_eps(TARGET_WAVELENGTH_NM)

        rho0 = initialize_rho(nx, ny, seed).double()
        beta = 5.0

        def fom_of(rho_t):
            _, rho_tilda = filter_and_project(rho_t, g_fft, beta)
            return evaluate_density(rho_tilda, P, Hh, TARGET_WAVELENGTH_NM,
                                    order, si_eps)

        rho = rho0.clone().requires_grad_(True)
        F_co, S_ED, S_MD = fom_of(rho)
        print(f"smoke: S_ED={float(S_ED):.6f} S_MD={float(S_MD):.6f} "
              f"F_co={float(F_co):+.6f}")
        assert F_co.requires_grad, 'F_co.requires_grad is False!'
        assert float(S_ED) > 0 and math.isfinite(float(S_ED)), 'S_ED not finite/positive'
        assert float(S_MD) > 0 and math.isfinite(float(S_MD)), 'S_MD not finite/positive'
        F_co.backward()
        assert rho.grad is not None, 'rho.grad is None!'
        g = rho.grad.detach().clone()
        assert torch.all(torch.isfinite(g)), 'gradient has non-finite entries!'
        gnorm = float(torch.linalg.norm(g))
        assert gnorm > 0, 'gradient is identically zero!'
        print(f"smoke: backward OK, |grad|={gnorm:.6e}, "
              f"max|grad|={float(g.abs().max()):.6e}, requires_grad=True")

        # finite-difference directional derivative
        torch.manual_seed(12345)
        d = torch.randn_like(rho0)
        d = d / torch.linalg.norm(d)
        analytic = float(torch.sum(g * d))
        print(f"smoke: analytic directional derivative = {analytic:+.8e}")
        ok = True
        for delta in [1e-3, 3e-4, 1e-4]:
            with torch.no_grad():
                pass
            Fp = fom_of((rho0 + delta * d).requires_grad_(False))[0]
            Fm = fom_of((rho0 - delta * d).requires_grad_(False))[0]
            fd = (float(Fp) - float(Fm)) / (2 * delta)
            rel = abs(fd - analytic) / (abs(analytic) + 1e-30)
            print(f"smoke: delta={delta:.0e}  FD={fd:+.8e}  rel.disagreement={rel:.3e}")
            if rel > 0.05:
                ok = False
        print('smoke: PASS' if ok else
              'smoke: WARNING - FD relative disagreement above 5%; inspect before sweeping')
        return ok
    finally:
        GEO_DTYPE, SIM_DTYPE = geo_bak, sim_bak


# ---------------------------------------------------------------------------
# Sweep / refine / verify / spectra drivers
# ---------------------------------------------------------------------------

def discovery_run_list():
    runs = []
    for P in PERIODS_NM:
        nsub_ratio, _ = diffraction_flags(P, TARGET_WAVELENGTH_NM)
        if nsub_ratio >= 1.0 and not ALLOW_PROPAGATING_HIGHER_ORDERS:
            print(f"EXCLUDED P={P} nm: propagating substrate order "
                  f"(n_sub*P/lam={nsub_ratio:.3f}) and "
                  f"ALLOW_PROPAGATING_HIGHER_ORDERS=False")
            continue
        for Hh in HEIGHTS_NM:
            for seed in SEEDS:
                runs.append((P, Hh, seed))
    return runs


def run_discovery_sweep(shard_idx=0, shard_total=1, iters=DISCOVERY_ITERS):
    runs = discovery_run_list()[shard_idx::shard_total]
    print(f"Discovery shard {shard_idx}/{shard_total}: {len(runs)} runs")
    for (P, Hh, seed) in runs:
        rid = run_identifier(P, Hh, seed)
        out = RESULTS_ROOT / 'discovery' / rid
        try:
            optimize_single_run(P, Hh, seed, out, iters=iters,
                                dx_nm=DISCOVERY_DX_NM, order=DISCOVERY_ORDER,
                                stage='discovery')
        except Exception:
            (out / 'run.log').parent.mkdir(parents=True, exist_ok=True)
            with open(out / 'run.log', 'a') as f:
                f.write('FATAL:\n' + traceback.format_exc() + '\n')
            print(f"{rid}: FATAL error, continuing sweep\n{traceback.format_exc()}")


def refine_candidate(discovery_dir, iters=REFINE_ITERS):
    """Refine one discovery result: upsample the raw density onto the finer
    grid (bilinear, consistent continuous representation), restart the
    projection continuation at a moderate beta, run at REFINE_ORDER."""
    discovery_dir = Path(discovery_dir)
    cfg = json.loads((discovery_dir / 'config.json').read_text())
    P, Hh, seed = cfg['period_nm'], cfg['height_nm'], cfg['seed']
    rho_d = torch.tensor(np.load(discovery_dir / 'rho_final.npy'),
                         dtype=GEO_DTYPE, device=DEVICE)
    nx_r, ny_r = configure_geometry(P, REFINE_DX_NM)
    rho_r = torch.nn.functional.interpolate(
        rho_d[None, None], size=(nx_r, ny_r), mode='bilinear',
        align_corners=False)[0, 0].contiguous()
    beta_schedule = BETA_START_REFINE * np.exp(
        np.arange(iters) * math.log(BETA_MAX_REFINE / BETA_START_REFINE) / iters)
    out = RESULTS_ROOT / 'refine' / (run_identifier(P, Hh, seed) + '_refined')
    return optimize_single_run(P, Hh, seed, out, iters=iters,
                               dx_nm=REFINE_DX_NM, order=REFINE_ORDER,
                               beta_schedule=beta_schedule, init_rho=rho_r,
                               stage='refine')


def verify_candidate(run_dir, orders=None):
    """Fourier-order convergence + binary-vs-projected verification of the
    FROZEN final geometry. Forward-only (no gradients)."""
    if orders is None:
        orders = VERIFY_ORDERS
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / 'config.json').read_text())
    P, Hh, lam = cfg['period_nm'], cfg['height_nm'], cfg['target_wavelength_nm']
    si_eps = silicon_eps(lam)
    rho_p = torch.tensor(np.load(run_dir / 'rho_projected.npy'),
                         dtype=GEO_DTYPE, device=DEVICE)
    rho_b = torch.tensor(np.load(run_dir / 'rho_binary.npy'),
                         dtype=GEO_DTYPE, device=DEVICE)
    rows = []
    with torch.no_grad():
        for order in orders:
            for tag, dens in [('projected', rho_p), ('binary', rho_b)]:
                t0 = time.time()
                F_co, S_ED, S_MD = evaluate_density(dens, P, Hh, lam, order, si_eps)
                rows.append({'order_x': order[0], 'order_y': order[1],
                             'geometry': tag, 'F_co': float(F_co),
                             'S_ED': float(S_ED), 'S_MD': float(S_MD),
                             'balance': float(min(S_ED, S_MD) / (max(S_ED, S_MD) + 1e-12)),
                             'eval_time_s': time.time() - t0})
                print(f"{run_dir.name}: order {order} {tag:9s} "
                      f"F_co={float(F_co):+.4f} S_ED={float(S_ED):.4f} "
                      f"S_MD={float(S_MD):.4f}", flush=True)
    with open(run_dir / 'verification.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows


def spectra_candidate(run_dir, span_nm=100.0, step_nm=5.0, order=None,
                      geometry='binary'):
    """Post-optimization DIAGNOSTIC wavelength sweep (never part of F_co):
    S_ED(lambda), S_MD(lambda) and 0th-order transmission."""
    if order is None:
        order = SPECTRA_ORDER
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / 'config.json').read_text())
    P, Hh, lam0 = cfg['period_nm'], cfg['height_nm'], cfg['target_wavelength_nm']
    rho = torch.tensor(np.load(run_dir / f'rho_{geometry}.npy'),
                       dtype=GEO_DTYPE, device=DEVICE)
    lams = np.arange(lam0 - span_nm, lam0 + span_nm + 0.1, step_nm)
    rows = []
    with torch.no_grad():
        for lam in lams:
            si_eps = silicon_eps(lam)
            sim = build_simulation(rho, P, Hh, float(lam), order, si_eps)
            Ex_s, Hz_s, _ = compute_fields(sim, P, Hh)
            S_ED, S_MD = compute_mode_scores(Ex_s, Hz_s)
            txx = sim.S_parameters(orders=[0, 0], direction='forward',
                                   port='transmission', polarization='xx',
                                   ref_order=[0, 0])
            tyx = sim.S_parameters(orders=[0, 0], direction='forward',
                                   port='transmission', polarization='yx',
                                   ref_order=[0, 0])
            T0 = float(torch.abs(txx) ** 2 + torch.abs(tyx) ** 2)
            rows.append({'wavelength_nm': float(lam), 'S_ED': float(S_ED),
                         'S_MD': float(S_MD), 'T0': T0,
                         'order_x': order[0], 'order_y': order[1],
                         'geometry': geometry})
            print(f"{run_dir.name}: lam={lam:7.1f}  S_ED={float(S_ED):8.4f}  "
                  f"S_MD={float(S_MD):8.4f}  T0={T0:.4f}", flush=True)
    with open(run_dir / 'spectra.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows


def save_multipole_data(run_dir, n_eval=64, nz=7, geometry='binary'):
    """Save the complex near fields + eps distribution inside the patterned
    layer that a later multipole decomposition needs (polarization current
    J = -i w eps0 (eps-1) E in physical units). No trustworthy multipole
    implementation exists in this workspace, so decomposition itself is left
    as a documented validation task."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / 'config.json').read_text())
    P, Hh, lam = cfg['period_nm'], cfg['height_nm'], cfg['target_wavelength_nm']
    si_eps = silicon_eps(lam)
    rho = torch.tensor(np.load(run_dir / f'rho_{geometry}.npy'),
                       dtype=GEO_DTYPE, device=DEVICE)
    z_fracs = [(k + 0.5) / nz for k in range(nz)]
    with torch.no_grad():
        sim = build_simulation(rho, P, Hh, lam, REFINE_ORDER, si_eps)
        _, _, allf = compute_fields(sim, P, Hh, n_eval=n_eval,
                                    z_fracs=z_fracs, keep_all=True)
        E = np.stack([np.stack([c.cpu().numpy() for c in Ef]) for Ef, _ in allf])
        Hf = np.stack([np.stack([c.cpu().numpy() for c in Hf]) for _, Hf in allf])
        # eps on the same sampling grid (nearest-pixel sampling of the mask)
        x_ax, _ = eval_axes(P, n_eval)
        idx = (x_ax / (P / rho.shape[0])).long().clamp(max=rho.shape[0] - 1)
        eps_map = (rho[idx][:, idx] * si_eps + (1 - rho[idx][:, idx])).cpu().numpy()
        np.savez_compressed(
            run_dir / 'multipole_data.npz',
            E_fields=E, H_fields=Hf, z_fracs=np.array(z_fracs),
            eps_map=eps_map, period_nm=P, height_nm=Hh, wavelength_nm=lam,
            note=('E,H complex fields [z,comp,x,y] in torcwa LH units '
                  '(E_inc=1, Z0=1); J = -i*omega*eps0*(eps-1)*E after unit '
                  'restoration. Multipole identity remains a validation task.'))
    print(f"{run_dir.name}: multipole_data.npz saved ({nz} z-slices, "
          f"{n_eval}x{n_eval})", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--mode', required=True,
                    choices=['preflight', 'smoke', 'single', 'sweep', 'refine',
                             'verify', 'spectra', 'multipole-data'])
    ap.add_argument('--period', type=float, default=870.0)
    ap.add_argument('--height', type=float, default=150.0)
    ap.add_argument('--seed', type=int, default=29)
    ap.add_argument('--iters', type=int, default=None)
    ap.add_argument('--shard', type=int, nargs=2, default=[0, 1],
                    metavar=('IDX', 'TOTAL'))
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--run-dir', type=str, default=None,
                    help='run directory for refine/verify/spectra/multipole-data')
    ap.add_argument('--geometry', type=str, default='binary',
                    choices=['binary', 'projected'])
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    if args.mode == 'preflight':
        preflight()
    elif args.mode == 'smoke':
        preflight()
        ok = gradient_smoke_test()
        sys.exit(0 if ok else 1)
    elif args.mode == 'single':
        preflight(iters=args.iters or DISCOVERY_ITERS)
        rid = run_identifier(args.period, args.height, args.seed)
        out = RESULTS_ROOT / 'discovery' / rid
        optimize_single_run(args.period, args.height, args.seed, out,
                            iters=args.iters or DISCOVERY_ITERS,
                            dx_nm=DISCOVERY_DX_NM, order=DISCOVERY_ORDER,
                            stage='discovery')
    elif args.mode == 'sweep':
        if args.shard[0] == 0:
            preflight(iters=args.iters or DISCOVERY_ITERS)
        run_discovery_sweep(args.shard[0], args.shard[1],
                            iters=args.iters or DISCOVERY_ITERS)
    elif args.mode == 'refine':
        refine_candidate(args.run_dir, iters=args.iters or REFINE_ITERS)
    elif args.mode == 'verify':
        verify_candidate(args.run_dir)
    elif args.mode == 'spectra':
        spectra_candidate(args.run_dir, geometry=args.geometry)
    elif args.mode == 'multipole-data':
        save_multipole_data(args.run_dir, geometry=args.geometry)


if __name__ == '__main__':
    main()
