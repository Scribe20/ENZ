"""Loading, momentum diagnostics, and grid mapping of the Phase-1 ENZ target.

Coordinate mapping (critical, verified against both codes):
- Phase-1 npz: z_p1 = 0 at ITO/glass, z_p1 = d at the air-side interface
  (z increases from glass toward air).
- TORCWA: layers are stacked from the input (air, top) toward the output
  (glass); field_xy(layer_num, ..., z_prop) measures z_prop from the layer's
  input-side face.  For the ITO layer, z_prop = 0 is the a-Si/ITO interface
  and z_prop = d is the ITO/glass interface.
- Therefore  z_p1 = d - z_prop, and the stored profile must be evaluated at
  flipped coordinates.  (The z-axis inversion flips the sign of Ez relative
  to Phase 1; this is a global phase of the target and does not change |a|.)

Complex-K treatment (documented approximation):
- The saved eigenmode has K = K' + iK'' with K'' large (overdamped mode).
  The periodic overlap target uses ONLY the phase factor exp(+i K' x); the
  lateral decay exp(K'' x) is NOT built into the periodic target (a periodic
  cell cannot represent it).  Im(K) is reported as a propagation-loss
  diagnostic: 1/|K''| ~ 19 nm << px, so the physical mode is a localized,
  heavily damped resonance and this overlap tests the generation of its
  field *pattern*, not lateral propagation.
"""

import numpy as np
import torch

import config


def load_target_npz(path=None):
    """Load and report the Phase-1 target file. Returns a plain dict."""
    path = path or config.TARGET_MODE_FILE
    d = np.load(path, allow_pickle=False)
    keys = list(d.keys())
    out = {k: d[k] for k in keys}
    print(f"[target] loaded {path}")
    print(f"[target] keys: {sorted(keys)}")
    lam = float(out["wavelength_nm"])
    K = complex(float(out["K_real_per_nm"]), float(out["K_imag_per_nm"]))
    k0 = float(out["k0_per_nm"])
    print(f"[target] lambda = {lam:.2f} nm, K/k0 = {K/k0:.4f}, "
          f"d_ITO = {float(out['ito_thickness_nm']):.1f} nm, "
          f"n_glass = {float(out['glass_index']):.4f}")
    print(f"[target] eps_ITO = {float(out['eps_ito_real']):+.4f} "
          f"+ {float(out['eps_ito_imag']):.4f}i, "
          f"normalization = {out['normalization']}")
    return out


def momentum_diagnostic(tgt, px_nm=None, py_nm=None, max_order=8,
                        k_parallel=(0.0, 0.0), verbose=True):
    """Check whether a reciprocal-lattice harmonic can supply Re(K_ENZ).

    Returns (m, n), delta_K_rel.  Uses |k_parallel + G_mn| vs Re(K).
    """
    px = px_nm or config.PX_NM
    py = py_nm or config.PY_NM
    K_re = float(tgt["K_real_per_nm"])
    K_im = float(tgt["K_imag_per_nm"])
    k0 = float(tgt["k0_per_nm"])

    best = None
    for m in range(-max_order, max_order + 1):
        for n in range(-max_order, max_order + 1):
            gx = k_parallel[0] + 2 * np.pi * m / px
            gy = k_parallel[1] + 2 * np.pi * n / py
            g = np.hypot(gx, gy)
            err = abs(K_re - g)
            if best is None or err < best[0]:
                best = (err, (m, n), g)
    err, (m, n), g = best
    delta = err / K_re
    if verbose:
        print(f"[momentum] target Re(K)/k0 = {K_re/k0:.4f}; "
              f"best harmonic (m,n) = ({m},{n}) with |k_par+G|/k0 = {g/k0:.4f}")
        print(f"[momentum] relative mismatch delta_K = {delta:.2e} "
              f"(threshold {config.MOMENTUM_MISMATCH_MAX})")
        if K_im != 0.0:
            print(f"[momentum] Im(K) diagnostic: 1/|Im K| = {1/abs(K_im):.1f} "
                  f"nm (vs period {px:.0f} nm) - overdamped mode; Im K is NOT "
                  "built into the periodic target (see module docstring)")
        else:
            print("[momentum] target K is real (real-K/complex-omega pole "
                  "formulation - self-consistent with the periodic cell); "
                  "modal damping lives in Im(omega): Q = "
                  f"{float(tgt.get('pole_Q', np.nan)):.2f}")
        if delta > config.MOMENTUM_MISMATCH_MAX:
            print("[momentum] WARNING: the current fixed lattice period is not "
                  "well matched to the target ENZ momentum, so the present "
                  "optimization should be interpreted as a preliminary "
                  "field-overlap test.")
    return (m, n), delta


def ito_z_slices(d_ito, n_z):
    """Midpoint-rule z_prop positions (TORCWA convention) inside the ITO."""
    return (np.arange(n_z) + 0.5) * d_ito / n_z


def build_target_field(tgt, x_nm, y_nm, z_prop_nm, direction="+x"):
    """Ez_target(x, y, z) on the TORCWA grid, normalized on that grid.

    x_nm, y_nm: 1-D arrays (nm); z_prop_nm: 1-D array of TORCWA z_prop values
    inside the ITO.  Returns a detached torch tensor of shape (Nz, Nx, Ny)
    plus the grid volume element dV (nm^3).
    """
    d = float(tgt["ito_thickness_nm"])
    K_re = float(tgt["K_real_per_nm"])

    # z-profile: interpolate the stored complex Ez(z_p1) at z_p1 = d - z_prop
    z_p1 = d - np.asarray(z_prop_nm)
    zs = tgt["z_nm"]
    Ez = tgt["Ez"]
    prof = np.interp(z_p1, zs, Ez.real) + 1j * np.interp(z_p1, zs, Ez.imag)

    sign = +1.0 if direction == "+x" else -1.0
    x = np.asarray(x_nm)[:, None]
    phase = np.exp(1j * sign * K_re * x)                       # (Nx, 1)
    T = prof[:, None, None] * phase[None, :, :]                # (Nz, Nx, 1)
    T = np.broadcast_to(T, (len(z_prop_nm), len(x_nm), len(y_nm))).copy()

    dx = float(x_nm[1] - x_nm[0])
    dy = float(y_nm[1] - y_nm[0])
    dz = d / len(z_prop_nm)
    dV = dx * dy * dz

    norm = np.sum(np.abs(T) ** 2) * dV
    T = T / np.sqrt(norm)
    resid = abs(np.sum(np.abs(T) ** 2) * dV - 1.0)
    print(f"[target] grid normalization: integral_ITO |Ez_t|^2 dV = 1, "
          f"residual = {resid:.2e} (dV = {dV:.3f} nm^3, direction {direction})")

    Tt = torch.as_tensor(T, dtype=config.SIM_DTYPE, device=config.DEVICE)
    return Tt.detach(), dV
