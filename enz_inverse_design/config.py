"""Configuration for the ENZ-overlap freeform inverse design (Phase 2).

Single source of truth for every parameter.  No magic numbers elsewhere.

Provenance of the fixed values (honesty ledger):
- wavelength, ITO thickness, eps_ITO, n_glass, target K: from the Phase-1
  output target_enz_mode.npz (../enz_target/).
- a-Si thickness 140 nm: Karimi et al. EDR metasurface thickness.
- eps_aSi: the supplied TORCWA Materials_data/aSiH.txt covers only
  192-999 nm; at 1527 nm the supplied Materials.aSiH class would silently
  clamp to the 999-nm value (n = 2.99), which is not physical for NIR a-Si.
  Therefore eps_aSi is an EXPLICIT INPUT PARAMETER here: n_aSi = 3.48
  (typical PECVD a-Si:H at 1.5 um, negligible loss below the gap).
  This is an assumption of this analysis, not a supplied-data value.
- period: chosen so that a reciprocal-lattice harmonic matches the target
  ENZ momentum: Re(K)/k0 = 5.9463 -> the (3,0) harmonic of a 770 nm square
  lattice gives |G|/k0 = 3*lambda/px = 5.9494 (mismatch 5e-4, diagnosed and
  printed at run time - see target_mode.momentum_diagnostic).
"""

from pathlib import Path
import torch

_HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
TORCWA_DIR = _HERE / "third_party"          # vendored supplied torcwa package
# v2 target (review fix): real K = 3G, complex-omega pole of the bare slab
# (~1470.8 nm), produced by ../enz_target/solve_periodic_target.py.  The
# original complex-K target (1527 nm) is kept for reference as
# target_enz_mode.npz but is NOT self-consistent with a periodic cell.
TARGET_MODE_FILE = _HERE.parent / "enz_target" / "target_enz_mode_periodic_850_g10.npz"
ENZ_TARGET_DIR = _HERE.parent / "enz_target"   # Phase-1 package (ITO material)
ASI_NK_FILE = _HERE / "data_aSi_H_measured_Postech_extended_to_2000nm.txt"
OUT_DIR = _HERE / "outputs"

# --------------------------------------------------------------------------
# Hardware / dtypes  (same scheme as the supplied Example6)
# --------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# complex128: the finite-difference gradient check agrees with autograd to
# ~1e-8 in double precision but only ~20% in complex64 (validate_gradient.py);
# at this problem size the 2x runtime cost is worth exact gradients.
SIM_DTYPE = torch.complex128
GEO_DTYPE = torch.float64
N_THREADS = 4

# --------------------------------------------------------------------------
# Fixed physical configuration (NOT optimized in this phase)
# --------------------------------------------------------------------------
WAVELENGTH_NM = None      # None -> read from the target npz (1527 nm)
PX_NM = 850.0             # FROZEN benchmark period (Karimi EDR lattice)
PY_NM = 850.0             # FROZEN benchmark period
ASI_THICKNESS_NM = 140.0  # Karimi EDR a-Si thickness
ITO_THICKNESS_NM = None   # None -> read from target npz (23 nm)
N_GLASS = None            # None -> read from target npz (1.4446)
# a-Si permittivity: resolved at startup from the measured POSTECH n,k file
# (ASI_NK_FILE) at the design wavelength; the old n=3.48 assumption is
# retired (review fix - at 1471 nm the measured n is ~2.963, eps ~ 8.78).
EPS_ASI = None            # None -> from ASI_NK_FILE at the design wavelength
INC_ANGLE_RAD = 0.0
AZI_ANGLE_RAD = 0.0
POLARIZATION = "x"        # incident E along x (couples to x-propagating TM ENZ)

# --------------------------------------------------------------------------
# Target-mode handling
# --------------------------------------------------------------------------
# Primary objective is the +-K modal subspace (review fix): a reciprocal,
# mirror-symmetric cell at normal incidence excites +K and -K degenerately
# (the v1 run gave |a-|/|a+| = 0.9995), so optimizing |a+|^2 alone renames a
# standing wave as directional. "bidir" makes the actual physics explicit.
TARGET_DIRECTION = "bidir"   # "+x" | "-x" | "bidir"
# Optical objective mode:
#   "qnm_overlap"   - historical: F = (|a+|^2+|a-|^2)/P_inc vs the QNM target
#   "ito_ez_volume" - direct ENZ excitation: F_ENZ = <|Ez/E_inc|^2>_ITO,
#                     the volume-averaged TOTAL longitudinal intensity in the
#                     ITO (all harmonics, no prescribed momentum channel);
#                     |E_inc| = 1 in the TORCWA source convention.
#   "ito_absorption" - resonant power-transfer campaign: maximize
#                     A_ITO(lambda_E) = 1-R-T (identity: ITO is the only
#                     lossy layer; cross-validated vs the volume integral in
#                     enz_absorption_campaign/target_audit.py) subject to the
#                     differentiable resonance surrogate
#                     A(lambda_E +/- W/2) <= A(lambda_E)/2 with
#                     W = lambda_E/Q_MIN (i.e. spectral Q >= Q_MIN), enforced
#                     as a normalized penalty (see PENALTY_MU).
OBJECTIVE = "qnm_overlap"
Q_MIN = 5.0        # from trusted refs: bare ENZ QNM Q=5.80, hybrid pole 5.04
PENALTY_MU = 10.0  # normalized penalty weight; sensitivity {3,10,30} planned
MOMENTUM_MISMATCH_MAX = 0.05   # allowed |ReK - |G|| / ReK before warning hard
Z_SAMPLES_ITO = 7         # midpoint z-slices inside ITO for the overlap

# --------------------------------------------------------------------------
# RCWA numerics
# --------------------------------------------------------------------------
FOURIER_ORDER = [7, 7]    # full run
NX_DESIGN = 128           # design/overlap grid (dx = 6.0 nm)
NY_DESIGN = 128

# --------------------------------------------------------------------------
# Optimization (mirrors the supplied Example6 scheme)
# --------------------------------------------------------------------------
N_ITER = 150
LR_INITIAL = 0.02         # Adam step, cosine-decayed to 0 (as Example6)
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8
BETA_PROJ_MAX = 1000.0    # tanh projection sharpness ramp: exp(0 -> ln 1000)
FILTER_RADIUS_NM = 40.0   # Gaussian blur radius (Example6 used 20 nm at 532)
MIRROR_SYMMETRY_Y = True  # Example6 enforced y-mirror; target has no y phase
RANDOM_SEED = 333         # same seed convention as Example6
SAVE_EVERY = 10

# loss = -F_ENZ  (default);  set True for  -log(F_ENZ + eps)
USE_LOG_LOSS = False
LOG_LOSS_EPS = 1e-12

# --------------------------------------------------------------------------
# Smoke test overrides
# --------------------------------------------------------------------------
SMOKE = dict(FOURIER_ORDER=[4, 4], NX_DESIGN=64, NY_DESIGN=64, N_ITER=3,
             Z_SAMPLES_ITO=5, SAVE_EVERY=1)


def apply_smoke(module=None):
    """Mutate this config module in place with the smoke-test overrides."""
    import sys
    m = module or sys.modules[__name__]
    for k, v in SMOKE.items():
        setattr(m, k, v)
    return m
