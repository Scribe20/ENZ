"""Central configuration for the bare air / ITO / glass ENZ target-mode calculation.

Conventions (declared once, used everywhere)
--------------------------------------------
Time convention:      exp(-i * omega * t)   (same as Vassant et al., Opt. Express 20,
                      23971 (2012), and the same as TORCWA's exp(-j*omega*t)).
Field ansatz:         F(x, z, t) = F(z) * exp(+i*K*x) * exp(-i*omega*t)
                      with K = K' + i*K'' the complex in-plane propagation constant.
kz definition:        kz_j = sqrt(eps_j * k0^2 - K^2),  k0 = 2*pi/lambda0 (vacuum).
Branch selection:     see tm_slab_mode.kz_branch (Vassant prescription
                      Re(kz)+Im(kz) > 0 on the "proper" sheet; sheets are made
                      explicit everywhere).
Units:                lengths in nm internally; SI factors (eps0, c) drop out of
                      every ratio we report, and the saved fields use the
                      normalization recorded in the .npz metadata.

Layer indexing (matches Vassant Fig. 1 with 1=superstrate, 2=film, 3=substrate):
      z > d      medium 1: air        (eps1 = 1)
      0 < z < d  medium 2: ITO        (eps2 = eps_ITO(lambda), from CSV)
      z < 0      medium 3: glass      (eps3 = n_glass^2)
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
# ITO film thickness. The Karimi et al. main text and SI (Nano Lett. 2023, 23,
# 11555) repeatedly specify "a 23 nm-thick ITO substrate" / "23 nm-thick ITO on
# SiO2 ... obtained from a commercial source".  23 nm is therefore a paper value.
D_ITO_NM = 23.0

# Glass (substrate) refractive index.
# The supplied papers specify the substrate material as SiO2 ("23 nm-thick ITO
# on SiO2", Karimi SI section S2) but give NO numeric index value anywhere in
# the supplied text.  n_glass is therefore an *input parameter of this analysis*,
# NOT a paper value.  We use the fused-silica (Malitson 1965 Sellmeier) value at
# 1.45 um, n = 1.4446, rounded to 4 digits.  Change it here to test sensitivity.
N_GLASS = 1.4446
N_GLASS_IS_PAPER_VALUE = False   # kept in saved metadata for honesty

EPS_AIR = 1.0

# ---------------------------------------------------------------------------
# Material data
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
ITO_CSV = _HERE / "data" / "ito_digitized_dense_1nm_physical.csv"

# ---------------------------------------------------------------------------
# Solver settings
# ---------------------------------------------------------------------------
LAMBDA_MIN_NM = 1200.0
LAMBDA_MAX_NM = 1700.0
LAMBDA_STEP_NM = 1.0          # continuation step

# Seed grid for the initial (first-wavelength) root scan, in units of K/k0.
SEED_RE = (0.3, 0.6, 0.9, 1.05, 1.2, 1.45, 1.7, 2.0, 2.5, 3.0, 3.5)
SEED_IM = (0.0, 0.05, 0.15, 0.35, 0.7, 1.2)

ROOT_XTOL = 1e-13             # scipy.optimize.root tolerance on x = (K'/k0, K''/k0)
RESIDUAL_OK = 1e-9            # |D| accepted as a converged pole
DEDUP_TOL = 1e-6              # roots closer than this in K/k0 are duplicates

# z-grid for field reconstruction / saving
N_Z_ITO = 231                 # points across the ITO film (0.1 nm steps)
Z_PAD_FACTOR = 3.0            # cladding extent = factor * local decay length
N_Z_CLAD = 600                # points per cladding

FIG_DIR = _HERE / "figures"
OUT_NPZ = _HERE / "target_enz_mode.npz"
OUT_BRANCH_CSV = _HERE / "enz_branch.csv"
