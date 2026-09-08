"""JZ freeform longitudinal-dissipation (F_z) inverse-design campaign.

Single source of truth for every parameter of enz_jz_inverse_design.
Nothing in the historical campaigns (enz_*) is modified; the vendored
upstream torcwa in ../enz_inverse_design/third_party is imported verbatim.

Physics (fixed by the task):
    stack        air / freeform a-Si:H (height h, density rho(x,y)) /
                 ITO 23 nm / soda-lime glass (semi-infinite)
    source       plane wave from air, normal incidence, lab-frame x
                 polarization, |E_inc| = 1 (TORCWA Lorentz-Heaviside units,
                 c = eps0 = mu0 = 1, time convention exp(-j w t))
    wavelength   lambda_ZE = Re[eps_ITO] zero crossing of the SUPPLIED
                 ITO_nk.csv (found by interpolation in materials.py, not
                 hard-coded)
    objective    F_z = (w/2) Im[eps_ITO] int_ITO |E_z|^2 dV / P_inc,cell
                 P_inc,cell = 0.5 cos(theta) |E_inc|^2 P^2  (n_air = 1)
    diagnostics  F_x, F_y (same normalization), F_tot = F_x+F_y+F_z,
                 A = 1 - R_tot - T_tot (all propagating orders),
                 eta_z,abs = F_z / F_tot, <|Ez/E_inc|^2>_ITO, max|Ez/E_inc|^2
"""

from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TORCWA_DIR = ROOT / "enz_inverse_design" / "third_party"   # vendored upstream torcwa (read-only)
MAT_DIR = HERE / "materials_data"
OUT = HERE / "outputs"

ITO_FILE = MAT_DIR / "ITO_nk.csv"
ASI_FILE = MAT_DIR / "aSi_H_measured_Postech.txt"
GLASS_FILE = MAT_DIR / "SiO2_substrate_measured.txt"

# ---------------------------------------------------------------------------
# numerics
# ---------------------------------------------------------------------------
DEVICE = torch.device("cpu")          # this session: 4-core CPU, no CUDA
SIM_DTYPE = torch.complex128          # exact autograd (validated in the repo)
GEO_DTYPE = torch.float64
N_THREADS_DEFAULT = 4

D_ITO_NM = 23.0
NX = 128                              # topology grid (dx = P / 128)
Z_SAMPLES_ITO = 7                     # midpoint slices inside the ITO (start)
THETA_DEG, PHI_DEG = 0.0, 0.0         # normal incidence
POL = "labx"

# Example6-derived optimizer scheme (validated in the repo)
FILTER_RADIUS_NM = 40.0
LR_INITIAL = 0.02
BETA_PROJ_MAX = 1000.0
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8

ORDER_SCREEN = [5, 5]
ORDER_FULL = [7, 7]
ORDER_CERT = [[9, 9], [11, 11]]

# ---------------------------------------------------------------------------
# outer geometric search (Stage 1)
# ---------------------------------------------------------------------------
P_SCREEN = [550.0, 650.0, 750.0, 825.0, 850.0]        # below lambda/n_glass
H_SCREEN = [100.0, 150.0, 200.0, 250.0, 300.0, 400.0]  # broad; no fixed fab value found
PAD_SCREEN = [0.04, 0.08, 0.12]                        # hard air ring, fraction of P per side
SEEDS_SCREEN = [333, 1001, 7777]                       # from-scratch seeds
PAD_MIN = 0.03

# ---------------------------------------------------------------------------
# predeclared preflight tolerances (hard gates)
# ---------------------------------------------------------------------------
TOL = dict(
    ito_eps_columns_abs=2e-5,      # |(n+ik)^2 - (eps1 + i eps2)| over the file (6-decimal data)
    lambda_ze_sanity_nm=1.0,       # expected ~1302.3 nm from the supplied file header
    eps2_ze_sanity=0.01,           # expected ~0.432
    ez_bare_max=1e-12,             # |Ez| in the ITO of the planar stack at normal incidence
    energy_abs=1e-9,               # |R+T-1| for lossless stacks (all orders)
    identity_abs=1e-3,             # |F_tot - (1-R-T)| at n_z = 7
    identity_rel=5e-3,             # relative
    components_abs=1e-12,          # |Fx+Fy+Fz - F_vol(|E|^2)|
    grad_rel=1e-4,                 # autograd vs central FD
    pad_leak=0.0,                  # exactly empty air ring
    s_flip_min=0.05,               # no hidden symmetry
)
