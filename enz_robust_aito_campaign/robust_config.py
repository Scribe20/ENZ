"""ROBUST ENZ ENERGY-TRANSFER campaign: single source of truth.

Objective (fixed by the task): maximize the angle-robust ITO absorption
    A_ITO(lambda_E; theta, phi) = 1 - R_total - T_total   (all orders)
through the calibrated differentiable smooth-min over the angular set
    J_robust = -(1/beta) log sum_m w_m exp(-beta A_m).
No Q, Ez, harmonic, QNM-overlap, multipole, Kerker, BIC, polariton or
critical-coupling term enters the loss.  Design variables: rho(x,y),
period P, a-Si height h, air padding p_pad > 0 (fraction of P).  Fixed:
ITO 23 nm (measured CSV eps at lambda_E), glass 1.4446, lambda_E.

lambda_E = 1433.488 nm: real part of the bare air/ITO(23)/glass TM QNM at
K = G10(850 nm) (enz_target/target_enz_mode_periodic_850_g10.npz).  It is the
ENZ-band anchor inherited from the frozen 850-nm benchmark, kept fixed here
by the task statement; it is NOT re-derived for the variable-P cells.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

LAMBDA_E = 1433.488

# ---------------------------------------------------------------------------
# Angular domain.  Repo search (grep -i "numerical aperture|high-NA|NA=" over
# *.md/*.py/*.txt) found NO authoritative NA / acceptance-angle requirement;
# only the planar Berreman check uses oblique incidence.  The sets below are
# therefore an ASSUMPTION of this campaign (documented in PREFLIGHT.md):
# a modest +-30 deg cone (NA ~ 0.5 in air), lab-frame x polarization.
# ---------------------------------------------------------------------------
ANGLES_SCREEN = [(0.0, 0.0), (20.0, 0.0), (20.0, 90.0)]           # Stages 2-3
ANGLES_FULL = [(0.0, 0.0), (15.0, 0.0), (30.0, 0.0),
               (15.0, 90.0), (30.0, 90.0)]                        # Stage 4
ANGLES_EVAL_PLANES = {"phi0": [(t, 0.0) for t in range(0, 41, 5)],
                      "phi90": [(t, 90.0) for t in range(0, 41, 5)],
                      "phi45": [(t, 45.0) for t in range(0, 41, 5)]}
WEIGHTS = "uniform"          # w_m = 1/M (assumption; no angular spectrum given)

# smooth-min sharpness: calibrated in preflight from the angular spread of the
# reference structures: beta = ln(10) / median_m spread(A_m) so that a
# structure whose worst angle is one reference-spread below the mean is
# weighted 10x more than the mean angle; clipped to BETA_RANGE.
BETA_RULE = "ln(10)/median_reference_angular_spread"
BETA_RANGE = (5.0, 200.0)

# ---------------------------------------------------------------------------
# Geometry search space
# ---------------------------------------------------------------------------
NX = 128                     # normalized design grid (dx = P/128)
P_SCREEN = [750.0, 850.0, 950.0, 1050.0]   # 1050 > lambda_E/n_glass: (+-1,0)
                                            # propagate in glass at normal inc.
H_SCREEN = [120.0, 160.0]    # brackets the 140-nm benchmark; Stage 3 steps +-20
PAD_SCREEN = [0.05, 0.10]    # fraction of P per side (>0 always)
PAD_MIN = 0.03               # never below (positive padding required)
SEEDS_SCREEN = [333, 1001]   # Example6 seed convention + one more
P_BOUNDS, H_BOUNDS, PAD_BOUNDS = (650.0, 1150.0), (90.0, 240.0), (0.03, 0.15)

# ---------------------------------------------------------------------------
# Optimizer (Example6 architecture; BOTH fliplr symmetry projections removed)
# ---------------------------------------------------------------------------
FILTER_RADIUS_NM = 40.0
LR_INITIAL = 0.02
BETA_PROJ_MAX = 1000.0
ORDER_SCREEN = [5, 5]
ORDER_FULL = [7, 7]
ORDER_REFINE = [[7, 7], [9, 9], [11, 11]]
N_ITER_FULL = 150

# wall-clock budgets (hours) used to size the shortened runs automatically
# from the preflight timing benchmark
BUDGET_H = dict(stage2=3.0, stage3=1.5, stage4=5.0)
N_ITER_SCREEN_RANGE = (25, 60)
N_ITER_REFINE_RANGE = (30, 80)
N_TOP_STAGE2 = 4
N_FINALISTS = 2

# post-hoc resonance gate (weak, only reported; NOT in the loss)
RES_PROBE_OFFSET_NM = 80.0
