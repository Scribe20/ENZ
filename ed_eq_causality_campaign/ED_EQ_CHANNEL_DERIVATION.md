# ED–EQ radiation-channel derivation (frozen before discovery)

Illumination: normal incidence, E ∥ x, propagation +z (TORCWA/Example6
convention: 'forward' runs from the input silica half-space, below, through
the patterned layer, into air above; +z is the forward propagation
direction; the layer occupies 0 ≤ z ≤ h).

## 1. Which multipoles radiate into the specular (0,0) channels?

For a periodic array (square cell, area A = P²) of identical induced cell
currents J(r) at normal incidence, the scattered 0th-order plane-wave
amplitude in a homogeneous background of index n_b (k_b = n_b k₀,
impedance Z_b = Z₀/n_b) is the transverse Fourier component of the cell
current at the outgoing wavevector:

    E_α^{±}(scattered) = −(Z_b / 2A) · ∫_cell J_α(r) · e^{∓ i k_b z} dV,
    α ∈ {x, y},  '+' = upward (+z), '−' = downward (−z).

(Standard periodic-sheet result; longitudinal components do not radiate at
normal emergence.) Expanding e^{∓ik_b z} about the layer-center origin:

    ∫ J_x e^{∓ik_b z} dV
      = ∫J_x dV  ∓  i k_b ∫ z J_x dV  −  (k_b²/2) ∫ z² J_x dV + …

Using the long-wavelength current-moment identities (MENP/toroidal
conventions; primitive Q̃_ij = ∫(r_i J_j + r_j J_i) dV, Qe_xz =
−(3/iω) Q̃_xz, m_y = ½∫(r×J)_y = ½∫(z J_x − x J_z)):

    ∫ J_x dV     = −iω p_x
    ∫ z J_x dV   = ½ Q̃_xz + m_y = −(iω/6) Qe_xz + m_y

so, to first order in k_b z,

    E_x^{±} ∝ −iω p_x ∓ i k_b [ −(iω/6) Qe_xz + m_y ]  + O(k²z²).

**Selection rules at the (0,0) order (k_t = 0):**

| multipole | x-pol specular channel | parity (+z vs −z) |
|---|---|---|
| p_x | YES | even |
| m_y | YES | odd |
| **Qe_xz** | **YES** | **odd** |
| p_y, p_z, m_x, m_z | no (y-pol or longitudinal) | — |
| Qe_xy, Qe_xx−Qe_yy, Qe_yy, Qe_zz(traceless part alone) | no — they enter only through in-plane phase factors e^{−i k_t·r}, which vanish at k_t = 0; they radiate into ±1 diffraction orders once those propagate | — |
| Qe_yz | y-pol channel only (partner of p_y) | odd |
| Qm_xy-type | y-pol/higher order | — |

The same conclusion follows from the free-space far-field operator
n̂×(n̂×(Q·n̂)) at n̂ = ±ẑ: (Q·ẑ)_t = (Q_xz, Q_yz), so along z only Q_xz
radiates x-polarized field, with odd parity. Two independent derivations
agree.

**Frozen target: Qe_xz.** It is the unique EQ Cartesian component that
shares the x-polarized specular channel with p_x. (Confirmed, not assumed.)

## 2. Physical consequences that shape the campaign hypothesis

1. **Channel degeneracy Q_xz ↔ m_y.** At 0th order the odd channel carries
   both m_y and Qe_xz (the k_b z expansion cannot distinguish them at this
   order — they differ in the ±1-order and angular patterns only). The
   qualification must therefore report m_y alongside Q_xz, and the
   channel-amplitude reconstruction must include both.
2. **Parity obstruction in a symmetric background.** With up/down mirror
   symmetry, total radiated power |E⁺|² + |E⁻|² = 2(|even|² + |odd|²):
   ED–EQ cross terms cancel in the SUM — pure p_x/Q_xz interference can
   only redistribute power (Kerker-type directionality), not reduce it.
   **The silica substrate breaks this parity**: the two channels acquire
   different impedances/Fresnel weights and interference between the even
   and odd contents CAN change total leakage. The prior campaign showed
   these resonances are substrate-bound; the Q-emergence hypothesis is
   therefore physically nontrivial exactly because of the substrate
   asymmetry — this is a testable, falsifiable statement, recorded here
   BEFORE discovery.
3. Independent of interference, EQ content is intrinsically subradiant at
   0th order (its coupling carries the extra k_b·(z-extent)/6 factor
   ~ k_b h/6 « 1), so ED–EQ hybrid states may show narrow linewidths from
   weak EQ leakage alone. Distinguishing "narrow because EQ is dark" from
   "narrow because ED and EQ destructively interfere" is precisely the
   §18-19 reconstruction + detuning program.

## 3. Layered-background reconstruction model (for §18)

In the real substrate geometry the scattered upward amplitude in air is
approximated by the two-path model

    E_x^{up,rec} = E_hom^{+}(air) + r_ba · e^{2 i k_z,b d} · E_hom^{−}(sub-side)

(direct up-radiation plus down-radiation reflected once from the bare
silica-air stack), with the exact TORCWA scattered amplitudes
(t_full − t_bare, r_full − r_bare) as the authority. The reconstruction
residual |full − multipolar| quantifies everything the p_x/m_y/Q_xz (+
higher moments) model misses. TORCWA channel amplitudes — never MENP
free-space cross sections — are the authority for periodic radiated power
P_rad (campaign contract §16).

## 4. Numerical cross-check (validation gate)

`method_validation.py` verifies, at the new material model:
(i) **exact-channel identity** — for a FREESTANDING patterned layer
(air background, where the homogeneous periodic-sheet formula is exact),
the scattered 0th-order amplitudes computed as
−(Z₀/2A)·∫J_x e^{∓ik z}dV from the dense induced current reproduce the
TORCWA scattered amplitudes (t_full − 1, r_full) with small quantified
residual; (ii) the multipole truncation p_x ∓ ik[−(iω/6)Q_xz + m_y]
approximates that exact channel integral, quantifying the truncation
error; (iii) the up/down combinations isolate even (p_x-like) and odd
(m_y/Q_xz-like) parts exactly as derived; and (iv) the torch moments
match the validated corrected MENP port at machine precision on identical
fields. Note: directional illumination breaks z-parity even for
z-uniform patterns, so Q_xz ≠ 0 generically — no structural symmetry is
assumed to null it. Results are recorded in METHOD_VALIDATION.md before
the pilot launches.
