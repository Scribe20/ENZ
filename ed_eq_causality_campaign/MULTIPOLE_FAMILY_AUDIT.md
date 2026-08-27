# MULTIPOLE_FAMILY_AUDIT — complete 4-family partition of the Stage-A ensemble

Forensic audit of the Stage-A multipole-family bookkeeping. No new
optimization was run; every number below is an exact re-decomposition of
the frozen Stage-A geometries (binary densities, canonical origin,
order [9,9], validated 48x48x7 moment grid) with the magnetic quadrupole
restored to the partition. Machine-readable: results/candidate_ledger_v2.csv
(lam0 rows) and results/families_at_wavelengths.csv (lam0 / px-peak /
Qxz-peak / fitted-pole rows).

## 1. OLD definition vs NEW definition — side by side

| | Stage-A (`family_weights`, ledger v1) | Audit (`family_weights4`, ledger v2) |
|---|---|---|
| families in denominator | ED + MD + EQ (**MQ absent — never computed**) | ED + MD + EQ + MQ (complete through quadrupole order) |
| reported fractions | `EDEQ_frac = (Cp+CQe)/(Cp+Cm+CQe)`, `MD_frac = Cm/(Cp+Cm+CQe)` | `f_ED, f_MD, f_EQ, f_MQ` with `f_ED+f_MD+f_EQ+f_MQ = 1` exactly |
| weights | Alaee/MENP radiation constants: `Cp = k^4/(6 pi eps0^2) |p|^2`, `Cm = Cp-const/c^2 |m|^2`, `CQe = k^6/(720 pi eps0^2) sum w|Qe|^2` (off-diag w=2) | same constants, plus `CQm = k^6/(720 pi eps0^2 c^2) sum w|Qm|^2` (off-diag w=2) |
| Qm moments | not present in `torch_moments` at all | exact-kernel Qm added to `torch_moments` (j2-kernel symmetrized form), validated to 2.2e-7 relative vs the corrected MENP port |
| toroidal | diagnostic only | unchanged: diagnostic only (`CT_diag_over_CED = |k T_x|^2-weighted`), NEVER a 5th family — the toroidal contribution is part of the exact-kernel p (Alaee exact form), so adding it would double count |
| sum rule | summed to 1 **by construction over 3 families** — hid the MQ omission | `sum_check = 1.000000` exactly for all 64 audited rows (max deviation 0.0) |

Why the old numbers were wrong-but-mostly-close: for 15/18 candidates at
lam0 the MQ family is genuinely small (f_MQ <= 0.03), so dropping it
shifted ED+EQ by <= 0.026 on average. The failure mode it hid is real,
though: **P0650_H0350_seed029 carries f_MQ = 0.298 at lam0 (0.492 at its
fitted pole)** — under the old bookkeeping its `EDEQ_frac = 0.806`
overstated the audited f_ED+f_EQ = 0.564 by 0.241. The old fractions were
upper bounds, not partitions.

## 2. Complete family fractions at lam0 = 1332.5 nm (all 18 pilot candidates)

old = Stage-A 3-family fractions (v1 ledger definition, recomputed on
identical moments); new = complete 4-family partition. `f_ED+f_EQ` is the
audited quantity comparable to old `EDEQ_frac`.

| run_id | old EDEQ_frac | old MD_frac | old f_MQ | new f_ED | new f_MD | new f_EQ | new f_MQ | new f_ED+f_EQ | class (new) |
|---|---|---|---|---|---|---|---|---|---|
| P0550_H0150_seed011 | 0.743 | 0.257 | — | 0.697 | 0.250 | 0.024 | 0.028 | 0.721 | ED_dominated |
| P0550_H0150_seed029 | 0.959 | 0.041 | — | 0.918 | 0.041 | 0.035 | 0.005 | 0.953 | ED_dominated |
| P0550_H0250_seed011 | 0.948 | 0.052 | — | 0.491 | 0.052 | 0.457 | 0.000 | 0.948 | clean_balanced_ED_EQ |
| P0550_H0250_seed029 | 0.376 | 0.624 | — | 0.047 | 0.623 | 0.329 | 0.001 | 0.376 | MD_contaminated |
| P0550_H0350_seed011 | 0.364 | 0.636 | — | 0.034 | 0.634 | 0.328 | 0.004 | 0.363 | MD_contaminated |
| P0550_H0350_seed029 | 0.409 | 0.591 | — | 0.066 | 0.586 | 0.340 | 0.008 | 0.406 | MD_contaminated |
| P0650_H0150_seed011 | 1.000 | 0.000 | — | 0.973 | 0.000 | 0.022 | 0.004 | 0.996 | ED_dominated |
| P0650_H0150_seed029 | 0.980 | 0.020 | — | 0.959 | 0.020 | 0.019 | 0.001 | 0.979 | ED_dominated |
| P0650_H0250_seed011 | 0.570 | 0.430 | — | 0.312 | 0.427 | 0.254 | 0.007 | 0.566 | MD_contaminated |
| P0650_H0250_seed029 | 0.580 | 0.420 | — | 0.269 | 0.416 | 0.305 | 0.010 | 0.574 | MD_contaminated |
| P0650_H0350_seed011 | 0.322 | 0.678 | — | 0.007 | 0.676 | 0.314 | 0.004 | 0.321 | MD_contaminated |
| P0650_H0350_seed029 | 0.806 | 0.194 | — | 0.531 | 0.137 | 0.033 | 0.298 | 0.564 | mixed_higher_order |
| P0750_H0150_seed011 | 0.990 | 0.010 | — | 0.875 | 0.010 | 0.106 | 0.009 | 0.981 | ED_dominated |
| P0750_H0150_seed029 | 0.994 | 0.006 | — | 0.886 | 0.006 | 0.102 | 0.006 | 0.988 | ED_dominated |
| P0750_H0250_seed011 | 0.386 | 0.614 | — | 0.063 | 0.598 | 0.312 | 0.027 | 0.375 | MD_contaminated |
| P0750_H0250_seed029 | 0.567 | 0.433 | — | 0.333 | 0.430 | 0.230 | 0.007 | 0.563 | MD_contaminated |
| P0750_H0350_seed011 | 0.958 | 0.042 | — | 0.609 | 0.038 | 0.263 | 0.091 | 0.871 | clean_balanced_ED_EQ |
| P0750_H0350_seed029 | 0.845 | 0.155 | — | 0.408 | 0.144 | 0.376 | 0.071 | 0.785 | mixed_higher_order |

All rows: `sum_check = 1` exactly; `C_total_exact` spans 4.7e-13 to
8.4e-11 (SI radiation-weight units, per-cell).

## 3. Component purities WITHIN each family at lam0 (special attention: m_y)

`px|ED = C_px/C_ED` etc.; the last three columns are single-component
fractions of the TOTAL 4-family sum.

| run_id | px given ED | my given MD | Qxz given EQ | C_px/total | C_my/total | C_Qxz/total |
|---|---|---|---|---|---|---|
| P0550_H0150_seed011 | 1.000 | 0.004 | 0.180 | 0.697 | 0.001 | 0.004 |
| P0550_H0150_seed029 | 1.000 | 1.000 | 1.000 | 0.918 | 0.041 | 0.035 |
| P0550_H0250_seed011 | 1.000 | 0.784 | 0.847 | 0.491 | 0.040 | 0.387 |
| P0550_H0250_seed029 | 0.988 | 0.993 | 0.992 | 0.046 | 0.619 | 0.327 |
| P0550_H0350_seed011 | 0.262 | 0.769 | 0.753 | 0.009 | 0.487 | 0.247 |
| P0550_H0350_seed029 | 0.821 | 0.984 | 0.994 | 0.054 | 0.577 | 0.338 |
| P0650_H0150_seed011 | 1.000 | 0.835 | 0.886 | 0.973 | 0.000 | 0.020 |
| P0650_H0150_seed029 | 0.997 | 0.088 | 0.572 | 0.956 | 0.002 | 0.011 |
| P0650_H0250_seed011 | 0.971 | 0.991 | 0.990 | 0.303 | 0.423 | 0.252 |
| P0650_H0250_seed029 | 0.940 | 0.994 | 0.999 | 0.253 | 0.413 | 0.305 |
| P0650_H0350_seed011 | 0.749 | 0.985 | 0.991 | 0.005 | 0.666 | 0.311 |
| P0650_H0350_seed029 | 0.975 | 0.744 | 0.462 | 0.518 | 0.102 | 0.015 |
| P0750_H0150_seed011 | 1.000 | 0.278 | 0.982 | 0.875 | 0.003 | 0.104 |
| P0750_H0150_seed029 | 1.000 | 0.513 | 0.984 | 0.886 | 0.003 | 0.100 |
| P0750_H0250_seed011 | 0.529 | 0.964 | 0.974 | 0.034 | 0.576 | 0.304 |
| P0750_H0250_seed029 | 0.978 | 0.917 | 0.924 | 0.326 | 0.394 | 0.212 |
| P0750_H0350_seed011 | 0.871 | 0.819 | 0.852 | 0.530 | 0.031 | 0.224 |
| P0750_H0350_seed029 | 0.938 | 0.033 | 0.696 | 0.383 | 0.005 | 0.262 |

**m_y finding.** m_y is channel-degenerate with Q_xz in the x-polarized
specular channel (ED_EQ_CHANNEL_DERIVATION.md: at 0th order the odd
integral ∫z J_x contributes -(i omega/6) Qe_xz + m_y). The audit confirms
this degeneracy is not hypothetical: in every MD_contaminated candidate
the MD family is 92-99% pure m_y — i.e. the *specific* magnetic component
that the objective's Q_xz channel cannot be distinguished from at 0th
order is the one that grew. The objective maximized |Q_xz|^2 exactly (the
moment, not the channel), so m_y was not rewarded directly; it rides in
because the same current pattern (odd-in-z J_x) feeds both moments.
Consequence for classification: f_EQ alone is NOT sufficient evidence of
an EQ state — the audited tables always carry f_MD and my|MD next to it.

## 4. Families at the four key wavelengths (5 finalists / special cases)

`which`: lam0 = 1332.5 nm; px_peak / Qxz_peak = maxima of C_px, C_Qxz on
the 81-point qualification scan; pole = fitted pole wavelength (where a
resolved fit exists). **Caveat: 17 of 36 peak rows sit at 1292.5 nm, the
short-wavelength EDGE of the qualification window — those are
edge-of-window maxima (the true peak lies outside the scanned range), not
resonant peaks; they are retained for completeness and marked by
lam = 1292.5.**

| run_id | which | lam (nm) | f_ED | f_MD | f_EQ | f_MQ | px given ED | Qxz given EQ | balance | class |
|---|---|---|---|---|---|---|---|---|---|---|
| P0550_H0250_seed011 | lam0 | 1332.5 | 0.491 | 0.052 | 0.457 | 0.000 | 1.000 | 0.847 | 0.931 | clean_balanced_ED_EQ |
| P0550_H0250_seed011 | px_peak | 1292.5 | 0.372 | 0.281 | 0.346 | 0.001 | 0.999 | 0.680 | 0.931 | mixed_higher_order |
| P0550_H0250_seed011 | Qxz_peak | 1302.5 | 0.437 | 0.162 | 0.400 | 0.001 | 0.999 | 0.751 | 0.917 | mixed_higher_order |
| P0750_H0250_seed011 | lam0 | 1332.5 | 0.063 | 0.598 | 0.312 | 0.027 | 0.529 | 0.974 | 0.203 | MD_contaminated |
| P0750_H0250_seed011 | px_peak | 1331.5 | 0.064 | 0.618 | 0.281 | 0.037 | 0.375 | 0.974 | 0.228 | MD_contaminated |
| P0750_H0250_seed011 | Qxz_peak | 1330.5 | 0.069 | 0.635 | 0.247 | 0.049 | 0.232 | 0.973 | 0.282 | MD_contaminated |
| P0750_H0250_seed011 | pole | 1330.3 | 0.072 | 0.638 | 0.237 | 0.053 | 0.201 | 0.972 | 0.302 | MD_contaminated |
| P0750_H0350_seed011 | lam0 | 1332.5 | 0.609 | 0.038 | 0.263 | 0.091 | 0.871 | 0.852 | 0.432 | clean_balanced_ED_EQ |
| P0750_H0350_seed011 | px_peak | 1319.5 | 0.554 | 0.044 | 0.278 | 0.124 | 0.831 | 0.867 | 0.502 | clean_balanced_ED_EQ |
| P0750_H0350_seed011 | Qxz_peak | 1292.5 | 0.377 | 0.077 | 0.268 | 0.278 | 0.634 | 0.907 | 0.711 | mixed_higher_order |
| P0750_H0350_seed029 | lam0 | 1332.5 | 0.408 | 0.144 | 0.376 | 0.071 | 0.938 | 0.696 | 0.922 | mixed_higher_order |
| P0750_H0350_seed029 | px_peak | 1292.5 | 0.246 | 0.166 | 0.488 | 0.100 | 0.960 | 0.723 | 0.503 | mixed_higher_order |
| P0750_H0350_seed029 | Qxz_peak | 1292.5 | 0.246 | 0.166 | 0.488 | 0.100 | 0.960 | 0.723 | 0.503 | mixed_higher_order |
| P0750_H0350_seed029 | pole | 1327.6 | 0.394 | 0.142 | 0.374 | 0.090 | 0.911 | 0.701 | 0.949 | mixed_higher_order |
| P0650_H0350_seed029 | lam0 | 1332.5 | 0.531 | 0.137 | 0.033 | 0.298 | 0.975 | 0.462 | 0.062 | mixed_higher_order |
| P0650_H0350_seed029 | px_peak | 1327.5 | 0.596 | 0.144 | 0.033 | 0.227 | 0.981 | 0.538 | 0.056 | mixed_higher_order |
| P0650_H0350_seed029 | Qxz_peak | 1292.5 | 0.712 | 0.204 | 0.049 | 0.034 | 0.992 | 0.796 | 0.069 | ED_dominated |
| P0650_H0350_seed029 | pole | 1343.3 | 0.296 | 0.177 | 0.035 | 0.492 | 0.938 | 0.356 | 0.118 | mixed_higher_order |

## 5. Clean/balanced ED–EQ criterion and re-assessment

Criterion (fixed before recomputation):
`clean_balanced_ED_EQ` iff f_ED + f_EQ >= 0.80 AND f_ED >= 0.20 AND
f_EQ >= 0.20 AND px|ED >= 0.80 AND Qxz|EQ >= 0.80. Balance metric
B_ED_EQ = min(f_ED,f_EQ)/max(f_ED,f_EQ) reported alongside.

* **P0550_H0250_seed011 — the Stage-A "cleanest co-excitation" claim
  SURVIVES the complete partition at lam0**: f_ED = 0.491, f_EQ = 0.457
  (f_ED+f_EQ = 0.948), f_MD = 0.052, f_MQ = 0.0004, B = 0.931,
  px|ED = 1.000, Qxz|EQ = 0.847. This is the best balanced ED–EQ state in
  the ensemble and its my|MD = 0.78 of a 5% family is immaterial. Its
  class holds at lam0 but degrades off-resonance (mixed at the scan-edge
  "peaks" — see caveat above).
* **P0750_H0350_seed011** also qualifies at lam0 (f_ED = 0.609,
  f_EQ = 0.263, B = 0.432, purities 0.82/0.85) — ED-leaning but clean.
* **P0650_H0350_seed011** qualifies ONLY at its px_peak (1292.5 nm, edge
  row): f_ED = 0.489, f_EQ = 0.484, B = 0.990 — remarkable balance, but at
  lam0 it is MD_contaminated (f_MD = 0.676); not a lam0 result.
* Nothing else qualifies anywhere. Final count: **2 candidates clean and
  balanced at lam0** (P0550_H0250_seed011, P0750_H0350_seed011).

## 6. Champion re-classification (P0750_H0250_seed011)

At every audited wavelength (lam0, px_peak, Qxz_peak, fitted pole
1330.25 nm) the Stage-A champion is `MD_contaminated`: at the pole
f_MD = 0.638 (my|MD = 0.971), f_EQ = 0.237 (Qxz|EQ = 0.972), f_ED = 0.072,
f_MQ = 0.053. The Stage-A description of its sharp resonance as an "EQ
dark mode on an ED background" is **corrected** to: a dark mode of MIXED
m_y/Q_xz odd-channel character (MD-majority by radiation weight) over a
predominantly non-px background at the pole. The Stage-A causal
conclusions (Q tracks darkness, not alignment; no destructive-phase
regime) are unaffected — they concern the odd-channel resonance as a
whole — but every "EQ mode" label for the champion is superseded by
"m_y/Q_xz hybrid odd mode". The champion's px|ED collapses to 0.20 at the
pole, so it is NOT the vehicle for the ED–EQ hypothesis; the balanced
candidates in §5 are.

## 7. MQ and toroidal exceptions

* **P0650_H0350_seed029** is a genuine higher-order object: f_MQ = 0.298
  at lam0 rising to 0.492 at its fitted pole (1343.3 nm) — an
  MQ-codominant state the old bookkeeping could not see; additionally its
  toroidal diagnostic CT/C_ED = 1.13 (toroidal comparable to the Cartesian
  p contribution), flagging strong exact-kernel corrections. Any Stage-A
  statement about this candidate based on 3-family fractions is void.
* All other candidates: f_MQ <= 0.09 at lam0; toroidal diagnostic well
  below 1 except P0650_H0350_seed011 (0.35) and P0550_H0150_seed011
  (0.24) — monitored, not partitioned.

## 8. Verdicts

1. Old EDEQ_frac/MD_frac: arithmetically consistent within their 3-family
   universe but **incomplete as a physical partition**; superseded by
   f_ED/f_MD/f_EQ/f_MQ everywhere. v1 ledger columns retained for
   traceability only.
2. Toroidal multipole: correctly kept diagnostic in Stage A and in this
   audit (exact kernels already contain it inside p); never a 5th family.
3. The pilot's headline co-excitation claim survives (2 clean balanced
   lam0 candidates); the champion's family label does not (m_y/Q_xz
   hybrid, MD-majority).
4. One candidate (P0650_H0350_seed029) is disqualified from any
   dipole/EQ narrative (MQ-codominant).
