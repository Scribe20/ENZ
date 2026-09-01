# MULTIPOLAR_PORT_STATE_MAP — existing-data search and predictors (Phases G, M-P)

Data: existing_candidate_tmin_ranking.csv,
same_composition_opposite_function_pairs.csv, composition_port_map.csv,
port_state_summary.json; Figures ps_fig6, ps_fig7.

## G/H. Low-T census of all existing qualified data
Sources: 18 Stage-A qualify spectra (1-nm, 1292.5-1372.5), P0550
thickness family, P0750 master+fine scans. Old ED/MD-campaign
candidates EXCLUDED (clamped material model - not comparable; recorded).
Classification: NEAR_TRANSMISSION_ZERO (T <= 0.01): P0550_H0350_s29
(0.0034 @1343.5, R 0.9965, A 2e-4), P0550_H0250_s29 (0.0067 @1340.5) -
both 1-nm-sampled, MEDIUM confidence pending sub-nm scans;
P0650_H0350_s29 / P0650_H0250_s29 (0.0069 @1372.5) - EDGE minima,
flagged. VERY_STRONG (T <= 0.03): P0750 (0.0108, certified at 0.2 nm),
P0650_H0250_s11. All are my+Qxz (or Qxz+px) odd-channel resonant types.
The clean-ED-EQ P0550 family never drops below T = 0.365.

## M/N. Same composition, opposite function
D_comp = euclidean distance in (f_ED, f_MD, f_EQ, f_MQ). Within the
P0550 family (identity interval h = 215-262.5): R spans 0.020-0.283 at
D_comp <= 0.10 from the reference. Best pairs
(same_composition_opposite_function_pairs.csv): h = 235 vs 260 -
D_comp = 0.069, dR = dT = 0.219; multiple pairs with D_comp < 0.05 and
dR > 0.12. Composition does not uniquely determine functionality -
quantified.

## O/P. Predictor maps
Composition map (fig 7A): at nearly constant (f_ED-f_EQ, f_MD+f_MQ)
coordinates the P0550 family spans T = 0.72-0.98; the P0750 track moves
through the mirror at its own composition corner - compositions
correlate with STATE TYPE (resonant odd-channel states host mirrors)
but not with the port value. Predictor correlations on the h-family:
|wrapped Dphi - 180| vs R: -0.93 (best); f_ED vs R: -0.82 (collinear
passenger of h); B_ED_EQ vs R: +0.17 (no predictive power); scattered-
vs-background phase alone: -0.29 (magnitude matters too). Channel-
normalized complex phase is the stronger predictor; composition alone
is not sufficient.
