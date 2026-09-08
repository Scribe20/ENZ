# Stage 5 - physics certification

lambda_ZE = 1302.28 nm. Spectra 1100-1400 nm (a-Si:H data end at 1400 nm; nothing is extrapolated). Poles: AAA rational fits of r_xx(omega) and t_xx(omega), accepted only when present in both with relative distance < 2%, damped, and with Lorentzian peak fraction >= 3% of the observable. Loss scaling: gamma(s) = gamma_rad + s gamma_nr on the field-overlap-tracked branch, s = scale of Im(eps_ITO).

## Working point and peaks

| design | Fz(lam_ZE) | Fz_max @ lam | FWHM (nm) / Q_eff | A(lam_ZE) | A_max @ lam | no-ITO T_min @ lam | bare-film A(lam_ZE) |
|---|---|---|---|---|---|---|---|
| final0_F2_P850_h600_pad0.12_s333_P825 | 0.9571 | 0.9571 @ 1302 | 150 / 8.7 | 0.9973 | 0.9985 @ 1298 | 0.000 @ 1314 | 0.0293 |

## Poles (order [7,7])

| design | with ITO (lambda nm, Q) | without ITO | lossless ITO |
|---|---|---|---|
| final0_F2_P850_h600_pad0.12_s333_P825 | 1138.8 (Q 43.0); 1152.6 (Q 15.0); 1238.2 (Q 42.0); 1270.9 (Q 9.1) | 1159.1 (Q 19.7); 1257.7 (Q 31.6); 1264.8 (Q 19.9); 1318.5 (Q 8.4) | 1148.4 (Q 15.0); 1237.0 (Q 56.3); 1266.2 (Q 19.9); 1274.9 (Q 12.3); 1321.4 (Q 243.9); 1358.5 (Q 423.1); 1366.0 (Q 859.6); 1374.6 (Q 1347.3); 1396.2 (Q 249.2) |

## Q / loss-scaling table

| design | tracked pole (nm) | Q_loaded | Q_rad | Q_nr | gamma_rad (rad/fs) | gamma_nr (rad/fs) | gamma_rad/gamma_nr | linearity resid | lossless pole | levels |
|---|---|---|---|---|---|---|---|---|---|---|
| final0_F2_P850_h600_pad0.12_s333_P825 | 1270.9 | 9.05 | 23.45 | 17.97 | 0.03161 | 0.04123 | **0.767** | 0.205 | 1266.2 nm, Q 19.9 | 6 |
| final0_F2_P850_h600_pad0.12_s333_P825 | 1238.2 | 41.97 | 55.77 | 163.35 | 0.01364 | 0.00466 | **2.929** | 0.018 | 1237.0 nm, Q 56.3 | 6 |

## Multipole character of the induced a-Si current (Cartesian, isolated-scatterer power weights; diagnostic)

| design | at | ED (p+ikT) | p only | toroidal | MD | EQ | MQ | dominant |
|---|---|---|---|---|---|---|---|---|
| final0_F2_P850_h600_pad0.12_s333_P825 | lambda_ZE | 0.452 | 0.093 | 0.141 | 0.155 | 0.389 | 0.004 | ED |
| final0_F2_P850_h600_pad0.12_s333_P825 | Fz peak | 0.452 | 0.092 | 0.141 | 0.154 | 0.390 | 0.004 | ED |

## Reference designs under the new materials

| design | P | h | Fz(lam_ZE) | Fz_max @ lam | A_max @ lam | poles (nm, Q) |
|---|---|---|---|---|---|---|
| Karimi EDR cuboid (560x500x140, P850) | 850 | 140 | 0.2619 | 0.3086 @ 1386 | 0.3431 @ 1388 | 1298 (16.5) |
| direct-Ez winner (P850,h140,pad86) | 850 | 140 | 0.2337 | 0.3242 @ 1390 | 0.4682 @ 1302 | 1285 (9.9); 1290 (100.3); 1294 (26.4); 1297 (59.7) |
| robust A finalist0 (P800,h200,pad4%) | 800 | 200 | 0.1770 | 0.3572 @ 1214 | 0.3751 @ 1214 | 1201 (9.3); 1217 (24.4) |
