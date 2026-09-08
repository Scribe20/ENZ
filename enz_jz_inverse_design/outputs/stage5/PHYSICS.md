# Stage 5 - physics certification

lambda_ZE = 1302.28 nm. Spectra 1100-1400 nm (a-Si:H data end at 1400 nm; nothing is extrapolated). Poles: AAA rational fits of r_xx(omega) and t_xx(omega), accepted only when present in both with relative distance < 2%, damped, and with Lorentzian peak fraction >= 3% of the observable. Loss scaling: gamma(s) = gamma_rad + s gamma_nr on the field-overlap-tracked branch, s = scale of Im(eps_ITO).

## Working point and peaks

| design | Fz(lam_ZE) | Fz_max @ lam | FWHM (nm) / Q_eff | A(lam_ZE) | A_max @ lam | no-ITO T_min @ lam | bare-film A(lam_ZE) |
|---|---|---|---|---|---|---|---|
| final3_div_F1_P825_h500_pad0.12_s333_h525 | 0.9559 | 0.9563 @ 1300 | 150 / 8.7 | 0.9935 | 0.9958 @ 1298 | 0.001 @ 1314 | 0.0293 |
| scratch_s8080_P825_h600_pad0.12 | 0.9521 | 0.9521 @ 1302 | 150 / 8.7 | 0.9959 | 0.9973 @ 1298 | 0.002 @ 1316 | 0.0293 |

## Poles (order [7,7])

| design | with ITO (lambda nm, Q) | without ITO | lossless ITO |
|---|---|---|---|
| final3_div_F1_P825_h500_pad0.12_s333_h525 | 1218.2 (Q 13.1); 1261.7 (Q 173.4); 1272.5 (Q 11.3) | 1197.0 (Q 21.8); 1272.8 (Q 181.7); 1337.6 (Q 7.9) | 1183.0 (Q 14.9); 1229.2 (Q 20.7); 1232.0 (Q 26.1); 1261.4 (Q 359.1); 1278.9 (Q 18.0); 1352.5 (Q 1209.6); 1367.8 (Q 479.8); 1384.9 (Q 694.2); 1398.8 (Q 163.6) |
| scratch_s8080_P825_h600_pad0.12 | 1120.6 (Q 37.2); 1195.3 (Q 22.5); 1261.8 (Q 22.7); 1285.7 (Q 9.5) | 1210.6 (Q 19.1); 1267.5 (Q 21.8); 1270.2 (Q 22.7); 1330.6 (Q 8.1) | 1121.6 (Q 41.2); 1162.9 (Q 21.3); 1200.1 (Q 26.4); 1261.4 (Q 28.3); 1278.4 (Q 13.8); 1335.7 (Q 441.3); 1362.9 (Q 375.7); 1370.5 (Q 446.8); 1379.0 (Q 549.4); 1380.9 (Q 85.3) |

## Q / loss-scaling table

| design | tracked pole (nm) | Q_loaded | Q_rad | Q_nr | gamma_rad (rad/fs) | gamma_nr (rad/fs) | gamma_rad/gamma_nr | linearity resid | lossless pole | levels |
|---|---|---|---|---|---|---|---|---|---|---|
| final3_div_F1_P825_h500_pad0.12_s333_h525 | 1272.5 | 11.27 | 18.49 | 32.32 | 0.04003 | 0.02290 | **1.748** | 0.096 | 1278.9 nm, Q 18.0 | 6 |
| final3_div_F1_P825_h500_pad0.12_s333_h525 | 1261.7 | 172.80 | 354.54 | 333.02 | 0.00211 | 0.00224 | **0.939** | 0.010 | 1261.4 nm, Q 357.9 | 6 |
| scratch_s8080_P825_h600_pad0.12 | 1285.7 | 9.46 | 31.07 | 12.95 | 0.02358 | 0.05655 | **0.417** | 0.143 | 1261.4 nm, Q 28.3 | 6 |
| scratch_s8080_P825_h600_pad0.12 | 1261.8 | 22.67 | 28.39 | 115.64 | 0.02629 | 0.00645 | **4.073** | 0.025 | 1261.4 nm, Q 28.3 | 6 |

## Multipole character of the induced a-Si current (Cartesian, isolated-scatterer power weights; diagnostic)

| design | at | ED (p+ikT) | p only | toroidal | MD | EQ | MQ | dominant |
|---|---|---|---|---|---|---|---|---|
| final3_div_F1_P825_h500_pad0.12_s333_h525 | lambda_ZE | 0.560 | 0.178 | 0.113 | 0.156 | 0.280 | 0.004 | ED |
| final3_div_F1_P825_h500_pad0.12_s333_h525 | Fz peak | 0.557 | 0.169 | 0.119 | 0.151 | 0.288 | 0.004 | ED |
| scratch_s8080_P825_h600_pad0.12 | lambda_ZE | 0.349 | 0.066 | 0.119 | 0.191 | 0.455 | 0.006 | EQ |
| scratch_s8080_P825_h600_pad0.12 | Fz peak | 0.349 | 0.065 | 0.119 | 0.188 | 0.458 | 0.006 | EQ |

## Reference designs under the new materials

| design | P | h | Fz(lam_ZE) | Fz_max @ lam | A_max @ lam | poles (nm, Q) |
|---|---|---|---|---|---|---|
