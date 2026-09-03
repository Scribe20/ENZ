# Perfect-R workspace mining

184 historical geometries re-evaluated at theta=0, order [9,9], corrected conventions (exact Jones, F_ideal).
9 recovered from checkpoint-only (incomplete) run directories.

## Partial-solution leaderboards (top 3 each; "useful" boards require R_cross >= 0.30)

### highest R_cross
- A_P271_H185_s11_g96_o9 (rtfree/refinement, final): Rcross=0.527 | F=0.527 Rc=0.527 T=0.081 co=0.081 A=0.312 |rx|,|ry|=0.80,0.76 err=43 isl=1
- A_P271_H200_s11_g96_o9 (rtfree/refinement, final): Rcross=0.526 | F=0.526 Rc=0.526 T=0.054 co=0.065 A=0.355 |rx|,|ry|=0.75,0.78 err=39 isl=1
- A_P271_H200_wideangle (rtfree/finalists, final): Rcross=0.518 | F=0.518 Rc=0.518 T=0.016 co=0.071 A=0.395 |rx|,|ry|=0.71,0.82 err=40 isl=1

### highest total reflection
- B_P213_H140_s11_wf (widefov/coarse, final): Rtot=0.843 | F=0.000 Rc=0.000 T=0.002 co=0.843 A=0.155 |rx|,|ry|=0.92,0.92 err=179 isl=1
- B_P226_H140_s11_wf (widefov/coarse, final): Rtot=0.826 | F=0.000 Rc=0.000 T=0.000 co=0.826 A=0.173 |rx|,|ry|=0.91,0.91 err=179 isl=1
- B_P239_H140_s11_wf (widefov/coarse, final): Rtot=0.820 | F=0.060 Rc=0.060 T=0.007 co=0.761 A=0.173 |rx|,|ry|=0.92,0.89 err=149 isl=1

### largest min(|r_x|,|r_y|)
- B_P213_H140_s11_wf (widefov/coarse, final): min_r=0.918 | F=0.000 Rc=0.000 T=0.002 co=0.843 A=0.155 |rx|,|ry|=0.92,0.92 err=179 isl=1
- B_P226_H140_s11_wf (widefov/coarse, final): min_r=0.908 | F=0.000 Rc=0.000 T=0.000 co=0.826 A=0.173 |rx|,|ry|=0.91,0.91 err=179 isl=1
- B_P239_H140_s11_wf (widefov/coarse, final): min_r=0.894 | F=0.060 Rc=0.060 T=0.007 co=0.761 A=0.173 |rx|,|ry|=0.92,0.89 err=149 isl=1

### smallest transmission
- A_P271_H200_wideangle (rtfree/finalists, final): T=0.016 | F=0.518 Rc=0.518 T=0.016 co=0.071 A=0.395 |rx|,|ry|=0.71,0.82 err=40 isl=1
- A_P252_H185_s11_wf (widefov/coarse, final): T=0.029 | F=0.342 Rc=0.342 T=0.029 co=0.313 A=0.316 |rx|,|ry|=0.80,0.82 err=87 isl=1
- A_P252_H170_s11_wf (widefov/coarse, final): T=0.036 | F=0.304 Rc=0.304 T=0.036 co=0.409 A=0.251 |rx|,|ry|=0.88,0.81 err=99 isl=1

### smallest R_co
- B_P271_H215_s11_g96_o9 (rtfree/refinement, final): co=0.005 | F=0.505 Rc=0.505 T=0.057 co=0.005 A=0.434 |rx|,|ry|=0.69,0.74 err=10 isl=1
- B_P271_H230_s11_g96_o9 (rtfree/refinement, final): co=0.005 | F=0.425 Rc=0.425 T=0.125 co=0.005 A=0.445 |rx|,|ry|=0.68,0.63 err=12 isl=1
- B_P262_H230_s11_g96_o9 (rtfree/refinement, final): co=0.012 | F=0.406 Rc=0.406 T=0.170 co=0.012 A=0.412 |rx|,|ry|=0.70,0.59 err=17 isl=1

### smallest absorption
- B_P239_H170_s11_wf (widefov/coarse, final): A=0.151 | F=0.308 Rc=0.308 T=0.376 co=0.165 A=0.151 |rx|,|ry|=0.38,0.90 err=65 isl=1
- B_P252_H155_s11_wf (widefov/coarse, final): A=0.159 | F=0.330 Rc=0.330 T=0.345 co=0.165 A=0.159 |rx|,|ry|=0.44,0.89 err=65 isl=1
- B_P239_H170_s11_wf (widefov/refinement, final): A=0.163 | F=0.326 Rc=0.326 T=0.340 co=0.171 A=0.163 |rx|,|ry|=0.43,0.90 err=66 isl=1

### smallest phase error from pi
- B_P271_H215_s11_g96_o9 (rtfree/refinement, final): phase_err_deg=10.182 | F=0.505 Rc=0.505 T=0.057 co=0.005 A=0.434 |rx|,|ry|=0.69,0.74 err=10 isl=1
- B_P271_H230_s11_g96_o9 (rtfree/refinement, final): phase_err_deg=11.910 | F=0.425 Rc=0.425 T=0.125 co=0.005 A=0.445 |rx|,|ry|=0.68,0.63 err=12 isl=1
- B_P262_H230_s11_g96_o9 (rtfree/refinement, final): phase_err_deg=16.801 | F=0.406 Rc=0.406 T=0.170 co=0.012 A=0.412 |rx|,|ry|=0.70,0.59 err=17 isl=1

### highest F_ideal
- A_P271_H185_s11_g96_o9 (rtfree/refinement, final): F=0.527 | F=0.527 Rc=0.527 T=0.081 co=0.081 A=0.312 |rx|,|ry|=0.80,0.76 err=43 isl=1
- A_P271_H200_s11_g96_o9 (rtfree/refinement, final): F=0.526 | F=0.526 Rc=0.526 T=0.054 co=0.065 A=0.355 |rx|,|ry|=0.75,0.78 err=39 isl=1
- A_P271_H200_wideangle (rtfree/finalists, final): F=0.518 | F=0.518 Rc=0.518 T=0.016 co=0.071 A=0.395 |rx|,|ry|=0.71,0.82 err=40 isl=1

## Checkpoint-only states (never recorded as finals)

- A_P271_H215_s11_g96_o9 [checkpoint@50]: F=0.498 T=0.092 co=0.017 A=0.393
- A_P262_H200_s11_g96_o9 [checkpoint@50]: F=0.489 T=0.100 co=0.069 A=0.343
- A_P262_H215_s11_g96_o9 [checkpoint@50]: F=0.461 T=0.121 co=0.042 A=0.375
- A_P253_H185_s11_g96_o9 [checkpoint@50]: F=0.449 T=0.114 co=0.148 A=0.288
- A_P253_H215_s11_g96_o9 [checkpoint@25]: F=0.411 T=0.173 co=0.060 A=0.356

## Corrected small angular set (theta 0/20/40/50 x phi 0/45/90, exact p/s basis)

| tag | F(0) | min F | mean F | max T | max co |
|---|---|---|---|---|---|
| A_P252_H170_s11_wf | 0.304 | 0.091 | 0.241 | 0.340 | 0.472 |
| A_P252_H185_s11_wf | 0.342 | 0.070 | 0.287 | 0.358 | 0.417 |
| A_P271_H185_s11_g96_o9 | 0.526 | 0.006 | 0.268 | 0.501 | 0.215 |
| A_P271_H200_s11_g96_o9 | 0.526 | 0.062 | 0.256 | 0.365 | 0.266 |
| A_P271_H200_wideangle | 0.518 | 0.020 | 0.253 | 0.507 | 0.273 |
| A_P271_H215_s11_g96_o9 | 0.498 | 0.077 | 0.238 | 0.316 | 0.270 |
| B_P239_H170_s11_wf | 0.308 | 0.084 | 0.289 | 0.710 | 0.165 |
| B_P252_H155_s11_wf | 0.330 | 0.061 | 0.293 | 0.710 | 0.165 |
| B_P271_H215_s11_g96_o9 | 0.505 | 0.045 | 0.251 | 0.387 | 0.296 |
| B_P271_H215_s23_g96_o9 | 0.495 | 0.045 | 0.244 | 0.295 | 0.266 |