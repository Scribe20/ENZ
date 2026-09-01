# R_TYPE_MULTIPOLE_COMPARISON — exact forensics + cancellation

Data: multipole_finalists.csv, t_argand_finalists.csv, figures/targand_*.

1. Exact 4-family fractions (48x48x7, order [9,9], device illumination):
   Method A finalists: x->ED (0.45-0.58, px-pure), y->MD (0.71-0.73,
   mx-pure). Method B finalists: x->EQ (0.48-0.58) except B_P262_H215
   (mixed, ED 0.37); y->MD (0.68-0.71) for all.
2. Forward-cancellation attribution (t-plane ladder, per-row exact
   coupling; 1st-order ladder + explicit truncation residual):
   x-pol: A champion |t_ED|=0.67 |t_EQ|=0.69 |t_MD|=0.49 -> three-way;
          B champion |t_EQ|=0.81 |t_ED|=0.48 |t_MD|=0.41 -> EQ-led.
   y-pol (both): |t_MD|~0.75-0.77 |t_ED|~0.5 |t_EQ|~0.03 -> MD-led,
          exactly the paper's magnetic mechanism.
   Full |t|^2 reaches 0.007 (x) / 0.10 (y); ladder models 0.03-0.09
   (truncation documented - the mechanism reading is family-level).
3. Integrity: at no point is "ED+MD dominance" equated with high R; all
   attributions are complex-amplitude cancellations against t_bg.
