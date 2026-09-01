# R_TYPE_EDMD_METHOD_REPORT — Method A (mode-constrained ED/MD)

Objective: circular-basis |r_cross|^2 maximization + T/co-pol/absorption
penalties + soft mode gates (f_ED^x >= 0.55, px|ED >= 0.8, f_MD^y >=
0.55, m-transverse|MD >= 0.8; verified against solver convention: y-pol
couples to m_x). Exact differentiable current multipoles per iteration.

Outcome: the constraint is achievable and productive. Champion
A_P271_H200 (exact 4-family at [9,9], 48x48x7):
x-pol f_ED=0.45, f_MD=0.24, f_EQ=0.28, f_MQ=0.02 (px|ED=1.00);
y-pol f_ED=0.29, f_MD=0.71, f_EQ=0.00, f_MQ=0.01 (mx|MD=1.00).
The x-pol ED gate sat slightly below target (0.45-0.58 across finalists)
because pushing f_ED harder cost R_cross - recorded per spec: a mildly
mixed ED state gives better function. Mechanism note: despite the
ED/MD radiated-power dominance, the FORWARD-CANCELLATION is three-way
(|t_ED|=0.67, |t_EQ|=0.69, |t_MD|=0.49 at x-pol) - dominance and
cancellation are distinct facts (see R_TYPE_MULTIPOLE_COMPARISON.md).
Best-basin trajectory: P226->P244->P262->P271 (optimum at the largest
diffraction-safe period), H 185-200. Seeds: attractor reproduced by 1
of 2 extra seeds (0.380); one seed fell to a poor local state.
