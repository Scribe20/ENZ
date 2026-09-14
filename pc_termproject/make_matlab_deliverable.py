"""Emit the submission script Nano_opt_code.m (and the .mat) for a given binary design.
The script (i) rebuilds the parametric seed, (ii) decodes the optimised design embedded as hex rows,
(iii) verifies both with an embedded independent PWEM (same formulation as the grader), (iv) saves ONLY epsr."""
import numpy as np, sys, os, json
from pwem import *

def mask_to_hex_rows(mask):
    rows = []
    for iy in range(mask.shape[0]):
        bits = ''.join('1' if b else '0' for b in mask[iy])
        rows.append(''.join(f"{int(bits[i:i+4], 2):X}" for i in range(0, len(bits), 4)))
    return rows

def emit(mask, seed_desc, out_dir, initial='INITIAL', student='STUDENTID', score_info=None, method_notes='', octave=False):
    os.makedirs(out_dir, exist_ok=True)
    rows = mask_to_hex_rows(mask)
    mat_name = f"Nano_P_{initial}_{student}.mat"
    pw = open(os.path.join(os.path.dirname(__file__), 'matlab', 'pc_pwem_official.m')).read()
    # turn the function file into local functions (strip nothing; MATLAB scripts may define local functions at the end)
    hexblock = "\n".join(f"    '{r}'" for r in rows)
    script = f"""%% Nano_opt_code.m  --  Nanophotonics term project: 2D square-lattice Si/air photonic crystal with a complete
%% (TE + TM) photonic band gap centred on lambda = 1550 nm (a/lambda = 0.40903).
%%
%% Running this script derives the 96 x 96 binary permittivity map `epsr` and saves it (and nothing else) to
%%     {mat_name}
%% Runtime: well under one minute (design reconstruction is deterministic; the verification is an independent PWEM).
%%
%% HOW THE DESIGN WAS DERIVED (summary; full code and logs are in the project repository):
%%   1. Physics-guided seed: {seed_desc}
%%      Large dielectric islands open TM gaps (Ez concentrated in the islands, Dz discontinuity), thin connected veins
%%      open TE gaps (in-plane E of the low band concentrated in the veins); the hybrid opens both around a/lambda = 0.409.
%%   2. Gradient (Hellmann-Feynman) max-min optimisation of the official score over a C4v-symmetric density,
%%      followed by an exact discrete refinement (pixel-orbit flips accepted only if the exactly re-evaluated score improves).
%%      The band-edge sensitivities are  d(omega)/d(eps_p) ~ -|Ez(r_p)|^2  (TM)  and  d(omega)/d(1/eps_p) ~ +|D_t(r_p)|^2  (TE).
%%   3. The result is embedded below as a hex-encoded bit map (row = y, column = x, 1 = silicon).
%% {method_notes}
%%
%% AI-use disclosure: the solver, optimiser, this script and the analysis were produced with an AI assistant
%% (Claude); see the presentation for the full disclosure.

clear; clc;
EPS_SI = 12.1104; EPS_AIR = 1.0; N = 96; a_nm = 634.0; target = a_nm / 1550.0;

%% ---- (1) parametric seed (for reference / comparison; not saved) ----------------------------------------------
x = ((0:N-1) + 0.5) / N - 0.5;            % pixel centres in units of a
[X, Y] = meshgrid(x, x);                   % X(iy,ix), Y(iy,ix)
r_seed = 0.32; w_seed = 0.08;
seed_mask = (X.^2 + Y.^2 <= r_seed^2) | (abs(X) <= w_seed/2) | (abs(Y) <= w_seed/2);
epsr_seed = EPS_AIR + (EPS_SI - EPS_AIR) * double(seed_mask);

%% ---- (2) optimised design (embedded) --------------------------------------------------------------------------
hex_rows = {{
{hexblock}
}};
mask = false(N, N);
for iy = 1:N
    v = hex2dec(num2cell(hex_rows{{iy}}));            % 24 values 0..15
    bits = dec2bin(v, 4);                               % 24 x 4 char
    mask(iy, :) = reshape(bits.', 1, N) == '1';
end
epsr = EPS_AIR + (EPS_SI - EPS_AIR) * double(mask);     % exactly 1.0 or 12.1104

%% ---- (3) independent verification (same PWEM formulation as the grader; not pc_evaluate.p) --------------------
fprintf('Seed design  : '); Rs = pc_pwem_official(epsr_seed, 9, target, 10);
fprintf('Final design : '); Rf = pc_pwem_official(epsr, 9, target, 10);
fprintf('fill fraction of Si: seed %.4f, final %.4f\\n', mean(seed_mask(:)), mean(mask(:)));
assert(isequal(size(epsr), [96 96])); assert(all(epsr(:) == EPS_AIR | epsr(:) == EPS_SI));

%% ---- (4) save ONLY epsr ------------------------------------------------------------------------------------------
save('{mat_name}', 'epsr', '-v7');     % MATLAB v7 MAT-file containing only epsr
fprintf('saved %s (variable epsr, %d x %d)\\n', '{mat_name}', size(epsr,1), size(epsr,2));

%% ================================================================================================================
%% local functions: independent plane-wave expansion evaluator (TM: eps-matrix, TE: 1/eps coefficients)
{pw if not octave else ''}
"""
    if octave:   # Octave needs the functions defined before they are called: put them right after the first statement
        script = script.replace("clear; clc;\n", "1;\n" + pw + "\n", 1)
    with open(os.path.join(out_dir, 'Nano_opt_code.m'), 'w') as f:
        f.write(script)
    save_mat(os.path.join(out_dir, mat_name), mask_to_eps(mask))
    return os.path.join(out_dir, 'Nano_opt_code.m'), os.path.join(out_dir, mat_name)

if __name__ == '__main__':
    mask = np.load(sys.argv[1]).astype(bool)
    out = sys.argv[2]
    octave = len(sys.argv) > 3 and sys.argv[3] == 'octave'
    m, mat = emit(mask, "circular Si rods r = 0.32a joined by veins w = 0.08a (square lattice, a = 634 nm)", out, octave=octave)
    print("wrote", m, mat)
