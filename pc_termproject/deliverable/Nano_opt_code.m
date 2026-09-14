%% Nano_opt_code.m  --  Nanophotonics term project: 2D square-lattice Si/air photonic crystal with a complete
%% (TE + TM) photonic band gap centred on lambda = 1550 nm (a/lambda = 0.40903).
%%
%% Running this script derives the 96 x 96 binary permittivity map `epsr` and saves it (and nothing else) to
%%     Nano_P_INITIAL_STUDENTID.mat
%% Runtime: well under one minute (design reconstruction is deterministic; the verification is an independent PWEM).
%%
%% HOW THE DESIGN WAS DERIVED (summary; full code and logs are in the project repository):
%%   1. Physics-guided seed: circular Si rods r = 0.33a joined by veins w = 0.08a (square lattice, a = 634 nm), refined pixel-by-pixel
%%      Large dielectric islands open TM gaps (Ez concentrated in the islands, Dz discontinuity), thin connected veins
%%      open TE gaps (in-plane E of the low band concentrated in the veins); the hybrid opens both around a/lambda = 0.409.
%%   2. Gradient (Hellmann-Feynman) max-min optimisation of the official score over a C4v-symmetric density,
%%      followed by an exact discrete refinement (pixel-orbit flips accepted only if the exactly re-evaluated score improves).
%%      The band-edge sensitivities are  d(omega)/d(eps_p) ~ -|Ez(r_p)|^2  (TM)  and  d(omega)/d(1/eps_p) ~ +|D_t(r_p)|^2  (TE).
%%   3. The result is embedded below as a hex-encoded bit map (row = y, column = x, 1 = silicon).
%% Optimisation history: rods r=0.32a+veins 0.08a (9.50 %) -> exact discrete refinement from the r=0.33a seed (10.63 %);
%%   gradient/continuous runs and basin hopping converge to the same optimum (10.61-10.63 %); an unconstrained (non-C4v)
%%   refinement gains only 0.14 % (4 pixels) and was not adopted.
%%
%% AI-use disclosure: the solver, optimiser, this script and the analysis were produced with an AI assistant
%% (Claude); see the presentation for the full disclosure.

clear; clc;
EPS_SI = 12.1104; EPS_AIR = 1.0; N = 96; a_nm = 634.0; target = a_nm / 1550.0;

%% ---- (1) parametric seed (for reference / comparison; not saved) ----------------------------------------------
x = ((0:N-1) + 0.5) / N - 0.5;            % pixel centres in units of a
[X, Y] = meshgrid(x, x);                   % X(iy,ix), Y(iy,ix)
r_seed = 0.33; w_seed = 0.08;
seed_mask = (X.^2 + Y.^2 <= r_seed^2) | (abs(X) <= w_seed/2) | (abs(Y) <= w_seed/2);
epsr_seed = EPS_AIR + (EPS_SI - EPS_AIR) * double(seed_mask);

%% ---- (2) optimised design (embedded) --------------------------------------------------------------------------
hex_rows = {
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000001FF80000000000'
    '00000000001FF80000000000'
    '00000000007FFE0000000000'
    '000007BFFFFFFFFFFDE00000'
    '00000BFFFFFFFFFFFFD00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '000007FFFFFFFFFFFFE00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '00001FFFFFFFFFFFFFF80000'
    '00001FFFFFFFFFFFFFF80000'
    '00007FFFFFFFFFFFFFFE0000'
    'FFFFFFFFFFFFFFFFFFFFFFFF'
    'FFFFFFFFFFFFFFFFFFFFFFFF'
    'FFFFFFFFFFFFFFFFFFFFFFFF'
    'FFFFFFFFFFFFFFFFFFFFFFFF'
    'FFFFFFFFFFFFFFFFFFFFFFFF'
    'FFFFFFFFFFFFFFFFFFFFFFFF'
    'FFFFFFFFFFFFFFFFFFFFFFFF'
    'FFFFFFFFFFFFFFFFFFFFFFFF'
    '00007FFFFFFFFFFFFFFE0000'
    '00001FFFFFFFFFFFFFF80000'
    '00001FFFFFFFFFFFFFF80000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '000007FFFFFFFFFFFFE00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000FFFFFFFFFFFFFF00000'
    '00000BFFFFFFFFFFFFD00000'
    '000007BFFFFFFFFFFDE00000'
    '00000000007FFE0000000000'
    '00000000001FF80000000000'
    '00000000001FF80000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
    '00000000000FF00000000000'
};
mask = false(N, N);
for iy = 1:N
    v = hex2dec(num2cell(hex_rows{iy}));            % 24 values 0..15
    bits = dec2bin(v, 4);                               % 24 x 4 char
    mask(iy, :) = reshape(bits.', 1, N) == '1';
end
epsr = EPS_AIR + (EPS_SI - EPS_AIR) * double(mask);     % exactly 1.0 or 12.1104

%% ---- (3) independent verification (same PWEM formulation as the grader; not pc_evaluate.p) --------------------
fprintf('Seed design  : '); Rs = pc_pwem_official(epsr_seed, 9, target, 10);
fprintf('Final design : '); Rf = pc_pwem_official(epsr, 9, target, 10);
fprintf('fill fraction of Si: seed %.4f, final %.4f\n', mean(seed_mask(:)), mean(mask(:)));
assert(isequal(size(epsr), [96 96])); assert(all(epsr(:) == EPS_AIR | epsr(:) == EPS_SI));

%% ---- (4) save ONLY epsr ------------------------------------------------------------------------------------------
save('Nano_P_INITIAL_STUDENTID.mat', 'epsr', '-v7');     % MATLAB v7 MAT-file containing only epsr
fprintf('saved %s (variable epsr, %d x %d)\n', 'Nano_P_INITIAL_STUDENTID.mat', size(epsr,1), size(epsr,2));

%% ================================================================================================================
%% local functions: independent plane-wave expansion evaluator (TM: eps-matrix, TE: 1/eps coefficients)
function R = pc_pwem_official(epsr, Mmax, target, nb)
% PC_PWEM_OFFICIAL  Independent plane-wave-expansion check of a 2D square-lattice photonic crystal.
%   R = pc_pwem_official(epsr)            uses the grading settings: Mmax = 9 (19x19 = 361 plane waves),
%                                          21 x 11 k-points over half the Brillouin zone, target a/lambda = 634/1550.
%   R = pc_pwem_official(epsr, Mmax, target, nb)
%
%   epsr : N x N permittivity map sampled at pixel centres (row = y, column = x), values 1.0 or 12.1104.
%   Conventions (photonic-crystal convention):  TM = (Ez,Hx,Hy),  TE = (Hz,Ex,Ey).
%   TM eigenproblem : |k+G|^2 c_G = k0^2 sum_G' eps_{G-G'} c_G'          (Fourier coefficients of eps)
%   TE eigenproblem : sum_G' (k+G).(k+G') eta_{G-G'} c_G' = k0^2 c_G    (Fourier coefficients of 1/eps)
%   Frequencies are returned as omega*a/(2*pi*c) = a/lambda.
%   Score = 2/w_t * min(w_high - w_t, w_t - w_low) for the complete (TE and TM) gap containing the target.
%
%   This is NOT the official evaluator (pc_evaluate.p); it is an independent implementation of the same
%   textbook method, written to mirror the Python solver used for the optimisation.
if nargin < 2 || isempty(Mmax),   Mmax = 9;          end
if nargin < 3 || isempty(target), target = 634/1550; end
if nargin < 4 || isempty(nb),     nb = 10;           end
N = size(epsr, 1);
m = -Mmax:Mmax;
[M1, M2] = meshgrid(m, m);           % M1 varies along columns
m1 = M1(:); m2 = M2(:);
G = 2*pi*[m1, m2];                   % reciprocal lattice vectors (a = 1): b1 = 2*pi*x, b2 = 2*pi*y
nG = numel(m1);
% Fourier tables of eps and 1/eps for all differences G-G' (indices -2Mmax..2Mmax)
Etab = ftable(epsr, Mmax, N);
Htab = ftable(1 ./ epsr, Mmax, N);
d1 = repmat(m1, 1, nG) - repmat(m1.', nG, 1) + 2*Mmax + 1;
d2 = repmat(m2, 1, nG) - repmat(m2.', nG, 1) + 2*Mmax + 1;
lin = sub2ind(size(Etab), d1, d2);
E = Etab(lin);                       % eps_{G-G'}
H = Htab(lin);                       % eta_{G-G'}
Einv = inv(E);
% k-points: 21 x 11 over half the Brillouin zone (kx in [-pi,pi], ky in [0,pi]), units 1/a
kx = linspace(-pi, pi, 21); ky = linspace(0, pi, 11);
[KX, KY] = meshgrid(kx, ky); K = [KX(:), KY(:)];
nk = size(K, 1);
w_tm = zeros(nk, nb); w_te = zeros(nk, nb);
for ik = 1:nk
    kG = [K(ik,1) + G(:,1), K(ik,2) + G(:,2)];
    mag = sqrt(sum(kG.^2, 2));
    Atm = (mag * mag.') .* Einv;  Atm = (Atm + Atm') / 2;      % symmetrised form, same eigenvalues as E\D
    Ate = (kG * kG.') .* H;       Ate = (Ate + Ate') / 2;
    ltm = sort(real(eig(Atm))); lte = sort(real(eig(Ate)));
    w_tm(ik, :) = sqrt(max(ltm(1:nb), 0)).' / (2*pi);
    w_te(ik, :) = sqrt(max(lte(1:nb), 0)).' / (2*pi);
end
% gaps between consecutive sorted bands
gtm = gaps(w_tm); gte = gaps(w_te);
best = []; score = 0;
for i = 1:size(gtm, 1)
    for j = 1:size(gte, 1)
        lo = max(gtm(i,1), gte(j,1)); hi = min(gtm(i,2), gte(j,2));
        if hi > lo && lo < target && target < hi
            s = 2 / target * min(hi - target, target - lo);
            if s > score, score = s; best = [lo, hi, gtm(i,3), gte(j,3)]; end
        end
    end
end
R.score = score; R.score_percent = 100*score;
R.gap = best;                        % [w_low, w_high, TM band below gap, TE band below gap]
R.tm_gaps = gtm; R.te_gaps = gte; R.w_tm = w_tm; R.w_te = w_te; R.K = K; R.target = target;
if isempty(best)
    fprintf('pc_pwem_official: no complete gap contains the target -> score 0\n');
else
    fprintf('pc_pwem_official: score = %.3f %%   complete gap [%.5f, %.5f]  (TM bands %d-%d, TE bands %d-%d)\n', ...
        100*score, best(1), best(2), best(3), best(3)+1, best(4), best(4)+1);
end
end

function T = ftable(f, Mmax, N)
% Fourier coefficients c(m1,m2) = (1/N^2) sum_{ix,iy} f(iy,ix) exp(-i 2 pi (m1 x_ix + m2 y_iy)),
% pixel centres x_ix = (ix - 1 + 0.5)/N - 0.5, for m1, m2 in [-2Mmax, 2Mmax]; T(m1+2Mmax+1, m2+2Mmax+1).
F = fft2(f.') / N^2;                 % rows <-> x index (m1), columns <-> y index (m2)
idx = -2*Mmax:2*Mmax;
ph = exp(-1i*pi*idx/N) .* exp(1i*pi*idx);
T = F(mod(idx, N) + 1, mod(idx, N) + 1) .* (ph.' * ph);
end

function g = gaps(w)
mx = max(w, [], 1); mn = min(w, [], 1); g = zeros(0, 3);
for n = 1:size(w, 2) - 1
    if mn(n+1) > mx(n), g(end+1, :) = [mx(n), mn(n+1), n]; end %#ok<AGROW>
end
end

