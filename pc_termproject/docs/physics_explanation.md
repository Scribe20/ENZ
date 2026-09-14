# Why the final structure opens both gaps at 1550 nm — causal chain

(Numbers refer to the C4v-optimised "rounded-block + veins" design family; final values are in RESULTS.md.)

## 1. Periodicity → Bloch waves → plane-wave eigenproblem (lectures PC-I, PC-II)
* The 96 × 96 cell tiles the plane: ε(r + m a1 + n a2) = ε(r). The translation operators commute with the Maxwell
  operator, so eigenmodes are Bloch waves ψ_k(r) = e^{ik·r} u_k(r) with u_k periodic (PC-I p.16–20).
* u_k and ε (or 1/ε) are Fourier series on the reciprocal lattice G = 2π(m1, m2)/a (b_p·a_q = 2π δ_pq, PC-II p.9).
  Truncation |m1|,|m2| ≤ 9 gives 361 plane waves; the grader's Fourier coefficients are the pixel sums of PC-II p.13.
* For each k the Maxwell operator becomes a Hermitian matrix: TM (Ez): |k+G|² c_G = (ω/c)² Σ ε_{G−G'} c_{G'};
  TE (Hz): Σ (k+G)·(k+G') η_{G−G'} c_{G'} = (ω/c)² c_G, η = 1/ε. Eigenvalues sorted at every k give the bands.
* Bragg condition: bands fold and split at the zone boundary (X, M); the size of the splitting at a given zone-boundary
  point is set by the Fourier component ε_G that couples the two degenerate plane waves — a gap is a Bragg splitting that
  survives over the whole zone.

## 2. Why TM and TE want different geometries (PC-III p.27–29)
* Electric energy density ~ D*·E. Low bands lower their frequency by concentrating electric energy in ε_Si.
* TM: Ez is continuous across interfaces, Dz = εEz jumps. A mode can put nearly all of its Ez inside an isolated Si island
  (dielectric band), while the next band must have a node inside the island and is pushed into air (air band): large
  index contrast between consecutive bands → large TM gaps for isolated islands (rods).
* TE: the in-plane E is discontinuous (E_t continuous, D_n continuous). To concentrate energy the field must follow
  connected dielectric paths; in isolated rods the field lines cross Si/air interfaces and the "dielectric" band is only
  weakly concentrated (23 % vs 83 % in the table of p.28) → tiny TE gaps for rods, large TE gaps for connected veins.
* Connected dielectric hurts TM: the air-band Ez penetrates the veins (77 % of energy in Si for the vein air band, p.29),
  so the two TM bands have similar effective index and the TM gap closes.

## 3. The tension at a fixed target frequency
a/λ = 0.409 is fixed. The lowest TM gap of small rods lies there only for r ≲ 0.2a (fill ≈ 12 %), but such a sparse
lattice has no TE gap at all (Figure 1 of the project). Making the lattice connected enough for a TE gap raises the
average index and drags the TM 1-2 gap far below 0.409. Therefore the complete gap cannot be the fundamental gap of both
polarisations; a higher TM gap must be used. Stage A shows every one-inclusion family (rods, holes, crosses, rings) fails
(H1/H2 confirmed), and that only hybrids succeed (H3/H4).

## 4. Mechanism of the final family: large island + thin connecting veins
Geometry: a Si block of width ≈ 0.6a (rounded/chamfered corners) at each lattice site, connected to its four neighbours
by veins ≈ 0.08a wide; Si fill ≈ 0.41.
* TM 3-4 gap (0.371–0.431): the isolated-island resonances of a large block. Band 3 at its maximum (X) is the block's
  dipole-like Ez resonance (98 % of electric energy in Si); band 4 at its minimum (Γ) is the quadrupole-like resonance with an
  extra nodal line (92 % in Si). The two resonances are separated because the block is large enough that its
  "Mie"-type resonances fall near a/λ ≈ 0.4; the thin veins carry little Ez, so they do not spoil the island character
  (a wider vein w = 0.10 already kills the score: 0.37 %).
* TE 2-3 gap (0.387–0.450): band 2 at Γ keeps its |D|² in a ring inside the block (92 % in Si, dielectric-like); band 3 at M
  is forced out of the block into the veins and the air pockets (44 % in Si). The connected veins are what allow the
  low TE bands to be dielectric-concentrated while the block's size sets where band 3 must leave the dielectric.
* Balance: the lower edge of the complete gap is the TE band-2 maximum (Γ), the upper edge the TM band-4 minimum (Γ).
  Increasing the island size lowers the TM edge (bad) and raises the TE band-2 (good); widening the veins raises the TM
  air-band penetration (bad for TM) but lowers the TE band-3 minimum (bad). The optimiser found the block size, corner
  shape and vein width that put the two limiting edges symmetric about 0.409 (margins 0.0220 / 0.0217): the score is
  the min of the two margins, so the optimum is where they are equal.
* Fill fraction: the gap centre scales roughly as 1/√ε_avg; the optimum fill 0.41 is the value that centres the complete
  gap on the target — larger fill (e.g. rods 0.32 + veins 0.10, fill 0.40 → 0.37 %) shifts the TM 4 minimum below target.

## 5. Symmetry
C4v is not required (the grader samples half the zone, i.e. it allows any structure) but the optimum is essentially
symmetric: removing the constraint gains only +0.14 % (10.62 → 10.75 %). With C4v the band extrema occur at the
high-symmetry points (Γ, X, M), the irreducible wedge of the official grid is exact, and the design is robust to the
orientation of the crystal.

## 6. Numerics vs physics
* The TM band edges are converged already at Mmax = 5 (eps-matrix form); the TE band edges of the grader's 1/ε form drift
  by ≈ 0.01 between Mmax = 9 and 15 (slow Laurent-rule convergence, PC-II p.14–16). The gap must therefore survive a
  basis increase; see the convergence table in RESULTS.md (both formulations, Mmax 5–15) and the fine k-grid check.
* Sub-pixel dithering could fool the Mmax = 9 evaluator (it averages ε for TM and 1/ε for TE inside a 5-pixel scale);
  the final design was produced with a filter radius ≥ 2 px, a discrete refinement restricted to boundary moves and
  checked at higher Mmax — the gap is a property of the geometry, not of the truncation.
