# Fresh no-ITO runs — why the 6 ps runs were rejected

Numerical diagnostics only.  Every statement below is a measurement on the simulation output; no
physical interpretation of the structures is offered or implied.

## 1. The residual field at the end of the run tracks the closure error

`A = 1 - R - T` has the exact value 0 at every wavelength here (all media lossless), so `max |A|` is a
direct measure of the numerical error.  Across the four 6 ps runs it orders exactly like the field
amplitude still present in the domain when the DFT window closes:

| design | `final max\|E\|` at 6 ps | `max\|A\|` |
|---|---|---|
| final3 | 3.121e-05 | 0.0023 |
| final2 | 1.030e-03 | 0.0239 |
| final0 | 9.874e-04 | 0.0941 |
| final1 | 1.110e-03 | 0.1349 |

## 2. The affected band narrows like 1/T

The wavelength interval over which the closure fails shrinks as the DFT window lengthens, at the rate
expected for window leakage (`dlambda = lambda^2 / (c T)`):

| run length | predicted leakage width | observed width of the failing band (final1) |
|---|---|---|
| 900 fs | 6.6 nm | ~8 nm (1326-1334 nm all fail) |
| 6000 fs | 0.98 nm | ~1 nm (only 1330 nm fails) |

so the error is DFT-window leakage from a field that has not finished decaying, not a mis-set monitor
or normalisation.

## 3. Three cheaper explanations were excluded

* **Weak source at those wavelengths.**  The incident spectrum measured by the empty-domain reference
  run is 0.85 (1330 nm) and 0.76 (1340 nm) of its peak, so R and T there are not small-denominator
  quantities.
* **TFSF leakage subtraction.**  `max |leak| / P_inc = 1.34e-07`.
* **Undamped Lorentz poles in the a-Si:H and SiO2 fits.**  They sit at 405 nm and 74 nm; the source is
  a Gaussian pulse of 13 THz spectral width centred at 230.6 THz, which has no content there.

## 4. The frequency-domain solver shows no feature at those wavelengths

The same parent stack (a-Si:H on glass, ITO removed) was solved with the campaign's RCWA code at the
FDTD-relevant wavelengths, for both the native 128 x 128 pattern and the 64 x 64 area-fraction
rasterisation that FDTDX actually meshes:

| scan | result |
|---|---|
| final1, 1315-1360 nm, 0.5 nm steps, order [7,7], 128 x 128 | smooth, `max` step 0.0038 |
| final1, 1320-1345 nm, 0.25 nm steps, 128 x 128 | smooth, `max` step 0.0018 |
| final1, 1320-1345 nm, 0.25 nm steps, 64 x 64 | smooth, `max` step 0.0019 |
| final1, 1328-1332 nm, **0.05 nm** steps, 64 x 64 | `R` in [0.94513, 0.96553], `max` step 0.00028 |

i.e. at 1330 nm the frequency-domain answer is a smooth `R ~ 0.955`, while the 6 ps FDTD run reports
`R = 0.483`, `T = 0.382`.  Nothing resolvable down to a 0.05 nm linewidth sits there.

## 5. The flux-plane near field is anomalous exactly at the failing wavelength

Fraction of the `|Ex|^2` on the reflection flux plane that is **not** in the specular (0,0) order
(the plane sits 425 nm above the a-Si, where every other order is evanescent):

| design | worst-`\|A\|` wavelength | non-specular fraction there | median over the band | ratio |
|---|---|---|---|---|
| final3 | 1258 nm | 0.0108 | 0.0036 | 3.0x |
| final0 | 1340 nm | 0.0242 | 0.0036 | 6.8x |
| final2 | 1308 nm | 0.0972 | 0.0036 | 27x |
| final1 | 1330 nm | 0.1406 | 0.0036 | 40x |

For final1 the strongest non-specular component is (0,-1) at 0.278 of the specular amplitude on the
plane, against ~0.03 at the neighbouring wavelengths.

## 6. What was changed for the reruns

* **final3** — its defect is spatial, not temporal: `R > 1` with a compensating `T < 0` at 1254-1258 nm,
  just above the glass Rayleigh cut-off (`n_glass P = 1251.2 nm`), where the (+-1,0) orders are
  evanescent in the glass with 1.3-1.9 um decay lengths, so 35-51% of their amplitude still reaches the
  CPML face; the 3 -> 6 ps drift there is only -0.02.  Rerun with 4.5 um of extra glass, which drops that
  amplitude to 4.9%.  Its with-ITO partner and the empty-domain reference are rerun on the identical
  deeper mesh so the pair and its normalisation stay comparable.
* **final1, final0** — rerun at 24 ps (from 6 ps), same mesh, with DFT windows closed at 6, 12 and 18 ps.
* **final2** — rerun at 12 ps with windows at 3, 6 and 9 ps.

The window series is the measurement that decides the outcome: an exponentially decaying residual means
the longer run converges, while a residual falling like 1/T means the field is still ringing at the end
of any window and the remaining error is quantified rather than removed.  Whatever the series shows is
reported as it is; no spectrum is smoothed, clipped, renormalised or otherwise modified.
