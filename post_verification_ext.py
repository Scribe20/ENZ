"""
post_verification_ext.py
========================
Post-pipeline extensions:

1. [15,15] Fourier-order extension of the frozen-binary verification for the
   three most order-sensitive and three most robust refined candidates
   (appended to each run's verification.csv).
2. Fine local diagnostic wavelength sweeps (lam0 +/- 12 nm, 0.5 nm step,
   order [9,9], binary geometry) for the top high-fidelity candidates,
   written to spectra_fine.csv, to resolve linewidths and true ED/MD peak
   splitting below the 5-nm coarse-sweep resolution.

Diagnostic only - F_co is never modified.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coexcite_ed_md_sweep import (verify_candidate, spectra_candidate,
                                  RESULTS_ROOT)

FRAGILE = ['P0900_H0200_seed029_lam1332p5', 'P0900_H0175_seed047_lam1332p5',
           'P0900_H0100_seed029_lam1332p5']
ROBUST = ['P0830_H0100_seed047_lam1332p5', 'P0790_H0125_seed029_lam1332p5',
          'P0830_H0100_seed011_lam1332p5']
FINE_SWEEP = ['P0830_H0100_seed047_lam1332p5', 'P0790_H0125_seed029_lam1332p5',
              'P0900_H0100_seed029_lam1332p5']


def main():
    for rid in FRAGILE + ROBUST:
        d = RESULTS_ROOT / 'refine' / (rid + '_refined')
        out = d / 'verification_o15.csv'
        if out.exists():
            print(f'skip [15,15] {rid}')
            continue
        rows = verify_candidate(d, orders=[[15, 15]])
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)

    for rid in FINE_SWEEP:
        d = RESULTS_ROOT / 'refine' / (rid + '_refined')
        out = d / 'spectra_fine.csv'
        if out.exists():
            print(f'skip fine sweep {rid}')
            continue
        # spectra_candidate writes spectra.csv in-place: preserve the coarse
        # sweep and move the fine result to spectra_fine.csv afterwards.
        coarse = (d / 'spectra.csv').read_text()
        rows = spectra_candidate(d, span_nm=12.0, step_nm=0.5,
                                 order=[9, 9], geometry='binary')
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        (d / 'spectra.csv').write_text(coarse)


if __name__ == '__main__':
    main()
