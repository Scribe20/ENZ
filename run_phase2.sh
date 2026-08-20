#!/bin/bash
# Phase 2 of the ED/MD co-excitation search:
#   rank + diversity-select discovery results, refine selected candidates in
#   parallel, then run binary/Fourier-order verification, diagnostic spectra,
#   and multipole-data export for every refined candidate.
# Usage: bash run_phase2.sh [n_parallel]
set -u
cd "$(dirname "$0")"
NPAR=${1:-4}
R=results_ed_md_coexcitation

echo "=== Phase 2: summary + selection ==="
python analyze_coexcite_results.py --mode summary || exit 1
python analyze_coexcite_results.py --mode select || exit 1
python analyze_coexcite_results.py --mode contact --stage discovery

SELECTED=$(python - <<'EOF'
import json
sel = json.load(open('results_ed_md_coexcitation/selection_diversity.json'))
print('\n'.join(s['run_id'] for s in sel['selected']))
EOF
)
echo "=== Refining $(echo "$SELECTED" | wc -l) candidates (${NPAR} workers) ==="
# up to 5 rounds: interrupted runs (e.g. OOM kills) resume from checkpoints;
# completed runs are skipped on re-entry, so extra rounds are cheap no-ops.
for round in 1 2 3 4 5; do
  echo "--- refinement round $round ---"
  echo "$SELECTED" | OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2 xargs -P "$NPAR" -I{} \
    python coexcite_ed_md_sweep.py --mode refine --run-dir "$R/discovery/{}" --threads 1
  pending=0
  for rid in $SELECTED; do
    st=$(python -c "import json;print(json.load(open('$R/refine/${rid}_refined/config.json'))['status'])" 2>/dev/null || echo none)
    [ "$st" = "completed" ] || pending=$((pending+1))
  done
  [ "$pending" -eq 0 ] && break
  echo "--- $pending refinements incomplete after round $round, retrying ---"
done

echo "=== Verification (orders 9/11/13, binary+projected) ==="
ls -d "$R"/refine/*_refined | OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 xargs -P "$NPAR" -I{} \
  python coexcite_ed_md_sweep.py --mode verify --run-dir {} --threads 1

echo "=== Diagnostic wavelength sweeps (binary geometry) ==="
ls -d "$R"/refine/*_refined | OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 xargs -P "$NPAR" -I{} \
  python coexcite_ed_md_sweep.py --mode spectra --run-dir {} --geometry binary --threads 1

echo "=== Multipole-decomposition data export ==="
ls -d "$R"/refine/*_refined | OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 xargs -P "$NPAR" -I{} \
  python coexcite_ed_md_sweep.py --mode multipole-data --run-dir {} --geometry binary --threads 1

echo "=== Final analysis ==="
python analyze_coexcite_results.py --mode summary --stage refine
python analyze_coexcite_results.py --mode contact --stage refine
python analyze_coexcite_results.py --mode final --stage refine
echo "=== Phase 2 complete ==="
