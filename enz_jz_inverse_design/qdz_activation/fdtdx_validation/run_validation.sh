#!/usr/bin/env bash
# Independent FDTDX validation of the FROZEN finalists (run only after the TORCWA campaign closed and
# register_finalists.py wrote manifest.json).  One reference run per candidate mesh (the z mesh depends on h).
#   ./run_validation.sh <tag> [<tag> ...]
# Environment: /opt/venv-fdtdx (fdtdx main-branch snapshot 98aef1c, jax 0.10.2, CPU).
set -euo pipefail
cd "$(dirname "$0")"
PY=/opt/venv-fdtdx/bin/python
TIME_FS=${TIME_FS:-600}
for tag in "$@"; do
  echo "=== $tag: reference (empty cell, identical mesh) ==="
  $PY fx_run.py --design "$tag" --reference --tag prod --time-fs 250 --ito-cells 5 --nxy 64 2>&1 | grep -v -i warning | tail -n 2
  echo "=== $tag: production with ITO, ${TIME_FS} fs, early-window certificates at 300 / 450 fs, deep volume detector with E and H ==="
  $PY fx_run.py --design "$tag" --tag prod --time-fs "$TIME_FS" --ito-cells 5 --nxy 64 --vol-H --vol-glass-cells 24 --vol-air-cells 8 \
      --conv-check-fs 300 450 --segment-steps 8000 2>&1 | grep -v -i warning | tail -n 3
  echo "=== $tag: post-processing (raw phasors -> normalized fields, spectra vs TORCWA, profiles, slices, cuts) ==="
  $PY fx_fields_plots.py --design "$tag" 2>&1 | grep -v -i warning | tail -n 5
done
