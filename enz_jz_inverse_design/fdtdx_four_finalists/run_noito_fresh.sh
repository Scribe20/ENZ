#!/bin/bash
# Fresh no-ITO runs: identical geometry/materials/source/mesh/monitors to the with-ITO production runs,
# ITO layer removed (replaced by the substrate glass) as the ONLY change; simulation time extended until
# the lossless energy closure A = 1 - R - T is satisfied.  Each run also records an early-window (3 ps)
# copy of the R/T flux phasors as an in-run convergence certificate.
cd "$(dirname "$0")"
PY=/opt/venv-fdtdx/bin/python
FLN="1302.282 1280 1290 1300 1310 1320"
for d in "$@"; do
  $PY fx_sim.py --design $d --no-ito --nxy 64 --ito-cells 5 --time-fs 6000 --conv-check-fs 3000 \
      --tag noito6ps --field-lams $FLN > logs/${d}_noito6ps.log 2>&1
  echo "[$d] $(grep '\[fx\] done' logs/${d}_noito6ps.log)"
done
echo "[noito fresh $*] done"
