#!/bin/bash
# chain C: no-ITO runs at 900 fs (final3, final0); starts immediately
cd "$(dirname "$0")"
PY=/opt/venv-fdtdx/bin/python
FLN="1302.282 1280 1290 1300 1310 1320"
$PY fx_sim.py --design final3 --no-ito --nxy 64 --ito-cells 5 --time-fs 900 --tag noito --field-lams $FLN > logs/final3_noito.log 2>&1
$PY fx_sim.py --design final0 --no-ito --nxy 64 --ito-cells 5 --time-fs 900 --tag noito --field-lams $FLN > logs/final0_noito.log 2>&1
echo "[queue C] done"
