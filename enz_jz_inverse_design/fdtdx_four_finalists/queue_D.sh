#!/bin/bash
# chain D: waits for the running final1 production job, then final2 production, then no-ITO 900 fs (final1, final2)
cd "$(dirname "$0")"
PY=/opt/venv-fdtdx/bin/python
FL="1302.282 1294 1298 1306 1310"
FLN="1302.282 1280 1290 1300 1310 1320"
while pgrep -f "fx_sim.py --design final1 --nxy" > /dev/null; do sleep 30; done
$PY fx_sim.py --design final2 --nxy 64 --ito-cells 5 --time-fs 300 --tag prod --field-lams $FL > logs/final2_prod.log 2>&1
$PY fx_sim.py --design final1 --no-ito --nxy 64 --ito-cells 5 --time-fs 900 --tag noito --field-lams $FLN > logs/final1_noito.log 2>&1
$PY fx_sim.py --design final2 --no-ito --nxy 64 --ito-cells 5 --time-fs 900 --tag noito --field-lams $FLN > logs/final2_noito.log 2>&1
echo "[queue D] done"
