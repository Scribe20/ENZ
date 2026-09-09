#!/bin/bash
# queue B: final3 production, final3 no-ITO, final0 production, final1 no-ITO
cd "$(dirname "$0")"
PY=/opt/venv-fdtdx/bin/python
FL="1302.282 1294 1298 1306 1310"
FLN="1302.282 1280 1290 1300 1310 1320"
$PY fx_sim.py --design final3 --nxy 64 --ito-cells 5 --time-fs 300 --tag prod --field-lams $FL > logs/final3_prod.log 2>&1
$PY fx_sim.py --design final3 --no-ito --nxy 64 --ito-cells 5 --time-fs 450 --tag noito --field-lams $FLN > logs/final3_noito.log 2>&1
$PY fx_sim.py --design final0 --nxy 64 --ito-cells 5 --time-fs 300 --tag prod --field-lams $FL > logs/final0_prod.log 2>&1
$PY fx_sim.py --design final1 --no-ito --nxy 64 --ito-cells 5 --time-fs 450 --tag noito --field-lams $FLN > logs/final1_noito.log 2>&1
echo "[queue B] done"
