#!/bin/bash
# queue A: reference, final3 fine-ITO convergence run, final1, final2 production, no-ITO for final0/final2
cd "$(dirname "$0")"
PY=/opt/venv-fdtdx/bin/python
FL="1302.282 1294 1298 1306 1310"
FLN="1302.282 1280 1290 1300 1310 1320"
$PY fx_sim.py --reference --design final3 --nxy 64 --ito-cells 5 --time-fs 250 --tag ref64 --field-lams $FL > logs/ref64.log 2>&1
$PY fx_sim.py --design final3 --nxy 64 --ito-cells 10 --time-fs 300 --tag prod_ito10 --field-lams $FL > logs/final3_prod_ito10.log 2>&1
$PY fx_sim.py --design final1 --nxy 64 --ito-cells 5 --time-fs 300 --tag prod --field-lams $FL > logs/final1_prod.log 2>&1
$PY fx_sim.py --design final2 --nxy 64 --ito-cells 5 --time-fs 300 --tag prod --field-lams $FL > logs/final2_prod.log 2>&1
$PY fx_sim.py --design final0 --no-ito --nxy 64 --ito-cells 5 --time-fs 450 --tag noito --field-lams $FLN > logs/final0_noito.log 2>&1
$PY fx_sim.py --design final2 --no-ito --nxy 64 --ito-cells 5 --time-fs 450 --tag noito --field-lams $FLN > logs/final2_noito.log 2>&1
echo "[queue A] done"
