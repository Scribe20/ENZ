#!/bin/bash
# Worker B of the fresh no-ITO reruns: the long final1 job, then final2 at 12 ps.
cd "$(dirname "$0")"
PY=/opt/venv-fdtdx/bin/python
FLN="1302.282 1280 1290 1300 1310 1320"
$PY fx_sim.py --design final1 --no-ito --nxy 64 --ito-cells 5 --time-fs ${T1:-24000} \
    --conv-check-fs 6000 12000 18000 --tag noito${T1KP:-24}ps --field-lams $FLN \
    > logs/final1_noito${T1KP:-24}ps.log 2>&1
echo "[B final1 long] $(grep '\[fx\] done' logs/final1_noito${T1KP:-24}ps.log)"
$PY fx_sim.py --design final2 --no-ito --nxy 64 --ito-cells 5 --time-fs 12000 \
    --conv-check-fs 3000 6000 9000 --tag noito12ps --field-lams $FLN \
    > logs/final2_noito12ps.log 2>&1
echo "[B final2 12ps] $(grep '\[fx\] done' logs/final2_noito12ps.log)"
echo "[stage3 B] done"
