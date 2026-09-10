#!/bin/bash
# Worker A of the fresh no-ITO reruns.  Waits for the final3 deep-glass no-ITO run that is already
# in flight, then re-runs the final3 WITH-ITO case and the empty-domain reference on the identical
# deeper mesh (so the with/without pair and its normalisation stay directly comparable), and finally
# takes the long final0 job.
#
# final0 runs to 36 ps, not 24 ps: the band-integrated closure residual of its 0.9 / 3 / 6 ps windows
# decays with tau = 9.7-13.6 ps (against 1.5-4.6 ps for the other three designs), so on the
# conservative estimate 6 ps + 13.6 ps * ln(0.094/0.01) = 36.5 ps is what the acceptance tolerance
# needs.  Windows are closed at 9, 18 and 27 ps to record the decay inside the run.
cd "$(dirname "$0")"
PY=/opt/venv-fdtdx/bin/python
FLN="1302.282 1280 1290 1300 1310 1320"
FL="1302.282 1294 1298 1306 1310"
while pgrep -f "fx_sim.py --design final3 --no-ito.*noito6ps_deepglass" > /dev/null; do sleep 120; done
$PY fx_sim.py --design final3 --nxy 64 --ito-cells 5 --time-fs 300 --glass-extra-nm 4500 \
    --tag prod_deepglass --field-lams $FL > logs/final3_prod_deepglass.log 2>&1
echo "[A final3 deepglass with-ITO] $(grep '\[fx\] done' logs/final3_prod_deepglass.log)"
$PY fx_sim.py --reference --design final3 --nxy 64 --ito-cells 5 --time-fs 400 --glass-extra-nm 4500 \
    --tag ref64_deepglass --field-lams $FLN > logs/ref64_deepglass.log 2>&1
echo "[A reference deepglass] $(grep '\[fx\] done' logs/ref64_deepglass.log)"
$PY fx_sim.py --design final0 --no-ito --nxy 64 --ito-cells 5 --time-fs 36000 \
    --conv-check-fs 9000 18000 27000 --tag noito36ps --field-lams $FLN \
    > logs/final0_noito36ps.log 2>&1
echo "[A final0 36ps] $(grep '\[fx\] done' logs/final0_noito36ps.log)"
echo "[stage3 A] done"
