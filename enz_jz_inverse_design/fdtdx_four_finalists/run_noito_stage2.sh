#!/bin/bash
# Stage 2 of the fresh no-ITO reruns, queued behind the running 6-ps jobs.
#   final3 : SAME simulation time, DEEPER GLASS (+4.5 um).  Its only defect is R > 1 / T < 0 at
#            1254-1258 nm, where the (+-1,0) orders are evanescent in the glass with 1.3-1.9 um decay
#            lengths and still 35-51% of their amplitude reaches the PML face; the 3->6 ps drift there
#            is only -0.02, so the artifact is domain depth, not time.  Deepening the glass drops the
#            evanescent amplitude at the PML face to 5%.  The with-ITO case is re-run on the identical
#            deeper mesh so the pair stays directly comparable.
#   final2 : SAME mesh, LONGER TIME (12 ps).  Its defect is a genuine truncation residual at
#            1306-1310 nm (|A| up to 0.024, 3->6 ps drift still ~0.10) on the longest-lived mode
#            (envelope tau = 1088 fs).
cd "$(dirname "$0")"
PY=/opt/venv-fdtdx/bin/python
FLN="1302.282 1280 1290 1300 1310 1320"
FL="1302.282 1294 1298 1306 1310"
while pgrep -f "fx_sim.py --design final[10] --no-ito" > /dev/null; do sleep 120; done
$PY fx_sim.py --design final3 --no-ito --nxy 64 --ito-cells 5 --time-fs 6000 --conv-check-fs 3000 \
    --glass-extra-nm 4500 --tag noito6ps_deepglass --field-lams $FLN > logs/final3_noito6ps_deepglass.log 2>&1
echo "[final3 deepglass no-ITO] $(grep '\[fx\] done' logs/final3_noito6ps_deepglass.log)"
$PY fx_sim.py --design final3 --nxy 64 --ito-cells 5 --time-fs 300 --glass-extra-nm 4500 \
    --tag prod_deepglass --field-lams $FL > logs/final3_prod_deepglass.log 2>&1
echo "[final3 deepglass with-ITO] $(grep '\[fx\] done' logs/final3_prod_deepglass.log)"
$PY fx_sim.py --reference --design final3 --nxy 64 --ito-cells 5 --time-fs 400 --glass-extra-nm 4500 \
    --tag ref64_deepglass --field-lams $FLN > logs/ref64_deepglass.log 2>&1
echo "[reference deepglass] $(grep '\[fx\] done' logs/ref64_deepglass.log)"
$PY fx_sim.py --design final2 --no-ito --nxy 64 --ito-cells 5 --time-fs 12000 --conv-check-fs 6000 \
    --tag noito12ps --field-lams $FLN > logs/final2_noito12ps.log 2>&1
echo "[final2 12ps no-ITO] $(grep '\[fx\] done' logs/final2_noito12ps.log)"
echo "[stage2] done"
