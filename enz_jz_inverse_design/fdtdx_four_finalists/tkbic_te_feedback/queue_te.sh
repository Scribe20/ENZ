#!/bin/bash
# Te family queue: empty-domain reference (250 fs), then the nine final3 Te runs (400 fs), two at a time.
cd "$(dirname "$0")"
PY=${PY:-/tmp/claude-0/-home-user-ENZ/b2c620ce-f033-5087-9e63-aa48bfaee6e8/scratchpad/venv/bin/python}
mkdir -p logs
$PY fx_te_sim.py --reference --time-fs 250 > logs/ref.log 2>&1
run_pair () {
  $PY fx_te_sim.py --Te $1 --time-fs 400 > logs/Te$1.log 2>&1 &
  p1=$!
  if [ -n "$2" ]; then $PY fx_te_sim.py --Te $2 --time-fs 400 > logs/Te$2.log 2>&1 & p2=$!; wait $p2; fi
  wait $p1
}
run_pair 300 8000
run_pair 3000 1500
run_pair 600 1000
run_pair 2000 4500
run_pair 6000
echo "[queue_te] all done $(date)" >> logs/queue_te.log
