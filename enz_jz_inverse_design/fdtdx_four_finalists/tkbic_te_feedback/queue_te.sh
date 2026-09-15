#!/bin/bash
# Te family queue (second launch): nine final3 Te runs at 600 fs with the ITO flux planes, four at a time on the 4-core CPU
# (each FDTDX process is effectively single-threaded here), then the checks: 8000 K at 900 fs (time-convergence of the
# hottest run in the 1250-1265 nm near-cut-off region) and 300 K at 300 fs (isolates the ITO pole refit from the window
# length when comparing with the campaign's 300-fs final3/prod).  The first launch (400 fs, no ITO planes; runs
# Te300_400fs / Te8000_400fs kept) was stopped after the post-processing showed the 400-fs window insufficient.
cd "$(dirname "$0")"
PY=${PY:-/tmp/claude-0/-home-user-ENZ/b2c620ce-f033-5087-9e63-aa48bfaee6e8/scratchpad/venv/bin/python}
mkdir -p logs
batch () {                       # batch "Te:time:tag" ...
  pids=()
  for spec in "$@"; do
    IFS=: read -r te tf tag <<< "$spec"
    $PY fx_te_sim.py --Te "$te" --time-fs "$tf" --tag "$tag" > "logs/$tag.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
}
batch 300:600:Te300 8000:600:Te8000 3000:600:Te3000 1500:600:Te1500
batch 600:600:Te600 1000:600:Te1000 2000:600:Te2000 4500:600:Te4500
batch 6000:600:Te6000 8000:900:Te8000_900fs 300:300:Te300_300fs
echo "[queue_te] all done $(date)" >> logs/queue_te.log
