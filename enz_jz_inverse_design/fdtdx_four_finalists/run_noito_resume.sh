#!/bin/bash
# Resumable driver for the remaining fresh no-ITO runs.  Safe to re-run at any time: a job whose
# phasors.npz already exists is skipped, and a job with a saved state.npz continues from it, so after a
# container restart this script simply picks the campaign back up.  Each job runs in 50 000-step
# segments (~10 min), which is the most that a restart can cost.  Outputs are rewritten after every
# segment too, so an interrupted job already has a usable (shorter-window) spectrum on disk; a job is
# "done" only when its run_meta.json says complete.  Segmenting was verified to be exact:
# a run killed at 37 % and resumed reproduces the uninterrupted run's detector phasors bit for bit,
# because the state handed to the next segment is exactly the state the loop would have had.
#
# $1 = slot name ("A" or "B") selecting which jobs this worker takes.
cd "$(dirname "$0")"
PY=/opt/venv-fdtdx/bin/python
FLN="1302.282 1280 1290 1300 1310 1320"
SEG=20000

done_p () { [ -f "$1/$2/run_meta.json" ] && grep -q '"complete": true' "$1/$2/run_meta.json"; }

job () {   # job <design> <tag> <time_fs> <windows...>
  local d=$1 tag=$2 tfs=$3; shift 3
  if done_p "$d" "$tag"; then echo "[$d/$tag] already complete, skipping"; return; fi
  # Retry loop: the job can be killed by the OOM killer when two runs and an output write coincide
  # (each run holds ~3.5 GB of 15.7 GB, and saving a segment briefly materialises ~1 GB more).  It
  # resumes from its last checkpoint, so a kill costs one segment; without the loop it silently ended
  # the whole run, which is what happened to final0 at ~10:00 UTC.
  local attempt=0
  while ! done_p "$d" "$tag"; do
    attempt=$((attempt + 1))
    if [ $attempt -gt 200 ]; then echo "[$d/$tag] giving up after $attempt attempts"; return 1; fi
    echo "[$d/$tag] attempt $attempt, starting/resuming at $(date -u +%H:%M:%S)"
    $PY fx_sim.py --design "$d" --no-ito --nxy 64 --ito-cells 5 --time-fs "$tfs" \
        --conv-check-fs "$@" --tag "$tag" --segment-steps $SEG --field-lams $FLN \
        >> "logs/${d}_${tag}.log" 2>&1
    done_p "$d" "$tag" || { echo "[$d/$tag] exited without completing; retrying in 60 s"; sleep 60; }
  done
  echo "[$d/$tag] $(grep '\[fx\] done' logs/${d}_${tag}.log | tail -1)"
}

case "$1" in
  A) job final0 noito36ps 36000 9000 18000 27000 ;;
  B) job final2 noito12ps 12000 3000 6000 9000
     job final1 noito24ps 24000 6000 12000 18000 ;;
  *) echo "usage: $0 A|B"; exit 1 ;;
esac
echo "[resume $1] done"
