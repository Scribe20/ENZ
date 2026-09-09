"""Sequential job driver for the Q-Dz parent campaign (2 workers x 2 threads on the 4-core host)."""
import argparse, json, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="JSON file: list of argument dicts for optimize_parent.py")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    jobs = json.load(open(a.jobs))
    mine = [j for i, j in enumerate(jobs) if i % a.nworkers == a.worker]
    print(f"[worker {a.worker}] {len(mine)} of {len(jobs)} jobs", flush=True)
    for j in mine:
        cmd = [sys.executable, str(HERE / "optimize_parent.py"), "--threads", str(a.threads)]
        for k, v in j.items():
            cmd += [f"--{k.replace('_', '-')}"] + ([] if v is True else [str(v)])
        t = time.time()
        print(f"[worker {a.worker}] {' '.join(cmd[2:])}", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        tail = [l for l in r.stdout.splitlines() if l.startswith("[")][-1:] or r.stderr.splitlines()[-3:]
        print(f"[worker {a.worker}] rc={r.returncode} {time.time()-t:.0f}s :: {' | '.join(tail)}", flush=True)
    print(f"[worker {a.worker}] done", flush=True)


if __name__ == "__main__":
    main()
