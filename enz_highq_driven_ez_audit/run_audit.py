"""Driver: run the audit stages in order (each a subprocess with its own
log), committing after each.  Usage: python run_audit.py --stages 0,1,2,3,4
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOGS = HERE / "outputs" / "logs"
FOOTER = ("\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\n"
          "Claude-Session: https://claude.ai/code/session_01WD97gqkKje1C4R3Eqsk7cR")
STAGES = {
    "0": ("stage0_manifest.py", "HighQ/driven-Ez audit: Stage 0 manifest + config"),
    "1": ("stage1_harness.py", "HighQ/driven-Ez audit: Stage 1 common driven harness + field maps"),
    "2": ("stage2_poles.py", "HighQ/driven-Ez audit: Stage 2 pole reconnaissance + field-overlap loss-scaling tracking"),
    "3": ("stage3_driven_vs_modal.py", "HighQ/driven-Ez audit: Stage 3 driven vs modal table"),
    "4": ("stage4_pareto.py", "HighQ/driven-Ez audit: Stage 4 Pareto analysis"),
    "5": ("stage5_angular.py", "HighQ/driven-Ez audit: Stage 5 angular driven quantities + sparse pole tracking"),
    "6": ("stage6_alignment.py", "HighQ/driven-Ez audit: Stage 6 spectral alignment P/h sweeps"),
    "7": ("stage7_loss_sweep.py", "HighQ/driven-Ez audit: Stage 7 loss/coupling sweep"),
    "8": ("stage8_leakage.py", "HighQ/driven-Ez audit: Stage 8 controlled leakage tuning of padded QNM"),
    "9": ("stage9_p925_controls.py", "HighQ/driven-Ez audit: Stage 9 P925 controls (with/lossless/no ITO)"),
    "10": ("stage10_p800_narrow.py", "HighQ/driven-Ez audit: Stage 10 P800 narrow-pole physics"),
    "11": ("stage11_survivors.py", "HighQ/driven-Ez audit: Stages 11-14 proxies, normalization, fabrication, convergence"),
    "15": ("stage15_report.py", "HighQ/driven-Ez audit: Stage 15 decision, final table, report"),
}


def log(s):
    print(time.strftime("%H:%M:%S "), s, flush=True)
    with open(LOGS / "driver.log", "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + s + "\n")


def commit(msg):
    subprocess.run(["git", "add", "-A", str(HERE)], cwd=ROOT)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        log("[git] nothing to commit"); return
    p = subprocess.run(["git", "commit", "-q", "-F", "-"], cwd=ROOT, input=msg + FOOTER,
                       text=True, capture_output=True)
    log(f"[git] commit rc={p.returncode} {p.stderr.strip()[-120:]}")
    for wait in (0, 2, 4, 8):
        time.sleep(wait)
        r = subprocess.run(["git", "push", "-u", "origin", "claude/enz-eigenmode-target-u95j8m"],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode == 0:
            log("[git] pushed"); return
    log(f"[git] push failed: {r.stderr.strip()[-160:]}")


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    stages = sys.argv[sys.argv.index("--stages") + 1].split(",") if "--stages" in sys.argv else list(STAGES)
    for s in stages:
        script, msg = STAGES[s]
        if not (HERE / script).exists():
            log(f"[stage {s}] {script} missing - skipped"); continue
        log(f"[stage {s}] start {script}")
        t0 = time.time()
        with open(LOGS / f"stage{s}.log", "a") as f:
            r = subprocess.run([sys.executable, script], cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
        log(f"[stage {s}] rc={r.returncode} in {(time.time()-t0)/60:.1f} min")
        commit(msg + ("" if r.returncode == 0 else " (FAILED - partial)"))
        if r.returncode:
            log(f"[driver] stage {s} failed; stopping"); sys.exit(1)
    log("[driver] requested stages complete")


if __name__ == "__main__":
    main()
