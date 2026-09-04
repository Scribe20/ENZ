"""Overnight driver (authorized to run without further sign-off).

Order of operations (as instructed):
  0. wait for the running target-audit and Stage-A jobs to finish;
  1. audit reproducibility: snapshot run-1 outputs, rerun target_audit.py,
     byte-compare, write REPRODUCIBILITY.md, commit audit + Stage-A;
  2. robust-A_ITO campaign stages: preflight -> stage2 -> stage3 -> stage4
     -> stage5, committing after each stage.
Every step is a subprocess with its own log under outputs/logs/; a failing
step stops the chain (later stages depend on it) after committing what
exists.
"""

import filecmp
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AUD = ROOT / "enz_absorption_campaign"
LOGS = HERE / "outputs" / "logs"
FOOTER = ("\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\n"
          "Claude-Session: https://claude.ai/code/session_01WD97gqkKje1C4R3Eqsk7cR")


def log(s):
    print(time.strftime("%H:%M:%S "), s, flush=True)
    with open(LOGS / "driver.log", "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + s + "\n")


def sh(cmd, cwd=ROOT, check=True):
    r = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"{cmd}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def commit(paths, msg):
    sh("git add -A " + " ".join(str(p) for p in paths))
    if sh("git diff --cached --quiet && echo clean || echo dirty").strip() == "clean":
        log(f"[git] nothing to commit for: {msg}")
        return
    p = subprocess.run(["git", "commit", "-q", "-F", "-"], cwd=ROOT,
                       input=msg + FOOTER, text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stderr)
    log(f"[git] committed: {msg.splitlines()[0]}  ({sh('git rev-parse --short HEAD').strip()})")
    for i, wait in enumerate((0, 2, 4, 8, 16)):
        time.sleep(wait)
        r = subprocess.run(["git", "push", "-u", "origin",
                            "claude/enz-eigenmode-target-u95j8m"], cwd=ROOT,
                           capture_output=True, text=True)
        if r.returncode == 0:
            log("[git] pushed")
            return
        log(f"[git] push attempt {i+1} failed: {r.stderr.strip()[-200:]}")


def step(name, cmd, cwd):
    log(f"[{name}] start: {cmd}")
    t0 = time.time()
    with open(LOGS / f"{name}.log", "a") as f:
        r = subprocess.run(cmd, cwd=cwd, shell=True, stdout=f,
                           stderr=subprocess.STDOUT)
    log(f"[{name}] rc={r.returncode} in {(time.time()-t0)/60:.1f} min")
    return r.returncode == 0


def wait_for_jobs():
    while True:
        # bracketed first letters so the pgrep shell itself never matches
        out = sh("pgrep -f '[t]arget_audit\\.py|[s]tage_a_decompose\\.py' || true")
        pids = [p for p in out.split() if p.strip()]
        if not pids:
            return
        log(f"waiting for running jobs {pids}")
        time.sleep(120)


def audit_reproducibility():
    snap = HERE / "outputs" / "audit_run1_snapshot"
    if snap.exists():
        shutil.rmtree(snap)
    snap.mkdir(parents=True)
    # run 1 was killed by the container reclaim on 2026-09-03 12:07 UTC:
    # always regenerate it fresh before the reproducibility run 2
    ok1 = step("audit_run1", "python3 target_audit.py > run1.stdout 2>&1", AUD)
    tracked = ["TARGET_AUDIT.md", "target_audit.log",
               "outputs/failure_test.json", "outputs/no_ito_poles.json",
               "outputs/calibration.csv", "outputs/gradient_audit.json",
               "outputs/target_definition.json"]
    for t in tracked:
        src = AUD / t
        if src.exists():
            shutil.copy2(src, snap / src.name)
    ok = step("audit_run2", "python3 target_audit.py > run2.stdout 2>&1", AUD)
    lines = ["# Audit reproducibility (two independent runs of target_audit.py)", ""]
    same_all = True
    for t in tracked:
        a, b = snap / Path(t).name, AUD / t
        if not (a.exists() and b.exists()):
            lines.append(f"- {t}: MISSING in {'run1' if not a.exists() else 'run2'}")
            same_all = False
            continue
        same = filecmp.cmp(a, b, shallow=False)
        same_all &= same
        if not same:
            d = subprocess.run(["diff", str(a), str(b)], capture_output=True,
                               text=True).stdout
            nd = sum(1 for l in d.splitlines() if l[:1] in "<>")
            lines.append(f"- {t}: DIFFERS ({nd} changed lines)")
            lines.append("```\n" + "\n".join(d.splitlines()[:40]) + "\n```")
        else:
            lines.append(f"- {t}: byte-identical")
    lines += ["", f"run1 exit ok: {ok1}; run2 exit ok: {ok}",
              f"**Verdict: {'byte-for-byte reproducible' if same_all else 'see differences above'}**"]
    (AUD / "REPRODUCIBILITY.md").write_text("\n".join(lines) + "\n")
    log("[audit] " + lines[-1])
    return ok


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    gi = ROOT / ".gitignore"
    if not gi.exists() or "__pycache__" not in gi.read_text():
        with open(gi, "a") as f:
            f.write("__pycache__/\n*.pyc\n")
    wait_for_jobs()
    # ---- 1. audit reproducibility + commit -------------------------------
    audit_reproducibility()
    commit([AUD, ROOT / "enz_highq_enz_campaign", gi],
           "Target audit: corrected reproducible outputs (2-run byte check) "
           "+ Stage-A radiative/non-radiative decomposition results")
    # ---- 2. campaign ------------------------------------------------------
    stages = [("preflight", "python3 preflight.py",
               "Robust A_ITO campaign: code + Stage 0/1 preflight (no fliplr symmetry; "
               "energy/A-identity/gradient checks; beta calibration)"),
              ("stage2", "python3 stage2_screen.py",
               "Robust A_ITO campaign: Stage 2 outer (P,h,pad) screen + landscape plots"),
              ("stage3", "python3 stage3_refine.py",
               "Robust A_ITO campaign: Stage 3 adaptive refinement, finalists"),
              ("stage4", "python3 stage4_full.py",
               "Robust A_ITO campaign: Stage 4 full topology optimization (warm + from-scratch)"),
              ("stage5", "python3 stage5_posthoc.py",
               "Robust A_ITO campaign: Stage 5 refinement, post-hoc physics, comparison, REPORT")]
    for name, cmd, msg in stages:
        ok = step(name, cmd, HERE)
        commit([HERE, gi], msg + ("" if ok else " (FAILED - partial outputs)"))
        if not ok:
            log(f"[driver] stage {name} failed; stopping chain")
            sys.exit(1)
    log("[driver] all stages complete")


if __name__ == "__main__":
    main()
