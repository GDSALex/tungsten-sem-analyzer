"""Runs arch_train.py jobs, at most two at a time and within a GPU memory budget.

    python arch_queue.py jobs.txt      # one job per line: <variant> <scheme> <fold>

Control files in arch_runs/:
    PAUSE   no new job starts while it exists (running jobs finish normally)
    STOP    no new job starts; the queue exits when the running jobs end
Finished jobs (done.json present) are skipped, so the queue can be restarted at any time.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rev_common as RC  # noqa: E402

RUNS = RC.REV / "arch_runs"
PY = sys.executable
TRAIN = str(Path(__file__).parent / "arch_train.py")
BUDGET_GB = 10.0          # 12 GB card, about 1.4 GB taken by the desktop, some margin
CONTEXT_GB = 0.7          # CUDA context and workspace not counted by the caching allocator
MAX_JOBS = 2


def need_gb(variant: str) -> float:
    est = json.loads((RUNS / "memory_estimates.json").read_text())
    return est.get(variant, 6.0) + CONTEXT_GB


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(RUNS / "queue.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> None:
    jobs = [ln.split() for ln in Path(sys.argv[1]).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8",
               MPLBACKEND="Agg")
    running: list[tuple[subprocess.Popen, list, float, float]] = []
    pending = [j for j in jobs if not (RUNS / j[0] / f"{j[1]}_f{j[2]}" / "done.json").exists()]
    log(f"queue start: {len(jobs)} jobs, {len(pending)} pending")
    RC.set_low_priority_off()
    while pending or running:
        for item in list(running):
            p, job, gb, t0 = item
            if p.poll() is not None:
                running.remove(item)
                log(f"end   {' '.join(job)}  exit {p.returncode}  {(time.time() - t0) / 60:.1f} min")
        stop = (RUNS / "STOP").exists()
        if stop and not running:
            log("STOP file found, queue ends")
            break
        if not stop and not (RUNS / "PAUSE").exists() and len(running) < MAX_JOBS:
            used = sum(gb for _, _, gb, _ in running)
            for job in pending:
                gb = need_gb(job[0])
                if used + gb <= BUDGET_GB or not running:   # a large job runs alone
                    out = RUNS / job[0] / f"{job[1]}_f{job[2]}"
                    out.mkdir(parents=True, exist_ok=True)
                    fh = open(out / "train.log", "w", encoding="utf-8")
                    p = subprocess.Popen([PY, TRAIN, "--variant", job[0], "--scheme", job[1],
                                          "--fold", job[2]], stdout=fh, stderr=subprocess.STDOUT,
                                         env=env, cwd=str(Path(TRAIN).parent))
                    running.append((p, job, gb, time.time()))
                    pending.remove(job)
                    log(f"start {' '.join(job)}  ({gb:.1f} GB, {len(pending)} left)")
                    break
        time.sleep(20)
    log("queue finished")


if __name__ == "__main__":
    main()
