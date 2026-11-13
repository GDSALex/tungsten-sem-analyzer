"""Run the grouped cross-validation jobs, two at a time, resumably.

Launched from the Windows Task Scheduler (hidden window) so that the session's background
CPU throttling does not apply.  A job is skipped when its done.json exists, so the queue
can be stopped and restarted at any time.  Create a file named STOP in the logs folder to
let the running jobs finish and stop launching new ones.
"""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import rev_common as RC  # noqa: E402

PY = sys.executable
LOGS = RC.REV / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
MAX_PARALLEL = 2

JOBS = []
for cfg, scheme, folds in [("A", "edge5_r1", 5), ("A", "loso", 3), ("B", "edge5_r1", 5),
                           ("A", "edge5_r2", 5), ("C", "edge5_r1", 5), ("D", "edge5_r1", 5)]:
    for f in range(folds):
        JOBS.append((cfg, scheme, f))


def done(job):
    cfg, scheme, f = job
    return (RC.CV_RUNS / f"{cfg}_{scheme}_f{f}" / "done.json").exists()


def main():
    RC.set_low_priority_off()
    running = {}
    pending = [j for j in JOBS if not done(j)]
    with open(LOGS / "queue.log", "a", encoding="utf-8") as q:
        q.write(f"{time.ctime()} queue start, {len(pending)} pending\n")
        q.flush()
        while pending or running:
            for job, proc in list(running.items()):
                if proc.poll() is not None:
                    q.write(f"{time.ctime()} finished {job} rc={proc.returncode}\n")
                    q.flush()
                    del running[job]
            stop = (LOGS / "STOP").exists()
            while pending and len(running) < MAX_PARALLEL and not stop:
                job = pending.pop(0)
                cfg, scheme, f = job
                log = open(LOGS / f"{cfg}_{scheme}_f{f}.log", "a", encoding="utf-8")
                proc = subprocess.Popen(
                    [PY, str(HERE / "train_cv.py"), "--config", cfg, "--scheme", scheme,
                     "--fold", str(f), "--workers", "5"],
                    stdout=log, stderr=subprocess.STDOUT, cwd=str(HERE),
                    env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1",
                         "PYTHONIOENCODING": "utf-8"})
                running[job] = proc
                q.write(f"{time.ctime()} started {job} pid={proc.pid}\n")
                q.flush()
                time.sleep(20)   # stagger the start-up (worker spawn, cudnn autotune)
            if stop and not running:
                break
            time.sleep(15)
        q.write(f"{time.ctime()} queue end\n")


if __name__ == "__main__":
    main()
