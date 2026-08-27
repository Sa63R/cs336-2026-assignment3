from __future__ import annotations

import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from cs336_scaling.local.database import LocalDatabase
from cs336_scaling.local.settings import LocalSettings


class WorkerStop:
    requested = False

    @classmethod
    def request(cls, _signum: int, _frame: object) -> None:
        cls.requested = True


def acquire_worker_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    file = path.open("a+")
    try:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        file.close()
        raise RuntimeError("another local GPU worker already holds the worker lock")
    file.seek(0)
    file.truncate()
    file.write(f"{os.getpid()}\n")
    file.flush()
    return file


def run_worker(settings: LocalSettings) -> None:
    settings.ensure_directories()
    database = LocalDatabase(settings)
    database.initialize()
    lock_file = acquire_worker_lock(settings.worker_lock_path)
    signal.signal(signal.SIGINT, WorkerStop.request)
    signal.signal(signal.SIGTERM, WorkerStop.request)
    interrupted = database.mark_running_interrupted(
        "worker restarted while experiment was marked running"
    )
    if interrupted:
        print(f"Marked {interrupted} stale running experiment(s) interrupted")
    print(f"Local GPU worker ready; database={settings.database_path}")

    try:
        while not WorkerStop.requested:
            claimed = database.claim_next(os.getpid())
            if claimed is None:
                time.sleep(settings.worker_poll_seconds)
                continue
            claimed.output_dir.mkdir(parents=True, exist_ok=True)
            log_path = claimed.output_dir / "worker.log"
            environment = os.environ.copy()
            environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
            command = [
                sys.executable,
                "-m",
                "cs336_scaling.local.trainer",
                "--experiment-id",
                str(claimed.experiment_id),
            ]
            print(f"Starting experiment {claimed.experiment_id}: {' '.join(command)}")
            with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
                process = subprocess.Popen(
                    command,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    start_new_session=True,
                    text=True,
                )
                process_started_at = time.monotonic()
                cancel_sent_at: float | None = None
                while process.poll() is None:
                    database.heartbeat(claimed.experiment_id, os.getpid())
                    should_stop = WorkerStop.requested or database.cancel_requested(
                        claimed.experiment_id
                    )
                    if should_stop and cancel_sent_at is None:
                        os.killpg(process.pid, signal.SIGINT)
                        cancel_sent_at = time.monotonic()
                    elif (
                        cancel_sent_at is not None
                        and time.monotonic() - cancel_sent_at > 120
                    ):
                        os.killpg(process.pid, signal.SIGKILL)
                    time.sleep(settings.worker_poll_seconds)

            experiment = database.get_experiment(claimed.experiment_id)
            if experiment.status == "running":
                database.finish(
                    claimed.experiment_id,
                    status="failed",
                    used_runtime_seconds=max(
                        experiment.used_runtime_seconds,
                        time.monotonic() - process_started_at,
                    ),
                    error_message=f"trainer exited with code {process.returncode}",
                )
            print(
                f"Experiment {claimed.experiment_id} finished with process code "
                f"{process.returncode} and status "
                f"{database.get_experiment(claimed.experiment_id).status}"
            )
    finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def main() -> None:
    run_worker(LocalSettings.from_env())


if __name__ == "__main__":
    main()
