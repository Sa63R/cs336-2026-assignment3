from __future__ import annotations

import datetime as dt
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from cs336_scaling.local.schemas import (
    BudgetGroup,
    BudgetUsage,
    ExperimentResponse,
    ExperimentStatus,
    LocalExperimentConfig,
    MetricRecord,
)
from cs336_scaling.local.settings import LocalSettings


ACTIVE_STATUSES = ("queued", "running")
TERMINAL_STATUSES = ("completed", "failed", "cancelled", "interrupted")


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def to_timestamp(value: dt.datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


class DuplicateExperimentError(RuntimeError):
    def __init__(self, experiment_id: int):
        super().__init__(
            f"an experiment with this configuration already exists: {experiment_id}"
        )
        self.experiment_id = experiment_id


class InsufficientBudgetError(RuntimeError):
    def __init__(self, usage: BudgetUsage, requested_seconds: float):
        super().__init__(
            f"budget {usage.budget_group!r} has {usage.remaining_seconds:.1f}s "
            f"remaining but the experiment reserves {requested_seconds:.1f}s"
        )
        self.usage = usage
        self.requested_seconds = requested_seconds


class InvalidStateTransitionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ClaimedExperiment:
    experiment_id: int
    config: LocalExperimentConfig
    output_dir: Path
    resume_from_experiment_id: int | None


class LocalDatabase:
    """SQLite-backed source of truth for local experiments and metrics."""

    def __init__(self, settings: LocalSettings):
        self.settings = settings

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.settings.database_path,
            timeout=30,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.settings.ensure_directories()
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    attempt INTEGER NOT NULL DEFAULT 1,
                    config_hash TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    budget_group TEXT NOT NULL CHECK (budget_group IN ('scaling', 'final')),
                    priority INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL CHECK (
                        status IN ('queued', 'running', 'completed', 'failed',
                                   'cancelled', 'interrupted')
                    ),
                    output_dir TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    heartbeat_at TEXT,
                    worker_pid INTEGER,
                    max_runtime_seconds REAL NOT NULL CHECK (max_runtime_seconds > 0),
                    used_runtime_seconds REAL NOT NULL DEFAULT 0 CHECK (used_runtime_seconds >= 0),
                    cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK (cancel_requested IN (0, 1)),
                    resume_from_experiment_id INTEGER REFERENCES experiments(id),
                    result_json TEXT,
                    error_message TEXT,
                    UNIQUE(config_hash, attempt)
                );

                CREATE INDEX IF NOT EXISTS ix_experiments_queue
                    ON experiments(status, priority DESC, id ASC);
                CREATE INDEX IF NOT EXISTS ix_experiments_budget
                    ON experiments(budget_group, status);
                CREATE INDEX IF NOT EXISTS ix_experiments_config_hash
                    ON experiments(config_hash, attempt DESC);

                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
                    phase TEXT NOT NULL CHECK (phase IN ('train', 'validation', 'system')),
                    optimizer_step INTEGER NOT NULL,
                    tokens_seen INTEGER NOT NULL,
                    values_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(experiment_id, phase, optimizer_step)
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def submit(
        self,
        config: LocalExperimentConfig,
        *,
        config_hash: str,
        dataset_id: str,
    ) -> int:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            duplicate = connection.execute(
                """
                SELECT id FROM experiments
                WHERE config_hash = ?
                  AND status IN ('queued', 'running', 'completed')
                ORDER BY attempt DESC LIMIT 1
                """,
                (config_hash,),
            ).fetchone()
            if duplicate is not None:
                connection.rollback()
                raise DuplicateExperimentError(int(duplicate["id"]))

            usage = self._budget_usage(connection, config.budget_group)
            requested = config.training.max_runtime_seconds
            if requested > usage.remaining_seconds:
                connection.rollback()
                raise InsufficientBudgetError(usage, requested)

            cursor = connection.execute(
                """
                INSERT INTO experiments (
                    attempt, config_hash, config_json, dataset_id, budget_group,
                    priority, status, output_dir, created_at, max_runtime_seconds
                ) VALUES (1, ?, ?, ?, ?, ?, 'queued', '', ?, ?)
                """,
                (
                    config_hash,
                    config.model_dump_json(),
                    dataset_id,
                    config.budget_group,
                    config.priority,
                    to_timestamp(now),
                    requested,
                ),
            )
            if cursor.lastrowid is None:
                connection.rollback()
                raise RuntimeError("SQLite did not return an experiment ID")
            experiment_id = cursor.lastrowid
            output_dir = self.settings.runs_dir / f"{experiment_id:06d}"
            connection.execute(
                "UPDATE experiments SET output_dir = ? WHERE id = ?",
                (str(output_dir), experiment_id),
            )
            self._insert_event(
                connection,
                experiment_id,
                "queued",
                {"config_hash": config_hash, "attempt": 1},
                now,
            )
            connection.commit()
            return experiment_id

    def retry(self, experiment_id: int, *, resume: bool) -> int:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            source = connection.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
            if source is None:
                connection.rollback()
                raise KeyError(experiment_id)
            if source["status"] not in TERMINAL_STATUSES:
                connection.rollback()
                raise InvalidStateTransitionError(
                    f"only terminal experiments can be retried, got {source['status']!r}"
                )

            usage = self._budget_usage(connection, source["budget_group"])
            requested = float(source["max_runtime_seconds"])
            if requested > usage.remaining_seconds:
                connection.rollback()
                raise InsufficientBudgetError(usage, requested)

            attempt_row = connection.execute(
                "SELECT COALESCE(MAX(attempt), 0) AS attempt FROM experiments WHERE config_hash = ?",
                (source["config_hash"],),
            ).fetchone()
            attempt = int(attempt_row["attempt"]) + 1
            cursor = connection.execute(
                """
                INSERT INTO experiments (
                    attempt, config_hash, config_json, dataset_id, budget_group,
                    priority, status, output_dir, created_at, max_runtime_seconds,
                    resume_from_experiment_id
                ) VALUES (?, ?, ?, ?, ?, ?, 'queued', '', ?, ?, ?)
                """,
                (
                    attempt,
                    source["config_hash"],
                    source["config_json"],
                    source["dataset_id"],
                    source["budget_group"],
                    source["priority"],
                    to_timestamp(now),
                    requested,
                    experiment_id if resume else None,
                ),
            )
            if cursor.lastrowid is None:
                connection.rollback()
                raise RuntimeError("SQLite did not return a retry experiment ID")
            new_id = cursor.lastrowid
            output_dir = self.settings.runs_dir / f"{new_id:06d}"
            connection.execute(
                "UPDATE experiments SET output_dir = ? WHERE id = ?",
                (str(output_dir), new_id),
            )
            self._insert_event(
                connection,
                new_id,
                "queued_retry",
                {
                    "attempt": attempt,
                    "source_experiment_id": experiment_id,
                    "resume": resume,
                },
                now,
            )
            connection.commit()
            return new_id

    def claim_next(self, worker_pid: int) -> ClaimedExperiment | None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM experiments
                WHERE status = 'queued' AND cancel_requested = 0
                ORDER BY priority DESC, id ASC
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            updated = connection.execute(
                """
                UPDATE experiments
                SET status = 'running', started_at = ?, heartbeat_at = ?, worker_pid = ?
                WHERE id = ? AND status = 'queued'
                """,
                (to_timestamp(now), to_timestamp(now), worker_pid, row["id"]),
            )
            if updated.rowcount != 1:
                connection.rollback()
                return None
            self._insert_event(
                connection,
                int(row["id"]),
                "running",
                {"worker_pid": worker_pid},
                now,
            )
            connection.commit()
            return ClaimedExperiment(
                experiment_id=int(row["id"]),
                config=LocalExperimentConfig.model_validate_json(row["config_json"]),
                output_dir=Path(row["output_dir"]),
                resume_from_experiment_id=row["resume_from_experiment_id"],
            )

    def heartbeat(self, experiment_id: int, worker_pid: int) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE experiments SET heartbeat_at = ?
                WHERE id = ? AND status = 'running' AND worker_pid = ?
                """,
                (to_timestamp(utc_now()), experiment_id, worker_pid),
            )

    def request_cancel(self, experiment_id: int) -> ExperimentResponse:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError(experiment_id)
            if row["status"] == "queued":
                connection.execute(
                    """
                    UPDATE experiments
                    SET status = 'cancelled', cancel_requested = 1, finished_at = ?
                    WHERE id = ?
                    """,
                    (to_timestamp(now), experiment_id),
                )
                event_type = "cancelled"
            elif row["status"] == "running":
                connection.execute(
                    "UPDATE experiments SET cancel_requested = 1 WHERE id = ?",
                    (experiment_id,),
                )
                event_type = "cancel_requested"
            else:
                connection.rollback()
                raise InvalidStateTransitionError(
                    f"cannot cancel experiment in status {row['status']!r}"
                )
            self._insert_event(connection, experiment_id, event_type, {}, now)
            connection.commit()
        return self.get_experiment(experiment_id)

    def cancel_requested(self, experiment_id: int) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM experiments WHERE id = ?",
                (experiment_id,),
            ).fetchone()
        if row is None:
            raise KeyError(experiment_id)
        return bool(row["cancel_requested"])

    def finish(
        self,
        experiment_id: int,
        *,
        status: ExperimentStatus,
        used_runtime_seconds: float,
        result: dict[str, object] | None = None,
        error_message: str | None = None,
    ) -> None:
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"finish requires a terminal status, got {status!r}")
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError(experiment_id)
            if row["status"] != "running":
                connection.rollback()
                raise InvalidStateTransitionError(
                    f"cannot finish experiment from status {row['status']!r}"
                )
            connection.execute(
                """
                UPDATE experiments
                SET status = ?, finished_at = ?, heartbeat_at = ?, worker_pid = NULL,
                    used_runtime_seconds = ?, result_json = ?, error_message = ?
                WHERE id = ?
                """,
                (
                    status,
                    to_timestamp(now),
                    to_timestamp(now),
                    max(0.0, used_runtime_seconds),
                    json.dumps(result, sort_keys=True) if result is not None else None,
                    error_message,
                    experiment_id,
                ),
            )
            self._insert_event(
                connection,
                experiment_id,
                status,
                {"used_runtime_seconds": used_runtime_seconds, "error": error_message},
                now,
            )
            connection.commit()

    def mark_running_interrupted(self, reason: str) -> int:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT id, started_at, used_runtime_seconds FROM experiments WHERE status = 'running'"
            ).fetchall()
            for row in rows:
                started_at = dt.datetime.fromisoformat(row["started_at"])
                elapsed = max(0.0, (now - started_at).total_seconds())
                connection.execute(
                    """
                    UPDATE experiments
                    SET status = 'interrupted', finished_at = ?, heartbeat_at = ?,
                        worker_pid = NULL, used_runtime_seconds = ?, error_message = ?
                    WHERE id = ?
                    """,
                    (
                        to_timestamp(now),
                        to_timestamp(now),
                        max(float(row["used_runtime_seconds"]), elapsed),
                        reason,
                        row["id"],
                    ),
                )
                self._insert_event(
                    connection, int(row["id"]), "interrupted", {"reason": reason}, now
                )
            connection.commit()
            return len(rows)

    def add_metric(self, metric: MetricRecord) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO metrics (
                    experiment_id, phase, optimizer_step, tokens_seen,
                    values_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(experiment_id, phase, optimizer_step) DO UPDATE SET
                    tokens_seen = excluded.tokens_seen,
                    values_json = excluded.values_json,
                    created_at = excluded.created_at
                """,
                (
                    metric.experiment_id,
                    metric.phase,
                    metric.optimizer_step,
                    metric.tokens_seen,
                    json.dumps(metric.values, sort_keys=True),
                    to_timestamp(metric.created_at),
                ),
            )

    def list_metrics(self, experiment_id: int) -> list[MetricRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM metrics WHERE experiment_id = ?
                ORDER BY optimizer_step ASC, id ASC
                """,
                (experiment_id,),
            ).fetchall()
        return [
            MetricRecord(
                experiment_id=int(row["experiment_id"]),
                phase=row["phase"],
                optimizer_step=int(row["optimizer_step"]),
                tokens_seen=int(row["tokens_seen"]),
                values=json.loads(row["values_json"]),
                created_at=dt.datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def get_experiment(self, experiment_id: int) -> ExperimentResponse:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
        if row is None:
            raise KeyError(experiment_id)
        return self._experiment_from_row(row)

    def list_experiments(
        self,
        *,
        statuses: Sequence[ExperimentStatus] | None = None,
        limit: int = 100,
    ) -> list[ExperimentResponse]:
        query = "SELECT * FROM experiments"
        parameters: list[object] = []
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query += f" WHERE status IN ({placeholders})"
            parameters.extend(statuses)
        query += " ORDER BY id DESC LIMIT ?"
        parameters.append(limit)
        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._experiment_from_row(row) for row in rows]

    def budget_usage(self, budget_group: BudgetGroup) -> BudgetUsage:
        with self.connect() as connection:
            return self._budget_usage(connection, budget_group)

    def all_budget_usage(self) -> list[BudgetUsage]:
        return [self.budget_usage(group) for group in ("scaling", "final")]

    def _budget_usage(
        self, connection: sqlite3.Connection, budget_group: BudgetGroup
    ) -> BudgetUsage:
        row = connection.execute(
            """
            SELECT
                COALESCE(SUM(CASE
                    WHEN status IN ('completed', 'failed', 'cancelled', 'interrupted')
                    THEN MIN(max_runtime_seconds, used_runtime_seconds)
                    ELSE 0 END), 0) AS used_seconds,
                COALESCE(SUM(CASE
                    WHEN status IN ('queued', 'running')
                    THEN max_runtime_seconds
                    ELSE 0 END), 0) AS reserved_seconds
            FROM experiments WHERE budget_group = ?
            """,
            (budget_group,),
        ).fetchone()
        total = (
            self.settings.scaling_budget_seconds
            if budget_group == "scaling"
            else self.settings.final_budget_seconds
        )
        used = float(row["used_seconds"])
        reserved = float(row["reserved_seconds"])
        return BudgetUsage(
            budget_group=budget_group,
            used_seconds=used,
            reserved_seconds=reserved,
            remaining_seconds=max(0.0, total - used - reserved),
            total_seconds=total,
        )

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection,
        experiment_id: int,
        event_type: str,
        payload: dict[str, object],
        created_at: dt.datetime,
    ) -> None:
        connection.execute(
            """
            INSERT INTO events (experiment_id, event_type, payload_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                experiment_id,
                event_type,
                json.dumps(payload, sort_keys=True),
                to_timestamp(created_at),
            ),
        )

    @staticmethod
    def _experiment_from_row(row: sqlite3.Row) -> ExperimentResponse:
        return ExperimentResponse(
            experiment_id=int(row["id"]),
            attempt=int(row["attempt"]),
            config_hash=row["config_hash"],
            status=row["status"],
            config=LocalExperimentConfig.model_validate_json(row["config_json"]),
            dataset_id=row["dataset_id"],
            output_dir=Path(row["output_dir"]),
            created_at=dt.datetime.fromisoformat(row["created_at"]),
            started_at=(
                dt.datetime.fromisoformat(row["started_at"])
                if row["started_at"]
                else None
            ),
            finished_at=(
                dt.datetime.fromisoformat(row["finished_at"])
                if row["finished_at"]
                else None
            ),
            heartbeat_at=(
                dt.datetime.fromisoformat(row["heartbeat_at"])
                if row["heartbeat_at"]
                else None
            ),
            worker_pid=row["worker_pid"],
            used_runtime_seconds=float(row["used_runtime_seconds"]),
            cancel_requested=bool(row["cancel_requested"]),
            resume_from_experiment_id=row["resume_from_experiment_id"],
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            error_message=row["error_message"],
        )
