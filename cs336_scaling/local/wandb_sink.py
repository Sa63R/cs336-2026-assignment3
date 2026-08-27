from __future__ import annotations

import os
from pathlib import Path

from cs336_scaling.local.schemas import LocalExperimentConfig, MetricRecord


class SafeWandbSink:
    """Best-effort W&B mirror that never controls training correctness."""

    def __init__(
        self,
        *,
        config: LocalExperimentConfig,
        experiment_id: int,
        attempt: int,
        output_dir: Path,
    ):
        self.run = None
        mode = config.wandb.mode
        if mode == "disabled":
            return
        if mode == "online" and not os.environ.get("WANDB_API_KEY"):
            print("W&B online mode disabled because WANDB_API_KEY is not set")
            return
        try:
            import wandb

            settings = wandb.Settings(disable_git=True)
            self.run = wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity,
                group=config.wandb.group,
                name=f"local-{experiment_id}-attempt-{attempt}",
                id=f"local-{experiment_id}-attempt-{attempt}",
                mode=mode,
                dir=output_dir,
                config=config.model_dump(mode="json"),
                save_code=False,
                settings=settings,
            )
        except Exception as exc:
            print(f"W&B initialization failed; continuing locally: {exc}")
            self.run = None

    def log(self, metric: MetricRecord) -> None:
        if self.run is None:
            return
        try:
            self.run.log(
                {
                    "optimizer_step": metric.optimizer_step,
                    "tokens_seen": metric.tokens_seen,
                    **{
                        f"{metric.phase}/{name}": value
                        for name, value in metric.values.items()
                    },
                }
            )
        except Exception as exc:
            print(f"W&B metric upload failed; local record is intact: {exc}")

    def finish(self, *, exit_code: int = 0) -> None:
        if self.run is None:
            return
        try:
            self.run.finish(exit_code=exit_code)
        except Exception as exc:
            print(f"W&B finish failed; local record is intact: {exc}")
