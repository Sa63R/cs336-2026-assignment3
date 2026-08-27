from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypedDict, cast

import requests

from cs336_scaling.local.schemas import LocalExperimentConfig


class ExperimentConfigSummary(TypedDict):
    budget_group: str


class ExperimentSummary(TypedDict):
    experiment_id: int
    status: str
    config: ExperimentConfigSummary
    attempt: int
    used_runtime_seconds: float
    created_at: str


class BudgetSummary(TypedDict):
    budget_group: str
    used_seconds: float
    reserved_seconds: float
    remaining_seconds: float
    total_seconds: float


class LocalScalingClient:
    def __init__(self, base_url: str | None = None, timeout: float = 30):
        self.base_url = (
            base_url
            or os.environ.get("LOCAL_SCALING_API_URL")
            or "http://127.0.0.1:8765"
        ).rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        # This service is intentionally loopback-only. Corporate/system HTTP proxy
        # variables must never route localhost experiment traffic off the machine.
        self.session.trust_env = False

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self.session.request(
            method,
            f"{self.base_url}{path}",
            timeout=self.timeout,
            **kwargs,
        )
        if not response.ok:
            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            raise RuntimeError(f"API returned HTTP {response.status_code}: {detail}")
        return response.json()

    def submit(self, config_path: Path) -> dict[str, object]:
        config = LocalExperimentConfig.model_validate_json(
            config_path.read_text(encoding="utf-8")
        )
        return self._request(
            "POST",
            "/experiments",
            json={"config": config.model_dump(mode="json")},
        )

    def list_experiments(self, limit: int = 100) -> list[ExperimentSummary]:
        return cast(
            list[ExperimentSummary],
            self._request("GET", "/experiments", params={"limit": limit}),
        )

    def get(self, experiment_id: int) -> dict[str, object]:
        return self._request("GET", f"/experiments/{experiment_id}")

    def metrics(self, experiment_id: int) -> list[dict[str, object]]:
        return self._request("GET", f"/experiments/{experiment_id}/metrics")

    def cancel(self, experiment_id: int) -> dict[str, object]:
        return self._request("POST", f"/experiments/{experiment_id}/cancel")

    def retry(self, experiment_id: int, *, resume: bool) -> dict[str, object]:
        return self._request(
            "POST",
            f"/experiments/{experiment_id}/retry",
            json={"resume": resume},
        )

    def budget(self) -> list[BudgetSummary]:
        return cast(list[BudgetSummary], self._request("GET", "/budget"))
