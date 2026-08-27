from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, status

from cs336_scaling.local.database import (
    DuplicateExperimentError,
    InsufficientBudgetError,
    InvalidStateTransitionError,
    LocalDatabase,
)
from cs336_scaling.local.schemas import (
    BudgetUsage,
    DatasetManifest,
    ExperimentResponse,
    ExperimentStatus,
    LocalExperimentConfig,
    MetricRecord,
    RetryExperimentRequest,
    SubmitExperimentRequest,
    SubmitExperimentResponse,
)
from cs336_scaling.local.settings import LocalSettings


def validate_dataset(config: LocalExperimentConfig) -> DatasetManifest:
    manifest_path = config.dataset_manifest.expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError(f"dataset manifest does not exist: {manifest_path}")
    manifest = DatasetManifest.load(manifest_path)
    if not manifest.train_tokens_path.is_file():
        raise ValueError(
            f"training token file does not exist: {manifest.train_tokens_path}"
        )
    if not manifest.validation_tokens_path.is_file():
        raise ValueError(
            f"validation token file does not exist: {manifest.validation_tokens_path}"
        )
    training = config.training
    if manifest.seed != training.data_seed:
        raise ValueError(
            f"training data_seed {training.data_seed} must match dataset seed "
            f"{manifest.seed}"
        )
    if manifest.vocab_size != training.architecture_config.vocab_size:
        raise ValueError(
            "model vocab_size must match the dataset tokenizer: "
            f"{training.architecture_config.vocab_size} != {manifest.vocab_size}"
        )
    if manifest.train_tokens < training.total_train_tokens + 1:
        raise ValueError(
            f"dataset has {manifest.train_tokens} train tokens but the run needs "
            f"at least {training.total_train_tokens + 1}"
        )
    if manifest.validation_tokens < training.validation_tokens + 1:
        raise ValueError(
            f"dataset has {manifest.validation_tokens} validation tokens but the run "
            f"needs at least {training.validation_tokens + 1}"
        )
    return manifest


def create_app(settings: LocalSettings | None = None) -> FastAPI:
    settings = settings or LocalSettings.from_env()
    database = LocalDatabase(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        database.initialize()
        yield

    app = FastAPI(
        title="CS336 Local Scaling API",
        description=(
            "Single-GPU local experiment queue. SQLite and per-run files are the "
            "source of truth; the training worker is a separate process."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.local_settings = settings
    app.state.local_database = database

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "database": str(settings.database_path),
            "runs_dir": str(settings.runs_dir),
        }

    @app.post(
        "/experiments",
        response_model=SubmitExperimentResponse,
        status_code=status.HTTP_201_CREATED,
    )
    @app.post(
        "/submit",
        response_model=SubmitExperimentResponse,
        status_code=status.HTTP_201_CREATED,
        include_in_schema=False,
    )
    def submit(request: SubmitExperimentRequest) -> SubmitExperimentResponse:
        try:
            manifest = validate_dataset(request.config)
            config_hash = request.config.semantic_hash(manifest.dataset_id)
            experiment_id = database.submit(
                request.config,
                config_hash=config_hash,
                dataset_id=manifest.dataset_id,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc
        except DuplicateExperimentError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "message": str(exc),
                    "existing_experiment_id": exc.experiment_id,
                },
            ) from exc
        except InsufficientBudgetError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": str(exc),
                    "budget": exc.usage.model_dump(mode="json"),
                },
            ) from exc
        return SubmitExperimentResponse(
            experiment_id=experiment_id, config_hash=config_hash
        )

    @app.get("/experiments", response_model=list[ExperimentResponse])
    def list_experiments(
        experiment_status: list[ExperimentStatus] | None = Query(
            default=None, alias="status"
        ),
        limit: int = Query(default=100, ge=1, le=1_000),
    ) -> list[ExperimentResponse]:
        return database.list_experiments(statuses=experiment_status, limit=limit)

    @app.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
    @app.get(
        "/experiment/{experiment_id}",
        response_model=ExperimentResponse,
        include_in_schema=False,
    )
    def get_experiment(experiment_id: int) -> ExperimentResponse:
        try:
            return database.get_experiment(experiment_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="experiment not found"
            ) from exc

    @app.get("/experiments/{experiment_id}/metrics", response_model=list[MetricRecord])
    def get_metrics(experiment_id: int) -> list[MetricRecord]:
        try:
            database.get_experiment(experiment_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="experiment not found"
            ) from exc
        return database.list_metrics(experiment_id)

    @app.post("/experiments/{experiment_id}/cancel", response_model=ExperimentResponse)
    def cancel_experiment(experiment_id: int) -> ExperimentResponse:
        try:
            return database.request_cancel(experiment_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="experiment not found"
            ) from exc
        except InvalidStateTransitionError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(exc)
            ) from exc

    @app.post(
        "/experiments/{experiment_id}/retry",
        response_model=SubmitExperimentResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def retry_experiment(
        experiment_id: int, request: RetryExperimentRequest
    ) -> SubmitExperimentResponse:
        try:
            new_id = database.retry(experiment_id, resume=request.resume)
            experiment = database.get_experiment(new_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="experiment not found"
            ) from exc
        except InvalidStateTransitionError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(exc)
            ) from exc
        except InsufficientBudgetError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": str(exc),
                    "budget": exc.usage.model_dump(mode="json"),
                },
            ) from exc
        return SubmitExperimentResponse(
            experiment_id=new_id, config_hash=experiment.config_hash
        )

    @app.get("/budget", response_model=list[BudgetUsage])
    def budget() -> list[BudgetUsage]:
        return database.all_budget_usage()

    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = LocalSettings.from_env()
    if settings.api_host not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError(
            "the local API has no authentication and must bind to a loopback address"
        )
    uvicorn.run(
        "cs336_scaling.local.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
