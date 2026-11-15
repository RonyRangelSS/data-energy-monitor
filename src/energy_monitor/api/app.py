import secrets
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session, sessionmaker

from energy_monitor.api import queries
from energy_monitor.api.forecasting import forecast_instants
from energy_monitor.api.ingestion import ingest_batch
from energy_monitor.api.operational import OperationalModels, load_operational_models
from energy_monitor.api.schemas import SCENARIO_PATTERN, IngestionResponse, ReadingBatch
from energy_monitor.api.structured_logging import configure_logging, get_logger
from energy_monitor.common.settings import Settings, get_settings
from energy_monitor.database.session import create_db_engine, create_session_factory

DEFAULT_IDLE_THRESHOLD = 10.0
ScenarioQuery = Annotated[str, Query(pattern=SCENARIO_PATTERN)]
OptionalMoment = Annotated[datetime | None, Query()]


def get_session(request: Request) -> Iterator[Session]:
    factory: sessionmaker = request.app.state.session_factory
    with factory() as session:
        yield session


def get_models(request: Request) -> OperationalModels:
    return request.app.state.models


def require_api_key(request: Request, x_api_key: Annotated[str | None, Header()] = None) -> None:
    expected = request.app.state.settings.api_key.get_secret_value()
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        get_logger().warning("api_key_rejected", extra={"path": request.url.path})
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Chave de API ausente ou inválida")


SessionDependency = Annotated[Session, Depends(get_session)]
ModelsDependency = Annotated[OperationalModels, Depends(get_models)]


def create_app(
    settings: Settings | None = None,
    models: OperationalModels | None = None,
    engine: Engine | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    engine = engine or create_db_engine(settings.database_url)
    if models is None:
        models = load_operational_models(
            settings.operational_dataset, Path(settings.models_dir), Path(settings.experiments_dir)
        )
    app = FastAPI(title="Monitoramento do consumo energético", version="1.0.0")
    app.state.settings = settings
    app.state.models = models
    app.state.session_factory = create_session_factory(engine)
    register_routes(app)
    return app


def register_routes(app: FastAPI) -> None:
    @app.get("/health")
    def health(session: SessionDependency) -> dict[str, str]:
        session.execute(text("SELECT 1"))
        return {"status": "ok"}

    @app.post("/readings", dependencies=[Depends(require_api_key)])
    def post_readings(
        batch: ReadingBatch, session: SessionDependency, models: ModelsDependency
    ) -> IngestionResponse:
        try:
            result = ingest_batch(session, models, batch.scenario, batch.readings)
            forecasts = forecast_instants(session, models, batch.scenario, result.instants)
            session.commit()
        except Exception:
            session.rollback()
            get_logger().exception("ingestion_failed", extra={"scenario": batch.scenario})
            raise
        return IngestionResponse(
            accepted=result.accepted,
            duplicates=result.duplicates,
            rejected=result.rejected,
            forecast_instants=forecasts,
            alerts_opened=result.alerts.opened,
            alerts_closed=result.alerts.closed,
        )

    @app.get("/servers")
    def get_servers(scenario: ScenarioQuery, session: SessionDependency) -> list[dict[str, Any]]:
        return queries.list_servers(session, scenario)

    @app.get("/servers/{host_id}/series")
    def get_series(
        host_id: str,
        scenario: ScenarioQuery,
        session: SessionDependency,
        start: OptionalMoment = None,
        end: OptionalMoment = None,
    ) -> list[dict[str, Any]]:
        return queries.server_series(session, scenario, host_id, start, end)

    @app.get("/alerts")
    def get_alerts(
        scenario: ScenarioQuery,
        session: SessionDependency,
        alert_status: Annotated[str | None, Query(alias="status")] = None,
        host_id: str | None = None,
        start: OptionalMoment = None,
        end: OptionalMoment = None,
    ) -> list[dict[str, Any]]:
        alerts = queries.list_alerts(session, scenario, alert_status, host_id, start, end)
        return [alert_payload(alert) for alert in alerts]

    @app.get("/indicators")
    def get_indicators(
        request: Request,
        scenario: ScenarioQuery,
        session: SessionDependency,
        start: OptionalMoment = None,
        end: OptionalMoment = None,
        tariff: Annotated[float | None, Query(gt=0)] = None,
        idle_threshold: Annotated[float, Query(gt=0, le=100)] = DEFAULT_IDLE_THRESHOLD,
    ) -> dict[str, Any]:
        rate = tariff if tariff is not None else request.app.state.settings.tariff_kwh
        return queries.indicators(session, scenario, start, end, rate, idle_threshold)

    @app.get("/forecasts")
    def get_forecasts(
        scenario: ScenarioQuery,
        session: SessionDependency,
        host_id: str | None = None,
        horizon: str | None = None,
        start: OptionalMoment = None,
        end: OptionalMoment = None,
        limit: Annotated[int, Query(gt=0, le=5000)] = 500,
    ) -> list[dict[str, Any]]:
        forecasts = queries.list_forecasts(session, scenario, host_id, horizon, start, end, limit)
        return [forecast_payload(item) for item in forecasts]

    @app.get("/models")
    def get_models_description(models: ModelsDependency) -> dict[str, Any]:
        return models.describe()

    @app.delete("/scenarios/{name}", dependencies=[Depends(require_api_key)])
    def delete_scenario(name: str, session: SessionDependency) -> dict[str, Any]:
        removed = queries.delete_scenario(session, name)
        session.commit()
        get_logger().info("scenario_deleted", extra={"scenario": name, "removed": removed})
        return {"scenario": name, "removed": removed}


def alert_payload(alert: Any) -> dict[str, Any]:
    fields = (
        "id",
        "host_id",
        "status",
        "start_ts",
        "end_ts",
        "last_ts",
        "readings",
        "peak_z",
        "unexplained_energy_kwh",
        "justification",
        "detector",
        "expected_model",
    )
    return {name: getattr(alert, name) for name in fields}


def forecast_payload(forecast: Any) -> dict[str, Any]:
    fields = (
        "host_id",
        "horizon",
        "origin_ts",
        "target_ts",
        "status",
        "prediction",
        "lower",
        "upper",
        "hosts_included",
        "hosts_left_out",
        "model",
        "variant",
    )
    return {name: getattr(forecast, name) for name in fields}
