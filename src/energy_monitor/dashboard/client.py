from typing import Any

import httpx


class MonitorClient:
    def __init__(
        self, base_url: str, transport: httpx.BaseTransport | None = None, timeout: float = 10.0
    ) -> None:
        self._client = httpx.Client(base_url=base_url, transport=transport, timeout=timeout)

    def _get(self, path: str, **parameters: Any) -> Any:
        clean = {key: value for key, value in parameters.items() if value is not None}
        response = self._client.get(path, params=clean)
        response.raise_for_status()
        return response.json()

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def models(self) -> dict[str, Any]:
        return self._get("/models")

    def servers(self, scenario: str) -> list[dict[str, Any]]:
        return self._get("/servers", scenario=scenario)

    def series(
        self, scenario: str, host_id: str, start: str | None = None, end: str | None = None
    ) -> list[dict[str, Any]]:
        return self._get(f"/servers/{host_id}/series", scenario=scenario, start=start, end=end)

    def alerts(self, scenario: str, status: str | None = None) -> list[dict[str, Any]]:
        return self._get("/alerts", scenario=scenario, status=status)

    def indicators(
        self, scenario: str, tariff: float | None = None, idle_threshold: float | None = None
    ) -> dict[str, Any]:
        return self._get(
            "/indicators", scenario=scenario, tariff=tariff, idle_threshold=idle_threshold
        )

    def forecasts(
        self,
        scenario: str,
        host_id: str | None = None,
        horizon: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        return self._get(
            "/forecasts", scenario=scenario, host_id=host_id, horizon=horizon, limit=limit
        )
