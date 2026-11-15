from typing import Any

import pandas as pd

SERIES_COLUMNS = ("power", "expected", "upper", "forecast_1h", "forecast_24h")
SERIES_LABELS = {
    "power": "Real",
    "expected": "Esperado",
    "upper": "Limite superior",
    "forecast_1h": "Previsto (1 h)",
    "forecast_24h": "Previsto (24 h)",
}
ALERT_LABELS = {
    "host_id": "Servidor",
    "status": "Situação",
    "start_ts": "Início",
    "end_ts": "Fim",
    "peak_z": "z máximo",
    "unexplained_energy_kwh": "Consumo não explicado (kWh)",
    "justification": "Justificativa",
}


def to_local(values: pd.Series, timezone: str) -> pd.Series:
    return pd.to_datetime(values, utc=True).dt.tz_convert(timezone)


def series_frame(payload: list[dict[str, Any]], timezone: str) -> pd.DataFrame:
    frame = pd.DataFrame(payload, columns=["ts", *SERIES_COLUMNS])
    frame["ts"] = to_local(frame["ts"], timezone)
    return frame.set_index("ts")[list(SERIES_COLUMNS)].rename(columns=SERIES_LABELS)


def alerts_frame(payload: list[dict[str, Any]], timezone: str) -> pd.DataFrame:
    frame = pd.DataFrame(payload, columns=list(ALERT_LABELS))
    for column in ("start_ts", "end_ts"):
        frame[column] = to_local(frame[column], timezone)
    return frame.sort_values("start_ts", ascending=False).rename(columns=ALERT_LABELS)


def total_forecast_frame(payload: list[dict[str, Any]], timezone: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        payload, columns=["horizon", "target_ts", "prediction", "hosts_included", "hosts_left_out"]
    )
    frame = frame.dropna(subset=["prediction"])
    frame["target_ts"] = to_local(frame["target_ts"], timezone)
    return frame.pivot_table(index="target_ts", columns="horizon", values="prediction")


def servers_frame(payload: list[dict[str, Any]], timezone: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        payload, columns=["host_id", "last_ts", "last_power", "last_z", "open_alert", "readings"]
    )
    frame["last_ts"] = to_local(frame["last_ts"], timezone)
    return frame.rename(
        columns={
            "host_id": "Servidor",
            "last_ts": "Última leitura",
            "last_power": "Potência (W)",
            "last_z": "z",
            "open_alert": "Alerta aberto",
            "readings": "Leituras",
        }
    )
