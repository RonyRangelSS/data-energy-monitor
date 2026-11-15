from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from energy_monitor.models.detection.alerts import run_boundaries
from energy_monitor.replayer.injection import InjectedEvent, event_mask, host_values, label_rows


def harmonic_mean(first: float, second: float) -> float:
    return 2 * first * second / (first + second) if first + second > 0 else 0.0


def alert_runs(
    frame: pd.DataFrame, active: np.ndarray
) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    groups = frame["host_id"].astype(str).to_numpy()
    timestamps = frame["ts"]
    return [
        (groups[start], timestamps.iloc[start], timestamps.iloc[end])
        for start, end in run_boundaries(groups, active)
    ]


def run_touches_event(run: tuple[str, pd.Timestamp, pd.Timestamp], event: InjectedEvent) -> bool:
    host, start, end = run
    return host == event.host_id and start <= event.end and end >= event.start


def event_detection(
    frame: pd.DataFrame,
    active: np.ndarray,
    event: InjectedEvent,
    hosts: np.ndarray | None = None,
) -> tuple[bool, float | None]:
    rows = event_mask(frame, event, hosts)
    alerted = rows & active
    if not alerted.any():
        return False, None
    first = frame["ts"].to_numpy()[np.flatnonzero(alerted)[0]]
    return True, float((pd.Timestamp(first) - event.start) / pd.Timedelta(minutes=1))


def pointwise_scores(labels: np.ndarray, active: np.ndarray) -> dict[str, float]:
    true_positive = float((labels & active).sum())
    precision = true_positive / active.sum() if active.sum() else 0.0
    recall = true_positive / labels.sum() if labels.sum() else 0.0
    return {
        "point_precision": precision,
        "point_recall": recall,
        "point_f1": harmonic_mean(precision, recall),
    }


def evaluate_detection(
    frame: pd.DataFrame, active: np.ndarray, events: Sequence[InjectedEvent]
) -> dict[str, Any]:
    runs = alert_runs(frame, active)
    hosts = host_values(frame)
    outcomes = [event_detection(frame, active, event, hosts) for event in events]
    detected = sum(found for found, _ in outcomes)
    delays = [delay for found, delay in outcomes if found]
    true_runs = sum(any(run_touches_event(run, event) for event in events) for run in runs)
    precision = true_runs / len(runs) if runs else 0.0
    recall = detected / len(events) if events else 0.0
    return {
        "events": len(events),
        "detected_events": int(detected),
        "alert_episodes": len(runs),
        "false_alert_episodes": len(runs) - int(true_runs),
        "precision": precision,
        "recall": recall,
        "f1": harmonic_mean(precision, recall),
        "mean_delay_minutes": float(np.mean(delays)) if delays else None,
        **pointwise_scores(label_rows(frame, events), active),
    }


def evaluate_negative_control(
    frame: pd.DataFrame, active: np.ndarray, peaks: Sequence[InjectedEvent]
) -> dict[str, Any]:
    hosts = host_values(frame)
    alerted = [event_detection(frame, active, peak, hosts)[0] for peak in peaks]
    runs = alert_runs(frame, active)
    return {
        "legit_peaks": len(peaks),
        "peaks_with_alert": int(sum(alerted)),
        "false_positive_rate_on_peaks": float(np.mean(alerted)) if peaks else 0.0,
        "alert_episodes": len(runs),
    }


def episodes_per_host_day(frame: pd.DataFrame, active: np.ndarray) -> float:
    days = (frame["ts"].max() - frame["ts"].min()) / pd.Timedelta(days=1)
    hosts = frame["host_id"].astype(str).nunique()
    return len(alert_runs(frame, active)) / (hosts * days) if days > 0 else 0.0
