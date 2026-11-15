from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import yaml

from energy_monitor.common.schema import GRID_STEP

OFFSET: Final = "offset"
DRIFT: Final = "drift"
STUCK: Final = "stuck"
ZERO_LOAD_HIGH_POWER: Final = "zero_load_high_power"
LEGIT_PEAK: Final = "legit_peak"
ANOMALY_KINDS: Final = (OFFSET, DRIFT, STUCK, ZERO_LOAD_HIGH_POWER, LEGIT_PEAK)
NEGATIVE_KINDS: Final = frozenset({LEGIT_PEAK})
MIN_GAP_SLOTS: Final = 6
MAX_CPU_LOAD: Final = 100.0
MIN_CPU_SLOPE: Final = 0.01
ZEROED_LOAD: Final = ("cpu_load", "cores_used", "n_vms")


@dataclass(frozen=True)
class InjectedEvent:
    host_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    kind: str
    magnitude: float
    slots: int

    @property
    def is_anomaly(self) -> bool:
        return self.kind not in NEGATIVE_KINDS

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["start"], values["end"] = self.start.isoformat(), self.end.isoformat()
        return values


@dataclass(frozen=True)
class InjectionPlan:
    kinds: tuple[str, ...]
    magnitudes: tuple[float, ...]
    durations_slots: tuple[int, ...]
    count: int
    seed: int
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class HostReference:
    mean_power: pd.Series
    cpu_slope: pd.Series


def compute_host_reference(frame: pd.DataFrame) -> HostReference:
    data = pd.DataFrame(
        {
            "host": frame["host_id"].astype(str).to_numpy(),
            "power": frame["power"].to_numpy(np.float64),
            "cpu": frame["cpu_load"].to_numpy(np.float64),
        }
    )
    grouped = data.groupby("host")
    covariance = grouped.apply(lambda host: host["power"].cov(host["cpu"]), include_groups=False)
    slope = covariance / grouped["cpu"].var()
    fallback = slope[slope > MIN_CPU_SLOPE].median()
    slope = slope.where(slope > MIN_CPU_SLOPE, fallback if pd.notna(fallback) else 1.0)
    return HostReference(mean_power=grouped["power"].mean(), cpu_slope=slope)


def overlaps(occupied: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(
        start <= other_end + MIN_GAP_SLOTS and end + MIN_GAP_SLOTS >= other_start
        for other_start, other_end in occupied
    )


def plan_events(hosts: Sequence[str], plan: InjectionPlan) -> list[InjectedEvent]:
    generator = np.random.default_rng(plan.seed)
    total_slots = int((plan.end - plan.start) / GRID_STEP)
    occupied: dict[str, list[tuple[int, int]]] = {host: [] for host in hosts}
    events: list[InjectedEvent] = []
    for _ in range(plan.count * 100):
        if len(events) == plan.count:
            break
        host = str(generator.choice(list(hosts)))
        kind = str(generator.choice(plan.kinds))
        magnitude = float(generator.choice(plan.magnitudes))
        slots = int(generator.choice(plan.durations_slots))
        if total_slots <= slots:
            continue
        first = int(generator.integers(0, total_slots - slots))
        last = first + slots - 1
        if overlaps(occupied[host], first, last):
            continue
        occupied[host].append((first, last))
        start = plan.start + first * GRID_STEP
        events.append(
            InjectedEvent(host, start, start + (slots - 1) * GRID_STEP, kind, magnitude, slots)
        )
    return sorted(events, key=lambda event: (event.host_id, event.start))


def host_values(frame: pd.DataFrame) -> np.ndarray:
    return frame["host_id"].astype(str).to_numpy()


def event_mask(
    frame: pd.DataFrame, event: InjectedEvent, hosts: np.ndarray | None = None
) -> np.ndarray:
    hosts = host_values(frame) if hosts is None else hosts
    within = (frame["ts"] >= event.start) & (frame["ts"] <= event.end)
    return (hosts == event.host_id) & within.to_numpy()


def apply_event(
    frame: pd.DataFrame,
    event: InjectedEvent,
    reference: HostReference,
    hosts: np.ndarray | None = None,
) -> None:
    rows = np.flatnonzero(event_mask(frame, event, hosts))
    if len(rows) == 0:
        return
    power = frame.columns.get_loc("power")
    increase = event.magnitude * float(reference.mean_power[event.host_id])
    if event.kind == OFFSET:
        frame.iloc[rows, power] += increase
    elif event.kind == DRIFT:
        frame.iloc[rows, power] += increase * np.linspace(1 / len(rows), 1, len(rows))
    elif event.kind == STUCK:
        frame.iloc[rows, power] = frame.iloc[rows[0], power]
    elif event.kind == ZERO_LOAD_HIGH_POWER:
        frame.iloc[rows, power] += increase
        for column in ZEROED_LOAD:
            frame.iloc[rows, frame.columns.get_loc(column)] = 0.0
    elif event.kind == LEGIT_PEAK:
        cpu = frame.columns.get_loc("cpu_load")
        extra_cpu = increase / float(reference.cpu_slope[event.host_id])
        frame.iloc[rows, power] += increase
        frame.iloc[rows, cpu] = np.minimum(frame.iloc[rows, cpu] + extra_cpu, MAX_CPU_LOAD)
    else:
        raise ValueError(f"Tipo de anomalia desconhecido: {event.kind}")


def apply_events(
    frame: pd.DataFrame, events: Sequence[InjectedEvent], reference: HostReference
) -> pd.DataFrame:
    injected = frame.copy()
    hosts = host_values(injected)
    for event in events:
        apply_event(injected, event, reference, hosts)
    return injected


def label_rows(frame: pd.DataFrame, events: Sequence[InjectedEvent]) -> np.ndarray:
    labels = np.zeros(len(frame), dtype=bool)
    hosts = host_values(frame)
    for event in events:
        labels |= event_mask(frame, event, hosts)
    return labels


def load_injection_plan(path: Path) -> InjectionPlan:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    return InjectionPlan(
        kinds=tuple(values["kinds"]),
        magnitudes=tuple(float(value) for value in values["magnitudes"]),
        durations_slots=tuple(int(value) for value in values["durations_slots"]),
        count=int(values["count"]),
        seed=int(values["seed"]),
        start=pd.Timestamp(values["start"]),
        end=pd.Timestamp(values["end"]),
    )
