from dataclasses import asdict, dataclass
from typing import Any, Final

import numpy as np
import pandas as pd

from energy_monitor.common.schema import GRID_STEP

HOURS_PER_READING: Final = GRID_STEP / pd.Timedelta(hours=1)
WATTS_PER_KILOWATT: Final = 1000.0


@dataclass(frozen=True)
class AlertEpisode:
    host_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    readings: int
    peak_score: float
    unexplained_energy_kwh: float
    justification: str

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["start"] = self.start.isoformat()
        values["end"] = self.end.isoformat()
        return values


def unexplained_energy_kwh(power: np.ndarray, upper: np.ndarray) -> float:
    excess = np.clip(power - upper, 0, None)
    return float(excess.sum() * HOURS_PER_READING / WATTS_PER_KILOWATT)


def describe_episode(excess_percent: float, cpu_percent: float, minutes: int) -> str:
    return (
        f"potência {excess_percent:.0f}% acima do esperado para CPU de "
        f"{cpu_percent:.0f}%, durante {minutes} min"
    )


def run_boundaries(groups: np.ndarray, active: np.ndarray) -> list[tuple[int, int]]:
    starts = active & ~np.r_[False, active[:-1] & (groups[1:] == groups[:-1])]
    ends = active & ~np.r_[active[1:] & (groups[1:] == groups[:-1]), False]
    return list(zip(np.flatnonzero(starts).tolist(), np.flatnonzero(ends).tolist(), strict=True))


def summarize_episode(rows: pd.DataFrame, scores: np.ndarray) -> AlertEpisode:
    power = rows["power"].to_numpy(np.float64)
    expected = rows["expected"].to_numpy(np.float64)
    excess = float(np.mean((power - expected) / expected) * 100)
    minutes = int(len(rows) * GRID_STEP / pd.Timedelta(minutes=1))
    return AlertEpisode(
        host_id=str(rows["host_id"].iloc[0]),
        start=rows["ts"].iloc[0],
        end=rows["ts"].iloc[-1],
        readings=int(len(rows)),
        peak_score=float(np.nanmax(scores)),
        unexplained_energy_kwh=unexplained_energy_kwh(power, rows["upper"].to_numpy(np.float64)),
        justification=describe_episode(excess, float(rows["cpu_load"].mean()), minutes),
    )


def extract_episodes(
    frame: pd.DataFrame, active: np.ndarray, scores: np.ndarray
) -> list[AlertEpisode]:
    groups = frame["host_id"].astype(str).to_numpy()
    return [
        summarize_episode(frame.iloc[start : end + 1], scores[start : end + 1])
        for start, end in run_boundaries(groups, active)
    ]
