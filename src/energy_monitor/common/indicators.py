from typing import Any, Final

import numpy as np
import pandas as pd

from energy_monitor.common.schema import GRID_STEP

HOURS_PER_READING: Final = GRID_STEP / pd.Timedelta(hours=1)
WATTS_PER_KILOWATT: Final = 1000.0
MIN_IDLE_READINGS: Final = 3


def energy_kwh(power: pd.Series | np.ndarray) -> float:
    return float(
        np.nansum(np.asarray(power, dtype=np.float64)) * HOURS_PER_READING / WATTS_PER_KILOWATT
    )


def idle_mask(
    frame: pd.DataFrame, threshold: float, minimum: int = MIN_IDLE_READINGS
) -> np.ndarray:
    ordered = frame.sort_values(["host_id", "ts"], kind="stable")
    below = (ordered["cpu_load"] < threshold).to_numpy()
    host = ordered["host_id"].astype(str).to_numpy()
    new_run = np.r_[True, (below[1:] != below[:-1]) | (host[1:] != host[:-1])]
    run_id = np.cumsum(new_run)
    run_length = pd.Series(run_id).map(pd.Series(run_id).value_counts()).to_numpy()
    idle = below & (run_length >= minimum)
    return pd.Series(idle, index=ordered.index).reindex(frame.index).to_numpy()


def compute_indicators(
    frame: pd.DataFrame, tariff_kwh: float, idle_threshold: float
) -> dict[str, Any]:
    idle = idle_mask(frame, idle_threshold)
    data = pd.DataFrame(
        {
            "host_id": frame["host_id"].astype(str).to_numpy(),
            "power": frame["power"].to_numpy(np.float64),
            "idle": idle,
        }
    )
    data["energy"] = data["power"] * HOURS_PER_READING / WATTS_PER_KILOWATT
    data["idle_energy"] = data["energy"].where(data["idle"], 0.0)
    per_host = data.groupby("host_id").agg(
        energy_kwh=("energy", "sum"),
        idle_energy_kwh=("idle_energy", "sum"),
        idle_time_share=("idle", "mean"),
        readings=("power", "size"),
    )
    per_host["associated_cost"] = per_host["energy_kwh"] * tariff_kwh
    per_host["idle_associated_cost"] = per_host["idle_energy_kwh"] * tariff_kwh
    total_energy = float(data["energy"].sum())
    total_idle = float(data["idle_energy"].sum())
    return {
        "tariff_kwh": tariff_kwh,
        "idle_threshold": idle_threshold,
        "total": {
            "energy_kwh": total_energy,
            "idle_energy_kwh": total_idle,
            "idle_energy_share": total_idle / total_energy if total_energy else 0.0,
            "idle_time_share": float(data["idle"].mean()) if len(data) else 0.0,
            "associated_cost": total_energy * tariff_kwh,
            "idle_associated_cost": total_idle * tariff_kwh,
            "readings": int(len(data)),
        },
        "per_host": per_host.round(6).to_dict(orient="index"),
    }
