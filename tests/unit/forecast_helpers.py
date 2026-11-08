import numpy as np
import pandas as pd

GRID_START = pd.Timestamp("2019-03-01 00:00:00", tz="UTC")


def make_grid(
    slots: int = 400,
    hosts: tuple[str, ...] = ("h1", "h2"),
    seed: int = 7,
) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    frames = []
    moments = pd.date_range(GRID_START, periods=slots, freq="10min", unit="us")
    for offset, host in enumerate(hosts):
        position = np.arange(slots)
        cycle = 20 * np.sin(2 * np.pi * position / 144)
        cpu = 10 + 5 * np.sin(2 * np.pi * position / 144) + generator.normal(0, 0.1, slots)
        frames.append(
            pd.DataFrame(
                {
                    "host_id": host,
                    "ts": moments,
                    "power": 150 + 30 * offset + cycle + generator.normal(0, 1, slots),
                    "observed": True,
                    "quality_mask": 0,
                    "cpu_load": cpu,
                    "mem_pct": 50.0,
                    "n_vms": 5.0,
                    "cores": 128.0,
                    "cores_used": 60.0,
                    "net_rx": 1000.0 + position,
                    "net_tx": 500.0 + position,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
