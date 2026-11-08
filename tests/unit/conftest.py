from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest


def make_row(
    host: str, time: str, power: float | None = 100.0, cpu: float = 10.0
) -> dict[str, Any]:
    return {
        "Series": host,
        "Time": time,
        "Power": power,
        "CPU_Load": cpu,
        "Ram": 64.0,
        "Ram_Used": 32.0,
        "No_Of_Running_vms": 5.0,
        "CPU_cores": 64.0,
        "CPU_cores_used": 40.0,
        "Network_RX": 1000.0,
        "Network_TX": 500.0,
        "CPU1_Temperature": 40.0,
        "CPU2_Temperature": 41.0,
    }


RawFrameFactory = Callable[[list[dict[str, Any]]], pd.DataFrame]


@pytest.fixture
def raw_frame() -> RawFrameFactory:
    return pd.DataFrame


@pytest.fixture
def row() -> Callable[..., dict[str, Any]]:
    return make_row
