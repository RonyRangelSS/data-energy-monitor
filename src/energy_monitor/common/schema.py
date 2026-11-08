from enum import IntFlag
from typing import Final

import pandas as pd

SOURCE_TIMEZONE: Final = "Australia/Melbourne"
GRID_FREQUENCY: Final = "10min"
GRID_STEP: Final = pd.Timedelta(minutes=10)
RAW_TIME_FORMAT: Final = "%Y-%m-%d %H:%M:%S"
MIN_CPU_LOAD: Final = 0.0
MAX_CPU_LOAD: Final = 100.0

COLUMN_MAP: Final[dict[str, str]] = {
    "Series": "host_id",
    "Time": "ts_local",
    "Power": "power",
    "CPU_Load": "cpu_load",
    "Ram": "ram",
    "Ram_Used": "ram_used",
    "No_Of_Running_vms": "n_vms",
    "CPU_cores": "cores",
    "CPU_cores_used": "cores_used",
    "Network_RX": "net_rx",
    "Network_TX": "net_tx",
    "CPU1_Temperature": "cpu1_temp",
    "CPU2_Temperature": "cpu2_temp",
    "Inlet_Temperature": "inlet_temp",
    "Fan_speed1": "fan_speed1",
    "Fan_speed2": "fan_speed2",
    "Fan_speed3": "fan_speed3",
    "Fan_speed4": "fan_speed4",
}

OPTIONAL_RAW_COLUMNS: Final = frozenset(
    {"Inlet_Temperature", "Fan_speed1", "Fan_speed2", "Fan_speed3", "Fan_speed4"}
)
REQUIRED_RAW_COLUMNS: Final = frozenset(COLUMN_MAP) - OPTIONAL_RAW_COLUMNS
COLUMNS_DROPPED_FROM_GRID: Final = ("ts_local", "ram", "ram_used")
NOT_REPORTED_AS_MISSING: Final = frozenset({"host_id", "ts_local", "ts", "quality_mask"})


class QualityFlag(IntFlag):
    DUPLICATE = 1
    NON_POSITIVE_POWER = 2
    CPU_OUT_OF_RANGE = 4
    MISSING_TARGET = 8
    OFF_GRID = 16
