import numpy as np
import pandas as pd

from energy_monitor.common.expected_features import prepare_expected_rows
from tests.unit.forecast_helpers import make_grid


def make_expected_rows(slots: int = 600, hosts: tuple[str, ...] = ("h1", "h2")) -> pd.DataFrame:
    grid = make_grid(slots, hosts)
    grid["power"] = 100 + 3 * grid["cpu_load"] + np.where(grid["host_id"].eq("h2"), 40.0, 0.0)
    return prepare_expected_rows(grid, list(hosts))
