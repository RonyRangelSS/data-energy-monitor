from pydantic import BaseModel, Field

SCENARIO_PATTERN = r"^[A-Za-z0-9_.-]{1,100}$"


class ReadingIn(BaseModel):
    host_id: str = Field(min_length=1, max_length=64)
    ts: str
    power: float | None = None
    cpu_load: float | None = None
    mem_pct: float | None = None
    n_vms: float | None = None
    cores: float | None = None
    cores_used: float | None = None
    net_rx: float | None = None
    net_tx: float | None = None
    cpu1_temp: float | None = None
    cpu2_temp: float | None = None
    inlet_temp: float | None = None


class ReadingBatch(BaseModel):
    scenario: str = Field(pattern=SCENARIO_PATTERN)
    readings: list[ReadingIn] = Field(min_length=1, max_length=5000)


class RejectedReading(BaseModel):
    index: int
    host_id: str
    ts: str
    reasons: list[str]


class IngestionResponse(BaseModel):
    accepted: int
    duplicates: int
    rejected: list[RejectedReading]
    forecast_instants: int
    alerts_opened: int
    alerts_closed: int
