"""
dirac_sim.api.schemas
=====================
Pydantic schemas shared between the REST server and the forecast client.

These schemas mirror the challenge submission format for Task A so that
a solver trained on Task A can directly serve Task B via the same REST API.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

try:
    from pydantic import BaseModel, Field
    _HAS_PYDANTIC = True
except ImportError:
    # Lightweight fallback so the module can be imported without pydantic.
    # The REST server requires pydantic; the simulator core does not.
    _HAS_PYDANTIC = False
    from dataclasses import dataclass as _dc, field as _field

    class BaseModel:  # type: ignore
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def model_dump(self): return self.__dict__
        def model_dump_json(self): import json; return json.dumps(self.__dict__, default=str)

    def Field(default=None, **_):  # type: ignore
        return default


# ------------------------------------------------------------------ #
# Forecast request / response                                         #
# ------------------------------------------------------------------ #
class ForecastRequest(BaseModel):
    """Request body for POST /forecast."""
    series_ids: List[str] = Field(
        ..., description="List of series_id values to forecast."
    )
    reference_timestamp_utc: datetime = Field(
        ..., description="Current time t; forecast is for t+h."
    )
    horizons: List[str] = Field(
        default=["1h", "24h"],
        description="Forecast horizons to return.",
    )


class ForecastRecord(BaseModel):
    """Single forecast row — matches Task A submission schema."""
    series_id: str
    forecast_timestamp_utc: datetime
    horizon_steps_15m: int = Field(
        ..., description="4 for 1h-ahead, 96 for 24h-ahead."
    )
    energy_wh_pred: float
    cfp_g_pred: float
    energy_wh_lower: Optional[float] = None   # optional prediction interval
    energy_wh_upper: Optional[float] = None
    cfp_g_lower: Optional[float] = None
    cfp_g_upper: Optional[float] = None


class ForecastResponse(BaseModel):
    """Response body for POST /forecast."""
    reference_timestamp_utc: datetime
    forecasts: List[ForecastRecord]
    model_version: str = "unknown"
    latency_ms: Optional[float] = None


# ------------------------------------------------------------------ #
# Scheduling request / response                                       #
# ------------------------------------------------------------------ #
class JobSpec(BaseModel):
    """Minimal job spec accepted by POST /schedule."""
    job_id: str
    series_id: str
    arrival_time: datetime
    deadline: datetime
    cpu_minutes: float
    memory_gb: float = 2.0
    site_whitelist: List[str] = []
    priority: str = "normal"


class ScheduleRequest(BaseModel):
    """Request body for POST /schedule (used by real DIRAC/SLURM integrations)."""
    jobs: List[JobSpec]
    current_time_utc: datetime
    declared_objective: str = "carbon"
    delta_energy_budget: float = 0.10
    delta_carbon_budget: float = 0.10
    delta_makespan_budget: float = 0.10


class DispatchDecisionOut(BaseModel):
    job_id: str
    site_id: str
    dispatch_at: datetime
    rationale: str = ""


class ScheduleResponse(BaseModel):
    """Response body for POST /schedule."""
    current_time_utc: datetime
    decisions: List[DispatchDecisionOut]
    declared_objective: str
    n_held: int = 0    # jobs deferred to future green window


# ------------------------------------------------------------------ #
# Health / status                                                      #
# ------------------------------------------------------------------ #
class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = False
    last_forecast_at: Optional[datetime] = None
    version: str = "0.1.0"
