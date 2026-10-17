from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable


CANONICAL_INPUT_COLUMNS = [
    "series_id",
    "bucket_15m",
    "records",
    "energy_wh",
    "cfp_g",
]

FORECAST_COLUMNS = [
    "series_id",
    "forecast_timestamp_utc",
    "horizon_steps_15m",
    "energy_wh_pred",
    "cfp_g_pred",
]

DETECTION_COLUMNS = [
    "series_id",
    "bucket_15m",
    "valid_signal_score",
    "valid_signal_pred",
]

PEAK_COLUMNS = [
    "series_id",
    "bucket_15m",
    "peak_score",
    "peak_pred",
]


@dataclass(frozen=True)
class SeriesRow:
    series_id: str
    bucket_15m: datetime
    records: float
    energy_wh: float
    cfp_g: float


@dataclass(frozen=True)
class ForecastRow:
    series_id: str
    forecast_timestamp_utc: datetime
    horizon_steps_15m: int
    energy_wh_pred: float
    cfp_g_pred: float


@dataclass(frozen=True)
class DetectionRow:
    series_id: str
    bucket_15m: datetime
    valid_signal_score: float
    valid_signal_pred: int


@dataclass(frozen=True)
class PeakRow:
    series_id: str
    bucket_15m: datetime
    peak_score: float
    peak_pred: int


def parse_timestamp(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def format_timestamp(value: datetime) -> str:
    return parse_timestamp(value).isoformat()


def require_columns(actual: Iterable[str], required: Iterable[str], context: str) -> None:
    actual_set = set(actual)
    missing = [col for col in required if col not in actual_set]
    if missing:
        raise ValueError(f"{context} is missing required columns: {', '.join(missing)}")
