from __future__ import annotations

import csv
import math
from pathlib import Path

from .schemas import (
    DETECTION_COLUMNS,
    FORECAST_COLUMNS,
    PEAK_COLUMNS,
    DetectionRow,
    ForecastRow,
    PeakRow,
    format_timestamp,
    parse_timestamp,
    require_columns,
)


def write_forecasts(rows: list[ForecastRow], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FORECAST_COLUMNS)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (
            item.forecast_timestamp_utc, item.series_id, item.horizon_steps_15m
        )):
            writer.writerow({
                "series_id": row.series_id,
                "forecast_timestamp_utc": format_timestamp(row.forecast_timestamp_utc),
                "horizon_steps_15m": row.horizon_steps_15m,
                "energy_wh_pred": row.energy_wh_pred,
                "cfp_g_pred": row.cfp_g_pred,
            })


def validate_forecast_csv(path: str | Path) -> int:
    path = Path(path)
    seen = set()
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        require_columns(reader.fieldnames, FORECAST_COLUMNS, str(path))
        count = 0
        for row_number, row in enumerate(reader, start=2):
            horizon = int(row["horizon_steps_15m"])
            if horizon not in {4, 96}:
                raise ValueError(f"{path}:{row_number} unsupported horizon_steps_15m: {horizon}")
            timestamp = parse_timestamp(row["forecast_timestamp_utc"])
            key = (row["series_id"], timestamp.isoformat(), horizon)
            if key in seen:
                raise ValueError(f"{path}:{row_number} duplicate forecast key: {key}")
            seen.add(key)
            energy = _finite_float(row["energy_wh_pred"], path, row_number, "energy_wh_pred")
            cfp = _finite_float(row["cfp_g_pred"], path, row_number, "cfp_g_pred")
            if energy < 0 or cfp < 0:
                raise ValueError(f"{path}:{row_number} predictions must be non-negative")
            count += 1
    if count == 0:
        raise ValueError(f"{path} contains no forecast rows")
    return count


def write_detections(rows: list[DetectionRow], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=DETECTION_COLUMNS)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item.bucket_15m, item.series_id)):
            writer.writerow({
                "series_id": row.series_id,
                "bucket_15m": format_timestamp(row.bucket_15m),
                "valid_signal_score": row.valid_signal_score,
                "valid_signal_pred": row.valid_signal_pred,
            })


def write_peaks(rows: list[PeakRow], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PEAK_COLUMNS)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item.bucket_15m, item.series_id)):
            writer.writerow({
                "series_id": row.series_id,
                "bucket_15m": format_timestamp(row.bucket_15m),
                "peak_score": row.peak_score,
                "peak_pred": row.peak_pred,
            })


def validate_detection_csv(path: str | Path) -> int:
    return _validate_binary_csv(path, DETECTION_COLUMNS, "valid_signal_score", "valid_signal_pred")


def validate_peak_csv(path: str | Path) -> int:
    return _validate_binary_csv(path, PEAK_COLUMNS, "peak_score", "peak_pred")


def _finite_float(value: str, path: Path, row_number: int, column: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{path}:{row_number} invalid numeric value in {column}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{path}:{row_number} non-finite numeric value in {column}: {value!r}")
    return parsed


def _validate_binary_csv(
    path: str | Path,
    columns: list[str],
    score_column: str,
    pred_column: str,
) -> int:
    path = Path(path)
    seen = set()
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        require_columns(reader.fieldnames, columns, str(path))
        count = 0
        for row_number, row in enumerate(reader, start=2):
            timestamp = parse_timestamp(row["bucket_15m"])
            key = (row["series_id"], timestamp.isoformat())
            if key in seen:
                raise ValueError(f"{path}:{row_number} duplicate key: {key}")
            seen.add(key)
            score = _finite_float(row[score_column], path, row_number, score_column)
            if score < 0.0 or score > 1.0:
                raise ValueError(f"{path}:{row_number} {score_column} must be between 0 and 1")
            pred = int(row[pred_column])
            if pred not in {0, 1}:
                raise ValueError(f"{path}:{row_number} {pred_column} must be 0 or 1")
            count += 1
    if count == 0:
        raise ValueError(f"{path} contains no rows")
    return count
