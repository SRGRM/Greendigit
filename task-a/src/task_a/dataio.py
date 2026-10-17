from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from .schemas import CANONICAL_INPUT_COLUMNS, SeriesRow, parse_timestamp, require_columns


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_data_path() -> Path:
    return repository_root() / "data" / "raw_metrics" / "summary_sites_15m.csv"


def _to_float(row: dict, column: str) -> float:
    value = row.get(column, "")
    if value == "":
        raise ValueError(f"empty numeric value in column {column}")
    return float(value)


def load_series_csv(path: str | Path, aggregate_duplicates: bool = True) -> list[SeriesRow]:
    """Load canonical Task A rows.

    The official schema uses ``series_id``. The bundled root data currently
    stores anonymized sites as ``site_id`` and one row per VO, so this loader
    maps ``site_id`` to ``series_id`` and aggregates duplicate site/bucket rows.
    """
    path = Path(path)
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")

        fields = set(reader.fieldnames)
        series_col = "series_id" if "series_id" in fields else "site_id"
        required = ["bucket_15m", "records", "energy_wh", "cfp_g", series_col]
        require_columns(fields, required, str(path))

        grouped: dict[tuple[str, object], dict[str, float]] = defaultdict(
            lambda: {"records": 0.0, "energy_wh": 0.0, "cfp_g": 0.0}
        )
        seen: set[tuple[str, object]] = set()
        for row_number, row in enumerate(reader, start=2):
            series_id = row[series_col].strip()
            bucket = parse_timestamp(row["bucket_15m"])
            if bucket.minute % 15 != 0 or bucket.second or bucket.microsecond:
                raise ValueError(f"{path}:{row_number} bucket_15m is not 15-minute aligned")
            key = (series_id, bucket)
            if key in seen and not aggregate_duplicates:
                raise ValueError(f"{path}:{row_number} duplicate (series_id, bucket_15m): {key}")
            seen.add(key)
            grouped[key]["records"] += _to_float(row, "records")
            grouped[key]["energy_wh"] += _to_float(row, "energy_wh")
            grouped[key]["cfp_g"] += _to_float(row, "cfp_g")

    rows = [
        SeriesRow(
            series_id=series_id,
            bucket_15m=bucket,
            records=values["records"],
            energy_wh=values["energy_wh"],
            cfp_g=values["cfp_g"],
        )
        for (series_id, bucket), values in grouped.items()
    ]
    return sorted(rows, key=lambda item: (item.series_id, item.bucket_15m))


def write_series_csv(rows: list[SeriesRow], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CANONICAL_INPUT_COLUMNS)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item.series_id, item.bucket_15m)):
            writer.writerow({
                "series_id": row.series_id,
                "bucket_15m": row.bucket_15m.isoformat(),
                "records": row.records,
                "energy_wh": row.energy_wh,
                "cfp_g": row.cfp_g,
            })


def split_temporal(
    rows: list[SeriesRow],
    cutoff_utc: str = "2026-02-18T14:00:00+00:00",
) -> tuple[list[SeriesRow], list[SeriesRow]]:
    cutoff = parse_timestamp(cutoff_utc)
    train = [row for row in rows if row.bucket_15m <= cutoff]
    test = [row for row in rows if row.bucket_15m > cutoff]
    return train, test
