from __future__ import annotations

from datetime import timedelta

from .schemas import SeriesRow


STEP = timedelta(minutes=15)


def generate_complete_grid(rows: list[SeriesRow]) -> list[dict]:
    """Return a per-series 15-minute grid with explicit missing flags."""
    by_series: dict[str, dict] = {}
    for row in rows:
        by_series.setdefault(row.series_id, {})[row.bucket_15m] = row

    grid: list[dict] = []
    for series_id, indexed in sorted(by_series.items()):
        if not indexed:
            continue
        current = min(indexed)
        end = max(indexed)
        while current <= end:
            row = indexed.get(current)
            grid.append({
                "series_id": series_id,
                "bucket_15m": current,
                "records": row.records if row else None,
                "energy_wh": row.energy_wh if row else None,
                "cfp_g": row.cfp_g if row else None,
                "is_missing_row": row is None,
                "hour": current.hour,
                "day_of_week": current.weekday(),
                "is_weekend": current.weekday() >= 5,
            })
            current += STEP
    return grid
