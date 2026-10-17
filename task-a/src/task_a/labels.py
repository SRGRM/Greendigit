from __future__ import annotations

from .schemas import SeriesRow


def valid_signal_label(row: dict | SeriesRow) -> int:
    if isinstance(row, SeriesRow):
        missing = False
        records = row.records
        energy = row.energy_wh
        cfp = row.cfp_g
    else:
        missing = bool(row.get("is_missing_row", False))
        records = row.get("records")
        energy = row.get("energy_wh")
        cfp = row.get("cfp_g")
    if missing or records is None or energy is None or cfp is None:
        return 0
    if records < 0 or energy < 0 or cfp < 0:
        return 0
    if records > 0 and energy == 0 and cfp == 0:
        return 0
    return 1


def peak_threshold(rows: list[SeriesRow], quantile: float = 0.95) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    by_series: dict[str, list[float]] = {}
    for row in rows:
        by_series.setdefault(row.series_id, []).append(max(row.energy_wh, row.cfp_g))
    for series_id, values in by_series.items():
        ordered = sorted(values)
        if not ordered:
            thresholds[series_id] = 0.0
            continue
        index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
        thresholds[series_id] = ordered[index]
    return thresholds


def peak_label(row: SeriesRow, thresholds: dict[str, float]) -> int:
    threshold = thresholds.get(row.series_id, 0.0)
    return int(max(row.energy_wh, row.cfp_g) >= threshold)
