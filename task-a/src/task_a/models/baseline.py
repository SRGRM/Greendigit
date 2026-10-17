from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from task_a.schemas import ForecastRow, SeriesRow, parse_timestamp


@dataclass
class BaselineForecaster:
    by_series_time: dict[str, dict[str, tuple[float, float]]]
    latest_by_series: dict[str, tuple[str, float, float]]
    means_by_series: dict[str, tuple[float, float]]
    global_mean: tuple[float, float]

    @classmethod
    def fit(cls, rows: list[SeriesRow]) -> "BaselineForecaster":
        by_series_time: dict[str, dict[str, tuple[float, float]]] = {}
        latest_by_series: dict[str, tuple[str, float, float]] = {}
        totals: dict[str, list[float]] = {}
        global_values = [0.0, 0.0, 0.0]

        for row in sorted(rows, key=lambda item: (item.series_id, item.bucket_15m)):
            ts = row.bucket_15m.isoformat()
            by_series_time.setdefault(row.series_id, {})[ts] = (row.energy_wh, row.cfp_g)
            latest_by_series[row.series_id] = (ts, row.energy_wh, row.cfp_g)
            totals.setdefault(row.series_id, [0.0, 0.0, 0.0])
            totals[row.series_id][0] += row.energy_wh
            totals[row.series_id][1] += row.cfp_g
            totals[row.series_id][2] += 1.0
            global_values[0] += row.energy_wh
            global_values[1] += row.cfp_g
            global_values[2] += 1.0

        means = {
            series_id: (values[0] / values[2], values[1] / values[2])
            for series_id, values in totals.items()
            if values[2]
        }
        if global_values[2]:
            global_mean = (global_values[0] / global_values[2], global_values[1] / global_values[2])
        else:
            global_mean = (0.0, 0.0)
        return cls(by_series_time, latest_by_series, means, global_mean)

    def predict_one(self, series_id: str, forecast_timestamp_utc: datetime) -> tuple[float, float]:
        ts = parse_timestamp(forecast_timestamp_utc)
        series = self.by_series_time.get(series_id, {})
        for candidate in (
            ts - timedelta(days=1),
            ts - timedelta(hours=1),
        ):
            found = series.get(candidate.isoformat())
            if found is not None:
                return found
        latest = self.latest_by_series.get(series_id)
        if latest is not None:
            return latest[1], latest[2]
        return self.means_by_series.get(series_id, self.global_mean)

    def predict(
        self,
        series_ids: list[str],
        forecast_origin: datetime,
        horizons: list[int] | tuple[int, ...] = (4, 96),
    ) -> list[ForecastRow]:
        origin = parse_timestamp(forecast_origin)
        output = []
        for series_id in series_ids:
            for horizon in horizons:
                forecast_ts = origin + timedelta(minutes=15 * int(horizon))
                energy, cfp = self.predict_one(series_id, forecast_ts)
                output.append(ForecastRow(series_id, forecast_ts, int(horizon), energy, cfp))
        return output

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(asdict(self), fh)

    @classmethod
    def load(cls, path: str | Path) -> "BaselineForecaster":
        with Path(path).open() as fh:
            data = json.load(fh)
        return cls(
            by_series_time={
                series_id: {ts: tuple(values) for ts, values in values_by_ts.items()}
                for series_id, values_by_ts in data["by_series_time"].items()
            },
            latest_by_series={
                series_id: tuple(values)
                for series_id, values in data["latest_by_series"].items()
            },
            means_by_series={
                series_id: tuple(values)
                for series_id, values in data["means_by_series"].items()
            },
            global_mean=tuple(data["global_mean"]),
        )
