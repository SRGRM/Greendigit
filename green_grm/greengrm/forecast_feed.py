"""Forecasts served to the Task B scheduler (Sect. 4.6).

For each site the Task A models produce, from the features of an origin
bucket o, a 1-hour forecast for o + 4 and a 24-hour forecast for o + 96 (the
two horizons of ``POST /forecast``). The telemetry of bucket o is complete
only when the bucket ends, so these forecasts are issued at the next tick,
o + 15 min (``issue_delay_steps = 1``); the scheduler therefore never sees a
forecast that uses the bucket in progress.

``ForecastFeed`` precomputes these forecasts for a whole simulation window
so the simulator does not have to call a REST service. ``OracleFeed``
issues the measured signals instead of forecasts; it is used for the
oracle run of the forecast-to-dispatch value gap of the challenge
specification.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd

from .data import STEP


def bucket_key(ts) -> str:
    """Key of the 15-minute bucket that contains ``ts`` (UTC)."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC").floor("15min").strftime("%Y-%m-%dT%H:%M")


class ForecastFeed:
    """Forecasts indexed by the tick at which they are issued."""

    name = "green_grm_forecasts"

    def __init__(self, table: pd.DataFrame, issue_delay_steps: int = 1):
        """``table`` has the columns series_id, origin, forecast_timestamp_utc,
        horizon_steps_15m, energy_wh_pred and cfp_g_pred."""
        if issue_delay_steps < 0:
            raise ValueError("issue_delay_steps must be >= 0")
        self.issue_delay = issue_delay_steps * STEP
        self.table = table.sort_values(["origin", "series_id", "horizon_steps_15m"]).reset_index(drop=True)
        self._by_tick: Dict[str, List[Dict]] = {}
        for row in self.table.itertuples(index=False):
            target = pd.Timestamp(row.forecast_timestamp_utc)
            issued = pd.Timestamp(row.origin) + self.issue_delay
            self._by_tick.setdefault(bucket_key(issued), []).append({
                "series_id": row.series_id,
                "forecast_timestamp_utc": target.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                "horizon_steps_15m": int(row.horizon_steps_15m),
                "energy_wh_pred": float(row.energy_wh_pred),
                "cfp_g_pred": float(row.cfp_g_pred),
            })

    @classmethod
    def from_model(cls, model, grid: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                   warmup: pd.Timedelta = pd.Timedelta(days=1),
                   issue_delay_steps: int = 1) -> "ForecastFeed":
        """Forecasts of every origin whose issue tick lies in (start - warmup, end]."""
        delay = issue_delay_steps * STEP
        table = model.predict_forecasts(grid, start - warmup - delay, end - delay)
        return cls(table, issue_delay_steps)

    def issued_at(self, now: datetime) -> List[Dict]:
        """Records issued at tick ``now`` (1-hour and 24-hour forecasts of all sites)."""
        return self._by_tick.get(bucket_key(now), [])

    def issued_between(self, start: datetime, end: datetime) -> List[Dict]:
        """Records issued at the ticks in [start, end); used to fill the
        scheduler's timeline before the first simulated tick."""
        records: List[Dict] = []
        tick = pd.Timestamp(start)
        while tick < pd.Timestamp(end):
            records.extend(self.issued_at(tick))
            tick += STEP
        return records


class OracleFeed(ForecastFeed):
    """Measured signals presented as forecasts (upper bound of Task B)."""

    name = "oracle_signals"

    @classmethod
    def from_grid(cls, grid: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                  horizons=(4, 96), warmup: pd.Timedelta = pd.Timedelta(days=1),
                  issue_delay_steps: int = 1) -> "OracleFeed":
        """Measured values of the buckets the forecasts would cover, issued on
        the same schedule as ``ForecastFeed``."""
        observed = grid[grid["observed"].astype(bool)]
        delay = issue_delay_steps * STEP
        rows = []
        for horizon in horizons:
            shifted = observed.assign(origin=observed["bucket_15m"] - horizon * STEP)
            issued = shifted["origin"] + delay
            shifted = shifted[(issued > start - warmup) & (issued <= end)]
            rows.append(pd.DataFrame({
                "series_id": shifted["series_id"].to_numpy(),
                "origin": shifted["origin"].to_numpy(),
                "forecast_timestamp_utc": shifted["bucket_15m"].to_numpy(),
                "horizon_steps_15m": horizon,
                "energy_wh_pred": shifted["energy_wh"].to_numpy(),
                "cfp_g_pred": shifted["cfp_g"].to_numpy(),
            }))
        return cls(pd.concat(rows, ignore_index=True), issue_delay_steps)


def tick_range(start: datetime, end: datetime, step: timedelta = timedelta(minutes=15)):
    tick = start
    while tick <= end:
        yield tick
        tick += step
