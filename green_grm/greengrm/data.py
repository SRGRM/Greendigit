"""Task A data: loading, the per-site 15-minute grid, labels and splits.

The GreenDIGIT EIMPS telemetry is published as one row per site, virtual
organisation (VO) and 15-minute bucket (``data/raw_metrics/summary_sites_15m.csv``).
Task A works on site-level series, so rows are summed per (site, bucket),
exactly as the official ``task_a.dataio.load_series_csv`` does. The loader
also accepts the canonical ``task_a_series.csv`` and the granular event file
``summary_sites_raw.csv`` (available from the challenge VRE), which is
floored to 15-minute buckets first.

Missing buckets are kept explicit: every site gets a regular 15-minute grid
and buckets without telemetry hold NaN values. The gradient boosting models
handle NaN natively, so a gap is never confused with a zero reading.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .utils import DataLoadError, get_logger

STEP = pd.Timedelta(minutes=15)
SIGNAL_COLUMNS = ("energy_wh", "cfp_g")
WORKLOAD_COLUMNS = ("records", "work", "ncores")
OFFICIAL_CUTOFF = "2026-02-18T14:00:00+00:00"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _parse_utc(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True)
    if parsed.isna().any():
        raise DataLoadError(f"{int(parsed.isna().sum())} timestamps could not be parsed")
    return parsed


def load_site_buckets(path: Path, sites: Optional[Sequence[str]] = None,
                      chunk_size: int = 1_000_000) -> pd.DataFrame:
    """Load a GreenDIGIT telemetry file and return one row per (site, bucket).

    Returns the columns ``series_id, bucket_15m, records, energy_wh, cfp_g``
    plus ``work`` and ``ncores`` when the file provides them.
    """
    logger = get_logger()
    path = Path(path)
    if not path.is_file():
        raise DataLoadError(f"data file not found: {path}")

    header = pd.read_csv(path, nrows=0).columns.tolist()
    series_col = "series_id" if "series_id" in header else "site_id"
    if series_col not in header:
        raise DataLoadError(f"{path} has neither a 'series_id' nor a 'site_id' column")
    missing = [c for c in SIGNAL_COLUMNS if c not in header]
    if missing:
        raise DataLoadError(f"{path} is missing required columns: {missing}")

    raw_events = "event_start_timestamp" in header and "bucket_15m" not in header
    time_col = "event_start_timestamp" if raw_events else "bucket_15m"
    if time_col not in header:
        raise DataLoadError(f"{path} has no 'bucket_15m' (or 'event_start_timestamp') column")

    numeric = [c for c in ("records",) + SIGNAL_COLUMNS + ("work", "ncores") if c in header]
    usecols = [series_col, time_col] + numeric
    logger.info("[data] reading %s (%s file)", path.name,
                "event-level" if raw_events else "15-minute summary")

    parts: List[pd.DataFrame] = []
    n_rows = 0
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_size):
        n_rows += len(chunk)
        chunk = chunk.rename(columns={series_col: "series_id", time_col: "bucket_15m"})
        chunk["series_id"] = chunk["series_id"].astype(str).str.strip()
        if sites is not None:
            chunk = chunk[chunk["series_id"].isin(set(sites))]
        chunk["bucket_15m"] = _parse_utc(chunk["bucket_15m"])
        if raw_events:
            # Granular events are assigned to the bucket in which they start;
            # each event counts as one record.
            chunk["bucket_15m"] = chunk["bucket_15m"].dt.floor("15min")
            chunk["records"] = 1.0
        if chunk[list(SIGNAL_COLUMNS)].isna().any().any():
            raise DataLoadError(f"{path} contains empty energy_wh / cfp_g values")
        agg_cols = [c for c in ("records",) + SIGNAL_COLUMNS + ("work", "ncores") if c in chunk.columns]
        parts.append(chunk.groupby(["series_id", "bucket_15m"], as_index=False)[agg_cols].sum())
        if raw_events:
            logger.info("[data]   %s event rows read", f"{n_rows:,}")

    if not parts:
        raise DataLoadError(f"{path} is empty")
    frame = pd.concat(parts, ignore_index=True)
    agg_cols = [c for c in frame.columns if c not in ("series_id", "bucket_15m")]
    frame = frame.groupby(["series_id", "bucket_15m"], as_index=False)[agg_cols].sum()
    if frame.empty:
        raise DataLoadError(f"no rows left in {path} after the site filter {sites}")
    if "records" not in frame.columns:
        frame["records"] = np.nan

    misaligned = (frame["bucket_15m"].dt.minute % 15 != 0) | (frame["bucket_15m"].dt.second != 0)
    if misaligned.any():
        raise DataLoadError(f"{int(misaligned.sum())} buckets in {path} are not 15-minute aligned")

    frame = frame.sort_values(["series_id", "bucket_15m"]).reset_index(drop=True)
    logger.info("[data] %s input rows -> %s site buckets for %d sites (%s .. %s)",
                f"{n_rows:,}", f"{len(frame):,}", frame["series_id"].nunique(),
                frame["bucket_15m"].min(), frame["bucket_15m"].max())
    return frame


def build_grid(buckets: pd.DataFrame, end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Reindex every site to a regular 15-minute grid.

    Each site starts at its first observed bucket; all sites end at ``end``
    (default: the last bucket in the data). The column ``observed`` marks
    buckets that exist in the source file.
    """
    if end is None:
        end = buckets["bucket_15m"].max()
    value_cols = [c for c in buckets.columns if c not in ("series_id", "bucket_15m")]
    frames = []
    for site, group in buckets.groupby("series_id", sort=True):
        index = pd.date_range(group["bucket_15m"].min(), end, freq="15min", name="bucket_15m")
        site_frame = group.set_index("bucket_15m")[value_cols].reindex(index)
        site_frame["observed"] = site_frame.index.isin(group["bucket_15m"])
        site_frame = site_frame.reset_index()
        site_frame.insert(0, "series_id", site)
        frames.append(site_frame)
    grid = pd.concat(frames, ignore_index=True)
    get_logger().info("[data] 15-minute grid: %s buckets, %.1f%% observed",
                      f"{len(grid):,}", 100.0 * grid["observed"].mean())
    return grid


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def valid_signal_labels(frame: pd.DataFrame) -> np.ndarray:
    """A.1 labels: 1 for valid telemetry, 0 for missing or invalid buckets.

    Same rule as the official ``task_a.labels.valid_signal_label``: a bucket
    is invalid when it is missing, when a value is negative, or when records
    exist but both energy and carbon are exactly zero.
    """
    records = frame["records"]
    energy = frame["energy_wh"]
    cfp = frame["cfp_g"]
    missing = ~frame["observed"].astype(bool) | records.isna() | energy.isna() | cfp.isna()
    negative = (records < 0) | (energy < 0) | (cfp < 0)
    zero_filled = (records > 0) & (energy == 0) & (cfp == 0)
    return (~(missing | negative | zero_filled)).astype(int).to_numpy()


def _official_quantile(values: np.ndarray, quantile: float) -> float:
    """Order statistic used by ``task_a.labels.peak_threshold``."""
    ordered = np.sort(values)
    if ordered.size == 0:
        return 0.0
    index = min(ordered.size - 1, max(0, int(round((ordered.size - 1) * quantile))))
    return float(ordered[index])


def compute_peak_thresholds(train: pd.DataFrame, quantile: float = 0.95,
                            rule: str = "per_signal_or") -> Dict[str, Dict[str, float]]:
    """Per-site peak thresholds estimated on the training split only.

    rule="per_signal_or" (paper, Sect. 4.2): the ``quantile`` of energy and of
        carbon are computed separately for every site.
    rule="max_signal": the rule of the official evaluator, one threshold on
        max(energy_wh, cfp_g) per site.
    """
    observed = train[train["observed"].astype(bool)]
    thresholds: Dict[str, Dict[str, float]] = {}
    for site, group in observed.groupby("series_id"):
        if rule == "per_signal_or":
            thresholds[site] = {
                "energy_wh": float(np.quantile(group["energy_wh"], quantile)),
                "cfp_g": float(np.quantile(group["cfp_g"], quantile)),
            }
        elif rule == "max_signal":
            joint = np.maximum(group["energy_wh"].to_numpy(), group["cfp_g"].to_numpy())
            thresholds[site] = {"max_signal": _official_quantile(joint, quantile)}
        else:
            raise ValueError(f"unknown peak label rule: {rule}")
    return thresholds


def peak_labels(frame: pd.DataFrame, thresholds: Dict[str, Dict[str, float]],
                rule: str = "per_signal_or") -> pd.Series:
    """A.2 labels (1 = peak). Unobserved buckets get NaN and are not used."""
    labels = pd.Series(np.nan, index=frame.index, dtype=float)
    observed = frame["observed"].astype(bool)
    for site, thr in thresholds.items():
        mask = observed & (frame["series_id"] == site)
        if not mask.any():
            continue
        if rule == "per_signal_or":
            is_peak = (frame.loc[mask, "energy_wh"] > thr["energy_wh"]) | \
                      (frame.loc[mask, "cfp_g"] > thr["cfp_g"])
        else:
            joint = np.maximum(frame.loc[mask, "energy_wh"], frame.loc[mask, "cfp_g"])
            is_peak = joint >= thr["max_signal"]
        labels.loc[mask] = is_peak.astype(float)
    return labels


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SplitSpec:
    """Chronological split points.

    data_start    first bucket in the data
    inner_cutoff  end of the search-training part; the hold-out used during
                  the hyperparameter search covers (inner_cutoff, cutoff]
    cutoff        official train/test cut-off; the final model is trained on
                  everything up to it
    data_end      last bucket in the data
    """
    data_start: pd.Timestamp
    inner_cutoff: pd.Timestamp
    cutoff: pd.Timestamp
    data_end: pd.Timestamp

    def describe(self) -> str:
        return (f"search-train <= {self.inner_cutoff} < hold-out <= {self.cutoff} "
                f"< test <= {self.data_end}")


def make_split(grid: pd.DataFrame, cutoff: str = OFFICIAL_CUTOFF,
               holdout_fraction: float = 0.2) -> SplitSpec:
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1)")
    cutoff_ts = pd.Timestamp(cutoff)
    if cutoff_ts.tzinfo is None:
        cutoff_ts = cutoff_ts.tz_localize("UTC")
    observed = grid[grid["observed"].astype(bool)]
    data_start = observed["bucket_15m"].min()
    data_end = observed["bucket_15m"].max()
    if not data_start < cutoff_ts:
        raise DataLoadError(f"cut-off {cutoff_ts} is before the first bucket {data_start}")
    inner = data_start + (cutoff_ts - data_start) * (1.0 - holdout_fraction)
    inner = inner.floor("15min")
    return SplitSpec(data_start=data_start, inner_cutoff=inner, cutoff=cutoff_ts, data_end=data_end)


def site_ids(frame: pd.DataFrame) -> List[str]:
    return sorted(frame["series_id"].unique().tolist())


def availability_by_hour(train: pd.DataFrame) -> Dict[str, Dict[int, float]]:
    """Share of observed buckets per site and hour of day on the training
    split. Used by the Task B scheduler to judge the risk that a future
    bucket has no telemetry."""
    table: Dict[str, Dict[int, float]] = {}
    hours = train["bucket_15m"].dt.hour
    for site, group in train.groupby("series_id"):
        share = group["observed"].astype(float).groupby(hours.loc[group.index]).mean()
        table[site] = {int(h): float(v) for h, v in share.items()}
    return table

