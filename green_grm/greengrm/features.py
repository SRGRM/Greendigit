"""Leakage-safe features and direct multi-horizon targets (Sect. 4.2).

Every row of a design matrix is a forecast origin ``t`` of one site. The
features only use telemetry at or before ``t``:

* autoregressive lags     value of the target signal at t - k, k >= 0
* cross-variable lags     the same lags of the other signal (carbon in the
                          energy model and vice versa)
* seasonal lags           value one day / two days / one week before the
                          target time t + h; kept only when that time is at or
                          before the origin, i.e. when s >= h
* rolling means           trailing means over the last w buckets ending at t
                          (w = 12 gives the three-hour mean of the paper)
* calendar indicators     hour, quarter-hour slot, day of week and weekend
                          flag of the target bucket t + h, known in advance

Optional groups (off in the paper configuration) add lags of the workload
columns (records, work, ncores), a momentum term (lag 0 minus lag 4) and a
site indicator.

The target of horizon ``h`` is the signal shifted forward by ``h`` steps, so
each regressor learns a direct mapping from the features at the origin to
the value four or 96 steps ahead. The unit test
``tests/test_features.py`` checks that perturbing any value after the origin
leaves the features unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data import SIGNAL_COLUMNS, STEP

ID_COLUMNS = ("series_id", "origin", "target_time")


@dataclass
class FeatureConfig:
    target_lags: Tuple[int, ...] = (0, 1, 2, 3, 4, 8, 12)
    seasonal_lags: Tuple[int, ...] = (96, 192, 672)
    rolling_windows: Tuple[int, ...] = (12,)
    cross_variable: bool = True
    calendar: bool = True
    workload_lags: Tuple[int, ...] = ()
    momentum: bool = False
    site_indicator: bool = False
    missing_values: str = "native"          # "native" (NaN kept) | "zero" (NaN -> 0)

    @classmethod
    def from_dict(cls, values: Dict) -> "FeatureConfig":
        known = {k: v for k, v in values.items() if k in cls.__dataclass_fields__}
        for key in ("target_lags", "seasonal_lags", "rolling_windows", "workload_lags"):
            if key in known and known[key] is not None:
                known[key] = tuple(int(x) for x in known[key])
        config = cls(**known)
        if config.missing_values not in ("native", "zero"):
            raise ValueError("missing_values must be 'native' or 'zero'")
        if any(k < 0 for k in config.target_lags + config.workload_lags):
            raise ValueError("lags must be >= 0 (0 is the origin itself)")
        if any(s <= 0 for s in config.seasonal_lags):
            raise ValueError("seasonal lags must be positive")
        if any(w < 1 for w in config.rolling_windows):
            raise ValueError("rolling windows must cover at least one bucket")
        return config

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DesignMatrix:
    """Features, targets and labels for one horizon.

    ``frame`` holds the id columns, the feature columns, one target column
    ``y_<signal>`` per signal and, when requested, the peak label ``y_peak``
    of the target bucket.
    """
    horizon: int
    frame: pd.DataFrame
    feature_names: List[str]
    signal_features: Dict[str, List[str]] = field(default_factory=dict)
    categorical: Optional[List[bool]] = None

    def features_for(self, target: Optional[str] = None) -> List[str]:
        """Feature list for one target; without cross-variable features the
        columns of the other signal are dropped."""
        if target is None or target not in self.signal_features:
            return self.feature_names
        return self.signal_features[target]

    def categorical_mask(self, names: Sequence[str]) -> Optional[List[bool]]:
        if not self.categorical:
            return None
        lookup = dict(zip(self.feature_names, self.categorical))
        mask = [lookup[n] for n in names]
        return mask if any(mask) else None


class FeatureBuilder:
    """Builds design matrices on a regular per-site 15-minute grid."""

    def __init__(self, config: FeatureConfig, sites: Sequence[str]):
        self.config = config
        self.sites = list(sites)
        self._site_code = {site: code for code, site in enumerate(self.sites)}

    # ------------------------------------------------------------------
    def _site_features(self, site_frame: pd.DataFrame,
                       horizon: int) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
        """Feature columns for one site. Returns the columns and, for each
        column, the signal it was computed from (or '' for shared ones)."""
        cfg = self.config
        cols: Dict[str, np.ndarray] = {}
        owner: Dict[str, str] = {}

        for signal in SIGNAL_COLUMNS:
            series = site_frame[signal].astype(float)
            for lag in cfg.target_lags:
                name = f"{signal}_lag_{lag}"
                cols[name] = series.shift(lag).to_numpy()
                owner[name] = signal
            for season in cfg.seasonal_lags:
                shift = season - horizon
                if shift < 0 or shift in cfg.target_lags:
                    continue            # would look past the origin, or duplicates a lag
                name = f"{signal}_seasonal_{season}"
                cols[name] = series.shift(shift).to_numpy()
                owner[name] = signal
            for window in cfg.rolling_windows:
                name = f"{signal}_mean_{window}"
                cols[name] = series.rolling(window, min_periods=1).mean().to_numpy()
                owner[name] = signal
            if cfg.momentum:
                name = f"{signal}_momentum_4"
                cols[name] = (series - series.shift(4)).to_numpy()
                owner[name] = signal

        for column in ("records", "work", "ncores"):
            if column not in site_frame.columns:
                continue
            values = site_frame[column].astype(float)
            for lag in cfg.workload_lags:
                name = f"{column}_lag_{lag}"
                cols[name] = values.shift(lag).to_numpy()
                owner[name] = ""

        if cfg.calendar:
            target_time = site_frame["bucket_15m"] + horizon * STEP
            cols["target_hour"] = target_time.dt.hour.to_numpy()
            cols["target_slot"] = (target_time.dt.hour * 4 + target_time.dt.minute // 15).to_numpy()
            cols["target_dayofweek"] = target_time.dt.dayofweek.to_numpy()
            cols["target_weekend"] = (target_time.dt.dayofweek >= 5).astype(int).to_numpy()
            for name in ("target_hour", "target_slot", "target_dayofweek", "target_weekend"):
                owner[name] = ""

        if cfg.site_indicator:
            site = site_frame["series_id"].iloc[0]
            cols["site_code"] = np.full(len(site_frame), self._site_code.get(site, -1))
            owner["site_code"] = ""
        return cols, owner

    # ------------------------------------------------------------------
    def build(self, grid: pd.DataFrame, horizon: int,
              peak_labels: Optional[pd.Series] = None) -> DesignMatrix:
        """Design matrix for ``horizon`` over every origin of the grid.

        ``grid`` must be the regular grid produced by ``data.build_grid`` and
        sorted by (series_id, bucket_15m). ``peak_labels`` (aligned with the
        grid) adds the label of the target bucket as ``y_peak``.
        """
        if horizon < 1:
            raise ValueError("horizon must be at least one step")
        pieces = []
        owner: Dict[str, str] = {}
        for site, site_frame in grid.groupby("series_id", sort=True):
            site_frame = site_frame.sort_values("bucket_15m")
            if site_frame["bucket_15m"].diff().dropna().ne(STEP).any():
                raise ValueError(f"grid of {site} is not a regular 15-minute grid")
            cols, site_owner = self._site_features(site_frame, horizon)
            if owner and list(site_owner) != list(owner):
                raise ValueError(f"site {site} produced a different feature set")
            owner = site_owner
            piece = pd.DataFrame(cols, index=site_frame.index)
            piece.insert(0, "series_id", site)
            piece.insert(1, "origin", site_frame["bucket_15m"].to_numpy())
            piece.insert(2, "target_time", (site_frame["bucket_15m"] + horizon * STEP).to_numpy())
            for signal in SIGNAL_COLUMNS:
                piece[f"y_{signal}"] = site_frame[signal].astype(float).shift(-horizon).to_numpy()
            piece["y_observed"] = site_frame["observed"].astype(float).shift(-horizon).to_numpy()
            if peak_labels is not None:
                piece["y_peak"] = peak_labels.loc[site_frame.index].shift(-horizon).to_numpy()
            pieces.append(piece)

        frame = pd.concat(pieces, ignore_index=True)
        frame["y_observed"] = frame["y_observed"].fillna(0.0).astype(bool)
        feature_names = [c for c in owner]
        if self.config.missing_values == "zero":
            frame[feature_names] = frame[feature_names].fillna(0.0)

        signal_features = {}
        for signal in SIGNAL_COLUMNS:
            if self.config.cross_variable:
                signal_features[signal] = list(feature_names)
            else:
                signal_features[signal] = [n for n in feature_names if owner[n] in ("", signal)]
        categorical = [n == "site_code" for n in feature_names]
        return DesignMatrix(horizon=horizon, frame=frame, feature_names=feature_names,
                            signal_features=signal_features,
                            categorical=categorical if any(categorical) else None)
