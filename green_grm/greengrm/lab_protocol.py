"""Challenge-phase evaluation protocol ("challenge_lab").

The numbers reported in the paper (Table 2 and the challenge score) were
produced with the feature pipeline and evaluation of the challenge-phase
experiment grid (``challenge_code/task_a``). This module provides that
setting as an alternative evaluation protocol for the GRM search, so that
Algorithm 1 can be run exactly where those numbers come from:

* features of the challenge feature library (lags 4/8/96/672, workload
  lags, 3-hour rolling means of t-15..t-4, momentum, hour and weekday); a
  bucket only sees values from four or more steps earlier, so every model is
  a one-hour-ahead model whose forecast is used for both horizons,
* one regressor per target (energy, carbon) and two classifiers, the
  invalid-signal detector and the peak detector; as in the challenge code
  the classifiers take the number of iterations from lambda and keep the
  scikit-learn defaults otherwise (``classifier_hyperparameters="full"``
  passes the whole vector instead),
* labels and gaps as in the challenge code: gaps are zero-filled; for
  training, a bucket is valid when its energy was present and a peak when
  its energy exceeds the 95th percentile of its site over the whole series;
  the hold-out labels are recomputed by the challenge-phase evaluator on the
  zero-filled hold-out buckets (every bucket counts as valid and the peak
  threshold is the 95th percentile of the site within the hold-out),
* models are trained on the buckets up to the cut-off and evaluated on the
  public buckets after it, which served as the hold-out during the
  challenge (the organisers scored the hidden later months).

MAE(lambda) is the energy MAE on that hold-out (the "MAE" of the
challenge-phase reports and of Table 2) and F1(lambda) the F1 of the peak
detector. The composite score of the challenge-phase evaluator,
(F1_A.1 + F1_A.2 + 0.5/(1 + MAE_energy) + 0.5/(1 + MAE_cfp)) / 3 (higher is
better), is recorded with every evaluation.

``build_lab_frame`` reproduces ``challenge_code/task_a/feature_library.py``;
``tests/test_lab_protocol.py`` checks that both give identical frames.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (f1_score, mean_absolute_error, precision_score, recall_score,
                             roc_auc_score)

from .evaluation import EvaluationLog, SurrogateObjective, _checkpoints
from .models import ModelSettings
from .search_space import Hyperparameters
from .utils import DataLoadError, get_logger, read_json, write_json

LAB_SITES = ("site_ad3433ffac", "site_e726c7cce5", "site_f5bcf8a88a")

LAB_FEATURE_STRATEGIES = {
    "Base_Time_Lags": {"USE_TIME": True, "LAGS": [4, 96], "CROSS_VAR": False,
                       "ROLLING": False, "MOMENTUM": False},
    "AAAI_Cross_Rolling": {"USE_TIME": True, "LAGS": [4, 96], "CROSS_VAR": True,
                           "ROLLING": True, "MOMENTUM": False},
    "Deep_Momentum_Lags": {"USE_TIME": True, "LAGS": [4, 96, 672], "CROSS_VAR": True,
                           "ROLLING": True, "MOMENTUM": True},
}


# ---------------------------------------------------------------------------
# Feature frame
# ---------------------------------------------------------------------------
def build_lab_frame(data_path: Path, sites: Sequence[str] = LAB_SITES) -> pd.DataFrame:
    """Feature library of the challenge-phase code (zero-filled 15-minute grid)."""
    data_path = Path(data_path)
    if not data_path.is_file():
        raise DataLoadError(f"data file not found: {data_path}")
    df = pd.read_csv(data_path)
    df = df.rename(columns={"event_start_timestamp": "bucket_15m", "site_id": "series_id"})
    for column in ("bucket_15m", "series_id", "energy_wh", "cfp_g"):
        if column not in df.columns:
            raise DataLoadError(f"{data_path} has no column '{column}'")
    df["bucket_15m"] = pd.to_datetime(df["bucket_15m"], utc=True)
    df = df[df["series_id"].isin(list(sites))]
    if df.empty:
        raise DataLoadError(f"none of the sites {list(sites)} occurs in {data_path}")
    agg_cols = [c for c in ("energy_wh", "cfp_g", "work", "ncores", "pue", "ci_g") if c in df.columns]
    df = df.groupby(["series_id", "bucket_15m"]).agg(
        {c: "mean" if c in ("pue", "ci_g") else "sum" for c in agg_cols}).reset_index()

    frames = []
    for site in sites:
        site_df = df[df["series_id"] == site].copy()
        if site_df.empty:
            continue
        site_df = site_df.set_index("bucket_15m").resample("15min").asfreq()
        if site_df.index.tzinfo is None:
            site_df.index = site_df.index.tz_localize("UTC")
        site_df = site_df.reset_index()
        site_df["series_id"] = site
        frames.append(site_df)
    df = pd.concat(frames, ignore_index=True)

    df["F_hour"] = df["bucket_15m"].dt.hour
    df["F_dow"] = df["bucket_15m"].dt.dayofweek
    by_site = df.groupby("series_id")
    for h in (4, 8, 96, 672):
        df[f"F_energy_lag_{h}"] = by_site["energy_wh"].shift(h)
        df[f"F_cfp_lag_{h}"] = by_site["cfp_g"].shift(h)
    for column in ("work", "ncores"):
        if column in df.columns:
            for h in (4, 96):
                df[f"F_{column}_lag_{h}"] = by_site[column].shift(h)
    df["F_energy_roll_3h"] = by_site["energy_wh"].transform(
        lambda x: x.shift(4).rolling(12, min_periods=1).mean())
    df["F_cfp_roll_3h"] = by_site["cfp_g"].transform(
        lambda x: x.shift(4).rolling(12, min_periods=1).mean())
    df["F_energy_momentum"] = df["F_energy_lag_4"] - df["F_energy_lag_8"]
    df["F_cfp_momentum"] = df["F_cfp_lag_4"] - df["F_cfp_lag_8"]
    df["is_valid"] = df["energy_wh"].notna().astype(int)
    threshold = df.groupby("series_id")["energy_wh"].transform(lambda x: x.quantile(0.95))
    df["is_peak"] = ((df["energy_wh"] > threshold) & df["energy_wh"].notna()).astype(int)
    return df.fillna(0)


def lab_feature_columns(frame: pd.DataFrame, strategy: Dict) -> List[str]:
    selected: List[str] = []
    if strategy["USE_TIME"]:
        selected += ["F_hour", "F_dow"]
    for h in strategy["LAGS"]:
        selected += [f"F_energy_lag_{h}", f"F_cfp_lag_{h}"]
    if strategy["CROSS_VAR"]:
        selected += [c for c in frame.columns if c.startswith("F_work_lag") or c.startswith("F_ncores_lag")]
    if strategy["ROLLING"]:
        selected += ["F_energy_roll_3h", "F_cfp_roll_3h"]
    if strategy["MOMENTUM"]:
        selected += ["F_energy_momentum", "F_cfp_momentum"]
    return [c for c in selected if c in frame.columns]


def challenge_phase_labels(rows: pd.DataFrame):
    """Hold-out labels as the challenge-phase evaluator derives them from
    the zero-filled ground truth: (valid, peak)."""
    valid = rows["energy_wh"].notna().astype(int).to_numpy()
    threshold = rows.groupby("series_id")["energy_wh"].transform(lambda x: x.quantile(0.95))
    peak = ((rows["energy_wh"] > threshold) & rows["energy_wh"].notna()).astype(int).to_numpy()
    return valid, peak


def lab_composite_score(det_f1: float, peak_f1: float, mae_energy: float, mae_cfp: float) -> float:
    """Composite of the challenge-phase evaluator (higher is better)."""
    forecast = 0.5 * (1.0 / (1.0 + mae_energy)) + 0.5 * (1.0 / (1.0 + mae_cfp))
    return (det_f1 + peak_f1 + forecast) / 3.0


def _classifiers(hp: Hyperparameters, mode: str, seed: int):
    if mode == "max_iter_only":
        det = HistGradientBoostingClassifier(max_iter=hp.max_iter, random_state=seed)
        peak = HistGradientBoostingClassifier(max_iter=hp.max_iter, class_weight="balanced",
                                              random_state=seed)
    elif mode == "full":
        common = dict(learning_rate=hp.learning_rate, max_iter=hp.max_iter, max_depth=hp.max_depth,
                      l2_regularization=hp.l2_regularization, random_state=seed)
        det = HistGradientBoostingClassifier(**common)
        peak = HistGradientBoostingClassifier(class_weight="balanced", **common)
    else:
        raise ValueError("classifier_hyperparameters must be 'max_iter_only' or 'full'")
    return det, peak


def _lab_regressor(hp: Hyperparameters, seed: int) -> HistGradientBoostingRegressor:
    # Only the five searched hyperparameters are set; everything else keeps
    # the scikit-learn defaults, as in the challenge-phase code.
    return HistGradientBoostingRegressor(loss=hp.loss, learning_rate=hp.learning_rate,
                                         max_iter=hp.max_iter, max_depth=hp.max_depth,
                                         l2_regularization=hp.l2_regularization,
                                         random_state=seed)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
class LabHoldoutEvaluator:
    """o(lambda) in the challenge-phase setting (same interface as
    ``evaluation.HoldoutEvaluator``)."""

    def __init__(self, data_path: Path, cutoff: str, settings: ModelSettings,
                 feature_strategy: str = "Deep_Momentum_Lags",
                 classifier_hyperparameters: str = "max_iter_only",
                 sites: Sequence[str] = LAB_SITES,
                 stage_checkpoints: Sequence[float] = (0.5, 0.75, 1.0),
                 seed_policy: str = "fixed", save_dir: Optional[Path] = None,
                 save_models: bool = False, save_predictions: bool = False):
        logger = get_logger()
        if feature_strategy not in LAB_FEATURE_STRATEGIES:
            raise ValueError(f"unknown feature strategy '{feature_strategy}'")
        self.settings = settings
        self.feature_strategy = feature_strategy
        self.classifier_hyperparameters = classifier_hyperparameters
        self.stage_checkpoints = tuple(stage_checkpoints)
        self.seed_policy = seed_policy
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.save_models = save_models
        self.save_predictions = save_predictions
        self.n_evaluations = 0

        self.frame = build_lab_frame(data_path, sites)
        cutoff_ts = pd.Timestamp(cutoff)
        cutoff_ts = cutoff_ts.tz_localize("UTC") if cutoff_ts.tzinfo is None else cutoff_ts
        self.cutoff = cutoff_ts
        self.features = lab_feature_columns(self.frame, LAB_FEATURE_STRATEGIES[feature_strategy])
        self.train = self.frame[self.frame["bucket_15m"] <= cutoff_ts]
        self.hold = self.frame[self.frame["bucket_15m"] > cutoff_ts]
        if self.train.empty or self.hold.empty:
            raise DataLoadError(f"the cut-off {cutoff_ts} leaves an empty training or hold-out part")
        self.eval_valid, self.eval_peak = challenge_phase_labels(self.hold)
        self.scales = {s: float(self.hold[s].abs().mean()) for s in ("energy_wh", "cfp_g")}
        self.site_scales = {
            s: {site: float(g[s].abs().mean()) for site, g in self.hold.groupby("series_id")}
            for s in ("energy_wh", "cfp_g")
        }
        logger.info("[lab] feature strategy %s (%d features); train %s buckets <= %s < hold-out %s buckets",
                    feature_strategy, len(self.features), f"{len(self.train):,}", cutoff_ts,
                    f"{len(self.hold):,}")

    def start_session(self, save_dir: Optional[Path]) -> None:
        self.n_evaluations = 0
        self.save_dir = Path(save_dir) if save_dir is not None else None

    def default_surrogate(self, a1="auto", a2: float = 1.0) -> SurrogateObjective:
        if a1 == "auto":
            a1 = 1.0 / self.scales["energy_wh"]
        return SurrogateObjective(a1=float(a1), a2=float(a2))

    def evaluate(self, hp: Hyperparameters) -> EvaluationLog:
        eval_id = self.n_evaluations
        self.n_evaluations += 1
        seed = self.settings.seed if self.seed_policy == "fixed" else self.settings.seed + eval_id
        start = time.perf_counter()
        X_train, X_hold = self.train[self.features], self.hold[self.features]
        samples: List[float] = []
        names: List[str] = []
        maes: Dict[str, float] = {}
        iterations: Dict[str, int] = {}
        preds = {}
        models = {}

        for signal in ("energy_wh", "cfp_g"):
            model = _lab_regressor(hp, seed).fit(X_train, self.train[signal])
            iterations[signal] = int(model.n_iter_)
            y_true = self.hold[signal].to_numpy()
            points = _checkpoints(model.n_iter_, self.stage_checkpoints)
            for stage, stage_pred in enumerate(model.staged_predict(X_hold), start=1):
                if stage in points:
                    stage_pred = np.clip(stage_pred, 0.0, None)
                    samples.append(float(np.mean(np.abs(stage_pred - y_true))) / self.scales[signal])
                    names.append(f"stage/{signal}/{stage}")
                if stage == model.n_iter_:
                    preds[signal] = np.clip(stage_pred, 0.0, None)
            errors = np.abs(preds[signal] - y_true)
            maes[signal] = float(errors.mean())
            for site, idx in self.hold.groupby("series_id").indices.items():
                samples.append(float(errors[idx].mean()) / self.site_scales[signal][site])
                names.append(f"site/{signal}/{site}")
            models[signal] = model

        det, peak = _classifiers(hp, self.classifier_hyperparameters, seed)
        det.fit(X_train, self.train["is_valid"])
        peak.fit(X_train, self.train["is_peak"])
        iterations["valid"] = int(det.n_iter_)
        iterations["peak"] = int(peak.n_iter_)
        y_peak = self.eval_peak
        points = _checkpoints(peak.n_iter_, self.stage_checkpoints)
        for stage, stage_pred in enumerate(peak.staged_predict(X_hold), start=1):
            if stage in points:
                samples.append(1.0 - float(f1_score(y_peak, stage_pred, zero_division=1.0)))
                names.append(f"stage/peak/{stage}")
        peak_pred = peak.predict(X_hold)
        peak_score = peak.predict_proba(X_hold)[:, 1]
        for site, idx in self.hold.groupby("series_id").indices.items():
            samples.append(1.0 - float(f1_score(y_peak[idx], peak_pred[idx], zero_division=1.0)))
            names.append(f"site/peak/{site}")
        det_pred = det.predict(X_hold)
        det_f1 = float(f1_score(self.eval_valid, det_pred, zero_division=0))
        peak_f1 = float(f1_score(y_peak, peak_pred, zero_division=0))
        models.update({"valid": det, "peak": peak})

        sample_array = np.asarray(samples, dtype=float)
        log = EvaluationLog(
            eval_id=eval_id, hyperparameters=hp, mae=maes["energy_wh"],
            mae_by_target={"energy_wh": maes["energy_wh"], "cfp_g": maes["cfp_g"]},
            peak_f1=peak_f1,
            peak_auroc=float(roc_auc_score(y_peak, peak_score)) if len(np.unique(y_peak)) > 1 else 0.5,
            peak_precision=float(precision_score(y_peak, peak_pred, zero_division=0)),
            peak_recall=float(recall_score(y_peak, peak_pred, zero_division=0)),
            metric_samples=sample_array, sample_names=names,
            p1=float(sample_array.mean()), p2=float(sample_array.std()),
            iterations=iterations, seconds=time.perf_counter() - start, seed=seed,
            extra={"detection_f1": det_f1,
                   "lab_composite_score": lab_composite_score(det_f1, peak_f1, maes["energy_wh"],
                                                              maes["cfp_g"])})
        self._save(log, preds, peak_pred, det_pred, models)
        return log

    def _save(self, log, preds, peak_pred, det_pred, models) -> None:
        if self.save_dir is None:
            return
        folder = self.save_dir / f"eval_{log.eval_id:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "log.json", log.to_json())
        if self.save_predictions:
            pd.DataFrame({
                "series_id": self.hold["series_id"].to_numpy(),
                "bucket_15m": self.hold["bucket_15m"].to_numpy(),
                "energy_wh_true": self.hold["energy_wh"].to_numpy(),
                "energy_wh_pred": preds["energy_wh"],
                "cfp_g_true": self.hold["cfp_g"].to_numpy(),
                "cfp_g_pred": preds["cfp_g"],
                "valid_label": self.eval_valid, "valid_pred": det_pred,
                "peak_label": self.eval_peak, "peak_pred": peak_pred,
            }).to_csv(folder / "holdout_predictions.csv.gz", index=False, compression="gzip")
        if self.save_models:
            joblib.dump(models, folder / "models.joblib", compress=3)


# ---------------------------------------------------------------------------
# Final model of the challenge-phase protocol
# ---------------------------------------------------------------------------
class LabModel:
    """Regressors and classifiers of one lambda in the challenge-phase setting."""

    def __init__(self, hp: Hyperparameters, feature_strategy: str = "Deep_Momentum_Lags",
                 classifier_hyperparameters: str = "max_iter_only", seed: int = 42):
        self.hp = hp
        self.feature_strategy = feature_strategy
        self.classifier_hyperparameters = classifier_hyperparameters
        self.seed = seed
        self.features: List[str] = []
        self.models: Dict[str, object] = {}
        self.training_end: Optional[pd.Timestamp] = None

    def fit(self, frame: pd.DataFrame, training_end: pd.Timestamp) -> "LabModel":
        logger = get_logger()
        self.features = lab_feature_columns(frame, LAB_FEATURE_STRATEGIES[self.feature_strategy])
        train = frame[frame["bucket_15m"] <= training_end]
        self.training_end = training_end
        X = train[self.features]
        for signal in ("energy_wh", "cfp_g"):
            self.models[signal] = _lab_regressor(self.hp, self.seed).fit(X, train[signal])
        det, peak = _classifiers(self.hp, self.classifier_hyperparameters, self.seed)
        self.models["valid"] = det.fit(X, train["is_valid"])
        self.models["peak"] = peak.fit(X, train["is_peak"])
        logger.info("[lab] trained %s on %s buckets <= %s", self.hp.label(), f"{len(train):,}", training_end)
        return self

    def predict_window(self, frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
        """Submission tables for the buckets in (start, end]. As in the
        challenge code, the one-hour forecast of a bucket is written for both
        horizons."""
        rows = frame[(frame["bucket_15m"] > start) & (frame["bucket_15m"] <= end)]
        X = rows[self.features]
        energy = np.clip(self.models["energy_wh"].predict(X), 0.0, None)
        cfp = np.clip(self.models["cfp_g"].predict(X), 0.0, None)
        forecasts = pd.concat([pd.DataFrame({
            "series_id": rows["series_id"].to_numpy(),
            "forecast_timestamp_utc": rows["bucket_15m"].to_numpy(),
            "horizon_steps_15m": h, "energy_wh_pred": energy, "cfp_g_pred": cfp,
        }) for h in (4, 96)], ignore_index=True)
        valid_score = self.models["valid"].predict_proba(X)[:, 1]
        detections = pd.DataFrame({
            "series_id": rows["series_id"].to_numpy(), "bucket_15m": rows["bucket_15m"].to_numpy(),
            "valid_signal_score": valid_score,
            "valid_signal_pred": self.models["valid"].predict(X).astype(int),
        })
        peak_score = self.models["peak"].predict_proba(X)[:, 1]
        peaks = pd.DataFrame({
            "series_id": rows["series_id"].to_numpy(), "bucket_15m": rows["bucket_15m"].to_numpy(),
            "peak_score": peak_score, "peak_pred": self.models["peak"].predict(X).astype(int),
        })
        return forecasts, detections, peaks

    def lab_metrics(self, frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, float]:
        """Challenge-phase metrics on the buckets in (start, end]."""
        rows = frame[(frame["bucket_15m"] > start) & (frame["bucket_15m"] <= end)]
        X = rows[self.features]
        valid, peak = challenge_phase_labels(rows)
        mae_e = mean_absolute_error(rows["energy_wh"], np.clip(self.models["energy_wh"].predict(X), 0, None))
        mae_c = mean_absolute_error(rows["cfp_g"], np.clip(self.models["cfp_g"].predict(X), 0, None))
        det_f1 = float(f1_score(valid, self.models["valid"].predict(X), zero_division=0))
        peak_f1 = float(f1_score(peak, self.models["peak"].predict(X), zero_division=0))
        return {"lab_composite_score": lab_composite_score(det_f1, peak_f1, mae_e, mae_c),
                "forecast_mae_energy_wh": float(mae_e), "forecast_mae_cfp_g": float(mae_c),
                "detection_f1": det_f1, "peak_f1": peak_f1}

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": self.models, "features": self.features}, directory / "models.joblib",
                    compress=3)
        write_json(directory / "model.json", {
            "protocol": "challenge_lab", "hyperparameters": self.hp.as_dict(),
            "feature_strategy": self.feature_strategy,
            "classifier_hyperparameters": self.classifier_hyperparameters,
            "seed": self.seed, "training_end": self.training_end, "features": self.features})

    @classmethod
    def load(cls, directory: Path) -> "LabModel":
        directory = Path(directory)
        manifest = read_json(directory / "model.json")
        model = cls(Hyperparameters.from_dict(manifest["hyperparameters"]), manifest["feature_strategy"],
                    manifest["classifier_hyperparameters"], manifest["seed"])
        stored = joblib.load(directory / "models.joblib")
        model.models, model.features = stored["models"], stored["features"]
        model.training_end = pd.Timestamp(manifest["training_end"]) if manifest["training_end"] else None
        return model
