"""Forecasting and detection models (Sect. 4.3).

All learning tasks use histogram-based gradient boosting from scikit-learn.
One ``HistGradientBoostingRegressor`` is trained for every combination of
target signal and horizon, and one ``HistGradientBoostingClassifier`` for the
peak-detection subtask A.2. The hyperparameter vector lambda sets the
regression loss of the regressors and the learning rate, number of
iterations, depth and L2 strength of all five models; the classifier always
uses the log-loss.

A fixed random seed is passed to every model, so any remaining variance of
an evaluation comes from sources outside the seed, such as the order of
floating-point reductions in multi-threaded runs. On the machine used for
the results in this repository repeated evaluations with the same seed were
identical; ``evaluation_seed_policy = "per_evaluation"`` draws a new seed for
every evaluation and exposes the seed-dependent early-stopping split.
Early stopping keeps the scikit-learn default ``"auto"``, which switches it
on for training sets above 10,000 rows, as in the challenge-phase models;
the number of iterations used can then be lower than T.

``GreenGRMModel`` bundles the five models trained with one vector and
produces the three Task A submission tables.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .data import SIGNAL_COLUMNS, valid_signal_labels
from .features import DesignMatrix, FeatureBuilder, FeatureConfig
from .search_space import Hyperparameters
from .utils import get_logger, read_json, write_json


@dataclass
class ModelSettings:
    """Settings that are fixed during the search (not part of lambda)."""
    seed: int = 42
    early_stopping: Union[str, bool] = "auto"     # "auto" | True | False
    validation_fraction: float = 0.1
    n_iter_no_change: int = 10
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 20
    peak_class_weight: Optional[str] = "balanced"  # "balanced" | None

    @classmethod
    def from_dict(cls, values: Dict) -> "ModelSettings":
        return cls(**{k: v for k, v in values.items() if k in cls.__dataclass_fields__})


def make_regressor(hp: Hyperparameters, settings: ModelSettings,
                   categorical: Optional[List[bool]] = None,
                   seed: Optional[int] = None) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss=hp.loss,
        learning_rate=hp.learning_rate,
        max_iter=hp.max_iter,
        max_depth=hp.max_depth,
        l2_regularization=hp.l2_regularization,
        max_leaf_nodes=settings.max_leaf_nodes,
        min_samples_leaf=settings.min_samples_leaf,
        early_stopping=settings.early_stopping,
        validation_fraction=settings.validation_fraction,
        n_iter_no_change=settings.n_iter_no_change,
        categorical_features=categorical,
        random_state=settings.seed if seed is None else seed,
    )


def make_peak_classifier(hp: Hyperparameters, settings: ModelSettings,
                         categorical: Optional[List[bool]] = None,
                         seed: Optional[int] = None) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=hp.learning_rate,
        max_iter=hp.max_iter,
        max_depth=hp.max_depth,
        l2_regularization=hp.l2_regularization,
        max_leaf_nodes=settings.max_leaf_nodes,
        min_samples_leaf=settings.min_samples_leaf,
        early_stopping=settings.early_stopping,
        validation_fraction=settings.validation_fraction,
        n_iter_no_change=settings.n_iter_no_change,
        class_weight=settings.peak_class_weight,
        categorical_features=categorical,
        random_state=settings.seed if seed is None else seed,
    )


# ---------------------------------------------------------------------------
# Row selection on design matrices
# ---------------------------------------------------------------------------
def regression_rows(matrix: DesignMatrix, signal: str, start: Optional[pd.Timestamp],
                    end: pd.Timestamp, by: str = "target_time",
                    origin_after: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Rows whose target is observed and whose ``by`` timestamp lies in
    (start, end]. ``origin_after`` additionally requires origin > value."""
    frame = matrix.frame
    mask = frame["y_observed"] & frame[f"y_{signal}"].notna() & (frame[by] <= end)
    if start is not None:
        mask &= frame[by] > start
    if origin_after is not None:
        mask &= frame["origin"] > origin_after
    return frame.loc[mask]


def peak_rows(matrix: DesignMatrix, start: Optional[pd.Timestamp], end: pd.Timestamp,
              origin_after: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    frame = matrix.frame
    mask = frame["y_peak"].notna() & (frame["target_time"] <= end)
    if start is not None:
        mask &= frame["target_time"] > start
    if origin_after is not None:
        mask &= frame["origin"] > origin_after
    return frame.loc[mask]


# ---------------------------------------------------------------------------
# Model bundle
# ---------------------------------------------------------------------------
class GreenGRMModel:
    """Five models trained with one hyperparameter vector."""

    def __init__(self, hp: Hyperparameters, feature_config: FeatureConfig,
                 settings: ModelSettings, horizons: Sequence[int] = (4, 96),
                 targets: Sequence[str] = SIGNAL_COLUMNS, peak_horizon: int = 4,
                 peak_decision_threshold: float = 0.5,
                 peak_rule: str = "per_signal_or", peak_quantile: float = 0.95):
        self.hp = hp
        self.feature_config = feature_config
        self.settings = settings
        self.horizons = tuple(int(h) for h in horizons)
        self.targets = tuple(targets)
        self.peak_horizon = int(peak_horizon)
        self.peak_decision_threshold = float(peak_decision_threshold)
        self.peak_rule = peak_rule
        self.peak_quantile = float(peak_quantile)
        self.regressors: Dict[Tuple[str, int], HistGradientBoostingRegressor] = {}
        self.regressor_features: Dict[Tuple[str, int], List[str]] = {}
        self.peak_model: Optional[HistGradientBoostingClassifier] = None
        self.peak_features: List[str] = []
        self.peak_thresholds: Dict[str, Dict[str, float]] = {}
        self.sites: List[str] = []
        self.training_end: Optional[pd.Timestamp] = None
        self.training_rows: Dict[str, int] = {}

    # ------------------------------------------------------------------
    def builder(self) -> FeatureBuilder:
        return FeatureBuilder(self.feature_config, self.sites)

    def build_matrices(self, grid: pd.DataFrame, peak_label_series: Optional[pd.Series]
                       ) -> Dict[object, DesignMatrix]:
        """One matrix per forecast horizon plus, under the key ``"peak"``,
        the matrix of the peak classifier with the label of the target bucket."""
        builder = self.builder()
        matrices: Dict[object, DesignMatrix] = {h: builder.build(grid, h) for h in self.horizons}
        if peak_label_series is not None:
            matrices["peak"] = builder.build(grid, self.peak_horizon, peak_label_series)
        return matrices

    def fit(self, grid: pd.DataFrame, training_end: pd.Timestamp,
            peak_thresholds: Dict[str, Dict[str, float]], peak_label_series: pd.Series,
            matrices: Optional[Dict[object, DesignMatrix]] = None) -> "GreenGRMModel":
        """Train all models on targets up to ``training_end``.

        ``matrices`` may be passed when they are already built (see
        ``build_matrices``).
        """
        logger = get_logger()
        self.sites = sorted(grid["series_id"].unique().tolist())
        self.training_end = training_end
        self.peak_thresholds = peak_thresholds
        if matrices is None:
            matrices = self.build_matrices(grid, peak_label_series)

        for horizon in self.horizons:
            matrix = matrices[horizon]
            for signal in self.targets:
                names = matrix.features_for(signal)
                rows = regression_rows(matrix, signal, None, training_end)
                if rows.empty:
                    raise ValueError(f"no training rows for {signal} at horizon {horizon}")
                model = make_regressor(self.hp, self.settings, matrix.categorical_mask(names))
                model.fit(rows[names], rows[f"y_{signal}"])
                self.regressors[(signal, horizon)] = model
                self.regressor_features[(signal, horizon)] = names
                self.training_rows[f"{signal}_h{horizon}"] = int(len(rows))
                logger.info("[model]   %-9s h=%-3d trained on %s rows (%d iterations)",
                            signal, horizon, f"{len(rows):,}", model.n_iter_)

        peak_matrix = matrices["peak"]
        names = peak_matrix.feature_names
        rows = peak_rows(peak_matrix, None, training_end)
        if rows["y_peak"].nunique() < 2:
            raise ValueError("the peak labels of the training rows contain a single class")
        self.peak_model = make_peak_classifier(self.hp, self.settings, peak_matrix.categorical_mask(names))
        self.peak_model.fit(rows[names], rows["y_peak"].astype(int))
        self.peak_features = names
        self.training_rows["peak"] = int(len(rows))
        logger.info("[model]   peak      h=%-3d trained on %s rows (%d iterations, %.1f%% peaks)",
                    self.peak_horizon, f"{len(rows):,}", self.peak_model.n_iter_,
                    100.0 * rows["y_peak"].mean())
        return self

    # ------------------------------------------------------------------
    def predict_forecasts(self, grid: pd.DataFrame, origin_start: pd.Timestamp,
                          origin_end: pd.Timestamp, target_end: Optional[pd.Timestamp] = None,
                          matrices: Optional[Dict[object, DesignMatrix]] = None) -> pd.DataFrame:
        """Forecasts for every site and every origin in (origin_start,
        origin_end]. Rows whose target lies after ``target_end`` are dropped
        (no ground truth exists there)."""
        if matrices is None:
            builder = self.builder()
            matrices = {h: builder.build(grid, h) for h in self.horizons}
        tables = []
        for horizon in self.horizons:
            frame = matrices[horizon].frame
            mask = (frame["origin"] > origin_start) & (frame["origin"] <= origin_end)
            if target_end is not None:
                mask &= frame["target_time"] <= target_end
            rows = frame.loc[mask]
            table = pd.DataFrame({
                "series_id": rows["series_id"].to_numpy(),
                "origin": rows["origin"].to_numpy(),
                "forecast_timestamp_utc": rows["target_time"].to_numpy(),
                "horizon_steps_15m": horizon,
            })
            for signal in self.targets:
                names = self.regressor_features[(signal, horizon)]
                pred = self.regressors[(signal, horizon)].predict(rows[names]) if len(rows) else []
                table[f"{signal}_pred"] = np.clip(np.asarray(pred, dtype=float), 0.0, None)
            tables.append(table)
        return pd.concat(tables, ignore_index=True)

    def predict_peak_scores(self, rows: pd.DataFrame) -> np.ndarray:
        if len(rows) == 0:
            return np.zeros(0)
        return self.peak_model.predict_proba(rows[self.peak_features])[:, 1]

    def predict_peaks(self, grid: pd.DataFrame, bucket_start: pd.Timestamp,
                      bucket_end: pd.Timestamp,
                      matrix: Optional[DesignMatrix] = None) -> pd.DataFrame:
        """A.2 scores for every bucket in (bucket_start, bucket_end], each
        predicted from the origin ``peak_horizon`` steps earlier."""
        if matrix is None:
            matrix = self.builder().build(grid, self.peak_horizon)
        frame = matrix.frame
        rows = frame.loc[(frame["target_time"] > bucket_start) & (frame["target_time"] <= bucket_end)]
        scores = self.predict_peak_scores(rows)
        return pd.DataFrame({
            "series_id": rows["series_id"].to_numpy(),
            "bucket_15m": rows["target_time"].to_numpy(),
            "peak_score": np.clip(scores, 0.0, 1.0),
            "peak_pred": (scores >= self.peak_decision_threshold).astype(int),
        })

    @staticmethod
    def predict_detection(grid: pd.DataFrame, bucket_start: pd.Timestamp,
                          bucket_end: pd.Timestamp) -> pd.DataFrame:
        """A.1: the label follows from the validity of the bucket telemetry
        (Sect. 4.2), so the prediction applies the validity rule to the
        bucket itself; the score equals the predicted label."""
        rows = grid.loc[(grid["bucket_15m"] > bucket_start) & (grid["bucket_15m"] <= bucket_end)]
        labels = valid_signal_labels(rows)
        return pd.DataFrame({
            "series_id": rows["series_id"].to_numpy(),
            "bucket_15m": rows["bucket_15m"].to_numpy(),
            "valid_signal_score": labels.astype(float),
            "valid_signal_pred": labels.astype(int),
        })

    # ------------------------------------------------------------------
    def manifest(self) -> Dict:
        return {
            "hyperparameters": self.hp.as_dict(),
            "feature_config": self.feature_config.to_dict(),
            "model_settings": asdict(self.settings),
            "horizons": list(self.horizons),
            "targets": list(self.targets),
            "peak_horizon": self.peak_horizon,
            "peak_decision_threshold": self.peak_decision_threshold,
            "peak_rule": self.peak_rule,
            "peak_quantile": self.peak_quantile,
            "peak_thresholds": self.peak_thresholds,
            "sites": self.sites,
            "training_end": self.training_end,
            "training_rows": self.training_rows,
            "iterations": {f"{s}_h{h}": int(m.n_iter_) for (s, h), m in self.regressors.items()},
            "peak_iterations": int(self.peak_model.n_iter_) if self.peak_model is not None else None,
            "sklearn_version": sklearn.__version__,
        }

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "regressors": self.regressors,
            "regressor_features": self.regressor_features,
            "peak_model": self.peak_model,
            "peak_features": self.peak_features,
        }, directory / "models.joblib", compress=3)
        write_json(directory / "model.json", self.manifest())

    @classmethod
    def load(cls, directory: Path) -> "GreenGRMModel":
        directory = Path(directory)
        manifest_path = directory / "model.json"
        models_path = directory / "models.joblib"
        if not manifest_path.is_file() or not models_path.is_file():
            raise FileNotFoundError(f"{directory} does not contain model.json and models.joblib")
        manifest = read_json(manifest_path)
        model = cls(
            hp=Hyperparameters.from_dict(manifest["hyperparameters"]),
            feature_config=FeatureConfig.from_dict(manifest["feature_config"]),
            settings=ModelSettings.from_dict(manifest["model_settings"]),
            horizons=manifest["horizons"],
            targets=manifest["targets"],
            peak_horizon=manifest["peak_horizon"],
            peak_decision_threshold=manifest["peak_decision_threshold"],
            peak_rule=manifest["peak_rule"],
            peak_quantile=manifest["peak_quantile"],
        )
        stored = joblib.load(models_path)
        model.regressors = stored["regressors"]
        model.regressor_features = stored["regressor_features"]
        model.peak_model = stored["peak_model"]
        model.peak_features = stored["peak_features"]
        model.peak_thresholds = manifest["peak_thresholds"]
        model.sites = manifest["sites"]
        model.training_end = pd.Timestamp(manifest["training_end"]) if manifest["training_end"] else None
        model.training_rows = manifest.get("training_rows", {})
        return model
