"""Evaluation of a hyperparameter vector on the chronological hold-out
split (Sect. 4.5).

Evaluating lambda means training the four regressors and the peak classifier
with lambda on the search-training part of the training data, predicting the
hold-out part and recording an evaluation log o(lambda). The log contains

* MAE(lambda)   mean absolute error of the forecasts (mean over the four
                target/horizon pairs),
* F1(lambda)    F1 score of the peak detector,
* m_1..m_n      metric samples recorded during the evaluation.

The metric samples are dimensionless so that errors of different targets can
be pooled:

* per-target, per-horizon and per-site hold-out errors, each divided by the
  mean absolute value of that hold-out target (normalised MAE),
* the normalised hold-out error of each regressor recorded over the boosting
  iterations at the checkpoints in ``stage_checkpoints`` (fractions of the
  iterations actually performed),
* the error 1 - F1 of the peak detector per site and over the boosting
  iterations.

From the samples the two penalty signals of the GRM are computed,
P1(o) = mean(m) and P2(o) = std(m) (population standard deviation).

The surrogate objective (Eq. 5) is J = a1 * MAE + a2 * (1 - F1).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .data import SIGNAL_COLUMNS, SplitSpec, compute_peak_thresholds, peak_labels
from .features import DesignMatrix, FeatureBuilder, FeatureConfig
from .models import (ModelSettings, make_peak_classifier, make_regressor, peak_rows,
                     regression_rows)
from .search_space import Hyperparameters
from .utils import get_logger, write_json


@dataclass
class EvaluationLog:
    """o(lambda): the outcome of one evaluation."""
    eval_id: int
    hyperparameters: Hyperparameters
    mae: float
    mae_by_target: Dict[str, float]
    peak_f1: float
    peak_auroc: float
    peak_precision: float
    peak_recall: float
    metric_samples: np.ndarray
    sample_names: List[str]
    p1: float
    p2: float
    iterations: Dict[str, int]
    seconds: float
    seed: int
    surrogate: Optional[float] = None
    extra: Dict = field(default_factory=dict)

    @property
    def penalty_signals(self) -> np.ndarray:
        return np.array([self.p1, self.p2], dtype=float)

    def to_record(self) -> Dict:
        """Flat dictionary for the CSV report."""
        record = {"eval_id": self.eval_id, "config": self.hyperparameters.label()}
        record.update(self.hyperparameters.as_dict())
        depth = self.hyperparameters.max_depth
        record["max_depth"] = "None" if depth is None else depth
        record.update({
            "mae": self.mae,
            **{f"mae_{k}": v for k, v in self.mae_by_target.items()},
            "peak_f1": self.peak_f1,
            "peak_auroc": self.peak_auroc,
            "peak_precision": self.peak_precision,
            "peak_recall": self.peak_recall,
            "p1": self.p1,
            "p2": self.p2,
            "n_samples": int(len(self.metric_samples)),
            "surrogate_J": self.surrogate,
            "seconds": self.seconds,
            "seed": self.seed,
        })
        record.update({f"iter_{k}": v for k, v in self.iterations.items()})
        record.update({k: v for k, v in self.extra.items() if np.isscalar(v)})
        return record

    def to_json(self) -> Dict:
        payload = self.to_record()
        payload["hyperparameters"] = self.hyperparameters.as_dict()
        payload["metric_samples"] = dict(zip(self.sample_names, self.metric_samples.tolist()))
        payload.update(self.extra)
        return payload


@dataclass(frozen=True)
class SurrogateObjective:
    """J(lambda) = a1 * MAE(lambda) + a2 * (1 - F1_peak(lambda)) (Eq. 5);
    lower is better."""
    a1: float
    a2: float

    def __call__(self, log: EvaluationLog) -> float:
        return self.a1 * log.mae + self.a2 * (1.0 - log.peak_f1)


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # A site without peaks in the window and without predicted peaks counts
    # as a perfect detection rather than as F1 = 0.
    return float(f1_score(y_true, y_pred, zero_division=1.0))


def _checkpoints(n_iter: int, fractions: Sequence[float]) -> List[int]:
    points = sorted({max(1, min(n_iter, int(round(f * n_iter)))) for f in fractions})
    return points


class HoldoutEvaluator:
    """Produces o(lambda) for any lambda of the search space.

    The design matrices, row selections and normalisation scales are built
    once; every call to ``evaluate`` only trains and scores the models.
    """

    def __init__(self, grid: pd.DataFrame, split: SplitSpec, feature_config: FeatureConfig,
                 settings: ModelSettings, horizons: Sequence[int] = (4, 96),
                 targets: Sequence[str] = SIGNAL_COLUMNS, peak_horizon: int = 4,
                 peak_rule: str = "per_signal_or", peak_quantile: float = 0.95,
                 peak_decision_threshold: float = 0.5,
                 stage_checkpoints: Sequence[float] = (0.5, 0.75, 1.0),
                 seed_policy: str = "fixed",
                 save_dir: Optional[Path] = None, save_models: bool = False,
                 save_predictions: bool = False):
        logger = get_logger()
        if seed_policy not in ("fixed", "per_evaluation"):
            raise ValueError("seed_policy must be 'fixed' or 'per_evaluation'")
        self.split = split
        self.settings = settings
        self.horizons = tuple(int(h) for h in horizons)
        self.targets = tuple(targets)
        self.peak_horizon = int(peak_horizon)
        self.peak_decision_threshold = float(peak_decision_threshold)
        self.stage_checkpoints = tuple(float(f) for f in stage_checkpoints)
        self.seed_policy = seed_policy
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.save_models = save_models
        self.save_predictions = save_predictions
        self.sites = sorted(grid["series_id"].unique().tolist())
        self.n_evaluations = 0

        # Peak thresholds come from the search-training part only, so that
        # the hold-out labels are not used to define themselves.
        search_train = grid[grid["bucket_15m"] <= split.inner_cutoff]
        self.peak_thresholds = compute_peak_thresholds(search_train, peak_quantile, peak_rule)
        labels = peak_labels(grid, self.peak_thresholds, peak_rule)

        builder = FeatureBuilder(feature_config, self.sites)
        self.matrices: Dict[object, DesignMatrix] = {h: builder.build(grid, h) for h in self.horizons}
        self.matrices["peak"] = builder.build(grid, self.peak_horizon, labels)

        # Rows: training targets up to the inner cut-off; hold-out origins
        # after it with targets up to the official cut-off.
        self.train_rows: Dict[object, pd.DataFrame] = {}
        self.holdout_rows: Dict[object, pd.DataFrame] = {}
        self.scales: Dict[str, float] = {}
        self.site_scales: Dict[str, Dict[str, float]] = {}
        for h in self.horizons:
            for signal in self.targets:
                key = (signal, h)
                self.train_rows[key] = regression_rows(self.matrices[h], signal, None, split.inner_cutoff)
                hold = regression_rows(self.matrices[h], signal, None, split.cutoff,
                                       origin_after=split.inner_cutoff)
                self.holdout_rows[key] = hold
                name = f"{signal}_h{h}"
                self.scales[name] = float(hold[f"y_{signal}"].abs().mean())
                self.site_scales[name] = {
                    site: float(g[f"y_{signal}"].abs().mean()) for site, g in hold.groupby("series_id")
                }
                logger.info("[eval] %-14s train %s rows | hold-out %s rows | mean |y| = %.2f",
                            name, f"{len(self.train_rows[key]):,}", f"{len(hold):,}", self.scales[name])
        self.train_rows["peak"] = peak_rows(self.matrices["peak"], None, split.inner_cutoff)
        self.holdout_rows["peak"] = peak_rows(self.matrices["peak"], None, split.cutoff,
                                              origin_after=split.inner_cutoff)
        logger.info("[eval] peak (h=%d)     train %s rows (%.1f%% peaks) | hold-out %s rows (%.1f%% peaks)",
                    self.peak_horizon, f"{len(self.train_rows['peak']):,}",
                    100 * self.train_rows["peak"]["y_peak"].mean(),
                    f"{len(self.holdout_rows['peak']):,}",
                    100 * self.holdout_rows["peak"]["y_peak"].mean())
        for key, rows in list(self.train_rows.items()) + list(self.holdout_rows.items()):
            if rows.empty:
                raise ValueError(f"empty row selection for {key}; check the split settings")

    # ------------------------------------------------------------------
    def start_session(self, save_dir: Optional[Path]) -> None:
        """Start a new search method: evaluation ids restart at 0 and logs
        go to ``save_dir`` (one folder per evaluation)."""
        self.n_evaluations = 0
        self.save_dir = Path(save_dir) if save_dir is not None else None

    def default_surrogate(self, a1="auto", a2: float = 1.0) -> SurrogateObjective:
        """a1 = "auto" scales the MAE by the mean absolute hold-out target so
        that both terms of J are of order one."""
        if a1 == "auto":
            a1 = 1.0 / float(np.mean(list(self.scales.values())))
        return SurrogateObjective(a1=float(a1), a2=float(a2))

    def _seed_for(self, eval_id: int) -> int:
        if self.seed_policy == "fixed":
            return self.settings.seed
        return self.settings.seed + eval_id

    # ------------------------------------------------------------------
    def evaluate(self, hp: Hyperparameters) -> EvaluationLog:
        eval_id = self.n_evaluations
        self.n_evaluations += 1
        seed = self._seed_for(eval_id)
        start = time.perf_counter()

        samples: List[float] = []
        names: List[str] = []
        mae_by_target: Dict[str, float] = {}
        iterations: Dict[str, int] = {}
        predictions = []
        models = {}

        for h in self.horizons:
            matrix = self.matrices[h]
            for signal in self.targets:
                key = (signal, h)
                name = f"{signal}_h{h}"
                features = matrix.features_for(signal)
                train = self.train_rows[key]
                hold = self.holdout_rows[key]
                model = make_regressor(hp, self.settings, matrix.categorical_mask(features), seed=seed)
                model.fit(train[features], train[f"y_{signal}"])
                iterations[name] = int(model.n_iter_)
                y_true = hold[f"y_{signal}"].to_numpy()

                # Hold-out error recorded over the boosting iterations.
                stage_points = _checkpoints(model.n_iter_, self.stage_checkpoints)
                y_pred = None
                for stage, stage_pred in enumerate(model.staged_predict(hold[features]), start=1):
                    if stage in stage_points:
                        stage_pred = np.clip(stage_pred, 0.0, None)
                        stage_mae = float(np.mean(np.abs(stage_pred - y_true)))
                        samples.append(stage_mae / self.scales[name])
                        names.append(f"stage/{name}/{stage}")
                    if stage == model.n_iter_:
                        y_pred = np.clip(stage_pred, 0.0, None)
                mae_by_target[name] = float(np.mean(np.abs(y_pred - y_true)))

                # Per-site errors of the final model.
                errors = np.abs(y_pred - y_true)
                for site, idx in hold.groupby("series_id").indices.items():
                    samples.append(float(errors[idx].mean()) / self.site_scales[name][site])
                    names.append(f"site/{name}/{site}")

                if self.save_predictions:
                    predictions.append(pd.DataFrame({
                        "series_id": hold["series_id"].to_numpy(),
                        "origin": hold["origin"].to_numpy(),
                        "target_time": hold["target_time"].to_numpy(),
                        "target": signal, "horizon_steps_15m": h,
                        "y_true": y_true, "y_pred": y_pred,
                    }))
                if self.save_models:
                    models[name] = model

        # Peak classifier.
        peak_matrix = self.matrices["peak"]
        features = peak_matrix.feature_names
        train = self.train_rows["peak"]
        hold = self.holdout_rows["peak"]
        classifier = make_peak_classifier(hp, self.settings, peak_matrix.categorical_mask(features),
                                          seed=seed)
        classifier.fit(train[features], train["y_peak"].astype(int))
        iterations["peak"] = int(classifier.n_iter_)
        y_peak = hold["y_peak"].astype(int).to_numpy()
        stage_points = _checkpoints(classifier.n_iter_, self.stage_checkpoints)
        scores = None
        for stage, proba in enumerate(classifier.staged_predict_proba(hold[features]), start=1):
            if stage in stage_points:
                pred = (proba[:, 1] >= self.peak_decision_threshold).astype(int)
                samples.append(1.0 - _f1(y_peak, pred))
                names.append(f"stage/peak/{stage}")
            if stage == classifier.n_iter_:
                scores = proba[:, 1]
        peak_pred = (scores >= self.peak_decision_threshold).astype(int)
        for site, idx in hold.groupby("series_id").indices.items():
            samples.append(1.0 - _f1(y_peak[idx], peak_pred[idx]))
            names.append(f"site/peak/{site}")
        peak_f1 = _f1(y_peak, peak_pred)
        peak_auroc = float(roc_auc_score(y_peak, scores)) if len(np.unique(y_peak)) > 1 else 0.5
        if self.save_predictions:
            predictions.append(pd.DataFrame({
                "series_id": hold["series_id"].to_numpy(),
                "origin": hold["origin"].to_numpy(),
                "target_time": hold["target_time"].to_numpy(),
                "target": "peak", "horizon_steps_15m": self.peak_horizon,
                "y_true": y_peak, "y_pred": scores,
            }))
        if self.save_models:
            models["peak"] = classifier

        sample_array = np.asarray(samples, dtype=float)
        log = EvaluationLog(
            eval_id=eval_id,
            hyperparameters=hp,
            mae=float(np.mean(list(mae_by_target.values()))),
            mae_by_target=mae_by_target,
            peak_f1=peak_f1,
            peak_auroc=peak_auroc,
            peak_precision=float(precision_score(y_peak, peak_pred, zero_division=0)),
            peak_recall=float(recall_score(y_peak, peak_pred, zero_division=0)),
            metric_samples=sample_array,
            sample_names=names,
            p1=float(sample_array.mean()),
            p2=float(sample_array.std()),
            iterations=iterations,
            seconds=time.perf_counter() - start,
            seed=seed,
        )
        self._save(log, predictions, models)
        return log

    # ------------------------------------------------------------------
    def _save(self, log: EvaluationLog, predictions: List[pd.DataFrame], models: Dict) -> None:
        if self.save_dir is None:
            return
        folder = self.save_dir / f"eval_{log.eval_id:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "log.json", log.to_json())
        if predictions:
            pd.concat(predictions, ignore_index=True).to_csv(
                folder / "holdout_predictions.csv.gz", index=False, compression="gzip")
        if models:
            joblib.dump(models, folder / "models.joblib", compress=3)
