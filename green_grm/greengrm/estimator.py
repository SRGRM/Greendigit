"""The GRM estimator E: O -> E and the reward estimator (Sect. 3.3, 4.5).

In the offline phase every evaluation log is summarised by its penalty
signals (P1, P2). A linear model with weights w = (w1, w2) relates them to
the surrogate objective,

    J_hat(o) = w1 * P1(o) + w2 * P2(o),        R_hat(o) = -J_hat(o)    (Eq. 6)

and the weights are fitted by least squares on the logs of randomly
sampled configurations (Algorithm 1, line 8). P1 and P2 are penalty
signals, so by default the fit is constrained to w >= 0 (non-negative least
squares): the agent then maximises reward by lowering both the error level
and its dispersion, as described in Sect. 4.5. ``fit_method="ols"`` gives the
unconstrained solution. The penalty-signal vectors of
the same logs are clustered into K performance clusters; the estimator maps
a new log to the cluster with the nearest centroid (line 9). The cluster
index is the estimated state of the agent.

Clusters are renumbered by the estimated objective of their centroid, so
state 0 is always the best region of the performance landscape. This does
not change the method, it only makes the logs easier to read.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.optimize import nnls
from sklearn.cluster import KMeans

from .evaluation import EvaluationLog


class GRMEstimator:
    def __init__(self, n_clusters: int = 4, fit_method: str = "nnls",
                 standardize: bool = True, seed: int = 42):
        if fit_method not in ("ols", "nnls", "ols_intercept"):
            raise ValueError("fit_method must be 'ols', 'nnls' or 'ols_intercept'")
        self.n_clusters = int(n_clusters)
        self.fit_method = fit_method
        self.standardize = standardize
        self.seed = seed
        self.weights = np.zeros(2)
        self.intercept = 0.0
        self.r2 = float("nan")
        self.centroids = np.zeros((0, 2))       # in penalty-signal units
        self._mean = np.zeros(2)
        self._std = np.ones(2)
        self.cluster_sizes: List[int] = []
        self.fitted = False

    # ------------------------------------------------------------------
    @staticmethod
    def signals(logs: Sequence[EvaluationLog]) -> np.ndarray:
        return np.vstack([log.penalty_signals for log in logs])

    def fit(self, logs: Sequence[EvaluationLog], objective: Sequence[float]) -> "GRMEstimator":
        X = self.signals(logs)
        y = np.asarray(objective, dtype=float)
        if len(X) < 3:
            raise ValueError("at least three offline evaluations are needed to fit the estimator")

        # Reward weights (Eq. 6) by least squares.
        if self.fit_method == "ols":
            self.weights, *_ = np.linalg.lstsq(X, y, rcond=None)
            self.intercept = 0.0
        elif self.fit_method == "nnls":
            self.weights, _ = nnls(X, y)
            self.intercept = 0.0
        else:
            design = np.column_stack([X, np.ones(len(X))])
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)
            self.weights, self.intercept = coef[:2], float(coef[2])
        fitted = X @ self.weights + self.intercept
        ss_res = float(np.sum((y - fitted) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        self.r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        # Performance clusters on the penalty-signal vectors.
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std == 0] = 1.0
        Z = self._scale(X)
        k = min(self.n_clusters, len(np.unique(Z.round(12), axis=0)))
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=self.seed).fit(Z)
        centroids = self._unscale(kmeans.cluster_centers_)
        order = np.argsort(centroids @ self.weights + self.intercept)
        self.centroids = centroids[order]
        labels = np.argsort(order)[kmeans.labels_]
        self.cluster_sizes = [int(np.sum(labels == i)) for i in range(k)]
        self.n_clusters = k
        self.fitted = True
        return self

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mean) / self._std if self.standardize else X

    def _unscale(self, Z: np.ndarray) -> np.ndarray:
        return Z * self._std + self._mean if self.standardize else Z

    # ------------------------------------------------------------------
    def estimated_objective(self, log: EvaluationLog) -> float:
        """J_hat(o) = w1 P1(o) + w2 P2(o)."""
        return float(log.penalty_signals @ self.weights + self.intercept)

    def reward(self, log: EvaluationLog) -> float:
        """R_hat(o) = -J_hat(o)."""
        return -self.estimated_objective(log)

    def state(self, log: EvaluationLog) -> int:
        """E(o): index of the performance cluster with the nearest centroid."""
        z = self._scale(log.penalty_signals[None, :])
        distances = np.linalg.norm(self._scale(self.centroids) - z, axis=1)
        return int(np.argmin(distances))

    def to_dict(self) -> Dict:
        return {
            "fit_method": self.fit_method,
            "weights": {"w1": float(self.weights[0]), "w2": float(self.weights[1])},
            "intercept": self.intercept,
            "r2": self.r2,
            "n_clusters": self.n_clusters,
            "standardize": self.standardize,
            "centroids": [
                {"state": i, "P1": float(c[0]), "P2": float(c[1]),
                 "J_hat": float(c @ self.weights + self.intercept),
                 "n_offline_logs": self.cluster_sizes[i]}
                for i, c in enumerate(self.centroids)
            ],
        }
