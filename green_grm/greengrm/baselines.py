"""Hyperparameter selection baselines (Sect. 5).

All baselines use the same search space and evaluation protocol as
Green-GRM, but they rank configurations by the scalar surrogate objective J
of each evaluation directly, without estimated states or rewards.

* default    the scikit-learn default configuration (squared error, eta 0.1,
             100 iterations, unrestricted depth, no L2 regularisation)
* random     random search (Bergstra and Bengio, 2012) without replacement
* bayesian   Bayesian optimisation (Snoek et al., 2012): a Gaussian process
             with a Matern-5/2 kernel on the encoded vectors and the
             expected-improvement acquisition, maximised over all
             configurations not evaluated yet
* tpe        Tree-structured Parzen Estimator (Bergstra et al., 2011): the
             observations are split at the gamma-quantile of J, every
             dimension gets a smoothed categorical density for the good
             (l) and bad (g) group, and the configuration with the largest
             l(x) / g(x) is evaluated next

The search space is a small finite grid, so the acquisition functions of the
Bayesian optimisation and TPE baselines are evaluated on every remaining
configuration instead of on random candidates.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .evaluation import EvaluationLog, HoldoutEvaluator, SurrogateObjective
from .search import SearchResult, rank_configurations
from .search_space import SKLEARN_DEFAULT, Hyperparameters, SearchSpace
from .utils import get_logger


class _ScalarSearch:
    """Shared bookkeeping: evaluate, record J, rank by J."""

    method = "base"

    def __init__(self, evaluator: HoldoutEvaluator, space: SearchSpace,
                 objective: SurrogateObjective, budget: int, seed: int = 42,
                 output_dir: Optional[Path] = None):
        self.evaluator = evaluator
        self.space = space
        self.objective = objective
        self.budget = int(budget)
        self.rng = np.random.default_rng(seed)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.logs: List[EvaluationLog] = []
        self.records: List[Dict] = []
        self.evaluated: Dict[int, float] = {}       # action index -> J

    def _evaluate(self, action: Optional[int], hp: Hyperparameters, note: str = "") -> EvaluationLog:
        log = self.evaluator.evaluate(hp)
        log.surrogate = self.objective(log)
        self.logs.append(log)
        if action is not None:
            self.evaluated[action] = log.surrogate
        record = {"method": self.method, "phase": note or "search", "step": len(self.logs),
                  "action": action}
        record.update(log.to_record())
        self.records.append(record)
        get_logger().info("[%s]   %2d/%d %-40s J=%.4f MAE=%.2f F1=%.3f (%.1fs)",
                          self.method, len(self.logs), self.budget, hp.label(), log.surrogate,
                          log.mae, log.peak_f1, log.seconds)
        return log

    def propose(self) -> int:
        raise NotImplementedError

    def run(self) -> SearchResult:
        start = time.perf_counter()
        self.evaluator.start_session(self.output_dir / "evaluations" if self.output_dir else None)
        n_steps = min(self.budget, len(self.space))
        for _ in range(n_steps):
            action = self.propose()
            self._evaluate(action, self.space.action(action))
        return self._result(start)

    def _result(self, start: float, extra: Optional[Dict] = None) -> SearchResult:
        history = pd.DataFrame(self.records)
        ranking = rank_configurations(history, "surrogate_J", higher_is_better=False)
        best = Hyperparameters.from_dict(ranking.iloc[0].to_dict())
        get_logger().info("[%s] selected %s (J=%.4f)", self.method, best.label(),
                          ranking.iloc[0]["surrogate_J"])
        result = SearchResult(method=self.method, best=best, history=history, ranking=ranking,
                              logs=list(self.logs), seconds=time.perf_counter() - start,
                              extra=dict(extra or {}, objective={"a1": self.objective.a1,
                                                                 "a2": self.objective.a2}))
        if self.output_dir is not None:
            result.save(self.output_dir)
        return result

    def _remaining(self) -> np.ndarray:
        return np.array([i for i in range(len(self.space)) if i not in self.evaluated], dtype=int)


class DefaultConfiguration(_ScalarSearch):
    """A single evaluation of the scikit-learn defaults."""

    method = "default"

    def run(self) -> SearchResult:
        start = time.perf_counter()
        self.evaluator.start_session(self.output_dir / "evaluations" if self.output_dir else None)
        self.budget = 1
        action = self.space.index(SKLEARN_DEFAULT) if self.space.contains(SKLEARN_DEFAULT) else None
        self._evaluate(action, SKLEARN_DEFAULT, note="default")
        return self._result(start)


class RandomSearch(_ScalarSearch):
    method = "random"

    def propose(self) -> int:
        return int(self.rng.choice(self._remaining()))


class BayesianOptimization(_ScalarSearch):
    method = "bayesian"

    def __init__(self, *args, n_initial: int = 8, xi: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_initial = int(n_initial)
        self.xi = float(xi)
        self.encoded = self.space.encode_all()
        self.n_dims = self.encoded.shape[1]

    def propose(self) -> int:
        remaining = self._remaining()
        if len(self.evaluated) < self.n_initial:
            return int(self.rng.choice(remaining))
        actions = np.array(list(self.evaluated.keys()))
        X = self.encoded[actions]
        y = np.array([self.evaluated[a] for a in actions])
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=np.ones(self.n_dims), length_scale_bounds=(1e-2, 1e2), nu=2.5)
                  + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e0)))
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3,
                                      random_state=int(self.rng.integers(1 << 31)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(X, y)
        mu, sigma = gp.predict(self.encoded[remaining], return_std=True)
        best = y.min()
        # Expected improvement for minimisation.
        improvement = best - mu - self.xi
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(sigma > 0, improvement / sigma, 0.0)
        ei = np.where(sigma > 0, improvement * norm.cdf(z) + sigma * norm.pdf(z), 0.0)
        top = np.flatnonzero(ei == ei.max())
        return int(remaining[self.rng.choice(top)])


class TPESearch(_ScalarSearch):
    method = "tpe"

    def __init__(self, *args, n_initial: int = 8, gamma: float = 0.25,
                 prior_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_initial = int(n_initial)
        self.gamma = float(gamma)
        self.prior_weight = float(prior_weight)
        names = list(self.space.grids)
        # Position of every action in every grid dimension.
        self.positions = np.array([
            [self.space.grids[name].index(getattr(hp, name)) for name in names]
            for hp in self.space
        ])
        self.sizes = [len(self.space.grids[name]) for name in names]

    def _log_density(self, members: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Sum over dimensions of the log of a categorical density estimated
        from ``members`` with a uniform prior of weight ``prior_weight``."""
        total = np.zeros(len(candidates))
        for dim, size in enumerate(self.sizes):
            counts = np.bincount(self.positions[members, dim], minlength=size).astype(float)
            probs = (counts + self.prior_weight / size) / (len(members) + self.prior_weight)
            total += np.log(probs[self.positions[candidates, dim]])
        return total

    def propose(self) -> int:
        remaining = self._remaining()
        if len(self.evaluated) < self.n_initial:
            return int(self.rng.choice(remaining))
        actions = np.array(list(self.evaluated.keys()))
        values = np.array([self.evaluated[a] for a in actions])
        n_good = max(1, int(np.ceil(self.gamma * len(actions))))
        order = np.argsort(values, kind="mergesort")
        good, bad = actions[order[:n_good]], actions[order[n_good:]]
        if len(bad) == 0:
            return int(self.rng.choice(remaining))
        score = self._log_density(good, remaining) - self._log_density(bad, remaining)
        top = np.flatnonzero(score == score.max())
        return int(remaining[self.rng.choice(top)])


BASELINES = {
    "default": DefaultConfiguration,
    "random": RandomSearch,
    "bayesian": BayesianOptimization,
    "tpe": TPESearch,
}
