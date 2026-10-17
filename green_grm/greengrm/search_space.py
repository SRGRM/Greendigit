"""Hyperparameter vectors and the finite action set of the agent (Sect. 4.4,
Table 1).

A hyperparameter vector is lambda = (loss, eta, T, d, rho): regression loss,
learning rate, maximum number of boosting iterations, maximum tree depth and
L2 regularisation. The search space is the Cartesian product of the discrete
grids below; every element is one action of the Q-learning agent.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Hyperparameters:
    loss: str                       # "absolute_error" | "squared_error"
    learning_rate: float            # eta
    max_iter: int                   # T
    max_depth: Optional[int]        # d (None = unrestricted)
    l2_regularization: float        # rho

    def as_dict(self) -> Dict:
        return asdict(self)

    def key(self) -> tuple:
        return (self.loss, float(self.learning_rate), int(self.max_iter),
                -1 if self.max_depth is None else int(self.max_depth),
                float(self.l2_regularization))

    def label(self) -> str:
        loss = {"absolute_error": "abs", "squared_error": "sq"}.get(self.loss, self.loss)
        depth = "None" if self.max_depth is None else str(self.max_depth)
        return (f"{loss}|eta={self.learning_rate:g}|T={self.max_iter}"
                f"|d={depth}|rho={self.l2_regularization:g}")

    @classmethod
    def from_dict(cls, values: Dict) -> "Hyperparameters":
        depth = values.get("max_depth")
        if depth is not None and (isinstance(depth, float) and math.isnan(depth)):
            depth = None
        return cls(loss=str(values["loss"]),
                   learning_rate=float(values["learning_rate"]),
                   max_iter=int(values["max_iter"]),
                   max_depth=None if depth in (None, "None", -1) else int(depth),
                   l2_regularization=float(values["l2_regularization"]))


# scikit-learn defaults of HistGradientBoostingRegressor: the "default
# configuration" baseline of Sect. 5.
SKLEARN_DEFAULT = Hyperparameters(loss="squared_error", learning_rate=0.1, max_iter=100,
                                  max_depth=None, l2_regularization=0.0)


class SearchSpace:
    """Discrete search space Lambda. Actions are indexed in a fixed order."""

    def __init__(self, losses: Sequence[str], learning_rates: Sequence[float],
                 max_iters: Sequence[int], max_depths: Sequence[Optional[int]],
                 l2_regularizations: Sequence[float]):
        self.grids = {
            "loss": list(losses),
            "learning_rate": [float(v) for v in learning_rates],
            "max_iter": [int(v) for v in max_iters],
            "max_depth": [None if v is None else int(v) for v in max_depths],
            "l2_regularization": [float(v) for v in l2_regularizations],
        }
        for name, values in self.grids.items():
            if not values:
                raise ValueError(f"empty grid for {name}")
        self._actions = [
            Hyperparameters(*combo)
            for combo in itertools.product(*self.grids.values())
        ]
        self._index = {hp.key(): i for i, hp in enumerate(self._actions)}

    @classmethod
    def from_dict(cls, values: Dict) -> "SearchSpace":
        return cls(values["loss"], values["learning_rate"], values["max_iter"],
                   values["max_depth"], values["l2_regularization"])

    def __len__(self) -> int:
        return len(self._actions)

    def __iter__(self):
        return iter(self._actions)

    def action(self, index: int) -> Hyperparameters:
        return self._actions[index]

    def index(self, hp: Hyperparameters) -> int:
        return self._index[hp.key()]

    def contains(self, hp: Hyperparameters) -> bool:
        return hp.key() in self._index

    def sample_indices(self, rng: np.random.Generator, n: int, replace: bool) -> List[int]:
        if not replace and n > len(self):
            raise ValueError(f"cannot draw {n} distinct actions from {len(self)}")
        return [int(i) for i in rng.choice(len(self), size=n, replace=replace)]

    # ------------------------------------------------------------------
    # Numeric encoding used by the Bayesian optimisation baseline
    # ------------------------------------------------------------------
    def encode(self, hp: Hyperparameters) -> np.ndarray:
        """Map a vector to [0, 1]^5: categorical choices become their grid
        position, numeric ones are min-max scaled (log scale for eta)."""
        def scaled(value, grid, log=False):
            lo, hi = min(grid), max(grid)
            if hi == lo:
                return 0.0
            if log:
                return (math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo))
            return (value - lo) / (hi - lo)

        depth_grid = self.grids["max_depth"]
        depth_pos = depth_grid.index(hp.max_depth)
        return np.array([
            self.grids["loss"].index(hp.loss) / max(1, len(self.grids["loss"]) - 1),
            scaled(hp.learning_rate, self.grids["learning_rate"], log=True),
            scaled(hp.max_iter, self.grids["max_iter"]),
            depth_pos / max(1, len(depth_grid) - 1),
            scaled(hp.l2_regularization, self.grids["l2_regularization"]),
        ], dtype=float)

    def encode_all(self) -> np.ndarray:
        return np.vstack([self.encode(hp) for hp in self._actions])

    def describe(self) -> str:
        parts = [f"{name}={values}" for name, values in self.grids.items()]
        return f"{len(self)} actions: " + ", ".join(parts)

