"""Algorithm 1: the Green-GRM hyperparameter search.

Phase 1 (offline calibration of the estimator)
    N0 configurations are sampled uniformly from the action set, each one
    is evaluated, the penalty signals and the surrogate objective J are
    recorded, the reward weights are fitted by least squares and the
    penalty-signal vectors are clustered into K performance clusters.

Phase 2 (online Q-learning search)
    Q(e, lambda) starts at q_init for every estimated state and action; the
    first state is the cluster of the first offline log. For t = 1..B the
    agent picks lambda_t epsilon-greedily, the configuration is evaluated,
    the estimator maps the new log to the estimated reward R_hat_t and the
    next state e', the reward machine issues R_hat_t and Q is updated with
    Eq. (4). Once the labelling function reports ``budget_exhausted`` the
    machine moves to u_done and the episode ends.

Selection
    The evaluated configurations are ranked by their estimated reward
    (averaged when a configuration was evaluated more than once) and the
    top-ranked one, lambda*, is returned for retraining on the full
    training split.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .agent import TabularQLearning
from .estimator import GRMEstimator
from .evaluation import EvaluationLog, HoldoutEvaluator, SurrogateObjective
from .reward_machine import BudgetRewardMachine
from .search_space import Hyperparameters, SearchSpace
from .utils import get_logger, write_json


@dataclass
class GRMSearchConfig:
    n_offline: int = 24                    # N0
    budget: int = 60                       # B
    n_clusters: int = 4                    # K
    alpha: float = 0.5
    gamma: float = 0.9
    epsilon: float = 0.2
    epsilon_min: float = 0.05
    epsilon_decay: float = 1.0
    q_init: object = 0.0                   # float, or "offline_mean"
    offline_sampling: str = "without_replacement"
    estimator_fit: str = "nnls"
    standardize_signals: bool = True
    rank_pool: str = "all"
    reuse_evaluations: bool = False
    seed: int = 42

    @classmethod
    def from_dict(cls, values: Dict) -> "GRMSearchConfig":
        return cls(**{k: v for k, v in values.items() if k in cls.__dataclass_fields__})


@dataclass
class SearchResult:
    """Outcome of one search method."""
    method: str
    best: Hyperparameters
    history: pd.DataFrame
    ranking: pd.DataFrame
    logs: List[EvaluationLog]
    seconds: float
    extra: Dict = field(default_factory=dict)

    def best_log_metrics(self) -> Dict[str, float]:
        """Hold-out metrics of the selected configuration (mean over its
        evaluations)."""
        rows = self.history[self.history["config"] == self.best.label()]
        return {
            "holdout_J": float(rows["surrogate_J"].mean()),
            "holdout_mae": float(rows["mae"].mean()),
            "holdout_peak_f1": float(rows["peak_f1"].mean()),
            "evaluations_of_best": int(len(rows)),
        }

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.history.to_csv(folder / "search_report.csv", index=False)
        self.ranking.to_csv(folder / "ranking.csv", index=False)
        summary = {
            "method": self.method,
            "best_configuration": self.best.as_dict(),
            "best_label": self.best.label(),
            "n_evaluations": len(self.logs),
            "seconds": self.seconds,
            **self.best_log_metrics(),
        }
        summary.update({k: v for k, v in self.extra.items() if k != "q_table"})
        write_json(folder / "summary.json", summary)


def rank_configurations(history: pd.DataFrame, score_column: str, higher_is_better: bool) -> pd.DataFrame:
    """Average repeated evaluations of a configuration and sort."""
    hp_cols = ["loss", "learning_rate", "max_iter", "max_depth", "l2_regularization"]
    aggregations = {c: (c, "first") for c in hp_cols}
    aggregations["n_evaluations"] = ("eval_id", "count")
    for column in (score_column, "surrogate_J", "mae", "peak_f1"):
        aggregations[column] = (column, "mean")
    grouped = history.groupby("config", sort=False).agg(**aggregations).reset_index()
    grouped = grouped.sort_values(score_column, ascending=not higher_is_better, kind="mergesort")
    grouped.insert(0, "rank", np.arange(1, len(grouped) + 1))
    return grouped.reset_index(drop=True)


class GreenGRMSearch:
    """Runs Algorithm 1 on a ``HoldoutEvaluator``."""

    method = "green_grm"

    def __init__(self, evaluator: HoldoutEvaluator, space: SearchSpace,
                 objective: SurrogateObjective, config: GRMSearchConfig,
                 output_dir: Optional[Path] = None):
        self.evaluator = evaluator
        self.space = space
        self.objective = objective
        self.config = config
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.rng = np.random.default_rng(config.seed)
        self._cache: Dict[tuple, EvaluationLog] = {}

    # ------------------------------------------------------------------
    def _evaluate(self, hp: Hyperparameters) -> EvaluationLog:
        if self.config.reuse_evaluations and hp.key() in self._cache:
            return self._cache[hp.key()]
        log = self.evaluator.evaluate(hp)
        log.surrogate = self.objective(log)
        self._cache[hp.key()] = log
        return log

    def run(self) -> SearchResult:
        logger = get_logger()
        cfg = self.config
        start = time.perf_counter()
        self.evaluator.start_session(self.output_dir / "evaluations" if self.output_dir else None)
        records: List[Dict] = []

        # ---------------- Phase 1: offline calibration ----------------
        logger.info("[grm] phase 1: evaluating %d configurations sampled uniformly from %d actions",
                    cfg.n_offline, len(self.space))
        replace = cfg.offline_sampling == "with_replacement"
        offline_actions = self.space.sample_indices(self.rng, cfg.n_offline, replace=replace)
        offline_logs: List[EvaluationLog] = []
        for k, action in enumerate(offline_actions, start=1):
            hp = self.space.action(action)
            log = self._evaluate(hp)
            offline_logs.append(log)
            logger.info("[grm]   offline %2d/%d  %-40s J=%.4f MAE=%.2f F1=%.3f P1=%.4f P2=%.4f (%.1fs)",
                        k, cfg.n_offline, hp.label(), log.surrogate, log.mae, log.peak_f1,
                        log.p1, log.p2, log.seconds)

        estimator = GRMEstimator(n_clusters=cfg.n_clusters, fit_method=cfg.estimator_fit,
                                 standardize=cfg.standardize_signals, seed=cfg.seed)
        estimator.fit(offline_logs, [log.surrogate for log in offline_logs])
        est = estimator.to_dict()
        logger.info("[grm] estimator: J_hat = %.4f*P1 + %.4f*P2%s, R^2 = %.3f, %d clusters %s",
                    est["weights"]["w1"], est["weights"]["w2"],
                    f" + {estimator.intercept:.4f}" if estimator.intercept else "",
                    estimator.r2, estimator.n_clusters, estimator.cluster_sizes)

        for k, (action, log) in enumerate(zip(offline_actions, offline_logs), start=1):
            records.append(self._record(log, phase="offline", step=k, action=action,
                                        state=None, next_state=estimator.state(log),
                                        estimator=estimator))

        # ---------------- Phase 2: online Q-learning ----------------
        q_init = self._initial_q(estimator, offline_logs)
        agent = TabularQLearning(
            n_states=estimator.n_clusters, n_actions=len(self.space), alpha=cfg.alpha,
            gamma=cfg.gamma, epsilon=cfg.epsilon, epsilon_min=cfg.epsilon_min,
            epsilon_decay=cfg.epsilon_decay, q_init=q_init, rng=self.rng)
        machine = BudgetRewardMachine(budget=cfg.budget)
        machine.reset()
        state = estimator.state(offline_logs[0])
        online_logs: List[EvaluationLog] = []
        logger.info("[grm] phase 2: Q-learning for B=%d evaluations (start state e=%d)", cfg.budget, state)

        for t in range(1, cfg.budget + 1):
            epsilon = agent.epsilon
            action, explored = agent.select_action(state)
            hp = self.space.action(action)
            log = self._evaluate(hp)
            online_logs.append(log)
            estimated_reward = estimator.reward(log)
            next_state = estimator.state(log)
            machine_state = machine.state
            reward, next_machine_state, done = machine.step(t, estimated_reward)
            td_error = agent.update(state, action, reward, next_state, terminal=done)
            agent.decay_epsilon()
            records.append(self._record(
                log, phase="online", step=t, action=action, state=state, next_state=next_state,
                estimator=estimator, explored=explored, epsilon=epsilon, reward=reward,
                td_error=td_error, machine_state=machine_state, next_machine_state=next_machine_state))
            logger.info("[grm]   step %2d/%d e=%d %s %-40s R_hat=%+.4f J=%.4f -> e'=%d%s",
                        t, cfg.budget, state, "explore" if explored else "greedy ", hp.label(),
                        estimated_reward, log.surrogate, next_state, " [u_done]" if done else "")
            state = next_state
            if done:
                break

        history = pd.DataFrame(records)
        pool = history if cfg.rank_pool == "all" else history[history["phase"] == "online"]
        ranking = rank_configurations(pool, "estimated_reward", higher_is_better=True)
        best = Hyperparameters.from_dict(ranking.iloc[0].to_dict())
        logger.info("[grm] lambda* = %s (estimated reward %.4f, surrogate J %.4f)",
                    best.label(), ranking.iloc[0]["estimated_reward"], ranking.iloc[0]["surrogate_J"])

        q_table = pd.DataFrame(agent.q, columns=[hp.label() for hp in self.space])
        q_table.insert(0, "state", np.arange(agent.n_states))
        visits = pd.DataFrame(agent.visits, columns=[hp.label() for hp in self.space])
        visits.insert(0, "state", np.arange(agent.n_states))
        result = SearchResult(
            method=self.method, best=best, history=history, ranking=ranking,
            logs=offline_logs + online_logs, seconds=time.perf_counter() - start,
            extra={"estimator": est, "search_config": asdict(cfg),
                   "objective": {"a1": self.objective.a1, "a2": self.objective.a2},
                   "distinct_configurations": int(history["config"].nunique()),
                   "q_table": q_table})
        if self.output_dir is not None:
            result.save(self.output_dir)
            q_table.to_csv(self.output_dir / "q_table.csv", index=False)
            visits.to_csv(self.output_dir / "q_visits.csv", index=False)
            write_json(self.output_dir / "estimator.json", est)
        return result

    def _initial_q(self, estimator: GRMEstimator, offline_logs: List[EvaluationLog]) -> float:
        """Initial Q value. Algorithm 1 starts at 0; since the estimated
        rewards are negative, untried actions then look better than tried
        ones and phase 2 explores the action set systematically (optimistic
        initialisation). ``"offline_mean"`` starts at the discounted return of
        an average offline configuration, so that the agent goes back to
        configurations that scored above average."""
        if self.config.q_init == "offline_mean":
            mean_reward = float(np.mean([estimator.reward(log) for log in offline_logs]))
            return mean_reward / (1.0 - self.config.gamma)
        return float(self.config.q_init)

    # ------------------------------------------------------------------
    def _record(self, log: EvaluationLog, phase: str, step: int, action: int,
                state: Optional[int], next_state: int, estimator: GRMEstimator,
                explored: Optional[bool] = None, epsilon: Optional[float] = None,
                reward: Optional[float] = None, td_error: Optional[float] = None,
                machine_state: Optional[str] = None,
                next_machine_state: Optional[str] = None) -> Dict:
        record = {"method": self.method, "phase": phase, "step": step, "action": action}
        record.update(log.to_record())
        record.update({
            "state": state,
            "next_state": next_state,
            "estimated_J": estimator.estimated_objective(log),
            "estimated_reward": estimator.reward(log),
            "rm_reward": reward,
            "explored": explored,
            "epsilon": epsilon,
            "td_error": td_error,
            "machine_state": machine_state,
            "next_machine_state": next_machine_state,
        })
        return record
