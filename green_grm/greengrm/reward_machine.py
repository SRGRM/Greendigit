"""Reward machine of the hyperparameter search (Sect. 3.2 and 4.5).

R = <U, u_I, F, AP, delta_u, delta_r> with

    U   = {u_search, u_done}       machine states
    u_I = u_search                 initial state
    F   = {u_done}                 terminal states
    AP  = {budget_exhausted}       atomic propositions

The labelling function L reports ``budget_exhausted`` once the number of
evaluations reaches the budget B. In u_search the reward function delta_r
issues the estimated reward R_hat of the newest evaluation log; the machine
then moves to u_done and the episode terminates. As in the GRM framework,
the propositions and rewards are evaluated on estimated transitions
(e, lambda, e') rather than on the hidden true state of the evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, FrozenSet, Tuple

BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass
class BudgetRewardMachine:
    budget: int
    SEARCH: ClassVar[str] = "u_search"
    DONE: ClassVar[str] = "u_done"

    def __post_init__(self) -> None:
        if self.budget < 1:
            raise ValueError("the search budget must be at least one evaluation")
        self.state = self.SEARCH

    @property
    def initial_state(self) -> str:
        return self.SEARCH

    @property
    def terminal_states(self) -> FrozenSet[str]:
        return frozenset({self.DONE})

    def reset(self) -> str:
        self.state = self.SEARCH
        return self.state

    def labeling(self, n_evaluations: int) -> FrozenSet[str]:
        """L: propositions that hold after ``n_evaluations`` online evaluations."""
        return frozenset({BUDGET_EXHAUSTED}) if n_evaluations >= self.budget else frozenset()

    def transition(self, state: str, propositions: FrozenSet[str]) -> str:
        """delta_u"""
        if state == self.DONE or BUDGET_EXHAUSTED in propositions:
            return self.DONE
        return self.SEARCH

    def reward(self, state: str, estimated_reward: float) -> float:
        """delta_r: the estimated reward is issued while searching; the
        terminal state issues no reward."""
        return float(estimated_reward) if state == self.SEARCH else 0.0

    def step(self, n_evaluations: int, estimated_reward: float) -> Tuple[float, str, bool]:
        """Advance the machine after one evaluation.

        Returns (reward, next machine state, episode finished).
        """
        reward = self.reward(self.state, estimated_reward)
        self.state = self.transition(self.state, self.labeling(n_evaluations))
        return reward, self.state, self.state in self.terminal_states
