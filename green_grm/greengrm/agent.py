"""Tabular Q-learning over the estimated states (Sect. 3.1, Eq. 4).

    Q(e, a) <- Q(e, a) + alpha * [r + gamma * max_a' Q(e', a') - Q(e, a)]

Actions are chosen epsilon-greedily; ties between equally valued actions
are broken at random. On the final step of the episode (the reward machine
reached its terminal state) the target is the reward alone.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class TabularQLearning:
    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.5, gamma: float = 0.9,
                 epsilon: float = 0.2, epsilon_min: float = 0.05, epsilon_decay: float = 1.0,
                 q_init: float = 0.0, rng: np.random.Generator = None):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must be in [0, 1)")
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q = np.full((n_states, n_actions), float(q_init))
        self.visits = np.zeros((n_states, n_actions), dtype=int)
        self.rng = rng if rng is not None else np.random.default_rng()

    def greedy_action(self, state: int) -> int:
        row = self.q[state]
        best = np.flatnonzero(row == row.max())
        return int(self.rng.choice(best))

    def select_action(self, state: int) -> Tuple[int, bool]:
        """Return (action, explored) where ``explored`` is True for a random
        exploratory choice."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions)), True
        return self.greedy_action(state), False

    def update(self, state: int, action: int, reward: float, next_state: int,
               terminal: bool) -> float:
        """One Q-learning update; returns the temporal-difference error."""
        bootstrap = 0.0 if terminal else self.gamma * float(self.q[next_state].max())
        td_error = reward + bootstrap - self.q[state, action]
        self.q[state, action] += self.alpha * td_error
        self.visits[state, action] += 1
        return float(td_error)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
