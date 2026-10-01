"""Baseline and learned planner adapters."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .actions import Action, MIN_COSTS, masked_argmax


class Planner:
    name = "planner"

    def select(self, state: list[float], mask: list[bool]) -> int:
        raise NotImplementedError


class ExistingTestsPlanner(Planner):
    name = "existing"

    def select(self, state: list[float], mask: list[bool]) -> int:
        return int(Action.EXPAND_COUNTEREXAMPLE) if mask[int(Action.EXPAND_COUNTEREXAMPLE)] else int(Action.STOP)


@dataclass
class FixedSchedulePlanner(Planner):
    name = "fixed"
    cursor: int = 0
    schedule: tuple[Action, ...] = (
        Action.LLM_SYNTHESIZE_INSTRUCTIONS,
        Action.GENERATE_BOUNDARY_TESTS,
        Action.PROPERTY_BASED_FUZZ,
        Action.LLM_GENERATE_ASSERTIONS,
        Action.STOP,
    )

    def select(self, state: list[float], mask: list[bool]) -> int:
        while self.cursor < len(self.schedule):
            action = self.schedule[self.cursor]
            self.cursor += 1
            if mask[int(action)]:
                return int(action)
        return int(Action.STOP)


class RandomPlanner(Planner):
    name = "random"

    def select(self, state: list[float], mask: list[bool]) -> int:
        valid = [idx for idx, ok in enumerate(mask) if ok]
        return random.choice(valid) if valid else int(Action.STOP)


class GreedyPlanner(Planner):
    name = "greedy"

    def select(self, state: list[float], mask: list[bool]) -> int:
        scores: list[float] = []
        for action in Action:
            cost = sum(MIN_COSTS[action].values()) or 0.1
            if action == Action.STOP:
                scores.append(0.0)
            elif action in {Action.GENERATE_BOUNDARY_TESTS, Action.PROPERTY_BASED_FUZZ}:
                scores.append(1.0 / cost)
            elif action.name.startswith("LLM"):
                scores.append(0.5 / cost)
            else:
                scores.append(0.25 / cost)
        return masked_argmax(scores, mask)


class ContextualBanditPlanner(Planner):
    """Online linear contextual bandit: ``Q(s, a) = w_a . s + b_a``.

    The per-action weight vector consumes the state (context) features, and the
    bias term ``b_a`` recovers the state-agnostic action mean as a special case.
    Weights are learned online with a gradient step on the squared prediction
    error, so the choice adapts to remaining budget, discovered failures, and
    action-family saturation rather than ranking actions on a global average.
    """

    name = "bandit"

    def __init__(self, lr: float = 0.05):
        self.lr = lr
        self.weights: list[list[float]] | None = None
        self.bias = [0.0 for _ in Action]
        self.last_state: list[float] | None = None
        self.last_action: int | None = None

    def _ensure_dim(self, state_dim: int) -> None:
        if self.weights is None:
            self.weights = [[0.0 for _ in range(state_dim)] for _ in Action]

    def _scores(self, state: list[float]) -> list[float]:
        assert self.weights is not None
        return [
            sum(w_i * s_i for w_i, s_i in zip(self.weights[a], state)) + self.bias[a]
            for a in range(len(Action))
        ]

    def select(self, state: list[float], mask: list[bool]) -> int:
        self._ensure_dim(len(state))
        self.last_state = list(state)
        action = masked_argmax(self._scores(state), mask)
        self.last_action = action
        return action

    def update(self, action: int, reward: float) -> None:
        if self.last_state is None or self.weights is None:
            return
        predicted = sum(w_i * s_i for w_i, s_i in zip(self.weights[action], self.last_state)) + self.bias[action]
        error = reward - predicted
        for i, s_i in enumerate(self.last_state):
            self.weights[action][i] += self.lr * error * s_i
        self.bias[action] += self.lr * error


class LinUCBPlanner(ContextualBanditPlanner):
    """Disjoint LinUCB (Li et al., 2010): per-action ridge regression plus an
    exploration bonus ``alpha * sqrt(x^T A^-1 x)``. ``A^-1`` is kept directly
    with Sherman-Morrison rank-1 updates, so no linear-algebra dependency is
    needed. STOP is never explored: it keeps a fixed score of 0."""

    name = "linucb"

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.a_inv: list[list[list[float]]] | None = None
        self.b: list[list[float]] | None = None

    def select(self, state: list[float], mask: list[bool]) -> int:
        x = list(state) + [1.0]
        if self.a_inv is None:
            d = len(x)
            self.a_inv = [[[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)] for _ in Action]
            self.b = [[0.0] * d for _ in Action]
        scores: list[float] = []
        for a in Action:
            if a == Action.STOP:
                scores.append(0.0)
                continue
            inv_x = [sum(r * xi for r, xi in zip(row, x)) for row in self.a_inv[a]]
            mean = sum(bi * v for bi, v in zip(self.b[a], inv_x))  # theta.x = b.(A^-1 x), A^-1 symmetric
            width = sum(xi * v for xi, v in zip(x, inv_x))
            scores.append(mean + self.alpha * math.sqrt(max(width, 0.0)))
        self.last_state = x
        return masked_argmax(scores, mask)

    def update(self, action: int, reward: float) -> None:
        if self.last_state is None or self.a_inv is None or action == Action.STOP:
            return
        x, inv = self.last_state, self.a_inv[action]
        inv_x = [sum(r * xi for r, xi in zip(row, x)) for row in inv]
        denom = 1.0 + sum(xi * v for xi, v in zip(x, inv_x))
        for i, vi in enumerate(inv_x):
            row = inv[i]
            for j, vj in enumerate(inv_x):
                row[j] -= vi * vj / denom
        self.b[action] = [bi + reward * xi for bi, xi in zip(self.b[action], x)]


def planner_from_name(name: str) -> Planner:
    if name == "existing":
        return ExistingTestsPlanner()
    if name == "fixed":
        return FixedSchedulePlanner()
    if name == "random":
        return RandomPlanner()
    if name == "greedy":
        return GreedyPlanner()
    if name == "bandit":
        return ContextualBanditPlanner()
    if name == "linucb":
        return LinUCBPlanner()
    if name == "dqn":
        from .dqn import DQNPlanner

        return DQNPlanner()
    raise ValueError(f"unknown planner {name!r}")

