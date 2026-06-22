"""Baseline and learned planner adapters."""

from __future__ import annotations

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
    if name == "dqn":
        from .dqn import DQNPlanner

        return DQNPlanner()
    raise ValueError(f"unknown planner {name!r}")

