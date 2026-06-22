"""Reward function with separately logged components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .models import ExecutionResult


# Per-resource scales that map heterogeneous cost units (tokens, seconds, fuzz
# iterations) onto a common dimensionless scale. Shared so reward and state
# features measure cost the same way.
COST_SCALES: dict[str, float] = {
    "tokens": 1000.0,
    "runtime_seconds": 60.0,
    "fuzz_iterations": 100.0,
    "solver_seconds": 60.0,
}


def normalized_cost(cost: Mapping[str, float]) -> float:
    """Dimensionless cost: each resource divided by its scale, then summed."""
    return sum(cost.get(key, 0.0) / scale for key, scale in COST_SCALES.items())


@dataclass(slots=True)
class RewardConfig:
    alpha_failure: float = 10.0
    beta_coverage: float = 1.0
    gamma_uncertainty: float = 1.0
    eta_risk: float = 1.0
    lambda_cost: float = 1.0
    rho_bad_tests: float = 0.5


def compute_reward(result: ExecutionResult, config: RewardConfig | None = None) -> tuple[float, dict[str, float]]:
    config = config or RewardConfig()
    normalized = normalized_cost(result.cost)
    bad_tests = result.invalid_tests + result.flaky_tests + result.duplicate_tests
    components = {
        "failure": config.alpha_failure * result.failures,
        "coverage": config.beta_coverage * result.coverage_delta,
        "uncertainty": config.gamma_uncertainty * result.uncertainty_delta,
        "risk": config.eta_risk * result.residual_risk_delta,
        "cost": -config.lambda_cost * normalized,
        "bad_tests": -config.rho_bad_tests * bad_tests,
        "repair_overfit": config.alpha_failure if result.repair_overfit_detected else 0.0,
    }
    return sum(components.values()), components

