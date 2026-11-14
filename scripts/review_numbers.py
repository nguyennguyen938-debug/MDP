"""Numbers added for the reviews: Wilson intervals, exact McNemar tests, the CQL-weight table,
and the LinUCB rows, recomputed from the replay logs in dqn_eval/.

    python scripts/review_numbers.py [dqn_eval]
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

LLM = {"LLM_SYNTHESIZE_INSTRUCTIONS", "LLM_GENERATE_ASSERTIONS", "REFINE_UNCERTAIN_ORACLE"}
PROFILES = {  # profile -> (baseline results dir, conservative Double-DQN dir)
    "Full pilot": ("results", "sweep_pilot_a0.5"),
    "Stress": ("results_stress", "sweep_stress_a0.5"),
    "Action-balanced": ("results_action_balanced", "sweep_action_balanced_a0.5"),
}
ALPHAS = {  # CQL weight -> Double-DQN dirs (full pilot, stress, action-balanced); 0 is the vanilla learner
    "0": ("results", "results_stress", "results_action_balanced"),
    "0.3": ("sweep_pilot_a0.3", "sweep_stress_a0.3", "sweep_action_balanced_a0.3"),
    "0.5": ("sweep_pilot_a0.5", "sweep_stress_a0.5", "sweep_action_balanced_a0.5"),
    "1.0": ("results_cql1.0", "results_stress_cql1.0", "results_action_balanced_cql1.0"),
    "3.0": ("results_cql3.0", "results_stress_cql3.0", "results_action_balanced_cql3.0"),
}


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score 95% interval for k successes out of n."""
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def episodes(path: Path) -> list[list[dict]]:
    return [
        [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        for p in sorted(path.rglob("events.jsonl"))
    ]


def found(eps: list[list[dict]]) -> set[str]:
    return {e[0]["subject_id"] for e in eps if any(r["result"]["failures"] > 0 for r in e)}


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def coverage(e: list[dict]) -> float:
    """The paper's coverage: cumulative coverage delta, capped at 1 per subject."""
    return min(1.0, sum(r["result"]["coverage_delta"] for r in e))


def main(root: Path) -> None:
    print("LinUCB rows: Fail, Cov., Return, Boundary, Fuzz, LLM")
    for profile, (res, _) in PROFILES.items():
        eps = episodes(root / res / "linucb")
        acts = Counter(r["action"] for e in eps for r in e)
        n = sum(acts.values())
        print(
            f"  {profile}: {100 * len(found(eps)) / len(eps):.1f} {100 * mean(map(coverage, eps)):.1f} "
            f"{mean(sum(r['reward'] for r in e) for e in eps):.2f} {100 * acts['GENERATE_BOUNDARY_TESTS'] / n:.1f} "
            f"{100 * acts['PROPERTY_BASED_FUZZ'] / n:.1f} {100 * sum(acts[a] for a in LLM) / n:.1f}"
        )

    print("Failure-found subjects, Wilson 95% CI, exact McNemar p of each Double-DQN seed")
    pooled = [[0, 0] for _ in range(3)]
    for profile, (res, cons) in PROFILES.items():
        base = {k: found(episodes(root / res / k)) for k in ("greedy", "fixed", "bandit")}
        n = len(episodes(root / res / "greedy"))
        for k, s in base.items():
            lo, hi = wilson(len(s), n)
            print(f"  {profile} {k}: {len(s)}/{n} [{100 * lo:.1f}, {100 * hi:.1f}]")
        for seed in range(3):
            dq = found(episodes(root / cons / "dqn" / f"seed_{seed}"))
            lo, hi = wilson(len(dq), n)
            tests = ", ".join(f"vs {k} p={mcnemar_exact(len(dq - s), len(s - dq)):.4f}" for k, s in base.items())
            print(f"  {profile} dqn seed {seed}: {len(dq)}/{n} [{100 * lo:.1f}, {100 * hi:.1f}] {tests}")
            if profile != "Full pilot":
                pooled[seed][0] += len(dq - base["greedy"])
                pooled[seed][1] += len(base["greedy"] - dq)
    for seed, (b, c) in enumerate(pooled):
        print(f"  Stress+Action-balanced dqn seed {seed} vs greedy: p={mcnemar_exact(b, c):.4f}")

    print("CQL weight: Full-pilot Fail, Cov.; Stress Fail, Tok.; Action-balanced Fail, Tok.")
    for alpha, dirs in ALPHAS.items():
        cells = []
        for i, d in enumerate(dirs):
            eps = episodes(root / d / "dqn")
            cells.append(f"{100 * mean(any(r['result']['failures'] > 0 for r in e) for e in eps):.1f}")
            if i == 0:
                cells.append(f"{100 * mean(map(coverage, eps)):.1f}")
            else:
                cells.append(f"{mean(sum(r['result']['cost'].get('tokens', 0.0) for r in e) for e in eps):.0f}")
        print(f"  {alpha}: " + " ".join(cells))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "dqn_eval")
