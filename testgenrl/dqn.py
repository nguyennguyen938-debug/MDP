"""Masked Double-DQN training and inference.

`train_double_dqn` learns a state-dependent action-value function from a replay
buffer using the masked Double-DQN target
``y = r + (1 - done) * gamma * Q_target(s', argmax_{a' valid} Q_online(s', a'))``
with separate online and target networks. When PyTorch is installed it trains a
small MLP; otherwise it trains a linear value function with the same update.
Either way the learned weights are serialized to JSON so the planner runs at
inference time without PyTorch.

`train_from_replay` keeps the lightweight mean-reward-per-action estimator used
as a state-agnostic fallback (and for environments where no model is wanted).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .actions import Action, masked_argmax
from .models import Transition
from .planners import Planner


@dataclass
class ReplayBuffer:
    transitions: list[Transition]

    def __init__(self) -> None:
        self.transitions = []

    def add(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.transitions, min(batch_size, len(self.transitions)))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(t) for t in self.transitions], indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ReplayBuffer":
        buf = cls()
        if path.exists():
            for row in json.loads(path.read_text(encoding="utf-8")):
                buf.add(Transition(**row))
        return buf


# --------------------------------------------------------------------------- #
# Q-network: portable weights + dependency-free forward pass for inference.
# --------------------------------------------------------------------------- #


@dataclass
class QModel:
    """A feed-forward Q-network stored as plain lists so inference needs no
    PyTorch. Each layer is ``{"W": out x in, "b": out, "act": "relu"|"none"}``."""

    layers: list[dict]
    in_dim: int
    out_dim: int

    def q(self, state: list[float]) -> list[float]:
        # Defensively match the trained input width.
        if len(state) < self.in_dim:
            x = list(state) + [0.0] * (self.in_dim - len(state))
        else:
            x = list(state[: self.in_dim])
        try:
            import numpy as np

            vec = np.asarray(x, dtype="float64")
            for layer in self.layers:
                vec = np.asarray(layer["W"], dtype="float64") @ vec + np.asarray(layer["b"], dtype="float64")
                if layer["act"] == "relu":
                    vec = np.maximum(vec, 0.0)
            return [float(v) for v in vec.tolist()]
        except Exception:
            for layer in self.layers:
                weight, bias, act = layer["W"], layer["b"], layer["act"]
                out = [sum(weight[o][i] * x[i] for i in range(len(x))) + bias[o] for o in range(len(bias))]
                x = [v if v > 0.0 else 0.0 for v in out] if act == "relu" else out
            return [float(v) for v in x]

    def to_json(self) -> dict:
        return {"layers": self.layers, "in_dim": self.in_dim, "out_dim": self.out_dim}

    @classmethod
    def from_json(cls, data: dict) -> "QModel":
        return cls(layers=list(data["layers"]), in_dim=int(data["in_dim"]), out_dim=int(data["out_dim"]))


class DQNPlanner(Planner):
    name = "dqn"

    def __init__(
        self,
        q_values: list[float] | None = None,
        epsilon: float = 0.0,
        model: QModel | None = None,
        gamma: float = 0.95,
    ):
        self.q_values = q_values or [0.0 for _ in Action]
        self.epsilon = epsilon
        self.model = model
        self.gamma = gamma

    def action_values(self, state: list[float]) -> list[float]:
        """State-dependent Q-values from the trained network, falling back to
        the state-agnostic action means when no model is available."""
        if self.model is not None:
            return self.model.q(state)
        return self.q_values

    def select(self, state: list[float], mask: list[bool]) -> int:
        valid = [i for i, ok in enumerate(mask) if ok]
        if valid and random.random() < self.epsilon:
            return random.choice(valid)
        return masked_argmax(self.action_values(state), mask)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "q_values": self.q_values,
            "gamma": self.gamma,
            "model": self.model.to_json() if self.model is not None else None,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DQNPlanner":
        data = json.loads(path.read_text(encoding="utf-8"))
        model = QModel.from_json(data["model"]) if data.get("model") else None
        return cls(
            q_values=list(data.get("q_values") or []),
            model=model,
            gamma=float(data.get("gamma", 0.95)),
        )


def masked_double_dqn_target(
    reward: float,
    gamma: float,
    online_next_q: list[float],
    target_next_q: list[float],
    next_mask: list[bool],
    done: bool = False,
) -> float:
    # Terminal states have no successor to bootstrap from: y = r.
    if done:
        return reward
    next_action = masked_argmax(online_next_q, next_mask)
    return reward + gamma * target_next_q[next_action]


# --------------------------------------------------------------------------- #
# Training.
# --------------------------------------------------------------------------- #


def _mean_reward_per_action(buffer: ReplayBuffer) -> list[float]:
    totals = [0.0 for _ in Action]
    counts = [0 for _ in Action]
    for t in buffer.transitions:
        totals[t.action] += t.reward
        counts[t.action] += 1
    return [totals[i] / counts[i] if counts[i] else 0.0 for i in range(len(Action))]


def train_from_replay(buffer: ReplayBuffer, out: Path, gamma: float = 0.95) -> DQNPlanner:
    """State-agnostic mean-reward-per-action estimator (no bootstrapping)."""
    planner = DQNPlanner(q_values=_mean_reward_per_action(buffer), gamma=gamma)
    planner.save(out)
    return planner


@dataclass
class DQNConfig:
    gamma: float = 0.95
    hidden: tuple[int, ...] = (64, 64)
    lr: float = 1e-3
    batch_size: int = 128
    steps: int = 2000
    target_sync: int = 200
    seed: int = 0
    epsilon: float = 0.0
    cql_alpha: float = 0.5
    history: list[float] = field(default_factory=list)


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _train_torch(buffer: ReplayBuffer, cfg: DQNConfig) -> tuple[QModel, list[float]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)

    transitions = buffer.transitions
    in_dim = len(transitions[0].state)
    out_dim = len(Action)

    states = torch.tensor([t.state for t in transitions], dtype=torch.float32)
    next_states = torch.tensor([t.next_state for t in transitions], dtype=torch.float32)
    actions = torch.tensor([t.action for t in transitions], dtype=torch.long)
    rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
    next_mask = torch.tensor([[bool(m) for m in t.next_mask] for t in transitions], dtype=torch.bool)
    done = torch.tensor([1.0 if t.done else 0.0 for t in transitions], dtype=torch.float32)

    def build() -> nn.Sequential:
        layers: list[nn.Module] = []
        prev = in_dim
        for width in cfg.hidden:
            layers += [nn.Linear(prev, width), nn.ReLU()]
            prev = width
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    online = build()
    target = build()
    target.load_state_dict(online.state_dict())
    target.eval()
    opt = torch.optim.Adam(online.parameters(), lr=cfg.lr)

    n = len(transitions)
    history: list[float] = []
    for step in range(cfg.steps):
        idx = torch.randint(0, n, (min(cfg.batch_size, n) if n else 1,), generator=gen)
        s, ns = states[idx], next_states[idx]
        a, r, d, nm = actions[idx], rewards[idx], done[idx], next_mask[idx]

        online_s = online(s)
        q = online_s.gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            online_next = online(ns)
            masked_next = online_next.masked_fill(~nm, float("-inf"))
            next_action = masked_next.argmax(dim=1)
            target_next = target(ns).gather(1, next_action.unsqueeze(1)).squeeze(1)
            y = r + cfg.gamma * (1.0 - d) * target_next

        loss = F.smooth_l1_loss(q, y)
        if cfg.cql_alpha > 0.0:
            # Conservative Q-Learning (CQL) regularizer: push down the soft-max
            # of all action-values while pulling up the dataset action, so the
            # value of under-covered (out-of-distribution) actions is not
            # over-estimated. Addresses offline overestimation on rare actions.
            cql = (torch.logsumexp(online_s, dim=1) - q).mean()
            loss = loss + cfg.cql_alpha * cql
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(online.parameters(), 10.0)
        opt.step()
        if step % cfg.target_sync == 0:
            target.load_state_dict(online.state_dict())
        if step % max(1, cfg.steps // 20) == 0:
            history.append(float(loss.detach()))

    layers: list[dict] = []
    modules = list(online)
    i = 0
    while i < len(modules):
        linear = modules[i]
        act = "relu" if i + 1 < len(modules) and isinstance(modules[i + 1], nn.ReLU) else "none"
        layers.append(
            {
                "W": linear.weight.detach().cpu().tolist(),
                "b": linear.bias.detach().cpu().tolist(),
                "act": act,
            }
        )
        i += 2 if act == "relu" else 1
    return QModel(layers=layers, in_dim=in_dim, out_dim=out_dim), history


def _train_linear(buffer: ReplayBuffer, cfg: DQNConfig) -> tuple[QModel, list[float]]:
    """Pure-Python linear Double-DQN: Q(s, a) = W[a] . s + b[a], SGD with a
    periodically synced target copy. Used when PyTorch is unavailable."""
    rng = random.Random(cfg.seed)
    transitions = buffer.transitions
    in_dim = len(transitions[0].state)
    out_dim = len(Action)

    weight = [[0.0 for _ in range(in_dim)] for _ in range(out_dim)]
    bias = [0.0 for _ in range(out_dim)]
    target_w = [row[:] for row in weight]
    target_b = bias[:]

    def q_row(w: list[list[float]], b: list[float], state: list[float], action: int) -> float:
        return sum(w[action][i] * state[i] for i in range(in_dim)) + b[action]

    def q_all(w: list[list[float]], b: list[float], state: list[float]) -> list[float]:
        return [sum(w[a][i] * state[i] for i in range(in_dim)) + b[a] for a in range(out_dim)]

    n = len(transitions)
    history: list[float] = []
    for step in range(cfg.steps):
        t = transitions[rng.randrange(n)]
        online_next = q_all(weight, bias, t.next_state)
        target_next = q_all(target_w, target_b, t.next_state)
        y = masked_double_dqn_target(
            t.reward, cfg.gamma, online_next, target_next, [bool(m) for m in t.next_mask], done=t.done
        )
        pred = q_row(weight, bias, t.state, t.action)
        err = pred - y
        # gradient of 0.5 * err^2 w.r.t. W[a]/b[a]
        for i in range(in_dim):
            weight[t.action][i] -= cfg.lr * err * t.state[i]
        bias[t.action] -= cfg.lr * err
        if step % cfg.target_sync == 0:
            target_w = [row[:] for row in weight]
            target_b = bias[:]
        if step % max(1, cfg.steps // 20) == 0:
            history.append(0.5 * err * err)

    layers = [{"W": weight, "b": bias, "act": "none"}]
    return QModel(layers=layers, in_dim=in_dim, out_dim=out_dim), history


def train_double_dqn(buffer: ReplayBuffer, out: Path, config: DQNConfig | None = None) -> DQNPlanner:
    """Train a masked Double-DQN value function from a replay buffer and save it.

    Falls back to the state-agnostic mean-reward planner when the buffer is
    empty. The mean-reward vector is always stored too, so the planner still has
    a sensible value for actions unseen by the network's masked argmax.
    """
    cfg = config or DQNConfig()
    mean_q = _mean_reward_per_action(buffer)
    if not buffer.transitions:
        planner = DQNPlanner(q_values=mean_q, gamma=cfg.gamma)
        planner.save(out)
        return planner

    model, history = _train_torch(buffer, cfg) if _torch_available() else _train_linear(buffer, cfg)
    cfg.history = history
    planner = DQNPlanner(q_values=mean_q, model=model, epsilon=cfg.epsilon, gamma=cfg.gamma)
    planner.save(out)
    return planner
