"""策略网络与 PPO 智能体。

和跑得快完全一样的架构：同一个打分网络逐个给候选动作打分，再在合法动作上 softmax。
动作空间变长无所谓，没见过的手牌组合也能靠特征泛化。

训练用 PPO：优势 = 折扣回报 − V(局面)，V 由一个小价值网络估计。裁剪概率比，
同一批数据能安全地反复用好几轮。REINFORCE 也能跑，做对照实验时用。
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .features import FEATURE_DIM, STATE_DIM, STATE_OFFSET, batch_features
from .game import Action, Observation

NORMS = ("none", "layer")


def make_norm(norm: str, width: int) -> List[nn.Module]:
    if norm == "none":
        return []
    if norm == "layer":
        return [nn.LayerNorm(width)]
    raise ValueError(f"未知的归一化方式 {norm!r}，可选: {', '.join(NORMS)}")


class ResidualBlock(nn.Module):
    """Linear + LayerNorm + ReLU，可选残差连接 x + f(x)。"""

    def __init__(self, dim: int, norm: str = "layer", residual: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm_layers = nn.Sequential(*make_norm(norm, dim)) if make_norm(norm, dim) else None
        self.relu = nn.ReLU()
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.norm_layers is not None:
            out = self.norm_layers(out)
        out = self.relu(out)
        if self.residual:
            out = x + out
        return out


class MoveScorer(nn.Module):
    """给单个候选动作打分。输入 (候选数, FEATURE_DIM)，输出 (候选数,)。"""

    def __init__(
        self, dim: int = FEATURE_DIM, hidden: int = 128, layers: int = 2,
        norm: str = "layer", residual: bool = False,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("至少要有一层隐藏层")

        self.dim = dim
        self.hidden = hidden
        self.layers = layers
        self.norm = norm
        self.residual = residual

        blocks: List[nn.Module] = []
        in_dim = dim
        for i in range(layers):
            if residual and i > 0:
                blocks.append(ResidualBlock(hidden, norm, True))
            else:
                blocks.append(nn.Linear(in_dim, hidden))
                blocks.extend(make_norm(norm, hidden))
                blocks.append(nn.ReLU())
                in_dim = hidden
        blocks.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*blocks)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ValueNet(nn.Module):
    """估计当前局面的期望回报，用作基线。"""

    def __init__(self, dim: int = STATE_DIM, hidden: int = 64, layers: int = 1,
                 norm: str = "layer", residual: bool = False) -> None:
        super().__init__()
        blocks: List[nn.Module] = []
        in_dim = dim
        for i in range(layers):
            if residual and i > 0:
                blocks.append(ResidualBlock(hidden, norm, True))
            else:
                blocks.append(nn.Linear(in_dim, hidden))
                blocks.extend(make_norm(norm, hidden))
                blocks.append(nn.ReLU())
                in_dim = hidden
        blocks.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class Step:
    """一个决策点。存特征矩阵（已 detach），更新时用当前网络重新前向。"""

    features: torch.Tensor   # (候选数, FEATURE_DIM)
    action: int              # 选中的下标
    log_prob: float          # 采样时旧策略给的 log π(a|s)


@dataclass
class Trajectory:
    steps: List[Step] = field(default_factory=list)

    def clear(self) -> None:
        self.steps.clear()

    def __len__(self) -> int:
        return len(self.steps)


class PolicyAgent:
    """用策略网络打牌的智能体。"""

    name = "policy"

    def __init__(
        self,
        scorer: MoveScorer,
        value: Optional[ValueNet] = None,
        device: torch.device | str = "cpu",
        training: bool = False,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.scorer = scorer
        self.value = value
        self.device = torch.device(device)
        self.training = training
        self.temperature = temperature
        self.rng = random.Random(seed)
        self.trajectory = Trajectory()

    def choose(self, obs: Observation) -> Action:
        if len(obs.legal) == 1:
            return obs.legal[0]

        x = torch.from_numpy(batch_features(obs)).to(self.device)

        with torch.no_grad():
            scores = self.scorer(x) / self.temperature
            if not self.training:
                return obs.legal[int(torch.argmax(scores).item())]

            dist = Categorical(logits=scores)
            sampled = dist.sample()
            index = int(sampled.item())
            self.trajectory.steps.append(
                Step(features=x, action=index, log_prob=float(dist.log_prob(sampled)))
            )

        return obs.legal[index]

    def eval_agent(self, temperature: float = 1.0) -> "PolicyAgent":
        return PolicyAgent(self.scorer, None, self.device, training=False,
                           temperature=temperature)


# ------------------------------------------------------------------ 存取

def save_agent(path: str, scorer: MoveScorer, value: Optional[ValueNet] = None,
               meta: Optional[dict] = None) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    torch.save(
        {
            "scorer": scorer.state_dict(),
            "value": value.state_dict() if value is not None else None,
            "hidden": scorer.hidden,
            "layers": scorer.layers,
            "norm": scorer.norm,
            "residual": scorer.residual,
            "feature_dim": scorer.dim,
            "meta": meta or {},
        },
        path,
    )


def load_agent(path: str, device: torch.device | str = "cpu") -> PolicyAgent:
    blob = torch.load(path, map_location=device, weights_only=False)
    if blob["feature_dim"] != FEATURE_DIM:
        raise ValueError(
            f"模型是用 {blob['feature_dim']} 维特征训练的，当前特征是 {FEATURE_DIM} 维"
        )
    scorer = MoveScorer(
        hidden=blob["hidden"], layers=blob.get("layers", 2), norm=blob.get("norm", "none"),
        residual=blob.get("residual", False),
    ).to(device)
    scorer.load_state_dict(blob["scorer"])
    scorer.eval()
    return PolicyAgent(scorer, device=device, training=False)


# ------------------------------------------------------------------ 批处理

@dataclass
class Batch:
    features: torch.Tensor   # (S, M, FEATURE_DIM)
    mask: torch.Tensor       # (S, M) bool
    actions: torch.Tensor    # (S,)
    old_log_probs: torch.Tensor  # (S,)

    def __len__(self) -> int:
        return self.features.shape[0]


def make_batch(steps: Sequence[Step], device: torch.device | str = "cpu") -> Batch:
    device = torch.device(device)
    n_steps = len(steps)
    widest = max(step.features.shape[0] for step in steps)
    dim = steps[0].features.shape[1]

    features = torch.zeros(n_steps, widest, dim, device=device)
    mask = torch.zeros(n_steps, widest, dtype=torch.bool, device=device)
    for i, step in enumerate(steps):
        n_moves = step.features.shape[0]
        features[i, :n_moves] = step.features
        mask[i, :n_moves] = True

    return Batch(
        features=features,
        mask=mask,
        actions=torch.tensor([step.action for step in steps], device=device),
        old_log_probs=torch.tensor([step.log_prob for step in steps], device=device),
    )


def evaluate_batch(
    scorer: MoveScorer, value: Optional[ValueNet], batch: Batch
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_steps, widest, dim = batch.features.shape

    scores = scorer(batch.features.reshape(-1, dim)).reshape(n_steps, widest)
    scores = scores.masked_fill(~batch.mask, float("-inf"))
    log_probs = torch.log_softmax(scores, dim=1)

    chosen = log_probs.gather(1, batch.actions.unsqueeze(1)).squeeze(1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs.masked_fill(~batch.mask, 0.0)).sum(dim=1)

    if value is None:
        values = torch.zeros(n_steps, device=batch.features.device)
    else:
        values = value(batch.features[:, 0, STATE_OFFSET:])

    return chosen, entropy, values


def discounted_returns(final_reward: float, n_steps: int, gamma: float) -> np.ndarray:
    steps = np.arange(n_steps - 1, -1, -1, dtype=np.float32)
    return final_reward * (gamma ** steps)
