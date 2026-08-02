"""策略网络与 REINFORCE 智能体。

关键设计：跑得快每一步的合法动作数量都不一样（本项目里平均 40 个、最多 80 多个），
所以不用"固定动作头"的网络。这里让同一个打分网络逐个给候选动作打分：

    score_i = f(特征(局面, 动作_i))          # 所有候选共享同一套参数
    π(动作_i) = softmax(score)_i             # 只在当前合法动作上归一化

这样动作空间变长也无所谓，而且没见过的牌型组合也能靠特征泛化。
训练用带基线的 REINFORCE：优势 = 回报 - V(局面)，V 由一个小的价值网络估计。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .features import FEATURE_DIM, STATE_OFFSET, batch_features
from .game import Action, Observation

STATE_DIM = FEATURE_DIM - STATE_OFFSET


class MoveScorer(nn.Module):
    """给单个候选动作打分。输入 (动作数, FEATURE_DIM)，输出 (动作数,)。

    `hidden` 是隐藏层宽度，`layers` 是隐藏层数量，一起决定模型大小。
    """

    def __init__(self, dim: int = FEATURE_DIM, hidden: int = 128, layers: int = 2) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("至少要有一层隐藏层")

        self.hidden = hidden
        self.layers = layers

        blocks: List[nn.Module] = [nn.Linear(dim, hidden), nn.ReLU()]
        for _ in range(layers - 1):
            blocks += [nn.Linear(hidden, hidden), nn.ReLU()]
        blocks.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*blocks)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ValueNet(nn.Module):
    """估计当前局面的期望回报，用作 REINFORCE 的基线。"""

    def __init__(self, dim: int = STATE_DIM, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class Step:
    """一个决策点，训练时用得到。"""

    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


@dataclass
class Trajectory:
    steps: List[Step] = field(default_factory=list)

    def clear(self) -> None:
        self.steps.clear()

    def __len__(self) -> int:
        return len(self.steps)


class PolicyAgent:
    """用策略网络打牌的智能体，接口和规则对手一致。"""

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
            # 只有一个选择时不产生梯度：这种决策点学不到东西，还会稀释信号
            return obs.legal[0]

        x = torch.from_numpy(batch_features(obs)).to(self.device)

        with torch.set_grad_enabled(self.training):
            scores = self.scorer(x) / self.temperature
            if self.training:
                dist = Categorical(logits=scores)
                index = int(dist.sample().item())
                value = self.value(x[0, STATE_OFFSET:]) if self.value is not None else torch.zeros((), device=self.device)
                self.trajectory.steps.append(
                    Step(
                        log_prob=dist.log_prob(torch.tensor(index, device=self.device)),
                        entropy=dist.entropy(),
                        value=value,
                    )
                )
            else:
                index = int(torch.argmax(scores).item())

        return obs.legal[index]

    # ------------------------------------------------------------------ 存取

    def eval_agent(self, temperature: float = 1.0) -> "PolicyAgent":
        """复制一个只做贪心决策、不记录轨迹的版本，用于评测。"""
        return PolicyAgent(self.scorer, None, self.device, training=False, temperature=temperature)


def save_agent(path: str, scorer: MoveScorer, value: Optional[ValueNet] = None, meta: Optional[dict] = None) -> None:
    """保存模型权重。"""
    torch.save(
        {
            "scorer": scorer.state_dict(),
            "value": value.state_dict() if value is not None else None,
            "hidden": scorer.hidden,
            "layers": scorer.layers,
            "feature_dim": FEATURE_DIM,
            "meta": meta or {},
        },
        path,
    )


def load_agent(path: str, device: torch.device | str = "cpu") -> PolicyAgent:
    """读取模型，返回可以直接对局的智能体。"""
    blob = torch.load(path, map_location=device, weights_only=False)
    if blob["feature_dim"] != FEATURE_DIM:
        raise ValueError(
            f"模型是用 {blob['feature_dim']} 维特征训练的，当前特征是 {FEATURE_DIM} 维，特征改过了就得重训"
        )
    # layers 是后加的字段，早期存的权重默认为 2 层
    scorer = MoveScorer(hidden=blob["hidden"], layers=blob.get("layers", 2)).to(device)
    scorer.load_state_dict(blob["scorer"])
    scorer.eval()
    return PolicyAgent(scorer, device=device, training=False)


def discounted_returns(final_reward: float, n_steps: int, gamma: float) -> np.ndarray:
    """回报只在终局给出，往前按 gamma 折扣分摊到每个决策点。"""
    steps = np.arange(n_steps - 1, -1, -1, dtype=np.float32)
    return final_reward * (gamma ** steps)
