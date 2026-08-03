"""策略网络与 PPO 智能体。

AlphaZero 风格二合一架构：共享局面编码器 + 策略头 + 价值头。
局面特征只算一次，策略和价值共享底层表示，互相促进。
动作空间变长无所谓，没见过的手牌组合也能靠特征泛化。

训练用 PPO：优势 = 折扣回报 − V(局面)，V 由价值头估计。裁剪概率比，
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


class UnifiedNet(nn.Module):
    """AlphaZero 风格二合一网络：共享局面编码 + 策略头 + 价值头。

    局面编码器只算一次；策略头把局面编码和每个候选动作的特征拼接后打分；
    价值头直接从局面编码预测期望回报。
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = STATE_OFFSET,
        hidden: int = 128,
        layers: int = 2,
        norm: str = "layer",
        residual: bool = False,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("至少要有一层隐藏层")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden
        self.layers = layers
        self.norm = norm
        self.residual = residual

        # --- 局面编码器：state_dim → hidden → ... → hidden ---
        enc: List[nn.Module] = []
        in_dim = state_dim
        for i in range(layers):
            if residual and i > 0:
                enc.append(ResidualBlock(hidden, norm, True))
            else:
                enc.append(nn.Linear(in_dim, hidden))
                enc.extend(make_norm(norm, hidden))
                enc.append(nn.ReLU())
                in_dim = hidden
        self.state_encoder = nn.Sequential(*enc)

        # --- 策略头：(hidden + action_dim) → hidden → 1 ---
        policy: List[nn.Module] = []
        policy_in = hidden + action_dim
        policy.append(nn.Linear(policy_in, hidden))
        policy.extend(make_norm(norm, hidden))
        policy.append(nn.ReLU())
        if residual:
            policy.append(ResidualBlock(hidden, norm, True))
        policy.append(nn.Linear(hidden, 1))
        self.policy_head = nn.Sequential(*policy)

        # --- 价值头：hidden → hidden//2 → 1 ---
        value: List[nn.Module] = []
        vh = hidden // 2
        value.append(nn.Linear(hidden, vh))
        value.extend(make_norm(norm, vh))
        value.append(nn.ReLU())
        value.append(nn.Linear(vh, 1))
        self.value_head = nn.Sequential(*value)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """批量前向。

        Args:
            features: (S, M, FEATURE_DIM)  padded 特征矩阵
            mask:     (S, M) bool          有效候选掩码

        Returns:
            logits: (S, M)  候选动作 logits（padding 位置为 -inf）
            values: (S,)    局面价值估计
        """
        S, M, _ = features.shape

        # 局面编码：取每步第一行的 state tail（所有候选共享）
        state = features[:, 0, self.action_dim:]          # (S, state_dim)
        state_emb = self.state_encoder(state)              # (S, hidden)

        # 价值估计
        values = self.value_head(state_emb).squeeze(-1)    # (S,)

        # 策略：局面编码扩展到所有候选，拼接动作特征
        state_expanded = state_emb.unsqueeze(1).expand(S, M, self.hidden)   # (S, M, hidden)
        action_feats = features[:, :, :self.action_dim]                     # (S, M, action_dim)
        combined = torch.cat([state_expanded, action_feats], dim=-1)        # (S, M, hidden+action_dim)

        logits = self.policy_head(combined.reshape(-1, self.hidden + self.action_dim))
        logits = logits.reshape(S, M)                       # (S, M)
        logits = logits.masked_fill(~mask, float("-inf"))

        return logits, values

    def forward_single(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """单决策点前向（推理/采样用，无 padding）。

        Args:
            x: (N, FEATURE_DIM) 特征矩阵

        Returns:
            logits: (N,)   候选动作 logits
            value:  scalar 局面价值估计
        """
        N = x.shape[0]

        state = x[0, self.action_dim:]                     # (state_dim,)
        state_emb = self.state_encoder(state.unsqueeze(0))  # (1, hidden)
        value = self.value_head(state_emb).squeeze()        # scalar

        state_expanded = state_emb.expand(N, self.hidden)   # (N, hidden)
        combined = torch.cat([state_expanded, x[:, :self.action_dim]], dim=-1)
        logits = self.policy_head(combined).squeeze(-1)     # (N,)

        return logits, value


# 保留别名，方便对照实验时切换
MoveScorer = UnifiedNet


class ValueNet(nn.Module):
    """独立价值网络——对照实验用，不参与 UnifiedNet 的训练。"""

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
    """用 UnifiedNet 打牌的智能体。"""

    name = "policy"

    def __init__(
        self,
        net: UnifiedNet,
        device: torch.device | str = "cpu",
        training: bool = False,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.net = net
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
            logits, _value = self.net.forward_single(x)
            scores = logits / self.temperature
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
        return PolicyAgent(self.net, self.device, training=False,
                           temperature=temperature)


# ------------------------------------------------------------------ 存取

def save_agent(path: str, net: UnifiedNet, meta: Optional[dict] = None) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    torch.save(
        {
            "state_dict": net.state_dict(),
            "hidden": net.hidden,
            "layers": net.layers,
            "norm": net.norm,
            "residual": net.residual,
            "state_dim": net.state_dim,
            "action_dim": net.action_dim,
            "feature_dim": net.action_dim + net.state_dim,
            "meta": meta or {},
        },
        path,
    )


def load_agent(path: str, device: torch.device | str = "cpu") -> PolicyAgent:
    blob = torch.load(path, map_location=device, weights_only=False)
    if blob.get("feature_dim", blob.get("state_dim", 0) + blob.get("action_dim", 0)) != FEATURE_DIM:
        raise ValueError(
            f"模型是用 {blob.get('feature_dim', '?')} 维特征训练的，当前特征是 {FEATURE_DIM} 维"
        )
    net = UnifiedNet(
        state_dim=blob.get("state_dim", STATE_DIM),
        action_dim=blob.get("action_dim", STATE_OFFSET),
        hidden=blob["hidden"],
        layers=blob.get("layers", 2),
        norm=blob.get("norm", "none"),
        residual=blob.get("residual", False),
    ).to(device)
    # 兼容旧格式 (scorer/value 分别存) 和新格式 (state_dict)
    if "state_dict" in blob:
        net.load_state_dict(blob["state_dict"])
    elif "scorer" in blob:
        # 旧格式：只加载 scorer 和 value 的权重到 encoder + heads
        # 由于架构不同，无法精确迁移，跳过
        raise ValueError("旧格式权重（独立 MoveScorer/ValueNet）无法加载到 UnifiedNet，请用新格式重新训练")
    net.eval()
    return PolicyAgent(net, device=device, training=False)


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
    net: UnifiedNet, batch: Batch
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """用 UnifiedNet 批量评估。

    Returns:
        chosen_log_probs: (S,)  选中动作的 log π
        entropy:          (S,)  策略熵
        values:           (S,)  局面价值
    """
    logits, values = net(batch.features, batch.mask)

    log_probs = torch.log_softmax(logits, dim=1)
    chosen = log_probs.gather(1, batch.actions.unsqueeze(1)).squeeze(1)

    probs = log_probs.exp()
    # 计算熵时 padding 位置不算
    entropy = -(probs * log_probs.masked_fill(~batch.mask, 0.0)).sum(dim=1)

    return chosen, entropy, values


def discounted_returns(final_reward: float, n_steps: int, gamma: float) -> np.ndarray:
    steps = np.arange(n_steps - 1, -1, -1, dtype=np.float32)
    return final_reward * (gamma ** steps)
