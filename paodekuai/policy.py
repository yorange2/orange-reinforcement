"""策略网络与 REINFORCE 智能体。

关键设计：跑得快每一步的合法动作数量都不一样（本项目里平均 40 个、最多 80 多个），
所以不用"固定动作头"的网络。这里让同一个打分网络逐个给候选动作打分：

    score_i = f(特征(局面, 动作_i))          # 所有候选共享同一套参数
    π(动作_i) = softmax(score)_i             # 只在当前合法动作上归一化

这样动作空间变长也无所谓，而且没见过的牌型组合也能靠特征泛化。
训练用带基线的 REINFORCE：优势 = 回报 - V(局面)，V 由一个小的价值网络估计。
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

from .features import FEATURE_DIM, STATE_OFFSET
from .game import Action, Observation

STATE_DIM = FEATURE_DIM - STATE_OFFSET

#: 可选的归一化层。
#:
#: 这里**不能用 BatchNorm**：MoveScorer 的"批"维度装的是候选动作而不是独立样本，
#: BatchNorm 会让一手牌的分数取决于当时碰巧还有哪些牌可出；训练时更糟，一批数据里
#: 混着大量补齐用的假零行（见 make_batch），它们会污染统计量。而且 PPO 的概率比
#: π_new/π_old 要求 π 只是"状态 + 参数"的函数，一旦依赖同批次的其他样本，
#: 采样时和更新时的批构成完全不同，这个比值就没意义了。
#:
#: LayerNorm 没有这些问题：它在单个样本的特征维度内归一化，候选之间互不影响，
#: 训练和推理行为一致。
NORMS = ("none", "layer")


def make_norm(norm: str, width: int) -> List[nn.Module]:
    if norm == "none":
        return []
    if norm == "layer":
        return [nn.LayerNorm(width)]
    raise ValueError(f"未知的归一化方式 {norm!r}，可选: {', '.join(NORMS)}")


class ResidualBlock(nn.Module):
    """x + f(x)。梯度可以顺着这条恒等路径直通，深一点的网络才好训。"""

    def __init__(self, width: int, norm: str) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(width, width), *make_norm(norm, width), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class MoveScorer(nn.Module):
    """给单个候选动作打分。输入 (动作数, dim)，输出 (动作数,)。

    `hidden` 是隐藏层宽度，`layers` 是隐藏层数量，一起决定模型大小。

    传 `grid` 就会先用一维卷积沿点数轴扫一遍再进 MLP：顺子和连对本来就是点数轴上的
    连续片段，卷积适合发现这种局部模式。但**默认不开**——实测在 PPO 更新的批量规模上
    （6.4 万行）卷积要 508ms，展平直接进 MLP 只要 16ms，慢 32 倍，训练吞吐从 380 局/秒
    掉到 27 局/秒。这个任务的点数轴只有 12 格，不值这个代价。
    """

    def __init__(
        self,
        dim: int = FEATURE_DIM,
        hidden: int = 128,
        layers: int = 2,
        norm: str = "layer",
        grid: Optional[Tuple[int, int]] = None,
        grid_channels: int = 6,
        conv_channels: int = 32,
        residual: bool = False,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("至少要有一层隐藏层")

        self.dim = dim
        self.hidden = hidden
        self.layers = layers
        self.norm = norm
        self.residual = residual
        self.grid = grid
        self.grid_channels = grid_channels
        self.conv_channels = conv_channels

        mlp_in = dim
        self.conv = None
        if grid is not None:
            start, end = grid
            n_ranks = (end - start) // grid_channels
            if n_ranks * grid_channels != end - start:
                raise ValueError(f"网格区间 {grid} 装不下 {grid_channels} 个通道")
            self.n_ranks = n_ranks
            self.conv = nn.Sequential(
                nn.Conv1d(grid_channels, conv_channels, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            # 网格换成卷积输出，其余维度原样拼在后面
            mlp_in = dim - (end - start) + conv_channels * n_ranks

        # 第一层要把输入投到 hidden 宽，维度不一致没法做残差
        blocks: List[nn.Module] = [nn.Linear(mlp_in, hidden), *make_norm(norm, hidden), nn.ReLU()]
        for _ in range(layers - 1):
            if residual:
                blocks.append(ResidualBlock(hidden, norm))
            else:
                blocks += [nn.Linear(hidden, hidden), *make_norm(norm, hidden), nn.ReLU()]
        blocks.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*blocks)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is not None:
            start, end = self.grid
            grid = x[:, start:end].reshape(-1, self.grid_channels, self.n_ranks)
            encoded = self.conv(grid).flatten(1)
            x = torch.cat([x[:, :start], encoded, x[:, end:]], dim=1)
        return self.net(x).squeeze(-1)


class ValueNet(nn.Module):
    """估计当前局面的期望回报，用作 REINFORCE 的基线。"""

    def __init__(self, dim: int = STATE_DIM, hidden: int = 64, norm: str = "layer") -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            *make_norm(norm, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class Step:
    """一个决策点。

    这里存的是**特征本身**而不是带计算图的张量：PPO 要拿同一批数据反复更新好几轮，
    每轮都得用当前策略重新前向一次，所以采样时只记录输入、选了哪个动作、以及旧策略
    当时给这个动作的对数概率。
    """

    features: torch.Tensor   # (候选动作数, FEATURE_DIM)，已 detach
    action: int              # 选中的候选下标
    log_prob: float          # 采样时旧策略给的 log π(a|s)


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
        sample: bool = False,
        encoder=None,
    ) -> None:
        from .encoding import make_encoder

        self.encoder = encoder if encoder is not None else make_encoder("handcrafted")
        self.scorer = scorer
        self.value = value
        self.device = torch.device(device)
        self.training = training
        #: 按概率抽动作而不是取最高分。`training` 必然要采样；自我对弈的快照对手
        #: 也要采样（不然它每局打得一模一样，提供不了多样的对局），但不记录轨迹。
        self.sample = sample or training
        self.temperature = temperature
        self.rng = random.Random(seed)
        self.trajectory = Trajectory()

    def choose(self, obs: Observation) -> Action:
        if len(obs.legal) == 1:
            # 只有一个选择时不产生梯度：这种决策点学不到东西，还会稀释信号
            return obs.legal[0]

        x = torch.from_numpy(self.encoder.build(obs)).to(self.device)

        with torch.no_grad():  # 采样不需要梯度，梯度在更新时重新前向来算
            scores = self.scorer(x) / self.temperature
            if not self.sample:
                return obs.legal[int(torch.argmax(scores).item())]

            dist = Categorical(logits=scores)
            sampled = dist.sample()
            index = int(sampled.item())
            if self.training:
                self.trajectory.steps.append(
                    Step(features=x, action=index, log_prob=float(dist.log_prob(sampled)))
                )

        return obs.legal[index]

    # ------------------------------------------------------------------ 存取

    def eval_agent(self, temperature: float = 1.0) -> "PolicyAgent":
        """复制一个只做贪心决策、不记录轨迹的版本，用于评测。"""
        return PolicyAgent(self.scorer, None, self.device, training=False,
                           temperature=temperature, encoder=self.encoder)


def save_agent(path: str, scorer: MoveScorer, value: Optional[ValueNet] = None, meta: Optional[dict] = None) -> None:
    """保存模型权重。目录不存在就建出来——训练跑了几分钟才发现存不下太亏。"""
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
            "encoder": getattr(scorer, "encoder_name", "handcrafted"),
            "conv": scorer.grid is not None,
            "residual": scorer.residual,
            "feature_dim": scorer.dim,
            "meta": meta or {},
        },
        path,
    )


def load_agent(path: str, device: torch.device | str = "cpu") -> PolicyAgent:
    """读取模型，返回可以直接对局的智能体。"""
    from .encoding import make_encoder

    blob = torch.load(path, map_location=device, weights_only=False)
    encoder = make_encoder(blob.get("encoder", "handcrafted"))
    if blob["feature_dim"] != encoder.dim:
        raise ValueError(
            f"模型是用 {blob['feature_dim']} 维输入训练的，当前 {encoder.name} 编码是 "
            f"{encoder.dim} 维，编码改过了就得重训"
        )
    # layers / norm / encoder 都是后加的字段，早期存的权重按当时的默认值来
    scorer = MoveScorer(
        dim=encoder.dim, hidden=blob["hidden"], layers=blob.get("layers", 2),
        norm=blob.get("norm", "none"),
        grid=encoder.grid_slice if blob.get("conv", False) else None,
        residual=blob.get("residual", False),
    ).to(device)
    scorer.encoder_name = encoder.name
    scorer.load_state_dict(blob["scorer"])
    scorer.eval()
    return PolicyAgent(scorer, device=device, training=False, encoder=encoder)


@dataclass
class Batch:
    """一批决策点，补齐成矩形好一次性前向。

    每个决策点的候选动作数量都不一样，所以按最大值补齐，再用 `mask` 把补出来的位置
    屏蔽掉——softmax 前填 -inf，它们的概率恒为 0，不会污染梯度。
    """

    features: torch.Tensor   # (S, M, dim)
    mask: torch.Tensor       # (S, M) bool，True 表示是真实候选
    actions: torch.Tensor    # (S,)
    old_log_probs: torch.Tensor  # (S,)
    state_offset: int        # 局面块从这里开始，价值网络只吃这一段

    def __len__(self) -> int:
        return self.features.shape[0]


def make_batch(
    steps: Sequence[Step],
    device: torch.device | str = "cpu",
    state_offset: int = STATE_OFFSET,
) -> Batch:
    """把若干决策点打包成一个可以整体前向的批。"""
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
        state_offset=state_offset,
    )


def evaluate_batch(
    scorer: MoveScorer, value: Optional[ValueNet], batch: Batch
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """用当前网络重新过一遍这批数据，返回 (选中动作的 log 概率, 熵, 状态价值)。"""
    n_steps, widest, dim = batch.features.shape

    scores = scorer(batch.features.reshape(-1, dim)).reshape(n_steps, widest)
    scores = scores.masked_fill(~batch.mask, float("-inf"))
    log_probs = torch.log_softmax(scores, dim=1)

    chosen = log_probs.gather(1, batch.actions.unsqueeze(1)).squeeze(1)
    # 被屏蔽的位置 log_prob 是 -inf，概率是 0，相乘会得到 nan，这里直接置零
    probs = log_probs.exp()
    entropy = -(probs * log_probs.masked_fill(~batch.mask, 0.0)).sum(dim=1)

    if value is None:
        values = torch.zeros(n_steps, device=batch.features.device)
    else:
        values = value(batch.features[:, 0, batch.state_offset :])

    return chosen, entropy, values


def discounted_returns(final_reward: float, n_steps: int, gamma: float) -> np.ndarray:
    """回报只在终局给出，往前按 gamma 折扣分摊到每个决策点。"""
    steps = np.arange(n_steps - 1, -1, -1, dtype=np.float32)
    return final_reward * (gamma ** steps)
