"""RosettaStone 环境的策略智能体。

复用心跳件（UnifiedNet、PPO、GAE、批处理）全部从 `hearthstone/policy.py` import，
这里只写一个适配 rosetta Observation/Action 的 `PolicyAgent`。
"""

from __future__ import annotations

import random
from typing import Optional

import torch
from torch.distributions import Categorical

# 心跳件：网络架构和训练工具都从 hearthstone 拿来
from hearthstone.policy import (  # noqa: F401 — 让 train.py 从这里 import
    Batch,
    Step,
    Trajectory,
    UnifiedNet,
    evaluate_batch,
    gae_advantages,
    make_batch,
    save_agent,
    load_agent,
)

from .features import STATE_DIM, STATE_OFFSET, batch_features


class PolicyAgent:
    """用 UnifiedNet 打 RosettaStone 的智能体。

    接口刻意和 `rosetta/bots.py` 的 bot 对齐：`choose(obs, actions)`。
    """

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

    def choose(self, obs, actions) -> object:
        """选一个合法动作。返回 rosetta Action 对象。

        训练模式下用 Categorical 采样并记 Step；推理模式下 argmax。
        oracle_dim=0——rosetta 拿不到对手手牌。
        """
        if len(actions) == 1:
            return actions[0]

        x = torch.from_numpy(batch_features(obs, actions)).to(self.device)

        with torch.no_grad():
            logits, _value = self.net.forward_single(x)
            scores = logits / self.temperature
            if not self.training:
                return actions[int(torch.argmax(scores).item())]

            dist = Categorical(logits=scores)
            sampled = dist.sample()
            index = int(sampled.item())
            self.trajectory.steps.append(
                Step(features=x, action=index,
                     log_prob=float(dist.log_prob(sampled)))
            )

        return actions[index]

    def eval_agent(self, temperature: float = 1.0) -> "PolicyAgent":
        return PolicyAgent(self.net, self.device, training=False,
                           temperature=temperature)
