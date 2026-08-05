"""批量对局（路线图 M4）：N 局镜像对局一次驱动。

底层是 `orange_stone.BatchEnv`（PyO3 批量类）：一次调用处理整批，Rust 侧
在 `allow_threads` 区域里迭代，摊薄 Python↔Rust 往返；每局独立 GameRng，
批量与单局逐 seed 结果一致，且可以再叠线程池（每线程一个批量块）。

与 `Env`（双实例锁步）的区别：`BatchEnv` 的**结构化观测直接给当前行动方
视角**（绑定层按 active player 编码），所以批量不需要 perspective 0/1 双
实例——每局一个 GameEnv 就够。奖励仍是每局构造时的固定视角（训练时
agent 坐哪边就用哪个 perspective）。

`battle_batch`（绑定层的 rayon 批量）是纯引擎吞吐的基准（内置 Greedy/Smart
bot，全 Rust 并行），见 `bench_bots`。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence

import orange_stone as _native

from .env import Action

__all__ = ["BatchedEnv", "bench_bots", "bench_battle_batch"]


class BatchedEnv:
    """N 局镜像对局的批量驱动。"""

    def __init__(
        self,
        n: int,
        deck: list[str],
        seeds: Optional[list[int]] = None,
        *,
        perspectives: Optional[list[int]] = None,
        bot: str = "none",
        hand_size: int = 3,
        second_player_coin: bool = True,
        terminal_reward: str = "sparse",
    ) -> None:
        self._seeds = list(seeds) if seeds is not None else list(range(n))
        self._batch = _native.BatchEnv(
            seeds=self._seeds,
            deck=list(deck),
            perspectives=perspectives,
            bot=bot,
            hand_size=hand_size,
            second_player_coin=second_player_coin,
            terminal_reward=terminal_reward,
        )
        self._last_rewards = [0.0] * n

    def __len__(self) -> int:
        return self._batch.len()

    # ------------------------------------------------------------ 四件套（批量）

    def reset(self, seeds: Sequence[int]) -> None:
        """整批重开（seeds 与构造时一一对应）。"""
        self._seeds = list(seeds)
        self._batch.reset(self._seeds)
        self._last_rewards = [0.0] * len(self)

    def reset_one(self, i: int, seed: int) -> None:
        """只重开第 i 局（批量训练里完成的局单独换 seed）。"""
        self._seeds[i] = seed
        self._batch.reset_one(i, seed)
        self._last_rewards[i] = 0.0

    def legal_actions(self) -> list[list[Action]]:
        """每局当前行动方的合法动作（Action 对象，index 可直接喂 step）。"""
        return [
            [Action.from_view(v) for v in views]
            for views in self._batch.structured_legal_actions()
        ]

    def observe(self) -> list:
        """每局**当前行动方**视角的结构化观测。"""
        return self._batch.structured_observations()

    def step(self, indices: Sequence[int]) -> None:
        """每局按自己的 index 走一步（长度必须等于局数）。"""
        _, rewards, _, _ = self._batch.step(list(indices))
        self._last_rewards = list(rewards)

    # ------------------------------------------------------------ 状态

    def done(self) -> list[bool]:
        return self._batch.done()

    def winners(self) -> list[Optional[int]]:
        """每局胜者（None = 未结束/平局；0 = P1, 1 = P2）。"""
        return self._batch.winners()

    def active_players(self) -> list[int]:
        """每局当前行动方（0 = P1, 1 = P2）。"""
        return [int(p) for p in self._batch.active_players()]

    def last_reward(self, i: int) -> float:
        """第 i 局上一步的奖励（该局构造时视角的终局奖励）。"""
        return self._last_rewards[i]


def bench_bots(
    games: int,
    deck: list[str],
    *,
    batch_size: int = 64,
    workers: int = 4,
    seed: int = 0,
) -> float:
    """BatchedEnv 驱动的 bot-vs-bot 吞吐（局/s）。

    每线程一个批量块（BatchEnv 内 allow_threads 释放 GIL，引擎工作并行）。
    随机策略 vs 内置 Greedy（与 M0 基线同口径）。
    """
    import random

    def run_chunk(seeds: list[int]) -> None:
        batch = BatchedEnv(len(seeds), deck, seeds, bot="greedy")
        rng = random.Random(seed)
        while True:
            legal = batch.legal_actions()
            if not any(legal):
                break
            batch.step([rng.randrange(len(l)) if l else 0 for l in legal])

    size = max(1, games // workers)
    chunks = [list(range(seed + i, seed + i + size))
              for i in range(0, games, size)]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(run_chunk, chunks))
    return games / (time.time() - t0)


def bench_battle_batch(games: int, deck: list[str], *, seed: int = 0) -> float:
    """绑定层 rayon 批量（全 Rust，内置 Greedy 对打）的吞吐（局/s）。"""
    t0 = time.time()
    _native.battle_batch(list(range(seed, seed + games)), deck, "greedy")
    return games / (time.time() - t0)
