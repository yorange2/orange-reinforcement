"""整回合搜索：把"这一整个回合怎么打"当成一个决策，而不是逐个动作贪心。

为什么是回合而不是 MCTS
----------------------
炉石里搜索的价值几乎全部集中在**本回合内的动作排序**——先用哪个随从换、攻击顺序
怎么排、有没有斩杀。跨回合往下搜会立刻撞上对手手牌未知和抽牌随机，回报衰减极快。

实测这个格式的分支因子只有 4.4（中位 3），每回合平均 3.4 个决策点，中位回合的完整
序列数只有几百条、穷举 10ms 以内。但尾部会炸到 30 万条以上（双方满场的后期回合），
所以这里用**定宽 beam** 兜底：搜索是精确的，直到宽度不够为止。

信息边界
--------
搜索要克隆真实 `Game`，也就顺带拿到了对手手牌和双方牌序。本模块靠两件事守住边界：

1. 克隆后**先把自己的牌堆洗一遍**——刀扇/奥术智慧/疾跑会在回合内抽牌，不洗就等于
   预知自己的牌序。
2. 叶子评估只用 `state_features`，不碰先知特征。

对手的手牌在自己回合内不影响任何结算（这个版本没有奥秘），所以不用额外处理。
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from .features import state_features
from .game import END, END_TURN, HERO_HEALTH, Action, Game, Minion, Observation
from .policy import UnifiedNet

#: beam 宽度默认值。实测中位回合远小于这个数（穷举），只有后期满场回合会被截断。
DEFAULT_BEAM = 24

#: 一个回合最多允许多少个动作，防御性上限——正常回合 3~4 个，满场极端情况也就十几个。
MAX_TURN_ACTIONS = 40


def _settle(game: Game, seat: int) -> Game:
    """把刚打完回合那一方的随从恢复成"能动"，再拿去给价值头评估。

    叶子局面是"对手回合开始"，此时我方随从刚攻击完，`attacks_left` 是 0、刚出场的
    还挂着召唤失调。但价值头是在**回合中**的局面上训出来的，那里"随从不能动"是真的
    坏消息；而在叶子上这纯粹是个瞬态——下个回合开始 `_begin_turn` 本来就会把它们全部
    重置。抹掉它，让价值头看到"局面本身"而不是"行动力刚好用完的那一瞬间"。

    实测收益 +1.9pp，但 4 个 seed 下**没到显著**（t≈0.9）。留着是因为它理论上站得住
    且不要钱，别把它当成已经验证过的东西。
    """
    for minion in game.boards[seat]:
        minion.attacks_left = Minion.max_attacks(minion.card)
        minion.just_played = False
    return game


def _terminal_value(game: Game, seat: int) -> Optional[float]:
    """局面已经结束就返回终局回报，否则 None。口径与 `arena.final_reward` 一致。"""
    if not game.finished:
        return None
    if game.winner == seat:
        return 1.0
    if game.winner is None:
        return 0.0
    return -game.hero_health[1 - seat] / HERO_HEALTH


class TurnSearchAgent:
    """用价值头当叶子评估的整回合 beam 搜索。

    一次搜出整个回合的动作序列，然后按序列逐个交出去——`play_game` 每次只要一个动作，
    所以这里缓存搜索结果，直到序列走完或者局面对不上再重新搜。
    """

    name = "search"

    def __init__(
        self,
        net: UnifiedNet,
        beam: int = DEFAULT_BEAM,
        device: torch.device | str = "cpu",
        seed: Optional[int] = None,
    ) -> None:
        self.net = net
        self.beam = beam
        self.device = torch.device(device)
        self.rng = random.Random(seed)
        self._game: Optional[Game] = None
        self._seat = 0
        self._plan: List[Action] = []
        self._plan_turn = -1        # 计划是为哪个半回合搜的

    # -------------------------------------------------------------- 接口

    def bind_game(self, game: Game, seat: int) -> None:
        self._game = game
        self._seat = seat
        self._plan = []
        self._plan_turn = -1

    def choose(self, obs: Observation) -> Action:
        # 计划只对搜它的那个回合有效。必须在下面任何提前返回之前作废，否则上个回合
        # 没走完的尾巴会漏到这个回合来——那会让智能体开局就打出一个陈旧动作、紧接着
        # 直接结束回合（实测胜率从 50% 崩到 2%）。
        if obs.turn != self._plan_turn:
            self._plan = []

        if len(obs.legal) == 1:
            return obs.legal[0]
        if self._game is None:
            raise RuntimeError("TurnSearchAgent 需要先 bind_game 才能搜索")

        if not self._plan:
            self._plan = self._search(self._game, obs.player)
            self._plan_turn = obs.turn

        # 计划可能因为随机卡（奥术飞弹/绝命乱斗）的实际结果而失效，对不上就重搜
        while self._plan:
            action = self._plan.pop(0)
            if action in obs.legal:
                return action
            self._plan = []

        self._plan = self._search(self._game, obs.player)
        self._plan_turn = obs.turn
        action = self._plan.pop(0) if self._plan else END_TURN
        return action if action in obs.legal else obs.legal[0]

    # -------------------------------------------------------------- 搜索

    def _root(self, game: Game, seat: int) -> Game:
        """搜索起点：克隆 + 洗掉自己的牌堆，避免预知牌序。"""
        twin = game.clone(rng=random.Random(self.rng.randrange(1 << 30)))
        twin.rng.shuffle(twin.decks[seat])
        return twin

    def _search(self, game: Game, seat: int) -> List[Action]:
        """搜出这一整个回合的动作序列。"""
        root = self._root(game, seat)

        # frontier: (局面, 走到这里的动作序列)
        frontier: List[Tuple[Game, List[Action]]] = [(root, [])]
        # leaves: (打完这个回合的局面, 完整动作序列)
        leaves: List[Tuple[Game, List[Action]]] = []

        for _ in range(MAX_TURN_ACTIONS):
            if not frontier:
                break
            children: List[Tuple[Game, List[Action]]] = []

            for node, path in frontier:
                for action in node.legal_actions(seat):
                    twin = node.clone(rng=node.rng)
                    twin.step(action)
                    branch = path + [action]
                    # 结束回合、打完了、或者行动权已经交出去 —— 都算这一支走完了
                    if action.kind == END or twin.finished or twin.current != seat:
                        leaves.append((twin, branch))
                    else:
                        children.append((twin, branch))

            frontier = self._prune(children, seat)

        if not leaves:
            return [END_TURN]
        return self._best(leaves, seat)

    def _prune(
        self, children: List[Tuple[Game, List[Action]]], seat: int
    ) -> List[Tuple[Game, List[Action]]]:
        """按价值头给中间局面打分，只留最好的 `beam` 个。

        中间局面都是"轮到自己动"，和价值头训练时见到的局面同分布，用来剪枝很合适。
        """
        if len(children) <= self.beam:
            return children
        scores = self._values([node for node, _ in children], seat)
        order = np.argsort(-scores)[: self.beam]
        return [children[i] for i in order]

    def _best(self, leaves: List[Tuple[Game, List[Action]]], seat: int) -> List[Action]:
        """在所有"打完整个回合"的候选里挑最好的一条。"""
        scores = np.empty(len(leaves), dtype=np.float32)
        pending: List[Tuple[int, Game]] = []

        for i, (node, _) in enumerate(leaves):
            terminal = _terminal_value(node, seat)
            if terminal is not None:
                scores[i] = terminal
            else:
                pending.append((i, node))

        if pending:
            # 回合已经交出去，轮到对手动。价值头给的是"该动的人"的期望回报，取负号
            # 换回自己的视角；评估前先抹掉自家随从的行动力瞬态，理由见 `_settle`。
            vals = self._values(
                [_settle(node, seat) for _, node in pending], 1 - seat
            )
            for (i, _), v in zip(pending, vals):
                scores[i] = -v

        return list(leaves[int(np.argmax(scores))][1])

    def _values(self, games: Sequence[Game], player: int) -> np.ndarray:
        """一次前向批量评估一批局面，视角是 `player`。"""
        rows = np.empty((len(games), self.net.state_dim), dtype=np.float32)
        for i, g in enumerate(games):
            rows[i] = state_features(g.observe(player))

        with torch.no_grad():
            x = torch.from_numpy(rows).to(self.device)
            emb = self.net.state_encoder(x)
            if self.net.oracle_dim:
                # 搜索不许用先知特征，补零占位（这条路径正常不会走到）
                pad = torch.zeros(len(games), self.net.oracle_dim, device=self.device)
                emb = torch.cat([emb, pad], dim=-1)
            return self.net.value_head(emb).squeeze(-1).cpu().numpy()
