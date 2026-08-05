"""整回合搜索：把"这一整个回合怎么打"当成一个决策（从 `hearthstone/search.py` 平移）。

为什么是回合而不是 MCTS
----------------------
炉石里搜索的价值几乎全部集中在**本回合内的动作排序**——先用哪个随从换、攻击顺序
怎么排、有没有斩杀。跨回合往下搜会立刻撞上对手手牌未知和抽牌随机，回报衰减极快。

**这是 rosetta 做不到、orange-stone 独有的收益**（路线图 D5）：搜索要反复克隆
局面，rosetta 的 `Game` 拷贝构造被 `= delete` 堵死，orange-stone 的 CoW
`GameState` 让 `Env.clone()` 廉价可恢复。

与 `hearthstone/search.py` 的差异：
- 底层是 `Env`（双 GameEnv 锁步），`clone()` 是绑定层透传；
- **不洗牌堆**：G9 子集卡池没有回合内抽牌（无战吼/法术），克隆搜索不会预知
  牌序；M5 卡池扩大后需要给绑定层加"洗自己的牌堆"的 API；
- **不做 `_settle`**：v6 会在叶子评估前抹掉自家随从的"行动力用尽"瞬态
  （实测 +1.9pp 但不显著），orange-stone 的视图不可变，无法从 Python 侧改，
  叶子价值直接按"对手回合开始"的局面估；
- 搜索从 `env.observe()`（当前行动方视角）取局面，叶子轮到对手动 → 视角是
  对手的，价值取负号换回自己。
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from .env import Action
from .features import state_features
from .policy import UnifiedNet

#: beam 宽度默认值。中位回合的序列数远小于这个数（等于穷举），只有双方满场的后期
#: 回合才真的被截断。
DEFAULT_BEAM = 8

#: 一个回合最多允许多少个动作，防御性上限。
MAX_TURN_ACTIONS = 40


def _terminal_value(env, seat: int) -> Optional[float]:
    """局面已经结束就返回终局回报，否则 None。口径与 `terminal_reward="health_scaled"` 一致。"""
    if not env.done:
        return None
    if env.winner == seat:
        return 1.0
    if env.winner == 0:
        return 0.0
    # 输：按赢家剩血给 0~−1（M1-G7 的口径）
    obs = env.observe()
    winner_health = obs.me.hero_health if env.winner == env.current_player else obs.opponent.hero_health
    return -winner_health / 30.0


class TurnSearchAgent:
    """用价值头当叶子评估的整回合 beam 搜索。

    一次搜出整个回合的动作序列，然后按序列逐个交出去——`play_game` 每次只要
    一个动作，所以这里缓存搜索结果，直到序列走完或者局面对不上再重新搜。
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
        self._env: Optional["Env"] = None
        self._seat = 0
        self._plan: List[Action] = []
        self._plan_turn = -1        # 计划是为哪个回合搜的

    # -------------------------------------------------------------- 接口

    def bind_env(self, env, seat: int) -> None:
        self._env = env
        self._seat = seat
        self._plan = []
        self._plan_turn = -1

    def choose(self, obs, actions: List[Action]) -> Action:
        # 计划只对搜它的那个回合有效。必须在下面任何提前返回之前作废，否则上个回合
        # 没走完的尾巴会漏到这个回合来——那会让智能体开局就打出一个陈旧动作、紧接着
        # 直接结束回合（v6 实测胜率从 50% 崩到 2%）。
        if obs.turn != self._plan_turn:
            self._plan = []

        if len(actions) == 1:
            return actions[0]
        if self._env is None:
            raise RuntimeError("TurnSearchAgent 需要先 bind_env 才能搜索")

        if not self._plan:
            self._plan = self._search(self._env, self._seat)
            self._plan_turn = obs.turn

        # 计划可能因为随机卡（G9 卡池没有，M5 会有）的实际结果而失效，对不上就重搜
        while self._plan:
            action = self._plan.pop(0)
            if action in actions:
                return action
            self._plan = []

        self._plan = self._search(self._env, self._seat)
        self._plan_turn = obs.turn
        action = self._plan.pop(0) if self._plan else actions[-1]
        return action if action in actions else actions[0]

    # -------------------------------------------------------------- 搜索

    def _search(self, env, seat: int) -> List[Action]:
        """搜出这一整个回合的动作序列。"""
        root = env.clone()

        # frontier: (局面, 走到这里的动作序列)
        frontier: List[Tuple["Env", List[Action]]] = [(root, [])]
        # leaves: (打完这个回合的局面, 完整动作序列)
        leaves: List[Tuple["Env", List[Action]]] = []

        for _ in range(MAX_TURN_ACTIONS):
            if not frontier:
                break
            children: List[Tuple["Env", List[Action]]] = []

            for node, path in frontier:
                for action in node.legal_actions():
                    twin = node.clone()
                    twin.step(action)
                    branch = path + [action]
                    # 结束回合、打完了、或者行动权已经交出去 —— 都算这一支走完了
                    if action.kind == "end_turn" or twin.done or twin.current_player != seat:
                        leaves.append((twin, branch))
                    else:
                        children.append((twin, branch))

            frontier = self._prune(children)

        if not leaves:
            return [next(a for a in env.legal_actions() if a.kind == "end_turn")]
        return self._best(leaves, seat)

    def _prune(
        self, children: List[Tuple["Env", List[Action]]]
    ) -> List[Tuple["Env", List[Action]]]:
        """按价值头给中间局面打分，只留最好的 `beam` 个。

        中间局面都是"轮到自己动"，和价值头训练时见到的局面同分布，用来剪枝很合适。
        """
        if len(children) <= self.beam:
            return children
        scores = self._values([node for node, _ in children])
        order = np.argsort(-scores)[: self.beam]
        return [children[i] for i in order]

    def _best(self, leaves: List[Tuple["Env", List[Action]]], seat: int) -> List[Action]:
        """在所有"打完整个回合"的候选里挑最好的一条。

        斩杀（终局 +1.0）无条件优先：价值头可能把"快赢的局面"估到 +1.0 以上
        （未校准），让"结束回合"排在真斩杀前面——实测搜索对着 1 血空场对手
        连续几千回合结束回合，游戏拖到步数上限。
        """
        scores = np.empty(len(leaves), dtype=np.float32)
        pending: List[Tuple[int, "Env"]] = []

        for i, (node, _) in enumerate(leaves):
            terminal = _terminal_value(node, seat)
            if terminal is not None:
                scores[i] = terminal
            else:
                pending.append((i, node))

        if pending:
            # 回合已经交出去，轮到对手动。`env.observe()` 给的是该动的人（对手）的
            # 视角，价值头在"行动方视角"的局面上训练，直接用它再取负号。
            vals = self._values([node for _, node in pending])
            for (i, _), v in zip(pending, vals):
                scores[i] = -v

        # 有直接斩杀的叶子无条件优先（分数被价值头超估的"结束回合"也不能压过它）
        for i, (node, _) in enumerate(leaves):
            if _terminal_value(node, seat) == 1.0:
                return list(leaves[i][1])

        return list(leaves[int(np.argmax(scores))][1])

    def _values(self, envs: Sequence["Env"]) -> np.ndarray:
        """一次前向批量评估一批局面（每个局面都是"行动方视角"）。

        输出裁剪到 [−1, +1]：终局回报的上限就是 ±1，价值头对"快赢局面"的
        超估（实测 +1.065）会让搜索把"结束回合"排在真斩杀前面。
        """
        rows = np.empty((len(envs), self.net.state_dim), dtype=np.float32)
        for i, env in enumerate(envs):
            obs = env.observe()
            going_first = 1.0 if env.current_player == 1 else 0.0
            rows[i] = state_features(obs, going_first)

        with torch.no_grad():
            x = torch.from_numpy(rows).to(self.device)
            emb = self.net.state_encoder(x)
            values = self.net.value_head(emb).squeeze(-1).cpu().numpy()
            return np.clip(values, -1.0, 1.0)
