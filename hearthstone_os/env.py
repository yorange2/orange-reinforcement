"""orange-stone 对局环境的 Python 门面（路线图 M2）。

API 形状复刻 `rosetta/env.py`：`reset(seed)` / `legal_actions()` / `step(action)`
/ `observe()` 四件套，`observe()` 返回**当前行动方**视角（`me` 带手牌，
`opponent` 手牌隐藏）。

与 rosetta 的三个本质区别（路线图 §1）：
- **clone()**：CoW GameState 克隆廉价，整回合搜索/回滚可以直接从 Python 分支；
- **每局独立 GameRng**：同 seed 逐位可复现、不同实例互不干扰（可多线程并行）；
- **无全局静态状态**：不需要 rosetta 那个 `gc.disable()` 的 hack。

实现细节：`orange_stone.GameEnv` 的 `perspective` 构造时固定，而双 bot 对局
需要**任意一方**都能看到自己的手牌。所以 `Env` 内部同时驱动两个 GameEnv
（perspective=0 和 1），同 seed、同动作序列走锁步——引擎确定性保证两边状态
一致，`observe()` 按当前行动方返回对应那一个的视图。
"""

from __future__ import annotations

import orange_stone as _native

__all__ = ["Action", "Env", "describe_action"]


class Action:
    """结构化动作的薄封装：字段照搬 `orange_stone.ActionView`。

    - `index`：传给 `Env.step` 的下标（与底层 `legal_actions()` 对齐）
    - `kind`：`"end_turn" | "play" | "attack" | "hero_power" | "choose"`
    - `card_index`：出牌的手牌下标（`play` 动作，否则 -1）
    - `entity_id` / `target_id`：实体槽下标；攻击动作的目标不在对方场上
      就是打脸（英雄不占 `opponent.field` 的槽位）
    """

    __slots__ = ("index", "kind", "card_index", "entity_id", "target_id",
                 "description")

    def __init__(self, index: int, kind: str, card_index: int = -1,
                 entity_id: int = -1, target_id: int = -1,
                 description: str = "") -> None:
        self.index = index
        self.kind = kind
        self.card_index = card_index
        self.entity_id = entity_id
        self.target_id = target_id
        self.description = description

    @classmethod
    def from_view(cls, view) -> "Action":
        """从 `orange_stone.ActionView` 构造。"""
        return cls(view.index, view.kind, view.card_index, view.entity_id,
                   view.target_id, view.description)

    def __repr__(self) -> str:
        return f"Action({self.kind}, index={self.index})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Action):
            return NotImplemented
        return (
            self.index, self.kind, self.card_index, self.entity_id,
            self.target_id,
        ) == (other.index, other.kind, other.card_index, other.entity_id,
              other.target_id)

    def __hash__(self) -> int:
        return hash((self.index, self.kind, self.card_index, self.entity_id,
                     self.target_id))


class Env:
    """一局炉石（arena / 人机 / 训练共用）。"""

    def __init__(
        self,
        deck: list[str] | None = None,
        *,
        seed: int = 0,
        bot: str = "none",
        hand_size: int = 3,
        second_player_coin: bool = True,
        terminal_reward: str = "sparse",
    ) -> None:
        """开一局。

        `deck=None` 时双方用全卡池随机卡组（M1-G2 保留的随机模式）；
        否则是双方镜像的固定卡组。`bot` 默认 `"none"`（双方外部可控，
        arena/对拍用），`"greedy"`/`"smart"` 会由内置 bot 代打对方回合。
        起手默认 3 张 + 后手硬币（简化炉石口径，路线图 G6）。
        """
        if deck is not None:
            deck = list(deck)
        self._deck = deck
        self._bot = bot
        self._hand_size = hand_size
        self._second_player_coin = second_player_coin
        self._terminal_reward = terminal_reward
        self._seed = seed
        # 双实例锁步：同 seed 构造 → 初始状态一致；后续所有 step 两边同步
        self._env0 = self._make(0)
        self._env1 = self._make(1)

    def _make(self, perspective: int) -> _native.GameEnv:
        return _native.GameEnv(
            seed=self._seed,
            perspective=perspective,
            deck=self._deck,
            bot=self._bot,
            hand_size=self._hand_size,
            second_player_coin=self._second_player_coin,
            terminal_reward=self._terminal_reward,
        )

    # ------------------------------------------------------------ 四件套

    def reset(self, seed: int | None = None) -> "_native.Observation":
        """开一局新的。`seed=None` 沿用构造时的 seed。"""
        if seed is not None:
            self._seed = seed
        self._env0.reset(self._seed)
        self._env1.reset(self._seed)
        return self.observe()

    def legal_actions(self) -> list[Action]:
        """当前行动方的合法动作（底层两个 GameEnv 枚举完全一致）。"""
        return [Action.from_view(v) for v in self._env0.structured_legal_actions()]

    def step(self, action: Action | int) -> "_native.Observation":
        """执行动作（接受 `Action` 对象或裸下标），返回行动方视角的新局面。"""
        index = action.index if isinstance(action, Action) else int(action)
        self._env0.step(index)
        self._env1.step(index)
        return self.observe()

    def observe(self) -> "_native.Observation":
        """当前**行动方**视角。`me` 带手牌，`opponent` 不带。

        注意 `current_player` 是 1-based（1=P1），`_view` 是 0-based。
        """
        return self._view(self.current_player - 1).structured_observation()

    def clone(self) -> "Env":
        """分支一份当前局面（M1-G5 clone 的透传）：搜索/回滚用。

        两个内部 GameEnv 各自 clone，互不影响原局。
        """
        twin = object.__new__(Env)
        twin._deck = self._deck
        twin._bot = self._bot
        twin._hand_size = self._hand_size
        twin._second_player_coin = self._second_player_coin
        twin._terminal_reward = self._terminal_reward
        twin._seed = self._seed
        twin._env0 = self._env0.clone()
        twin._env1 = self._env1.clone()
        return twin

    # ------------------------------------------------------------ 状态

    @property
    def done(self) -> bool:
        return self._view(0).structured_observation().done

    @property
    def winner(self) -> int:
        """0 = 未结束或平局，1 = P1，2 = P2。"""
        return self._view(0).structured_observation().winner

    @property
    def current_player(self) -> int:
        """当前行动方（1 = P1，2 = P2）。

        env0 是 P1 视角：`my_turn` 为真说明轮到 P1 行动。
        """
        return 1 if self._view(0).structured_observation().my_turn else 2

    @property
    def turn(self) -> int:
        return self._view(0).structured_observation().turn

    # ------------------------------------------------------------ 内部

    def _view(self, perspective: int) -> _native.GameEnv:
        """perspective 0 → env0（P1 视角），1 → env1（P2 视角）。"""
        return self._env0 if perspective == 0 else self._env1


def describe_action(action: Action, obs: "_native.Observation") -> str:
    """把一个动作渲染成人能读的一行（play.py 用）。"""
    if action.kind == "end_turn":
        return "结束回合"
    if action.kind == "choose":
        return f"选择实体 {action.target_id}"

    target = _describe_target(action, obs)

    if action.kind == "play":
        card = obs.me.hand[action.card_index]
        text = f"出 {card.name}({card.cost}费)"
        return f"{text} -> {target}" if target else text

    if action.kind == "hero_power":
        return f"英雄技能 -> {target}" if target else "英雄技能"

    source = _source_name(action, obs)
    return f"{source} 攻击 {target}"


def _source_name(action: Action, obs: "_native.Observation") -> str:
    """攻击者名字：英雄（entity_id 不在我方场上）或随从。"""
    for minion in obs.me.field:
        if minion.entity_id == action.entity_id:
            return minion.name
    return "英雄"


def _describe_target(action: Action, obs: "_native.Observation") -> str:
    """目标描述：对手随从 / 自己随从 / 英雄。"""
    if action.target_id < 0:
        return ""
    for minion in obs.opponent.field:
        if minion.entity_id == action.target_id:
            return f"对方 {minion.name}"
    for minion in obs.me.field:
        if minion.entity_id == action.target_id:
            return f"自己 {minion.name}"
    return "对方英雄"
