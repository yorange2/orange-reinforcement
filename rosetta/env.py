"""RosettaStone 对局环境的 Python 门面。

底下是 `rosetta_env` 这个 pybind11 扩展（见 native/），它把 RosettaStone 的
C++ 引擎包成了 reset / legal_actions / step / observe 四件套。官方的 pyRosetta
只导出卡牌数据库，打不了牌，所以这一层是我们自己写的。
"""

from __future__ import annotations

try:
    from . import rosetta_env as _native
except ImportError as exc:  # pragma: no cover - 只在没编译时触发
    raise ImportError(
        "rosetta_env 扩展还没编译。先跑 ./rosetta/build.sh"
    ) from exc

ActionType = _native.ActionType
Action = _native.Action

#: 平局
DRAW = 0

__all__ = ["Action", "ActionType", "Env", "DRAW", "describe_action"]


class Env:
    """一局炉石。

    和 `hearthstone/` 那个自研引擎最大的不同：**没有 clone()**。
    RosettaStone 的 `Game` 拷贝和移动构造函数都是 `= delete`，所以整回合搜索
    那套东西在这里没有直接对应物。
    """

    def __init__(
        self,
        player1_class: str = "MAGE",
        player2_class: str = "MAGE",
        player1_deck: list[str] | None = None,
        player2_deck: list[str] | None = None,
        *,
        start_player: str = "PLAYER1",
    ) -> None:
        self._env = _native.Env(
            player1_class=player1_class,
            player2_class=player2_class,
            player1_deck=player1_deck or [],
            player2_deck=player2_deck or [],
            skip_mulligan=True,
            start_player=start_player,
        )

    def reset(self, seed: int = -1) -> "_native.Observation":
        """开一局新的。

        seed 是**进程级**的：RosettaStone 用一个全局静态 RNG，所以并行采样
        只能多进程，不能多线程。
        """
        self._env.reset(seed)
        return self.observe()

    def legal_actions(self) -> list[Action]:
        return self._env.legal_actions()

    def step(self, action: Action) -> "_native.Observation":
        self._env.step(action)
        return self.observe()

    def observe(self) -> "_native.Observation":
        """当前**行动方**视角的局面。`me` 带手牌，`opponent` 不带。"""
        return self._env.observe()

    @property
    def done(self) -> bool:
        return self._env.done

    @property
    def winner(self) -> int:
        """0 = 未结束或平局，1 = player1，2 = player2。"""
        return self._env.winner

    @property
    def current_player(self) -> int:
        return self._env.current_player

    @property
    def turn(self) -> int:
        return self._env.turn


def describe_action(action: Action, obs: "_native.Observation") -> str:
    """把一个动作渲染成人能读的一行。"""
    if action.type == ActionType.END_TURN:
        return "结束回合"

    if action.type == ActionType.CHOOSE:
        return f"选择实体 {action.choice}"

    target = _describe_target(action, obs)

    if action.type == ActionType.PLAY_CARD:
        card = obs.me.hand[action.hand_idx]
        text = f"出 {card.name}({card.cost}费)"
        return f"{text} -> {target}" if target else text

    if action.type == ActionType.HERO_POWER:
        return f"英雄技能 -> {target}" if target else "英雄技能"

    source = (
        "英雄"
        if action.source_pos < 0
        else obs.me.field[action.source_pos].name
    )
    return f"{source} 攻击 {target}"


def _describe_target(action: Action, obs: "_native.Observation") -> str:
    if action.target_side < 0:
        return ""

    side = obs.me if action.target_side == 0 else obs.opponent
    label = "自己" if action.target_side == 0 else "对方"

    if action.target_pos < 0:
        return f"{label}英雄"

    if action.target_pos < len(side.field):
        return f"{label} {side.field[action.target_pos].name}"

    return f"{label} 场上 {action.target_pos} 号"
