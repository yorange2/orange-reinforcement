"""orange-stone 炉石模拟器的 Python RL 模块（路线图 M2）。

API 形状复刻 `rosetta/`（Env 门面 + bots/arena/play），底层是 orange_stone
PyO3 绑定。与 rosetta 的三个本质区别（路线图 §1）：
- **clone()**：CoW GameState 克隆廉价，整回合搜索/回滚可直接从 Python 分支
  （rosetta 的 Game 拷贝构造被 `= delete`，搜索做不了）；
- **每局独立 GameRng**：同 seed 逐位可复现、不同实例互不干扰，可多线程并行
  （rosetta 是全局静态 RNG，只能多进程）；
- **MIT 许可**：无 AGPL 传染风险。
"""

from .bots import BOTS, GreedyBot, RandomBot, RuleBot
from .decks import SUBSET_IDS, SUBSET_MAP, VANILLA_IDS, build_deck, vanilla
from .env import Action, Env, describe_action

__all__ = [
    "Action",
    "Env",
    "describe_action",
    "BOTS",
    "RandomBot",
    "GreedyBot",
    "RuleBot",
    "SUBSET_IDS",
    "SUBSET_MAP",
    "VANILLA_IDS",
    "build_deck",
    "vanilla",
]
