"""简化版炉石传说：只有白板随从，没有法术、武器、技能和任何卡牌文本。

    cards.py    白板随从卡池与套牌、关键词
    game.py     引擎：法力水晶、抽牌、疲劳、出随从/法术、攻击、关键词结算、判定胜负
    bots.py     三个手写规则对手 (random / greedy / rule)
    features.py (局面, 动作) -> 定长特征向量
    policy.py   打分网络、价值网络、智能体、变长动作集的批量前向、存取权重
    train.py    训练脚本 (PPO，默认 2000 局)
    bench.py    统一口径的胜率基准表
    arena.py    对局评测与胜率统计
    play.py     人机对战 / 观战
"""
