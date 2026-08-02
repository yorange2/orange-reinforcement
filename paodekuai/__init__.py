"""跑得快 + 强化学习：一个让模型学会击败规则算法的小项目。"""

from .cards import Card, deal, full_deck
from .combos import Combo, beats, classify, legal_moves
from .game import Game, Observation, play_game

__all__ = [
    "Card", "deal", "full_deck",
    "Combo", "beats", "classify", "legal_moves",
    "Game", "Observation", "play_game",
]
__version__ = "0.1.0"
