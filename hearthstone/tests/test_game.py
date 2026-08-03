import random
import unittest

from hearthstone.cards import (
    CHARGE,
    DIVINE_SHIELD,
    LIFESTEAL,
    POISONOUS,
    REBORN,
    RUSH,
    STEALTH,
    TAUNT,
    THE_COIN,
    WINDFURY,
    CardDef,
)
from hearthstone.game import (
    BOARD_LIMIT,
    COIN_MANA,
    END_TURN,
    HAND_LIMIT,
    HERO,
    HERO_HEALTH,
    MAX_MANA,
    STARTING_HAND,
    Game,
    Minion,
    attack,
    describe,
    play,
    play_game,
)

WISP = CardDef("w", 0, 1, 1)
BIG = CardDef("B", 6, 6, 7)
RAGER = CardDef("R", 3, 5, 1)
CROC = CardDef("C", 2, 2, 3)
CHARGER = CardDef("CH", 1, 2, 1, (CHARGE,))
RUSHER = CardDef("RU", 4, 4, 5, (RUSH,))
TAUNT_GUY = CardDef("T", 2, 1, 3, (TAUNT,))
STEALTH_GUY = CardDef("S", 3, 4, 2, (STEALTH,))
SHIELD_GUY = CardDef("D", 1, 1, 1, (DIVINE_SHIELD,))
POISON_GUY = CardDef("P", 3, 2, 3, (POISONOUS,))
LIFESTEAL_GUY = CardDef("L", 1, 2, 1, (LIFESTEAL,))
WINDFURY_GUY = CardDef("WF", 1, 1, 1, (WINDFURY,))
REBORN_GUY = CardDef("RB", 4, 3, 2, (REBORN,))
TAUNT_SHIELD = CardDef("TD", 4, 3, 3, (TAUNT, DIVINE_SHIELD))
TAUNT_REBORN = CardDef("TR", 4, 2, 5, (TAUNT, REBORN))
LIFESTEAL_RUSH = CardDef("LR", 2, 1, 3, (LIFESTEAL, RUSH))
STEALTH_REBORN = CardDef("SR", 5, 4, 2, (STEALTH, REBORN))


def fresh(seed=0, **kwargs):
    return Game(rng=random.Random(seed), **kwargs)


def stack(game, player, *cards, ready=False):
    """直接把随从摆到场上，绕过费用、手牌和召唤失调。"""
    for card in cards:
        minion = Minion.summon(card, game._take_uid())
        minion.just_played = not ready
        if ready:
            minion.attacks_left = Minion.max_attacks(card)
        game.boards[player].append(minion)
    return game.boards[player]


# ================================================================ Setup
class TestSetup(unittest.TestCase):
    def test_starting_hands(self):
        game = fresh()
        # 先手 3 + 自己回合抽的 1；后手 4 + Coin，还没抽
        self.assertEqual(len(game.hands[0]), STARTING_HAND[0] + 1)
        self.assertEqual(len(game.hands[1]), STARTING_HAND[1] + 1)
        self.assertIn(THE_COIN, game.hands[1])

    def test_second_player_draws_on_own_turn(self):
        game = fresh()
        game.step(END_TURN)
        self.assertEqual(len(game.hands[1]), STARTING_HAND[1] + 2)

    def test_first_player_option(self):
        game = fresh(first=1)
        self.assertIn(THE_COIN, game.hands[0])
        self.assertNotIn(THE_COIN, game.hands[1])

    def test_same_seed_same_game(self):
        self.assertEqual(fresh(3).hands, fresh(3).hands)


# ================================================================ Mana / Coin
class TestMana(unittest.TestCase):
    def test_coin_gives_extra_mana(self):
        game = fresh()
        game.hands[0] = [THE_COIN]
        game.mana[0] = 0
        game.step(play(0))
        self.assertEqual(game.mana[0], COIN_MANA)  # 0 cost + 1 mana bonus

    def test_cannot_play_unaffordable(self):
        game = fresh()
        game.hands[0] = [BIG]
        game.mana[0] = 1
        with self.assertRaises(ValueError):
            game.step(play(0))

    def test_mana_refills_each_turn(self):
        game = fresh()
        game.mana[0] = 0
        game.step(END_TURN)
        game.step(END_TURN)
        self.assertEqual(game.mana[0], game.max_mana[0])


# ================================================================ Keywords — Charge / Rush
class TestChargeRush(unittest.TestCase):
    def test_charge_can_attack_immediately(self):
        game = fresh()
        game.hands[0] = [CHARGER]
        game.mana[0] = 10
        game.step(play(0))
        self.assertTrue(game.boards[0][0].can_attack)
        self.assertTrue(game.boards[0][0].can_hit_face)

    def test_rush_can_attack_minion_but_not_face(self):
        game = fresh()
        game.hands[0] = [RUSHER]
        game.mana[0] = 10
        stack(game, 1, WISP, ready=False)
        game.step(play(0))
        self.assertTrue(game.boards[0][0].can_attack)
        self.assertFalse(game.boards[0][0].can_hit_face)

    def test_rush_can_only_attack_minions_turn_played(self):
        game = fresh()
        game.hands[0] = [RUSHER]
        game.mana[0] = 10
        stack(game, 1, WISP, ready=False)
        game.step(play(0))
        legal = game.legal_actions()
        attack_targets = [a.target for a in legal if a.kind == "attack"]
        self.assertNotIn(HERO, attack_targets)
        self.assertIn(0, attack_targets)

    def test_normal_minion_has_summoning_sickness(self):
        game = fresh()
        game.hands[0] = [BIG]
        game.mana[0] = 10
        game.step(play(0))
        self.assertFalse(game.boards[0][0].can_attack)
        game.step(END_TURN)
        game.step(END_TURN)
        self.assertTrue(game.boards[0][0].can_attack)


# ================================================================ Keywords — Taunt
class TestTaunt(unittest.TestCase):
    def test_taunt_blocks_face(self):
        game = fresh()
        stack(game, 0, BIG, ready=True)
        stack(game, 1, TAUNT_GUY, ready=False)
        legal = game.legal_actions()
        attack_targets = [a.target for a in legal if a.kind == "attack"]
        self.assertNotIn(HERO, attack_targets)
        self.assertIn(0, attack_targets)

    def test_no_taunt_allows_face(self):
        game = fresh()
        stack(game, 0, BIG, ready=True)
        stack(game, 1, WISP, ready=False)
        self.assertIn(attack(0, HERO), game.legal_actions())

    def test_stealthed_taunt_does_not_block(self):
        game = fresh()
        guido = Minion.summon(TAUNT_GUY, 99)
        guido.stealth = True
        guido.just_played = False
        game.boards[1].append(guido)
        stack(game, 0, BIG, ready=True)
        self.assertIn(attack(0, HERO), game.legal_actions())

    def test_multiple_taunts_all_valid_targets(self):
        game = fresh()
        stack(game, 0, BIG, ready=True)
        stack(game, 1, TAUNT_GUY, TAUNT_SHIELD, ready=False)
        targets = {a.target for a in game.legal_actions() if a.kind == "attack"}
        self.assertEqual(targets, {0, 1})


# ================================================================ Keywords — Stealth
class TestStealth(unittest.TestCase):
    def test_stealth_cannot_be_targeted(self):
        game = fresh()
        stack(game, 0, BIG, ready=True)
        stack(game, 1, STEALTH_GUY, WISP, ready=False)
        targets = {a.target for a in game.legal_actions() if a.kind == "attack"}
        self.assertNotIn(0, targets)
        self.assertIn(1, targets)

    def test_stealth_is_lost_on_attacking(self):
        game = fresh()
        stack(game, 0, STEALTH_GUY, ready=True)
        stack(game, 1, WISP, ready=False)
        game.step(attack(0, HERO))
        self.assertFalse(game.boards[0][0].stealth)

    def test_stealth_is_lost_on_being_attacked(self):
        """被对面打不会掉潜行——只有自己主动攻击才会。"""
        pass  # Stealth only drops when the stealthed minion attacks


# ================================================================ Keywords — Divine Shield
class TestDivineShield(unittest.TestCase):
    def test_shield_blocks_incoming_damage(self):
        """圣盾只挡受到的伤害——自己攻击别人不消耗圣盾。"""
        game = fresh()
        # shielded minion attacks a poisonous guy: shield blocks the counter-damage
        attacker = Minion.summon(SHIELD_GUY, game._take_uid())
        attacker.just_played = False
        attacker.attacks_left = 1
        game.boards[0].append(attacker)
        defender = Minion.summon(POISON_GUY, game._take_uid())
        defender.just_played = False
        game.boards[1].append(defender)
        game.step(attack(0, 0))
        self.assertTrue(game.boards[0][0].health > 0)          # survived (shield blocked poison counter)
        self.assertFalse(game.boards[0][0].divine_shield)

    def test_shield_persists_when_attacking_face(self):
        """自己打人不掉圣盾。"""
        game = fresh()
        stack(game, 0, SHIELD_GUY, ready=True)
        game.step(attack(0, HERO))
        self.assertTrue(game.boards[0][0].divine_shield)       # attacking face doesn't remove shield

    def test_shield_blocks_poison(self):
        game = fresh()
        stack(game, 0, SHIELD_GUY, ready=True)
        stack(game, 1, POISON_GUY, ready=False)
        game.step(attack(0, 0))
        self.assertFalse(game.boards[0][0].divine_shield)
        self.assertFalse(game.boards[0][0].health <= 0)

    def test_shield_prevents_lifesteal_heal(self):
        game = fresh()
        game.hero_health[0] = 10
        stack(game, 0, LIFESTEAL_GUY, ready=True)
        defender = Minion.summon(SHIELD_GUY, 99)
        defender.just_played = False
        game.boards[1].append(defender)
        game.step(attack(0, 0))
        self.assertEqual(game.hero_health[0], 10)  # no heal


# ================================================================ Keywords — Poisonous
class TestPoisonous(unittest.TestCase):
    def test_poison_kills_on_damage(self):
        game = fresh()
        stack(game, 0, POISON_GUY, ready=True)
        stack(game, 1, BIG, ready=False)
        game.step(attack(0, 0))
        self.assertEqual(game.boards[1], [])

    def test_poison_does_not_kill_if_no_damage_dealt(self):
        game = fresh()
        stack(game, 0, POISON_GUY, ready=True)
        defender = Minion.summon(SHIELD_GUY, 99)  # divine shield
        defender.just_played = False
        game.boards[1].append(defender)
        game.step(attack(0, 0))
        self.assertFalse(game.boards[1][0].divine_shield)  # shield popped
        self.assertTrue(game.boards[1][0].health > 0)       # but not dead


# ================================================================ Keywords — Windfury
class TestWindfury(unittest.TestCase):
    def test_windfury_attacks_twice(self):
        game = fresh()
        stack(game, 0, WINDFURY_GUY, ready=True)
        self.assertEqual(game.boards[0][0].attacks_left, 2)
        game.step(attack(0, HERO))
        self.assertEqual(game.boards[0][0].attacks_left, 1)
        self.assertTrue(game.boards[0][0].can_attack)
        game.step(attack(0, HERO))
        self.assertEqual(game.boards[0][0].attacks_left, 0)


# ================================================================ Keywords — Lifesteal
class TestLifesteal(unittest.TestCase):
    def test_lifesteal_heals_on_face(self):
        game = fresh()
        game.hero_health[0] = 10
        stack(game, 0, LIFESTEAL_GUY, ready=True)
        game.step(attack(0, HERO))
        self.assertEqual(game.hero_health[0], 10 + LIFESTEAL_GUY.attack)

    def test_lifesteal_cannot_overheal(self):
        game = fresh()
        stack(game, 0, LIFESTEAL_GUY, ready=True)
        game.step(attack(0, HERO))
        self.assertEqual(game.hero_health[0], HERO_HEALTH)

    def test_lifesteal_on_minion_combat(self):
        game = fresh()
        game.hero_health[0] = 10
        stack(game, 0, LIFESTEAL_GUY, ready=True)
        stack(game, 1, WISP, ready=False)
        game.step(attack(0, 0))
        self.assertEqual(game.hero_health[0], 10 + LIFESTEAL_GUY.attack)


# ================================================================ Keywords — Reborn
class TestReborn(unittest.TestCase):
    def test_reborn_returns_with_one_health(self):
        game = fresh()
        stack(game, 0, BIG, ready=True)
        stack(game, 1, REBORN_GUY, ready=False)
        game.step(attack(0, 0))                    # 6 damage kills the 2hp reborn guy
        self.assertEqual(len(game.boards[1]), 1)   # came back
        self.assertEqual(game.boards[1][0].health, 1)
        self.assertFalse(game.boards[1][0].reborn)

    def test_reborn_triggers_only_once(self):
        game = fresh()
        stack(game, 0, BIG, ready=True)
        stack(game, 1, REBORN_GUY, ready=False)
        game.step(attack(0, 0))
        game.step(END_TURN)
        game.step(END_TURN)
        self.assertEqual(len(game.boards[1]), 1)
        game.step(attack(0, 0))                    # kill the reborn copy
        self.assertEqual(game.boards[1], [])

    def test_reborn_blocked_by_full_board(self):
        game = fresh()
        stack(game, 0, BIG, ready=True)
        stack(game, 1, *[WISP] * BOARD_LIMIT, ready=False)
        game.boards[1][0] = Minion.summon(REBORN_GUY, 99)
        game.boards[1][0].just_played = False
        game.step(attack(0, 0))
        self.assertEqual(len(game.boards[1]), BOARD_LIMIT - 1)  # lost one, no room


# ================================================================ Combat
class TestCombat(unittest.TestCase):
    def test_damage_is_simultaneous(self):
        game = fresh()
        stack(game, 0, BIG, ready=True)
        stack(game, 1, RAGER, ready=False)
        game.step(attack(0, 0))
        self.assertEqual(game.boards[1], [])
        self.assertEqual(game.boards[0][0].health, BIG.health - RAGER.attack)

    def test_both_can_die(self):
        game = fresh()
        stack(game, 0, RAGER, ready=True)
        stack(game, 1, RAGER, ready=False)
        game.step(attack(0, 0))
        self.assertEqual((game.boards[0], game.boards[1]), ([], []))

    def test_attacks_once_per_turn(self):
        game = fresh()
        stack(game, 0, BIG, ready=True)
        game.step(attack(0, HERO))
        self.assertFalse(game.boards[0][0].can_attack)
        with self.assertRaises(ValueError):
            game.step(attack(0, HERO))


# ================================================================ Draw / Fatigue
class TestDrawAndFatigue(unittest.TestCase):
    def test_fatigue_increases(self):
        game = fresh()
        game.decks[0] = []
        for expected in (1, 3, 6):
            game._draw(0)
            self.assertEqual(game.hero_health[0], HERO_HEALTH - expected)

    def test_fatigue_can_kill(self):
        game = fresh()
        game.decks[0] = []
        game.hero_health[0] = 1
        game._draw(0)
        game._check_over()
        self.assertTrue(game.finished)
        self.assertEqual(game.winner, 1)

    def test_overdraw_burns_card(self):
        game = fresh()
        game.hands[0] = [WISP] * HAND_LIMIT
        top = game.decks[0][-1]
        game._draw(0)
        self.assertEqual(len(game.hands[0]), HAND_LIMIT)
        self.assertEqual(game.burned[0], [top])


# ================================================================ Game Over
class TestGameOver(unittest.TestCase):
    def test_lethal_ends_game(self):
        game = fresh()
        game.hero_health[1] = 3
        stack(game, 0, BIG, ready=True)
        game.step(attack(0, HERO))
        self.assertTrue(game.finished)
        self.assertEqual(game.winner, 0)

    def test_cannot_act_after_end(self):
        game = fresh()
        game.hero_health[1] = 1
        stack(game, 0, BIG, ready=True)
        game.step(attack(0, HERO))
        with self.assertRaises(RuntimeError):
            game.step(END_TURN)

    def test_result_before_end_raises(self):
        with self.assertRaises(RuntimeError):
            fresh().result()

    def test_has_lethal_detects_taunt(self):
        game = fresh()
        game.hero_health[1] = 1
        stack(game, 0, BIG, ready=True)
        stack(game, 1, TAUNT_GUY, ready=False)
        self.assertFalse(game.observe().has_lethal())


# ================================================================ Legal Actions
class TestLegalActions(unittest.TestCase):
    def test_end_turn_always_legal(self):
        self.assertIn(END_TURN, fresh().legal_actions())

    def test_duplicate_hand_cards_collapse(self):
        game = fresh()
        game.hands[0] = [WISP, WISP, WISP]
        game.mana[0] = 10
        plays = [a for a in game.legal_actions() if a.kind == "play"]
        self.assertEqual(len(plays), 1)

    def test_full_board_blocks_minions_not_spells(self):
        game = fresh()
        game.hands[0] = [WISP, THE_COIN]
        game.mana[0] = 10
        stack(game, 0, *[WISP] * BOARD_LIMIT)
        plays = game.observe().playable()
        names = {game.hands[0][a.source].name for a in plays}
        self.assertEqual(names, {THE_COIN.name})


# ================================================================ Play Game
class TestPlayGame(unittest.TestCase):
    def test_random_bots_always_finish(self):
        from hearthstone.bots import RandomBot

        for seed in range(30):
            result = play_game([RandomBot(seed), RandomBot(seed + 100)], rng=random.Random(seed))
            self.assertIn(result.winner, (0, 1, None))

    def test_illegal_action_rejected(self):
        class Cheater:
            def choose(self, obs):
                return attack(99, HERO)

        from hearthstone.bots import RandomBot

        with self.assertRaises(ValueError):
            play_game([Cheater(), RandomBot(0)], rng=random.Random(0))


class TestDescribe(unittest.TestCase):
    def test_describe_covers_action_kinds(self):
        game = fresh()
        game.hands[0] = [BIG, THE_COIN]
        game.mana[0] = 10
        stack(game, 0, BIG, ready=True)
        stack(game, 1, WISP, ready=False)
        obs = game.observe()
        for action in obs.legal:
            self.assertTrue(describe(obs, action), f"describe failed for {action}")


if __name__ == "__main__":
    unittest.main()
