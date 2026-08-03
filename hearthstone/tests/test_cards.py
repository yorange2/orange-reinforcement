import random
import unittest

from hearthstone.cards import (
    CARD_INDEX,
    CHARGE,
    COPIES,
    DECK_SIZE,
    DISTINCT,
    DIVINE_SHIELD,
    KEYWORD_INDEX,
    KEYWORDS,
    LIFESTEAL,
    POISONOUS,
    REBORN,
    RUSH,
    SPELL_DAMAGE,
    STEALTH,
    TAUNT,
    THE_COIN,
    WINDFURY,
    CardDef,
    build_decklist,
    by_keyword,
    hand_to_str,
    parse_card,
    shuffled,
)


class TestPool(unittest.TestCase):
    def test_pool_is_non_empty(self):
        self.assertGreater(len(CARD_INDEX), 20)

    def test_all_cards_have_valid_costs(self):
        from hearthstone.cards import POOL

        for card in POOL:
            self.assertGreaterEqual(card.cost, 0)

    def test_names_are_unique(self):
        from hearthstone.cards import POOL

        names = [card.name for card in POOL]
        self.assertEqual(len(names), len(set(names)))

    def test_card_index_matches_pool_order(self):
        from hearthstone.cards import POOL

        for i, card in enumerate(POOL):
            self.assertEqual(CARD_INDEX[card.name], i)

    def test_all_keywords_are_valid(self):
        from hearthstone.cards import POOL

        for card in POOL:
            for word in card.keywords:
                self.assertIn(word, KEYWORDS)

    def test_keyword_index_is_contiguous(self):
        self.assertEqual(KEYWORD_INDEX[CHARGE], 0)
        self.assertEqual(len(KEYWORD_INDEX), len(KEYWORDS))

    def test_by_keyword_returns_relevant_cards(self):
        for word in [TAUNT, CHARGE, DIVINE_SHIELD]:
            cards = by_keyword(word)
            self.assertTrue(all(card.has(word) for card in cards))

    def test_the_coin_is_a_spell(self):
        self.assertTrue(THE_COIN.spell)
        self.assertEqual(THE_COIN.cost, 0)
        self.assertEqual(THE_COIN.name, "幸运币")

    def test_card_has_method(self):
        card = CardDef("测试", 3, 2, 3, (TAUNT,))
        self.assertTrue(card.has(TAUNT))
        self.assertFalse(card.has(CHARGE))


class TestDeck(unittest.TestCase):
    def test_build_decklist_is_30_cards(self):
        deck = build_decklist(random.Random(0))
        self.assertEqual(len(deck), DECK_SIZE)

    def test_build_decklist_has_two_copies(self):
        deck = build_decklist(random.Random(0))
        names = [c.name for c in deck]
        for name in set(names):
            self.assertEqual(names.count(name), COPIES)

    def test_build_decklist_is_reproducible(self):
        self.assertEqual(build_decklist(random.Random(7)), build_decklist(random.Random(7)))

    def test_build_decklist_variance(self):
        decks = {tuple(c.name for c in build_decklist(random.Random(s))) for s in range(30)}
        self.assertGreater(len(decks), 1)

    def test_shuffled_is_a_permutation(self):
        decklist = build_decklist(random.Random(1))
        shuf = shuffled(decklist, random.Random(2))
        self.assertCountEqual(decklist, shuf)

    def test_shuffled_same_seed_same_order(self):
        decklist = build_decklist(random.Random(1))
        self.assertEqual(shuffled(decklist, random.Random(3)), shuffled(decklist, random.Random(3)))


class TestParse(unittest.TestCase):
    def test_parse_by_full_name(self):
        self.assertEqual(parse_card("暴风城骑士").name, "暴风城骑士")

    def test_parse_by_index(self):
        from hearthstone.cards import POOL

        self.assertEqual(parse_card("0"), POOL[0])

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            parse_card("阿古斯守护者")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            parse_card("   ")


class TestFormat(unittest.TestCase):
    def test_hand_to_str_sorts_by_cost(self):
        from hearthstone.cards import POOL

        # find a cheap and an expensive card
        cheap = [c for c in POOL if c.cost == 1 and not c.spell][0]
        expensive = [c for c in POOL if c.cost >= 6 and not c.spell][0]
        text = hand_to_str([expensive, cheap])
        self.assertLess(text.index(cheap.name), text.index(expensive.name))

    def test_card_str_includes_keywords(self):
        card = CardDef("测试", 3, 3, 3, (TAUNT, DIVINE_SHIELD))
        s = str(card)
        self.assertIn(TAUNT, s)
        self.assertIn(DIVINE_SHIELD, s)

    def test_card_str_for_spell(self):
        s = str(THE_COIN)
        self.assertIn(THE_COIN.name, s)


if __name__ == "__main__":
    unittest.main()
