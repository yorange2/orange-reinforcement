import random
import unittest

from paodekuai.cards import (DIAMOND_THREE, HAND_SIZE, RANKS, Card, deal,
                             full_deck, hand_to_str, parse_card, rank_counts)


class TestCards(unittest.TestCase):
    def test_deck_is_48_cards(self):
        deck = full_deck()
        self.assertEqual(len(deck), 48)
        self.assertEqual(len(set(deck)), 48)

    def test_no_twos_and_no_jokers(self):
        ranks = {card.rank for card in full_deck()}
        self.assertEqual(ranks, set(RANKS))
        self.assertNotIn(2, ranks)
        self.assertEqual(min(ranks), 3)
        self.assertEqual(max(ranks), 14)

    def test_each_rank_has_four_suits(self):
        counts = rank_counts(full_deck())
        self.assertTrue(all(count == 4 for count in counts.values()))

    def test_deal_gives_everyone_16_cards(self):
        hands = deal(random.Random(0))
        self.assertEqual([len(hand) for hand in hands], [HAND_SIZE] * 3)
        all_cards = [card for hand in hands for card in hand]
        self.assertEqual(len(set(all_cards)), 48)

    def test_deal_is_deterministic_per_seed(self):
        self.assertEqual(deal(random.Random(7)), deal(random.Random(7)))

    def test_deal_rejects_too_many_players(self):
        with self.assertRaises(ValueError):
            deal(random.Random(0), n_players=4, hand_size=16)

    def test_diamond_three_sorts_first_among_threes(self):
        threes = sorted(card for card in full_deck() if card.rank == 3)
        self.assertEqual(threes[0], DIAMOND_THREE)

    def test_cards_sort_by_rank_then_suit(self):
        self.assertLess(Card(3, 3), Card(4, 0))
        self.assertLess(Card(5, 0), Card(5, 1))

    def test_parse_card(self):
        self.assertEqual(parse_card("3"), Card(3, 0))
        self.assertEqual(parse_card("♠A"), Card(14, 3))
        self.assertEqual(parse_card("h10"), Card(10, 2))
        with self.assertRaises(ValueError):
            parse_card("2")  # 2 已经从牌堆里去掉了
        with self.assertRaises(ValueError):
            parse_card("")

    def test_hand_to_str(self):
        self.assertEqual(hand_to_str([Card(4, 0), Card(3, 1)]), "♣3 ♦4")


if __name__ == "__main__":
    unittest.main()
