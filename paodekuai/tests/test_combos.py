import random
import unittest

from paodekuai import combos as C
from paodekuai.cards import deal, parse_card


def cards(*names):
    return [parse_card(name) for name in names]


class TestClassify(unittest.TestCase):
    def kind_of(self, *names):
        combo = C.classify(cards(*names))
        return combo.kind if combo else None

    def test_basic_kinds(self):
        self.assertEqual(self.kind_of("3"), C.SINGLE)
        self.assertEqual(self.kind_of("3", "c3"), C.PAIR)
        self.assertEqual(self.kind_of("3", "c3", "h3"), C.TRIPLE)
        self.assertEqual(self.kind_of("3", "c3", "h3", "s3"), C.BOMB)

    def test_triple_with_attachments(self):
        self.assertEqual(self.kind_of("3", "c3", "h3", "4"), C.TRIPLE_ONE)
        self.assertEqual(self.kind_of("3", "c3", "h3", "4", "c4"), C.TRIPLE_TWO)
        self.assertIsNone(self.kind_of("3", "c3", "h3", "4", "5"))

    def test_straight(self):
        self.assertEqual(self.kind_of("3", "4", "5", "6", "7"), C.STRAIGHT)
        self.assertEqual(self.kind_of("10", "J", "Q", "K", "A"), C.STRAIGHT)
        self.assertIsNone(self.kind_of("3", "4", "5", "6"))       # 不足 5 张
        self.assertIsNone(self.kind_of("3", "4", "5", "6", "8"))  # 断开

    def test_straight_length_is_card_count(self):
        combo = C.classify(cards("3", "4", "5", "6", "7", "8"))
        self.assertEqual(combo.length, 6)
        self.assertEqual(combo.rank, 8)  # 用最大的一张比大小

    def test_pair_straight(self):
        self.assertEqual(self.kind_of("3", "c3", "4", "c4"), C.PAIR_STRAIGHT)
        self.assertEqual(self.kind_of("3", "c3", "4", "c4", "5", "c5"), C.PAIR_STRAIGHT)
        self.assertIsNone(self.kind_of("3", "c3", "5", "c5"))  # 不连续

    def test_plane(self):
        self.assertEqual(self.kind_of("3", "c3", "h3", "4", "c4", "h4"), C.PLANE)
        self.assertEqual(self.kind_of("3", "c3", "h3", "4", "c4", "h4", "5", "6"), C.PLANE_ONE)
        self.assertEqual(
            self.kind_of("3", "c3", "h3", "4", "c4", "h4", "5", "c5", "6", "c6"), C.PLANE_TWO
        )

    def test_junk_is_not_a_combo(self):
        self.assertIsNone(self.kind_of("3", "5"))
        self.assertIsNone(self.kind_of("3", "c3", "h3", "4", "c4", "h4", "5"))
        self.assertIsNone(C.classify([]))

    def test_duplicate_cards_rejected(self):
        self.assertIsNone(C.classify(cards("3", "3")))


class TestBeats(unittest.TestCase):
    def combo(self, *names):
        return C.classify(cards(*names))

    def test_higher_rank_wins(self):
        self.assertTrue(C.beats(self.combo("7"), self.combo("5")))
        self.assertFalse(C.beats(self.combo("5"), self.combo("7")))
        self.assertFalse(C.beats(self.combo("5"), self.combo("5")))

    def test_anything_beats_nothing(self):
        self.assertTrue(C.beats(self.combo("3"), None))

    def test_different_kinds_do_not_compare(self):
        self.assertFalse(C.beats(self.combo("A", "cA"), self.combo("5")))
        self.assertFalse(C.beats(self.combo("A"), self.combo("5", "c5")))

    def test_straight_length_must_match(self):
        long_straight = self.combo("3", "4", "5", "6", "7", "8")
        short_straight = self.combo("9", "10", "J", "Q", "K")
        self.assertFalse(C.beats(long_straight, short_straight))
        self.assertFalse(C.beats(short_straight, long_straight))

    def test_bomb_beats_everything_else(self):
        bomb = self.combo("3", "c3", "h3", "s3")
        self.assertTrue(C.beats(bomb, self.combo("A")))
        self.assertTrue(C.beats(bomb, self.combo("10", "J", "Q", "K", "A")))
        self.assertFalse(C.beats(self.combo("A"), bomb))

    def test_bombs_compare_by_rank(self):
        low = self.combo("3", "c3", "h3", "s3")
        high = self.combo("A", "cA", "hA", "sA")
        self.assertTrue(C.beats(high, low))
        self.assertFalse(C.beats(low, high))


class TestLegalMoves(unittest.TestCase):
    def test_generated_moves_round_trip_through_classify(self):
        rng = random.Random(0)
        for _ in range(150):
            for hand in deal(rng):
                for move in C.legal_moves(hand, None):
                    self.assertLessEqual(set(move.cards), set(hand))
                    self.assertEqual(len(set(move.cards)), len(move.cards))
                    got = C.classify(move.cards)
                    self.assertIsNotNone(got, f"生成了识别不了的牌 {move}")
                    self.assertEqual((got.kind, got.rank, got.length), (move.kind, move.rank, move.length))

    def test_following_moves_all_beat_the_requirement(self):
        rng = random.Random(1)
        for _ in range(80):
            for hand in deal(rng):
                free = C.legal_moves(hand, None)
                required = rng.choice(free)
                for move in C.legal_moves(hand, required):
                    self.assertTrue(C.beats(move, required), f"{move} 压不过 {required}")

    def test_bombs_are_always_available_as_a_response(self):
        hand = cards("9", "c9", "h9", "s9", "3")
        moves = C.legal_moves(hand, C.classify(cards("A")))
        self.assertTrue(any(move.kind == C.BOMB for move in moves))

    def test_no_moves_when_nothing_beats_it(self):
        hand = cards("3", "4", "5")
        self.assertEqual(C.legal_moves(hand, C.classify(cards("A"))), [])

    def test_only_bigger_bombs_beat_a_bomb(self):
        hand = cards("3", "c3", "h3", "s3", "K", "cK", "hK", "sK")
        moves = C.legal_moves(hand, C.classify(cards("9", "c9", "h9", "s9")))
        self.assertEqual([m.rank for m in moves], [13])


class TestEstimateTurns(unittest.TestCase):
    def test_empty_hand_needs_no_turns(self):
        self.assertEqual(C.estimate_turns([]), 0)

    def test_single_card_is_one_turn(self):
        self.assertEqual(C.estimate_turns(cards("5")), 1)

    def test_straight_is_one_turn(self):
        self.assertEqual(C.estimate_turns(cards("3", "4", "5", "6", "7")), 1)

    def test_bomb_is_one_turn(self):
        self.assertEqual(C.estimate_turns(cards("8", "c8", "h8", "s8")), 1)

    def test_scattered_cards_need_more_turns(self):
        scattered = C.estimate_turns(cards("3", "5", "7", "9", "J"))
        self.assertEqual(scattered, 5)
        self.assertGreater(scattered, C.estimate_turns(cards("3", "4", "5", "6", "7")))

    def test_triple_absorbs_a_single(self):
        # 三张 + 一张散牌 = 三带一，一轮就能走
        self.assertEqual(C.estimate_turns(cards("3", "c3", "h3", "9")), 1)


if __name__ == "__main__":
    unittest.main()
