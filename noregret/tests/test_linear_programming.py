from unittest import main, TestCase

import noregret as nr


class LinearProgrammingTestCase(TestCase):
    KER = nr.FPKer()
    GAME_VALUES = (
        (nr.MatchingPennies(KER), 0),
        (nr.RockPaperScissors(KER), 0),
        (nr.RockPaperScissorsPlus(KER), 0),
        (nr.RockPaperSuperscissors(KER), 0),
        (nr.to_efg(KER, nr.MatchingPennies(KER)), 0),
        (nr.to_efg(KER, nr.RockPaperScissors(KER)), 0),
        (nr.to_efg(KER, nr.RockPaperScissorsPlus(KER)), 0),
        (nr.to_efg(KER, nr.RockPaperSuperscissors(KER)), 0),
        (nr.to_efg(KER, nr.OpenSpielGame(KER, 'kuhn_poker')), -1 / 18),
        (nr.to_efg(KER, nr.OpenSpielGame(KER, 'leduc_poker')), -0.08560642418),
    )

    def test_linear_programming(self):
        dtype = self.KER.data_type

        for game, value in self.GAME_VALUES:
            x, y = nr.lp(game)
            epsilon = game.exploitability(x, y)
            v = game.expected_row_utility(x, y)

            self.assertAlmostEqual(epsilon, 0)
            self.assertAlmostEqual(v, value)
            self.assertEqual(epsilon.dtype, dtype)
            self.assertEqual(v.dtype, dtype)


if __name__ == '__main__':
    main()  # pragma: no cover
