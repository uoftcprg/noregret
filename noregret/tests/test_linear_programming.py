from unittest import main, TestCase

import noregret as nr


class LinearProgrammingTestCase(TestCase):
    KERNEL = nr.FloatingPointKernel()
    GAME_VALUES = (
        (nr.MatchingPennies(KERNEL), 0),
        (nr.RockPaperScissors(KERNEL), 0),
        (nr.RockPaperScissorsPlus(KERNEL), 0),
        (nr.RockPaperSuperscissors(KERNEL), 0),
        (nr.to_efg(nr.MatchingPennies(KERNEL)), 0),
        (nr.to_efg(nr.RockPaperScissors(KERNEL)), 0),
        (nr.to_efg(nr.RockPaperScissorsPlus(KERNEL)), 0),
        (nr.to_efg(nr.RockPaperSuperscissors(KERNEL)), 0),
        (nr.from_open_spiel(KERNEL, 'kuhn_poker'), -1 / 18),
        (nr.from_open_spiel(KERNEL, 'leduc_poker'), -0.08560642407800048),
    )

    def test_linear_programming(self):
        dtype = self.KERNEL.data_type

        for game, value in self.GAME_VALUES:
            x, y = nr.linear_programming(game)
            e = game.exploitability(x, y)
            v = game.expected_row_utility(x, y)

            self.assertAlmostEqual(e, 0)
            self.assertAlmostEqual(v, value)
            self.assertEqual(e.dtype, dtype)
            self.assertEqual(v.dtype, dtype)


if __name__ == '__main__':
    main()  # pragma: no cover
