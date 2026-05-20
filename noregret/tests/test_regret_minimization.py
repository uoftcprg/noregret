from functools import partial
from math import inf
from unittest import main, TestCase

import noregret as nr


class ProbabilitySimplexRegretMinimizationTestCase(TestCase):
    KERNEL = nr.FloatingPointKernel()
    SYMMETRIC_GAME_VALUES = (
        (nr.RockPaperScissors(KERNEL), 0),
        (nr.RockPaperScissorsPlus(KERNEL), 0),
        (nr.RockPaperSuperscissors(KERNEL), 0),
    )
    GAME_VALUES = (
        *SYMMETRIC_GAME_VALUES,
        (nr.MatchingPennies(KERNEL), 0),
    )
    REGRET_MINIMIZER_TYPES = (
        partial(nr.MWU, learning_rate=1e-3),
        partial(nr.ER, learning_rate=1e-3),
        partial(nr.OGD, learning_rate=1e-3),
        nr.RM,
        nr.RM_plus,
        nr.DRM,
    )
    ITERATION_COUNT = 1000000
    TARGET_EXPLOITABILITY = 1e-2
    DELTA = 2 * TARGET_EXPLOITABILITY

    def test_average_iterate_convergence(self):
        for game, value in self.GAME_VALUES:
            assert isinstance(game, nr.NFG_2p0s)

            for R_type in self.REGRET_MINIMIZER_TYPES:
                for alt in (True, False):
                    for pred in (True, False):
                        x_bar, y_bar = nr.regret_minimization(
                            game,
                            R_type(self.KERNEL, game.row_dimension),
                            R_type(self.KERNEL, game.column_dimension),
                            alternation=alt,
                            prediction=pred,
                            iteration_count=self.ITERATION_COUNT,
                            target_exploitability=self.TARGET_EXPLOITABILITY,
                            progress_bar=False,
                        )
                        e = game.exploitability(x_bar, y_bar)
                        v = game.expected_row_utility(x_bar, y_bar)

                        self.assertLess(e, self.TARGET_EXPLOITABILITY)
                        self.assertAlmostEqual(v, value, delta=self.DELTA)

    def test_last_iterate_convergence(self):
        for game, value in self.GAME_VALUES:
            assert isinstance(game, nr.NFG_2p0s)

            for R_type in self.REGRET_MINIMIZER_TYPES:
                for alt in (True, False):
                    x_bar, y_bar = nr.regret_minimization(
                        game,
                        R_type(self.KERNEL, game.row_dimension),
                        R_type(self.KERNEL, game.column_dimension),
                        alternation=alt,
                        prediction=True,
                        iteration_count=self.ITERATION_COUNT,
                        target_exploitability=self.TARGET_EXPLOITABILITY,
                        progress_bar=False,
                    )
                    e = game.exploitability(x_bar, y_bar)
                    v = game.expected_row_utility(x_bar, y_bar)

                    self.assertLess(e, self.TARGET_EXPLOITABILITY)
                    self.assertAlmostEqual(v, value, delta=self.DELTA)

    def test_frequent_iterate_convergence(self):
        for game, value in self.SYMMETRIC_GAME_VALUES:
            assert game.is_symmetric()
            assert isinstance(game, nr.NFG_2p0s)

            for R_type in self.REGRET_MINIMIZER_TYPES:
                R_type = partial(nr.BM, regret_minimizer_type=R_type)
                x_bar, y_bar = nr.symmetric_regret_minimization(
                    game,
                    R_type(self.KERNEL, game.row_dimension, gamma=inf),
                    iteration_count=self.ITERATION_COUNT,
                    target_exploitability=self.TARGET_EXPLOITABILITY,
                    progress_bar=False,
                )
                e = game.exploitability(x_bar, y_bar)
                v = game.expected_row_utility(x_bar, y_bar)

                self.assertLess(e, self.TARGET_EXPLOITABILITY)
                self.assertAlmostEqual(v, value, delta=self.DELTA)


class SequenceFormPolytopeRegretMinimizationTestCase(TestCase):
    KERNEL = nr.FloatingPointKernel()
    GAME_VALUES = (
        (nr.to_efg(nr.MatchingPennies(KERNEL)), 0),
        (nr.to_efg(nr.RockPaperScissors(KERNEL)), 0),
        (nr.to_efg(nr.RockPaperScissorsPlus(KERNEL)), 0),
        (nr.to_efg(nr.RockPaperSuperscissors(KERNEL)), 0),
        (nr.from_open_spiel(KERNEL, 'kuhn_poker'), -1 / 18),
        (nr.from_open_spiel(KERNEL, 'leduc_poker'), -0.08560642407800048),
    )
    REGRET_MINIMIZION_PARAMETERS = (
        (partial(nr.CFR, KERNEL), False, False),
        (partial(nr.CFR_plus, KERNEL), True, False),
        (partial(nr.DCFR, KERNEL), True, False),
        (partial(nr.CFR_plus, KERNEL, gamma=2), True, True),
        (partial(nr.CFR_plus, KERNEL, gamma=inf), True, True),
    )
    ITERATION_COUNT = 1000000
    TARGET_EXPLOITABILITY = 1e-2
    DELTA = 2 * TARGET_EXPLOITABILITY

    def test_convergence(self):
        for game, value in self.GAME_VALUES:
            assert isinstance(game, nr.EFG_2p0s)

            for (R_type, alt, pred) in self.REGRET_MINIMIZION_PARAMETERS:
                x_bar, y_bar = nr.regret_minimization(
                    game,
                    R_type(game.row_sequence_form_polytope),
                    R_type(game.column_sequence_form_polytope),
                    alternation=alt,
                    prediction=pred,
                    iteration_count=self.ITERATION_COUNT,
                    target_exploitability=self.TARGET_EXPLOITABILITY,
                    progress_bar=False,
                )
                e = game.exploitability(x_bar, y_bar)
                v = game.expected_row_utility(x_bar, y_bar)

                self.assertLess(e, self.TARGET_EXPLOITABILITY)
                self.assertAlmostEqual(v, value, delta=self.DELTA)


if __name__ == '__main__':
    main()  # pragma: no cover
