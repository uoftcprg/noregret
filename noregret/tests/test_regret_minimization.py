from functools import partial
from math import inf
from unittest import main, TestCase

import noregret as nr


class ProbabilitySimplexRegretMinimizationTestCase(TestCase):
    KER = nr.FPKer('single')
    SYMMETRIC_GAME_VALUES = (
        (nr.RockPaperScissors(KER), 0),
        (nr.RockPaperScissorsPlus(KER), 0),
        (nr.RockPaperSuperscissors(KER), 0),
    )
    GAME_VALUES = *SYMMETRIC_GAME_VALUES, (nr.MatchingPennies(KER), 0)
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
        dtype = self.KER.data_type

        for game, value in self.GAME_VALUES:
            assert isinstance(game, nr.NFG_2p0s)

            for R_type in self.REGRET_MINIMIZER_TYPES:
                for alt in (True, False):
                    for pred in (True, False):
                        x_bar, y_bar = nr.rm(
                            game,
                            R_type(self.KER, game.row_dimension),
                            R_type(self.KER, game.column_dimension),
                            alternation=alt,
                            prediction=pred,
                            iteration_count=self.ITERATION_COUNT,
                            target_exploitability=self.TARGET_EXPLOITABILITY,
                            progress_bar=False,
                        )
                        epsilon = game.exploitability(x_bar, y_bar)
                        v = game.expected_row_utility(x_bar, y_bar)

                        self.assertLess(epsilon, self.TARGET_EXPLOITABILITY)
                        self.assertAlmostEqual(v, value, delta=self.DELTA)
                        self.assertEqual(epsilon.dtype, dtype)
                        self.assertEqual(v.dtype, dtype)

    def test_last_iterate_convergence(self):
        dtype = self.KER.data_type

        for game, value in self.GAME_VALUES:
            assert isinstance(game, nr.NFG_2p0s)

            for R_type in self.REGRET_MINIMIZER_TYPES:
                for alt in (True, False):
                    x_bar, y_bar = nr.rm(
                        game,
                        R_type(self.KER, game.row_dimension),
                        R_type(self.KER, game.column_dimension),
                        alternation=alt,
                        prediction=True,
                        iteration_count=self.ITERATION_COUNT,
                        target_exploitability=self.TARGET_EXPLOITABILITY,
                        progress_bar=False,
                    )
                    epsilon = game.exploitability(x_bar, y_bar)
                    v = game.expected_row_utility(x_bar, y_bar)

                    self.assertLess(epsilon, self.TARGET_EXPLOITABILITY)
                    self.assertAlmostEqual(v, value, delta=self.DELTA)
                    self.assertEqual(epsilon.dtype, dtype)
                    self.assertEqual(v.dtype, dtype)

    def test_frequent_iterate_convergence(self):
        dtype = self.KER.data_type

        for game, value in self.SYMMETRIC_GAME_VALUES:
            assert game.is_symmetric
            assert isinstance(game, nr.NFG_2p0s)

            for R_type in self.REGRET_MINIMIZER_TYPES:
                R_type = partial(nr.BM, regret_minimizer_type=R_type)
                x_bar, y_bar = nr.symmetric_rm(
                    game,
                    R_type(self.KER, game.row_dimension, gamma=inf),
                    iteration_count=self.ITERATION_COUNT,
                    target_exploitability=self.TARGET_EXPLOITABILITY,
                    progress_bar=False,
                )
                epsilon = game.exploitability(x_bar, y_bar)
                v = game.expected_row_utility(x_bar, y_bar)

                self.assertLess(epsilon, self.TARGET_EXPLOITABILITY)
                self.assertAlmostEqual(v, value, delta=self.DELTA)
                self.assertEqual(epsilon.dtype, dtype)
                self.assertEqual(v.dtype, dtype)


class SequenceFormPolytopeRegretMinimizationTestCase(TestCase):
    KER = nr.FPKer('single')
    GAME_VALUES = (
        (nr.to_efg(KER, nr.MatchingPennies(KER)), 0),
        (nr.to_efg(KER, nr.RockPaperScissors(KER)), 0),
        (nr.to_efg(KER, nr.RockPaperScissorsPlus(KER)), 0),
        (nr.to_efg(KER, nr.RockPaperSuperscissors(KER)), 0),
        (nr.to_efg(KER, nr.OpenSpielGame(KER, 'kuhn_poker')), -1 / 18),
        (nr.to_efg(KER, nr.OpenSpielGame(KER, 'leduc_poker')), -0.08560642418),
    )
    REGRET_MINIMIZION_PARAMETERS = (
        (partial(nr.CFR, KER), False, False),
        (partial(nr.CFR_plus, KER), True, False),
        (partial(nr.DCFR, KER), True, False),
        (partial(nr.CFR_plus, KER, gamma=2), True, True),
        (partial(nr.CFR_plus, KER, gamma=inf), True, True),
    )
    ITERATION_COUNT = 1000000
    TARGET_EXPLOITABILITY = 1e-2
    DELTA = 2 * TARGET_EXPLOITABILITY

    def test_convergence(self):
        dtype = self.KER.data_type

        for game, value in self.GAME_VALUES:
            assert isinstance(game, nr.EFG_2p0s)

            for (R_type, alt, pred) in self.REGRET_MINIMIZION_PARAMETERS:
                x_bar, y_bar = nr.rm(
                    game,
                    R_type(game.row_sequence_form_polytope),
                    R_type(game.column_sequence_form_polytope),
                    alternation=alt,
                    prediction=pred,
                    iteration_count=self.ITERATION_COUNT,
                    target_exploitability=self.TARGET_EXPLOITABILITY,
                    progress_bar=False,
                )
                epsilon = game.exploitability(x_bar, y_bar)
                v = game.expected_row_utility(x_bar, y_bar)

                self.assertLess(epsilon, self.TARGET_EXPLOITABILITY)
                self.assertAlmostEqual(v, value, delta=self.DELTA)
                self.assertEqual(epsilon.dtype, dtype)
                self.assertEqual(v.dtype, dtype)


class SequenceFormPolytopeRegretMinimization2TestCase(TestCase):
    KER = nr.FPKer()
    GAMES = (
        nr.to_efg(KER, nr.MatchingPennies(KER)),
        nr.to_efg(KER, nr.RockPaperScissors(KER)),
        nr.to_efg(KER, nr.RockPaperScissorsPlus(KER)),
        nr.to_efg(KER, nr.RockPaperSuperscissors(KER)),
        nr.to_efg(KER, nr.OpenSpielGame(KER, 'kuhn_poker')),
        nr.to_efg(KER, nr.OpenSpielGame(KER, 'leduc_poker')),
    )
    PLACES = 2

    def test_equivalence(self):
        for game in self.GAMES:
            assert isinstance(game, nr.EFG_2p0s)

            x_bar, y_bar = nr.rm(
                game,
                nr.CFR(self.KER, game.row_sequence_form_polytope),
                nr.CFR(self.KER, game.column_sequence_form_polytope),
                progress_bar=False,
            )
            epsilon = game.exploitability(x_bar, y_bar)
            v = game.expected_row_utility(x_bar, y_bar)
            x_bar2, y_bar2 = nr.rm(
                game,
                nr.CFR2(self.KER, game.row_sequence_form_polytope),
                nr.CFR2(self.KER, game.column_sequence_form_polytope),
                progress_bar=False,
            )
            epsilon2 = game.exploitability(x_bar2, y_bar2)
            v2 = game.expected_row_utility(x_bar2, y_bar2)

            self.assertAlmostEqual(epsilon, epsilon2, self.PLACES)
            self.assertAlmostEqual(v, v2, self.PLACES)

            x_bar, y_bar = nr.rm(
                game,
                nr.CFR(self.KER, game.row_sequence_form_polytope),
                nr.CFR(self.KER, game.column_sequence_form_polytope),
                prediction=True,
                progress_bar=False,
            )
            epsilon = game.exploitability(x_bar, y_bar)
            v = game.expected_row_utility(x_bar, y_bar)
            x_bar2, y_bar2 = nr.rm(
                game,
                nr.CFR2(self.KER, game.row_sequence_form_polytope),
                nr.CFR2(self.KER, game.column_sequence_form_polytope),
                prediction=True,
                progress_bar=False,
            )
            epsilon2 = game.exploitability(x_bar2, y_bar2)
            v2 = game.expected_row_utility(x_bar2, y_bar2)

            self.assertAlmostEqual(epsilon, epsilon2, self.PLACES)
            self.assertAlmostEqual(v, v2, self.PLACES)


class StochasticRegretMinimizationTestCase(TestCase):
    KER = nr.FPKer()
    GAME = nr.OpenSpielGame(KER, 'kuhn_poker')
    SAMPLE_COUNT = 100000
    TARGET_EXPLOITABILITY = 1e-1
    SEED = 42

    def test_external_sampling(self):
        np = self.KER.numpy

        assert self.GAME.is_two_player and self.GAME.is_zero_sum

        np.random.seed(self.SEED)

        R = nr.MCCFR(self.KER, self.GAME)
        sigma = nr.stochastic_rm(
            self.GAME,
            R,
            alternation=True,
            sample_count=self.SAMPLE_COUNT,
            progress_bar=False,
        )
        epsilon = self.GAME.exploitability(sigma)

        self.assertLess(epsilon, self.TARGET_EXPLOITABILITY)

    def test_outcome_sampling(self):
        np = self.KER.numpy

        assert self.GAME.is_two_player and self.GAME.is_zero_sum

        np.random.seed(self.SEED)

        R = nr.MCCFR(
            self.KER,
            self.GAME,
            reference_strategy_profile=nr.UniformStrategyProfile(
                self.KER,
                self.GAME,
            ),
        )
        sigma = nr.stochastic_rm(
            self.GAME,
            R,
            alternation=True,
            sample_count=self.SAMPLE_COUNT,
            progress_bar=False,
        )
        epsilon = self.GAME.exploitability(sigma)

        self.assertLess(epsilon, self.TARGET_EXPLOITABILITY)


if __name__ == '__main__':
    main()  # pragma: no cover
