from abc import ABC, abstractmethod
from unittest import main, TestCase

import noregret as nr


class GameTestCaseMixin(ABC):
    KER = None
    GAMES = None

    @abstractmethod
    def uniform_strategy_profile(self, game):
        pass

    def test_equivalence(self):
        np = self.KER.numpy

        for game in self.GAMES:
            x, y = self.uniform_strategy_profile(game)

            self.assertEqual(
                nr.MultilinearGame.dimensions.fget(game),
                game.dimensions,
            )
            np.testing.assert_allclose(
                nr.Game.utilities(game, x, y),
                game.utilities(x, y),
            )
            np.testing.assert_allclose(
                nr.Game.expected_utilities(game, x, y),
                game.expected_utilities(x, y),
            )
            np.testing.assert_allclose(
                nr.Game.best_response_values(game, x, y),
                game.best_response_values(x, y),
            )
            np.testing.assert_allclose(
                nr.Game.nash_gap(game, x, y),
                game.nash_gap(x, y),
            )


class NormalFormGameTestCase(GameTestCaseMixin, TestCase):
    KER = nr.FPKer()
    GAMES = (
        nr.AssuranceGame(KER),
        nr.BattleOfTheSexes(KER),
        nr.Chicken(KER),
        nr.GiftExchangeGame(KER),
        nr.MatchingPennies(KER),
        nr.PrisonersDilemma(KER),
        nr.PureCoordination(KER),
        nr.RockPaperScissors(KER),
        nr.RockPaperScissorsPlus(KER),
        nr.RockPaperSuperscissors(KER),
        nr.StagHunt(KER),
    )

    def uniform_strategy_profile(self, game):
        np = self.KER.numpy
        dtype = self.KER.data_type

        for n in game.dimensions:
            yield np.full(n, 1 / n, dtype)

    def test_best_response_value(self):
        np = self.KER.numpy

        for game in self.GAMES:
            x, y = self.uniform_strategy_profile(game)

            np.testing.assert_allclose(
                nr.NFG.best_response_value(game, 0, y),
                game.best_response_value(0, y),
            )
            np.testing.assert_allclose(
                nr.NFG.best_response_value(game, 1, x),
                game.best_response_value(1, x),
            )

    def test_serialization(self):
        for game in self.GAMES:
            raw_game = game.dumps()
            game2 = type(game).loads(self.KER, raw_game)
            raw_game2 = game2.dumps()

            self.assertEqual(raw_game, raw_game2)
            self.assertTrue((game.payoffs == game2.payoffs).all())
            self.assertEqual(game.actions, game2.actions)

    def test_matrix_game(self):
        np = self.KER.numpy
        dtype = self.KER.data_type
        A = np.array([[3, 0, -3], [0, 3, -4], [0, 0, 1]], dtype)
        game = nr.matrix_game(self.KER, A)
        x, y = nr.lp(game)
        v = game.expected_row_utility(x, y)

        self.assertAlmostEqual(v, 0.25)


class ExtensiveFormGameTestCase(GameTestCaseMixin, TestCase):
    KER = nr.FPKer()
    GAMES = (
        nr.to_efg(KER, nr.MatchingPennies(KER)),
        nr.to_efg(KER, nr.RockPaperScissors(KER)),
        nr.to_efg(KER, nr.RockPaperScissorsPlus(KER)),
        nr.to_efg(KER, nr.RockPaperSuperscissors(KER)),
        nr.to_efg(KER, nr.open_spiel_game(KER, 'kuhn_poker')),
        nr.to_efg(KER, nr.open_spiel_game(KER, 'leduc_poker')),
    )

    def uniform_strategy_profile(self, game):
        for sfp in game.sequence_form_polytopes:
            yield sfp.to_sequence_form(sfp.behavioral_form_uniform_strategy)

    def test_best_response_value(self):
        np = self.KER.numpy

        for game in self.GAMES:
            x, y = self.uniform_strategy_profile(game)

            np.testing.assert_allclose(
                nr.EFG.best_response_value(game, 0, y),
                game.best_response_value(0, y),
            )
            np.testing.assert_allclose(
                nr.EFG.best_response_value(game, 1, x),
                game.best_response_value(1, x),
            )

    def test_serialization(self):
        for game in self.GAMES:
            raw_game = game.dumps()
            game2 = type(game).loads(self.KER, raw_game)
            raw_game2 = game2.dumps()

            self.assertEqual(raw_game, raw_game2)
            self.assertFalse((game.payoffs != game2.payoffs).count_nonzero())

            for sfp, sfp2 in zip(
                    game.sequence_form_polytopes,
                    game2.sequence_form_polytopes,
            ):
                self.assertEqual(sfp.actions, sfp2.actions)
                self.assertEqual(sfp.parent_sequences, sfp2.parent_sequences)


class SimulationTestCase(TestCase):
    KER = nr.FPKer()

    def test_sequences(self):
        np = self.KER.numpy
        dtype = self.KER.data_type
        sim = nr.Sim(
            self.KER,
            (0, None, 0, 1),
            ('', None, 'ab', 'b'),
            ('a', 'b', 'c', 'd'),
            np.array([1, -1], dtype),
        )

        self.assertEqual(
            tuple(sim.sequences()),
            (('', 'a'), ('ab', 'c'), ('b', 'd')),
        )
        self.assertEqual(tuple(sim.sequences(0)), (('', 'a'), ('ab', 'c')))
        self.assertEqual(tuple(sim.sequences(1)), (('b', 'd'),))

    def test_utility(self):
        np = self.KER.numpy
        dtype = self.KER.data_type
        sim = nr.Sim(self.KER, (), (), (), np.array([1, -1], dtype))

        self.assertEqual(sim.utility(0), 1)
        self.assertEqual(sim.utility(1), -1)


class BlackBoxGameTestCase(TestCase):
    KER = nr.FPKer()
    GAMES = (
        nr.open_spiel_game(KER, 'kuhn_poker'),
        nr.open_spiel_game(KER, 'leduc_poker'),
    )
    SEED = 42

    def test_actions_and_children(self):
        for game in self.GAMES:
            h = game.root_node
            A = game.actions(h)
            children = list(map(str, game.children(h)))
            A_children = game.actions_and_children(h)

            self.assertIsInstance(A_children, tuple)
            self.assertEqual(len(A_children), 2)

            A_children = A_children[0], list(map(str, A_children[1]))

            self.assertEqual((A, children), A_children)

            children2 = list(map(str, nr.BlackBoxGame.children(game, h)))
            A_children2 = nr.BlackBoxGame.actions_and_children(game, h)

            self.assertIsInstance(A_children, tuple)
            self.assertEqual(len(A_children), 2)

            A_children2 = A_children2[0], list(map(str, A_children2[1]))

            self.assertEqual((A, children2), A_children2)
            self.assertEqual(A_children, A_children2)

    def test_utilities(self):
        np = self.KER.numpy

        for game in self.GAMES:
            h = game.root_node
            us = game.utilities(h)
            us2 = nr.BlackBoxGame.utilities(game, h)

            np.testing.assert_equal(us, us2)

    def test_chance_probabilities(self):
        np = self.KER.numpy

        for game in self.GAMES:
            h = game.root_node
            ps = game.chance_probabilities(h)
            ps2 = nr.BlackBoxGame.chance_probabilities(game, h)

            np.testing.assert_equal(ps, ps2)

    def test_exploitability(self):
        for game in self.GAMES:
            sigma = nr.UniformStrategyProfile(self.KER, game)
            epsilon = game.exploitability(sigma)

            sigma = nr.TupleStrategyProfile(
                self.KER,
                game,
                (
                    nr.UniformStrategyProfile(self.KER, game),
                    nr.UniformStrategyProfile(self.KER, game),
                ),
            )
            epsilon2 = game.exploitability(sigma)

            self.assertAlmostEqual(epsilon, epsilon2)

            game2 = nr.to_efg(self.KER, game)
            sfps = game2.sequence_form_polytopes
            bs = [sfp.behavioral_form_uniform_strategy for sfp in sfps]
            sigma = [sfp.to_sequence_form(b) for sfp, b in zip(sfps, bs)]
            epsilon2 = game2.exploitability(*sigma)

            self.assertAlmostEqual(epsilon, epsilon2)

            seqs = [sfp.non_empty_sequences for sfp in sfps]
            sigma = nr.SequenceFormStrategyProfile(self.KER, game, seqs, sigma)
            epsilon2 = game.exploitability(sigma)

            self.assertAlmostEqual(epsilon, epsilon2)

    def test_simulation(self):
        np = self.KER.numpy

        for game in self.GAMES:
            np.random.seed(self.SEED)

            sigma = nr.UniformStrategyProfile(self.KER, game)

            game.simulate(sigma)


if __name__ == '__main__':
    main()  # pragma: no cover
