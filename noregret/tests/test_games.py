from abc import ABC, abstractmethod
from unittest import main, TestCase

import noregret as nr


class GameTestCaseMixin(ABC):
    @abstractmethod
    def uniform_strategy_profile(self, game):
        pass

    def test_equivalence(self):
        np = self.KERNEL.numpy

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
    KERNEL = nr.FloatingPointKernel()
    GAMES = (
        nr.AssuranceGame(KERNEL),
        nr.BattleOfTheSexes(KERNEL),
        nr.Chicken(KERNEL),
        nr.GiftExchangeGame(KERNEL),
        nr.MatchingPennies(KERNEL),
        nr.PrisonersDilemma(KERNEL),
        nr.PureCoordination(KERNEL),
        nr.RockPaperScissors(KERNEL),
        nr.RockPaperScissorsPlus(KERNEL),
        nr.RockPaperSuperscissors(KERNEL),
        nr.StagHunt(KERNEL),
    )

    def uniform_strategy_profile(self, game):
        np = self.KERNEL.numpy
        dtype = self.KERNEL.data_type

        for n in game.dimensions:
            yield np.full(n, 1 / n, dtype)

    def test_best_response_value(self):
        np = self.KERNEL.numpy

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
            game2 = type(game).loads(self.KERNEL, raw_game)
            raw_game2 = game2.dumps()

            self.assertEqual(raw_game, raw_game2)
            self.assertTrue((game.payoffs == game2.payoffs).all())
            self.assertEqual(game.actions, game2.actions)

    def test_from_matrix(self):
        np = self.KERNEL.numpy
        dtype = self.KERNEL.data_type
        A = np.array([[3, 0, -3], [0, 3, -4], [0, 0, 1]], dtype)
        game = nr.from_matrix(self.KERNEL, A)
        x, y = nr.linear_programming(game)
        v = game.expected_row_utility(x, y)

        self.assertAlmostEqual(v, 0.25)


class ExtensiveFormGameTestCase(GameTestCaseMixin, TestCase):
    KERNEL = nr.FloatingPointKernel()
    GAMES = (
        nr.to_efg(nr.MatchingPennies(KERNEL)),
        nr.to_efg(nr.RockPaperScissors(KERNEL)),
        nr.to_efg(nr.RockPaperScissorsPlus(KERNEL)),
        nr.to_efg(nr.RockPaperSuperscissors(KERNEL)),
        nr.to_efg(KERNEL, nr.from_open_spiel('kuhn_poker')),
        nr.to_efg(KERNEL, nr.from_open_spiel('leduc_poker')),
    )

    def uniform_strategy_profile(self, game):
        for sfp in game.sequence_form_polytopes:
            yield sfp.to_sequence_form(sfp.behavioral_form_uniform_strategy)

    def test_best_response_value(self):
        np = self.KERNEL.numpy

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
            game2 = type(game).loads(self.KERNEL, raw_game)
            raw_game2 = game2.dumps()

            self.assertEqual(raw_game, raw_game2)
            self.assertFalse((game.payoffs != game2.payoffs).count_nonzero())

            for sfp, sfp2 in zip(
                    game.sequence_form_polytopes,
                    game2.sequence_form_polytopes,
            ):
                self.assertEqual(sfp.actions, sfp2.actions)
                self.assertEqual(sfp.parent_sequences, sfp2.parent_sequences)


class BlackBoxGameTestCase(TestCase):
    GAMES = nr.from_open_spiel('kuhn_poker'), nr.from_open_spiel('leduc_poker')

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
        for game in self.GAMES:
            h = game.root_node
            us = game.utilities(h)
            us2 = nr.BlackBoxGame.utilities(game, h)

            self.assertEqual(us, us2)

    def test_chance_probabilities(self):
        for game in self.GAMES:
            h = game.root_node
            ps = game.chance_probabilities(h)
            ps2 = nr.BlackBoxGame.chance_probabilities(game, h)

            self.assertEqual(ps, ps2)


if __name__ == '__main__':
    main()  # pragma: no cover
