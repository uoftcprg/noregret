from unittest import main, TestCase

import noregret as nr


class NormalFormGameTestCase(TestCase):
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

    def test_serialization(self):
        for game in self.GAMES:
            raw_game = game.dumps()
            game2 = type(game).loads(self.KERNEL, raw_game)
            raw_game2 = game2.dumps()

            self.assertEqual(raw_game, raw_game2)
            self.assertTrue((game.payoffs == game2.payoffs).all())
            self.assertEqual(game.actions, game2.actions)


class ExtensiveFormGameTestCase(TestCase):
    KERNEL = nr.FloatingPointKernel()
    GAMES = (
        nr.to_efg(nr.MatchingPennies(KERNEL)),
        nr.to_efg(nr.RockPaperScissors(KERNEL)),
        nr.to_efg(nr.RockPaperScissorsPlus(KERNEL)),
        nr.to_efg(nr.RockPaperSuperscissors(KERNEL)),
        nr.from_open_spiel(KERNEL, 'kuhn_poker'),
        nr.from_open_spiel(KERNEL, 'leduc_poker'),
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


if __name__ == '__main__':
    main()  # pragma: no cover
