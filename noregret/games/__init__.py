"""Module for games."""
from noregret.games.extensive_form import (
    ExtensiveFormGame,
    TwoPlayerExtensiveFormGame,
    TwoPlayerZeroSumExtensiveFormGame,
)
from noregret.games.games import Game, TwoPlayerGame, TwoPlayerZeroSumGame
from noregret.games.multilinear import (
    MultilinearGame,
    TwoPlayerMultilinearGame,
    TwoPlayerZeroSumMultilinearGame,
)
from noregret.games.normal_form import (
    AssuranceGame,
    BattleOfTheSexes,
    Chicken,
    GiftExchangeGame,
    MatchingPennies,
    NormalFormGame,
    PrisonersDilemma,
    PureCoordination,
    RockPaperScissors,
    RockPaperScissorsPlus,
    RockPaperSuperscissors,
    StagHunt,
    TwoPlayerNormalFormGame,
    TwoPlayerZeroSumNormalFormGame,
)
from noregret.games.utilities import from_open_spiel, to_extensive_form

__all__ = (
    'AssuranceGame',
    'BattleOfTheSexes',
    'Chicken',
    'ExtensiveFormGame',
    'from_open_spiel',
    'Game',
    'GiftExchangeGame',
    'MatchingPennies',
    'MultilinearGame',
    'NormalFormGame',
    'PrisonersDilemma',
    'PureCoordination',
    'RockPaperScissors',
    'RockPaperScissorsPlus',
    'RockPaperSuperscissors',
    'StagHunt',
    'to_extensive_form',
    'TwoPlayerExtensiveFormGame',
    'TwoPlayerGame',
    'TwoPlayerMultilinearGame',
    'TwoPlayerNormalFormGame',
    'TwoPlayerZeroSumExtensiveFormGame',
    'TwoPlayerZeroSumGame',
    'TwoPlayerZeroSumMultilinearGame',
    'TwoPlayerZeroSumNormalFormGame',
)
