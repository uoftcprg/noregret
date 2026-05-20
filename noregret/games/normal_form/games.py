"""Module for normal-form games (NFGs)."""
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from ordered_set import OrderedSet
from orjson import dumps, loads, OPT_SERIALIZE_NUMPY

from noregret.games.multilinear import (
    MultilinearGame,
    TwoPlayerMultilinearGame,
    TwoPlayerZeroSumMultilinearGame,
)
from noregret.kernels import Serializable


@dataclass
class NormalFormGame(MultilinearGame, Serializable):
    """Class for normal-form games (NFGs).

    Every player optimizes over a probability simplex.
    """
    actions: tuple[OrderedSet[str], ...]
    """Actions."""

    @property
    def player_count(self):
        return len(self.actions)

    @property
    def dimensions(self):
        """Return the dimensions.

        :return: Dimensions.
        """
        return tuple(map(len, self.actions))

    def best_response_value(self, player, *strategies):
        return self.utility(player, *strategies).max()

    @classmethod
    def loads(cls, kernel, raw_data):
        data = loads(raw_data)
        np = kernel.numpy
        payoffs = np.ascontiguousarray(np.array(data['payoffs']))
        actions = tuple(map(OrderedSet, data['actions']))

        return cls(kernel, payoffs, actions)

    def dumps(self):
        data = {'payoffs': self.payoffs, 'actions': self.actions}

        return dumps(data, list, OPT_SERIALIZE_NUMPY)


@dataclass
class TwoPlayerNormalFormGame(TwoPlayerMultilinearGame, NormalFormGame):
    """Class for two-player (2p) normal-form games (NFGs)."""

    @property
    def row_actions(self):
        """Return the row actions.

        :return: Row actions.
        """
        return self.actions[0]

    @property
    def column_actions(self):
        """Return the column actions.

        :return: Column actions.
        """
        return self.actions[1]

    @property
    def row_dimension(self):
        """Return the dimension for the row player.

        :return: Dimension for the row player.
        """
        return len(self.row_actions)

    @property
    def column_dimension(self):
        """Return the dimension for the column player.

        :return: Dimension for the column player.
        """
        return len(self.column_actions)

    def row_best_response_value(self, player, column_strategy):
        return self.row_utility(column_strategy).max()

    def column_best_response_value(self, player, row_strategy):
        return self.column_utility(row_strategy).max()


@dataclass
class TwoPlayerZeroSumNormalFormGame(
        TwoPlayerZeroSumMultilinearGame,
        TwoPlayerNormalFormGame,
):
    """Class for two-player zero-sum (2p0s) normal-form games (NFGs)."""

    def _best_response_row_values(self, row_strategy, column_strategy):
        u, neg_v = self._row_utilities(row_strategy, column_strategy)
        u = u.max()
        neg_v = neg_v.min()

        return u, neg_v


def _2p_nfg(name, kernel):
    with open(Path(__file__).parent / f'{name}.json', 'rb') as file:
        return TwoPlayerNormalFormGame.loads(kernel, file.read())


def _2p0s_nfg(name, kernel):
    with open(Path(__file__).parent / f'{name}.json', 'rb') as file:
        return TwoPlayerZeroSumNormalFormGame.loads(kernel, file.read())


AssuranceGame = partial(_2p_nfg, 'assurance-game')
"""Assurance game."""
BattleOfTheSexes = partial(_2p_nfg, 'battle-of-the-sexes')
"""Battle of the sexes."""
Chicken = partial(_2p_nfg, 'chicken')
"""Chicken."""
GiftExchangeGame = partial(_2p_nfg, 'gift-exchange-game')
"""Gift exchange game."""
MatchingPennies = partial(_2p0s_nfg, 'matching-pennies')
"""Matching pennies."""
PrisonersDilemma = partial(_2p_nfg, 'prisoners-dilemma')
"""Prisoner's dilemma."""
PureCoordination = partial(_2p_nfg, 'pure-coordination')
"""Pure coordination."""
RockPaperScissors = partial(_2p0s_nfg, 'rock-paper-scissors')
"""Rock paper scissors"""
RockPaperScissorsPlus = partial(_2p0s_nfg, 'rock-paper-scissors-plus')
"""Rock paper scissors+."""
RockPaperSuperscissors = partial(_2p0s_nfg, 'rock-paper-superscissors')
"""RockPaperSuperscissors."""
StagHunt = partial(_2p_nfg, 'stag-hunt')
"""Stag hunt."""
