"""Module for multilinear games."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from noregret.games.games import Game, TwoPlayerGame, TwoPlayerZeroSumGame


@dataclass
class MultilinearGame(Game, ABC):
    """Abstract base class for multilinear games."""
    payoffs: Any
    """Payoffs."""

    def __post_init__(self):
        super().__post_init__()

        if not (
                self.payoffs.ndim - 1
                == self.payoffs.shape[0]
                == self.player_count
        ):
            raise ValueError('inconsistent number of players')
        elif self.payoffs.shape[1:] != self.dimensions:
            raise ValueError('inconsistent dimensions')

    @abstractmethod
    def dimension(self, player):
        """Return the dimension for a given player.

        :return: Dimension.
        """

    @property
    def dimensions(self):
        """Return the dimensions.

        :return: Dimensions.
        """
        return tuple(self.dimension(i) for i in range(self.player_count))

    def is_symmetric(self):
        raise NotImplementedError

    def utility(self, player, *strategies):
        raise NotImplementedError

    def expected_utility(self, player, *strategy_profile):
        raise NotImplementedError


@dataclass
class TwoPlayerMultilinearGame(TwoPlayerGame, MultilinearGame, ABC):
    """Abstract base class for two-player (2p) multilinear games."""

    @property
    @abstractmethod
    def row_dimension(self):
        """Return the dimension for the row player.

        :return: Dimension for the row player.
        """

    @property
    @abstractmethod
    def column_dimension(self):
        """Return the dimension for the column player.

        :return: Dimension for the column player.
        """

    def dimension(self, player):
        """Return the dimension for a given player.

        :return: Dimension for the given player.
        """
        match player:
            case 0:
                n_or_m = self.row_dimension
            case 1:
                n_or_m = self.column_dimension
            case _:
                raise ValueError('unknown player')

        return n_or_m

    @property
    def row_payoffs(self):
        """Return the payoffs for the row player.

        :return: Payoffs for the row player.
        """
        return self.payoffs[0]

    @property
    def column_payoffs(self):
        """Return the payoffs for the column player.

        :return: Payoffs for the column player.
        """
        return self.payoffs[1]

    def is_symmetric(self):
        np = self.kernel.numpy

        return (
            self.row_dimension == self.column_dimension
            and np.allclose(self.row_payoffs, self.column_payoffs.T)
        )

    def row_utility(self, column_strategy):
        return self.row_payoffs @ column_strategy

    def column_utility(self, row_strategy):
        return row_strategy @ self.column_payoffs

    def expected_row_utility(self, row_strategy, column_strategy):
        return row_strategy @ self.row_payoffs @ column_strategy

    def expected_column_utility(self, row_strategy, column_strategy):
        return row_strategy @ self.column_payoffs @ column_strategy

    def expected_utility(self, player, row_strategy, column_strategy):
        return row_strategy @ self.payoffs[player] @ column_strategy

    def expected_utilities(self, row_strategy, column_strategy):
        return row_strategy @ self.payoffs @ column_strategy


@dataclass
class TwoPlayerZeroSumMultilinearGame(
        TwoPlayerZeroSumGame,
        TwoPlayerMultilinearGame,
        ABC,
):
    """Abstract base class for two-player zero-sum (2p0s) multilinear
    games.

    The utility matrix is from the viewpoint of the row player.
    """

    def __post_init__(self):
        super(MultilinearGame, self).__post_init__()

        if self.payoffs.shape != (self.row_dimension, self.column_dimension):
            raise ValueError('inconsistent dimensions')

    @property
    def row_payoffs(self):
        return self.payoffs

    @property
    def column_payoffs(self):
        return -self.payoffs

    def _row_utilities(self, row_strategy, column_strategy):
        u = self.payoffs @ column_strategy
        neg_v = row_strategy @ self.payoffs

        return u, neg_v
