"""Module for games."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

from noregret.kernels import Kernel


@dataclass
class Game(ABC):
    """Abstract base class for games."""
    kernel: Kernel
    """Kernel."""

    def __post_init__(self):
        pass

    @property
    @abstractmethod
    def player_count(self):
        """Return the number of players.

        :return: Number of players.
        """

    @abstractmethod
    def is_symmetric(self):
        """Return whether the game is symmetric.

        :return: Whether the game is symmetric.
        """

    @abstractmethod
    def utility(self, player, *strategies):
        """Calculate the utility for a given player given strategies
        (except that of the given player).

        :param player: Player.
        :param strategies: Strategies (except that of the given player).
        :return: Utility for the given player.
        """

    def utilities(self, *strategy_profile):
        """Calculate the utilities given a strategy profile.

        :param strategy_profile: Strategy profile.
        :return: Utilities.
        """
        P = range(self.player_count)
        s = strategy_profile

        return [self.utility(i, *s[:i], *s[i + 1:]) for i in P]

    @abstractmethod
    def expected_utility(self, player, *strategy_profile):
        """Calculate the expected utility for a given player given a
        strategy profile.

        :param player: Player.
        :param strategy_profile: Strategy profile.
        :return: Expected utility for the given player.
        """

    def expected_utilities(self, *strategy_profile):
        """Calculate the expected utilities given a strategy profile.

        :param strategy_profile: Strategy profile.
        :return: Expected utilities.
        """
        P = range(self.player_count)

        return [self.expected_utility(i, *strategy_profile) for i in P]

    @abstractmethod
    def best_response_value(self, player, *strategies):
        """Calculate the best response values for a given player given
        strategies (except that of the given player).

        :param player: Player.
        :param strategies: Strategies (except that of the given player).
        :return: Best response value for the given player.
        """

    def best_response_values(self, *strategy_profile):
        """Calculate the best response values given a strategy profile.

        :param strategy_profile: Strategy profile.
        :return: Best response values.
        """
        P = range(self.player_count)
        s = strategy_profile

        return [self.best_response_value(i, *s[:i], *s[i + 1:]) for i in P]

    def nash_gap(self, *strategy_profile):
        """Calculate the Nash gap given a strategy profile.

        :param strategy_profile: Strategy profile.
        :return: Nash gap.
        """
        expected_utilities = self.expected_utilities(strategy_profile)
        best_response_values = self.best_response_values(strategy_profile)

        assert (best_response_values >= expected_utilities).all()

        nash_gap = (best_response_values - expected_utilities).sum()

        return nash_gap


@dataclass
class TwoPlayerGame(Game, ABC):
    """Abstract base class for two-player (2p) games.

    Row and column players are of indices 0 and 1, respectively.
    """

    @abstractmethod
    def row_utility(self, row_strategy, column_strategy):
        """Calculate the utility for the row player given a strategy for
        the column player.

        :param column_strategy: Column strategy.
        :return: Utility for the row player.
        """

    @abstractmethod
    def column_utility(self, row_strategy, column_strategy):
        """Calculate the utility for the column player given a strategy
        for the row player.

        :param row_strategy: Row strategy.
        :return: Utility for the column player.
        """

    def utility(self, player, row_or_column_strategy):
        match player:
            case 0:
                u_or_v = self.row_utility(row_or_column_strategy)
            case 1:
                u_or_v = self.column_utility(row_or_column_strategy)
            case _:
                raise ValueError('unknown player')

        return u_or_v

    @abstractmethod
    def expected_row_utility(self, row_strategy, column_strategy):
        """Calculate the expected utility for the row player given a
        strategy profile.

        :param row_strategy: Row strategy.
        :param column_strategy: Column strategy.
        :return: Expected utility for the row player.
        """

    @abstractmethod
    def expected_column_utility(self, row_strategy, column_strategy):
        """Calculate the expected utility for the column player given a
        strategy profile.

        :param row_strategy: Row strategy.
        :param column_strategy: Column strategy.
        :return: Expected utility for the column player.
        """

    def expected_utility(self, player, row_strategy, column_strategy):
        match player:
            case 0:
                u_or_v = self.expected_row_utility(
                    row_strategy,
                    column_strategy,
                )
            case 1:
                u_or_v = self.expected_column_utility(
                    row_strategy,
                    column_strategy,
                )
            case _:
                raise ValueError('unknown player')

        return u_or_v

    @abstractmethod
    def row_best_response_value(self, column_strategy):
        """Calculate the best response value for the row player given a
        strategy for the column player.

        :param column_strategy: Column strategy.
        :return: Best response value for the row player.
        """

    @abstractmethod
    def column_best_response_value(self, row_strategy):
        """Calculate the best response value for the column player given a
        strategy for the row player.

        :param row_strategy: Row strategy.
        :return: Best response value for the column player.
        """

    def best_response_value(self, player, row_or_column_strategy):
        match player:
            case 0:
                u_or_v = self.row_best_response_value(row_or_column_strategy)
            case 1:
                u_or_v = self.column_best_response_value(
                    row_or_column_strategy,
                )
            case _:
                raise ValueError('unknown player')

        return u_or_v


@dataclass
class TwoPlayerZeroSumGame(TwoPlayerGame, ABC):
    """Abstract base class for two-player zero-sum (2p0s) games."""

    @abstractmethod
    def _row_utilities(self, row_strategy, column_strategy):
        pass

    def utilities(self, row_strategy, column_strategy):
        u, v = self._row_utilities(row_strategy, column_strategy)

        return u, -v

    def expected_column_utility(self, row_strategy, column_strategy):
        return -self.expected_row_utility(row_strategy, column_strategy)

    def expected_utilities(self, row_strategy, column_strategy):
        u = self.expected_row_utility(row_strategy, column_strategy)

        return u, -u

    @abstractmethod
    def _best_response_row_values(self, row_strategy, column_strategy):
        pass

    def best_response_values(self, row_strategy, column_strategy):
        u, v = self._best_response_row_values(row_strategy, column_strategy)

        return u, -v

    def _assert_nash_gap(self, x, y, u, neg_v):
        u2 = self.expected_row_utility(x, y)

        return (
            (neg_v < u2 or self.kernel.numpy.isclose(neg_v, u2))
            and (u2 < u or self.kernel.numpy.isclose(u2, u))
        )

    def nash_gap(self, row_strategy, column_strategy):
        u, neg_v = self._best_response_row_values(
            row_strategy,
            column_strategy,
        )

        assert self._assert_nash_gap(row_strategy, column_strategy, u, neg_v)

        return u - neg_v

    def exploitability(self, row_strategy, column_strategy):
        return self.nash_gap(row_strategy, column_strategy) / 2
