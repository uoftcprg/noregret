"""Module for black box games."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any

from ordered_set import OrderedSet

from noregret.kernels import Kernel


@dataclass
class Simulation:
    """Class for simulations."""
    kernel: Kernel
    """Kernel."""
    players: list[int]
    """Players."""
    decision_points: list[str | None]
    """Decision points."""
    actions: list[str]
    """Actions."""
    utilities: Any
    """Utilities."""

    def sequences(self, player=None):
        """Return sequences given an optional player.

        :param player: Optional player.
        :return: Sequences.
        """
        for i, j, a in zip(self.players, self.decision_points, self.actions):
            if i is not None and (player is None or i == player):
                yield j, a

    def utility(self, player):
        """Return the utility given a player.

        :param player: Player.
        :return: Utility.
        """
        return self.utilities[player]


@dataclass
class BlackBoxGame(ABC):
    """Abstract base class for black box games."""
    kernel: Kernel
    """Kernel."""

    @property
    @abstractmethod
    def player_count(self):
        """Return the number of players.

        :return: Number of players.
        """

    @property
    def is_two_player(self):
        """Return whether the game is two-player.

        :return: Whether the game is two-player.
        """
        return self.player_count == 2

    @property
    @abstractmethod
    def is_zero_sum(self):
        """Return whether the game is zero-sum.

        :return: Whether the game is zero-sum.
        """

    @property
    @abstractmethod
    def root_node(self):
        """Return the root node.

        :return: Root node.
        """

    @abstractmethod
    def actions(self, node):
        """Return the actions given a node.

        :param node: Node.
        :return: Actions.
        """

    @abstractmethod
    def apply(self, node, action):
        """Return the child node given a node and an action.

        :param node: Node.
        :param action: Action.
        :return: Child node.
        """

    def children(self, node):
        """Return the children given a node.

        :return: Children.
        """
        return list(map(partial(self.apply, node), self.actions(node)))

    def actions_and_children(self, node):
        """Return the actions and children given a node.

        :return: Actions and children.
        """
        A = self.actions(node)

        return A, list(map(partial(self.apply, node), A))

    @abstractmethod
    def player(self, node):
        """Return the player given a node.

        :param node: Node.
        :return: Player.
        """

    @abstractmethod
    def utility(self, node, player):
        """Return the utility given a player and a node.

        :param node: Node.
        :param player: Player.
        :return: Utility.
        """

    def utilities(self, node):
        """Return the utilities given a node.

        :param node: Node.
        :return: Utilities.
        """
        np = self.kernel.numpy
        dtype = self.kernel.data_type
        us = list(map(partial(self.utility, node), range(self.player_count)))

        return np.array(us, dtype)

    @abstractmethod
    def information_set(self, node):
        """Return the information set given a node.

        :param node: Node.
        :return: information set.
        """

    @abstractmethod
    def chance_probability(self, node, action):
        """Return the chance probability given a node and an action.

        :param node: Node.
        :param action: Action.
        :return: Chance probability.
        """

    def chance_probabilities(self, node):
        """Return the chance probabilities given a node.

        :param node: Node.
        :return: Chance probabilities.
        """
        np = self.kernel.numpy
        dtype = self.kernel.data_type
        A = self.actions(node)
        ps = list(map(partial(self.chance_probability, node), A))

        return np.array(ps, dtype)

    def exploitability(self, strategy_profile):
        """Return exploitability given a strategy profile.

        :param strategy_profile: Strategy profile.
        :return: Exploitability.
        """
        if not self.is_two_player or not self.is_zero_sum:
            raise ValueError('not 2p0s')

        raise NotImplementedError

    def simulate(self, strategy_profile):
        """Run a simulation given a strategy profile.

        :param strategy_profile: Strategy profile.
        :return: Simulation.
        """
        np = self.kernel.numpy
        is_ = []
        js = []
        as_ = []
        h = self.root_node

        while A := self.actions(h):
            i = self.player(h)

            if i is None:
                j = None
                ps = self.chance_probabilities(h)
            else:
                j = self.information_set(h)
                ps = strategy_profile(h)

            a = np.random.choice(A, p=ps).item()
            h = self.apply(h, a)

            is_.append(i)
            js.append(j)
            as_.append(a)

        is_ = tuple(is_)
        js = tuple(js)
        as_ = tuple(as_)
        us = self.utilities(h)
        simulation = Simulation(self.kernel, is_, js, as_, us)

        return simulation


@dataclass
class StrategyProfile(ABC):
    """Abstract base class for strategy profiles."""
    kernel: Kernel
    """Kernel."""
    game: BlackBoxGame
    """Game."""

    def __post_init__(self):
        pass

    @abstractmethod
    def __call__(self, node):
        pass


@dataclass
class UniformStrategyProfile(StrategyProfile):
    """Class for uniform strategy profiles."""

    def __call__(self, node):
        np = self.kernel.numpy
        dtype = self.kernel.data_type
        n = len(self.game.actions(node))

        return np.full(n, 1 / n, dtype)


@dataclass
class TupleStrategyProfile(StrategyProfile):
    """Class for tuple strategy profiles."""
    strategies: tuple[Any, ...]
    """Strategies."""

    def __post_init__(self):
        super().__post_init__()

        if len(self.strategies) != self.game.player_count:
            raise ValueError('inconsistent number of players')

    def __call__(self, node):
        i = self.game.player(node)

        return self.strategies[i](node)


@dataclass
class SequenceFormStrategyProfile(StrategyProfile):
    """Class for tuple strategy profiles."""
    non_empty_sequences: tuple[OrderedSet[tuple[str, str]], ...]
    """Non-empty sequences."""
    strategy_profile: tuple[Any, ...]
    """Strategy profile."""

    def __post_init__(self):
        super().__post_init__()

        if (
                not (
                    len(self.non_empty_sequences)
                    == len(self.strategy_profile)
                    == self.game.player_count
                )
        ):
            raise ValueError('inconsistent number of players')

    def __call__(self, node):
        np = self.kernel.numpy
        dtype = self.kernel.data_type
        i = self.game.player(node)
        j = self.game.information_set(node)
        A_j = self.game.actions(node)
        ps = []

        for a in A_j:
            index = self.non_empty_sequences[i].index((j, a)) + 1

            ps.append(self.strategy_profile[i][index])

        ps = np.array(ps, dtype)

        if np.allclose(ps, 0):
            n = len(A_j)
            ps = np.full(n, 1 / n, dtype)
        else:
            ps /= ps.sum()

        return ps
