"""Module for black box games."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial

from ordered_set import OrderedSet
from pyspiel import GameType, load_game


@dataclass
class BlackBoxGame(ABC):
    """Abstract base class for black box games."""

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
        return list(map(partial(self.utility, node), range(self.player_count)))

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
        A = self.actions(node)

        return list(map(partial(self.chance_probability, node), A))


@dataclass
class _OpenSpielBlackBoxGame(BlackBoxGame):
    game: str
    _game: str = field(init=False)

    def __post_init__(self):
        self._game = load_game(self.game)

    @property
    def player_count(self):
        return self._game.num_players()

    @property
    def is_zero_sum(self):
        return self._game.get_type().utility == GameType.Utility.ZERO_SUM

    @property
    def root_node(self):
        return self._game.new_initial_state()

    def actions(self, node):
        return OrderedSet(map(node.action_to_string, node.legal_actions()))

    def apply(self, node, action):
        return node.child(node.string_to_action(action))

    def children(self, node):
        return list(node.child(a) for a in node.legal_actions())

    def actions_and_children(self, node):
        actions = []
        children = []

        for a in node.legal_actions():
            actions.append(node.action_to_string(a))
            children.append(node.child(a))

        return OrderedSet(actions), children

    def player(self, node):
        i = node.current_player()

        return None if i < 0 else i

    def utility(self, node, player):
        return node.player_reward(player)

    def utilities(self, node):
        return node.rewards()

    def information_set(self, node):
        return node.information_state_string()

    def chance_probability(self, node, action):
        return node.chance_outcomes()[self.actions(node).index(action)][1]

    def chance_probabilities(self, node):
        return [p for _, p in node.chance_outcomes()]


def open_spiel_game(game):
    """Load a game from OpenSpiel.

    :param game: Game in OpenSpiel.
    :return: Game.
    """
    return _OpenSpielBlackBoxGame(game)
