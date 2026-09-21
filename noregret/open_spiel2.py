"""Module for OpenSpiel."""
from dataclasses import dataclass, field

from ordered_set import OrderedSet
from pyspiel import exploitability, GameType, load_game

from noregret.games.black_box import BlackBoxGame


@dataclass
class OpenSpielGame(BlackBoxGame):
    """Class for OpenSpiel games."""
    name: str
    """Name."""
    _game: str = field(init=False)

    def __post_init__(self):
        self._game = load_game(self.name)

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
        return list(map(node.child, node.legal_actions()))

    def actions_and_children(self, node):
        A = node.legal_actions()
        actions = OrderedSet(map(node.action_to_string, A))
        children = list(map(node.child, A))

        return actions, children

    def player(self, node):
        i = node.current_player()

        return None if i < 0 else i

    def utility(self, node, player):
        np = self.kernel.numpy
        dtype = self.kernel.data_type

        return np.array(node.player_reward(player), dtype)

    def utilities(self, node):
        np = self.kernel.numpy
        dtype = self.kernel.data_type

        return np.array(node.rewards(), dtype)

    def information_set(self, node):
        return node.information_state_string()

    def chance_probability(self, node, action):
        np = self.kernel.numpy
        dtype = self.kernel.data_type
        p = node.chance_outcomes()[self.actions(node).index(action)][1]

        return np.array(p, dtype)

    def chance_probabilities(self, node):
        np = self.kernel.numpy
        dtype = self.kernel.data_type

        return np.array([p for _, p in node.chance_outcomes()], dtype)

    def _sigma(self, strategy_profile, h, sigma):
        A = h.legal_actions()
        h_primes = list(map(h.child, A))
        i = self.player(h)

        if A and i is not None and (j := self.information_set(h)) not in sigma:
            sigma[j] = list(zip(A, strategy_profile(h).tolist()))

        for h_prime in h_primes:
            self._sigma(strategy_profile, h_prime, sigma)

    def _sigma2(self, strategy_profile):
        sigma = {}

        self._sigma(strategy_profile, self.root_node, sigma)

        return sigma

    def exploitability(self, strategy_profile):
        return exploitability(self._game, self._sigma2(strategy_profile))
