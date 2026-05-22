"""Module for extensive-form games (EFGs)."""
from collections import defaultdict
from dataclasses import dataclass
from functools import partial, singledispatch
from io import BytesIO
from itertools import starmap

from ordered_set import OrderedSet
from orjson import dumps, loads
from scipy.sparse import lil_array, load_npz, save_npz

from noregret.games.black_box import BlackBoxGame
from noregret.games.games import Game
from noregret.games.multilinear import (
    MultilinearGame,
    TwoPlayerMultilinearGame,
    TwoPlayerZeroSumMultilinearGame,
)
from noregret.games.normal_form.games import (
    NormalFormGame,
    TwoPlayerNormalFormGame,
    TwoPlayerZeroSumNormalFormGame,
)
from noregret.kernels import Serializable
from noregret.sequence_form_polytopes import SequenceFormPolytope
from noregret.utilities import tuple_or_none


@dataclass
class ExtensiveFormGame(MultilinearGame, Serializable):
    """Extensive-form game (EFG).

    Every player optimizes over a sequence-form polytope.
    """
    sequence_form_polytopes: tuple[SequenceFormPolytope, ...]
    """Sequence-form polytopes."""

    @property
    def player_count(self):
        return len(self.sequence_form_polytopes)

    @property
    def dimensions(self):
        return tuple(sfp.column_count for sfp in self.sequence_form_polytopes)

    def best_response_value(self, player, *strategies):
        u = self.utility(player, *strategies)

        return self.sequence_form_polytopes[player].best_response_value(u)

    @classmethod
    def loads(cls, kernel, raw_data):
        scipy = kernel.scipy

        def sfp(raw_sfp):
            actions = raw_sfp['actions']
            J = actions.keys()
            A = actions.values()
            actions = dict(zip(J, map(OrderedSet, A)))
            parent_sequences = raw_sfp['parent_sequences']
            J = parent_sequences.keys()
            sequences = parent_sequences.values()
            parent_sequences = dict(zip(J, map(tuple_or_none, sequences)))

            return SequenceFormPolytope(kernel, actions, parent_sequences)

        data = loads(raw_data)
        io = BytesIO(bytes.fromhex(data['payoffs']))
        dtype = kernel.data_type
        payoffs = scipy.sparse.csr_array(load_npz(io), dtype=dtype)
        sfps = tuple(map(sfp, data['sequence_form_polytopes']))

        return cls(kernel, payoffs, sfps)

    def dumps(self):

        def raw_sfp(sfp):
            return {
                'actions': sfp.actions,
                'parent_sequences': sfp.parent_sequences,
            }

        io = BytesIO()
        sfps = self.sequence_form_polytopes

        save_npz(io, self.payoffs)

        data = {
            'payoffs': io.getvalue().hex(),
            'sequence_form_polytopes': list(map(raw_sfp, sfps)),
        }

        return dumps(data, list)


@dataclass
class TwoPlayerExtensiveFormGame(TwoPlayerMultilinearGame, ExtensiveFormGame):
    """Class for two-player (2p) extensive-form games (EFGs)."""

    @property
    def row_sequence_form_polytope(self):
        """Return the sequence-form polytope for the row player.

        :return: Sequence-form polytope for the row player.
        """
        return self.sequence_form_polytopes[0]

    @property
    def column_sequence_form_polytope(self):
        """Return the sequence-form polytope for the column player.

        :return: Sequence-form polytope for the column player.
        """
        return self.sequence_form_polytopes[1]

    @property
    def row_dimension(self):
        """Return the dimension for the row player.

        :return: Dimension for the row player.
        """
        return self.row_sequence_form_polytope.column_count

    @property
    def column_dimension(self):
        """Return the dimension for the column player.

        :return: Dimension for the column player.
        """
        return self.column_sequence_form_polytope.column_count

    def row_best_response_value(self, column_strategy):
        u = self.row_utility(column_strategy)

        return self.row_sequence_form_polytope.best_response_value(u)

    def column_best_response_value(self, row_strategy):
        v = self.column_utility(row_strategy)

        return self.column_sequence_form_polytope.best_response_value(v)


@dataclass
class TwoPlayerZeroSumExtensiveFormGame(
        TwoPlayerZeroSumMultilinearGame,
        TwoPlayerExtensiveFormGame,
):
    """Class for two-player zero-sum (2p0s) extensive-form games (EFGs)."""

    def _best_response_row_values(self, row_strategy, column_strategy):
        u, neg_v = self._row_utilities(row_strategy, column_strategy)
        u = self.row_sequence_form_polytope.best_response_value(u)
        neg_v = self.column_sequence_form_polytope.worst_response_value(neg_v)

        return u, neg_v


def _nfg2efg(kernel, game, decision_points='p{}'.format):
    np = kernel.numpy
    scipy = kernel.scipy
    dtype = kernel.data_type

    if isinstance(game, TwoPlayerZeroSumNormalFormGame):
        type_ = TwoPlayerZeroSumExtensiveFormGame
    elif isinstance(game, TwoPlayerNormalFormGame):
        type_ = TwoPlayerExtensiveFormGame
    else:
        type_ = ExtensiveFormGame

    d = game.dimensions

    if isinstance(game, TwoPlayerZeroSumNormalFormGame):
        payoffs = np.zeros(tuple(n + 1 for n in d), dtype)
        payoffs[tuple(slice(1, None) for _ in d)] = game.payoffs
    else:
        payoffs = np.zeros((game.player_count, *(n + 1 for n in d)), dtype)
        payoffs[:, *(slice(1, None) for _ in d)] = game.payoffs

    payoffs = scipy.sparse.csr_array(payoffs)
    sfps = []

    for i, A_j in enumerate(game.actions):
        j = decision_points(i)
        sfp = SequenceFormPolytope(kernel, {j: A_j}, {j: None})

        sfps.append(sfp)

    sfps = tuple(sfps)

    return type_(kernel, payoffs, sfps)


def _bbg2efg(kernel, game):
    scipy = kernel.scipy
    dtype = kernel.data_type
    P = range(game.player_count)
    A_js = [defaultdict(OrderedSet) for _ in P]
    p_js = [{} for _ in P]
    raw_payoffs = [defaultdict(int) for _ in P]

    def dfs(h, p, seqs, us):
        A_j, h_primes = game.actions_and_children(h)
        i = game.player(h)
        us = us.copy()

        for i_prime, v in enumerate(game.utilities(h)):
            us[i_prime] += v

        if not A_j:
            seqs = tuple(seqs)

            for i_prime, u in enumerate(us):
                raw_payoffs[i_prime][seqs] += p * u
        elif i is None:
            p_primes = game.chance_probabilities(h)

            for h_prime, p_prime in zip(h_primes, p_primes):
                dfs(h_prime, p_prime * p, seqs, us)
        else:
            j = game.information_set(h)
            p_j = seqs[i]
            p_js[i][j] = p_j

            for a, h_prime in zip(A_j, h_primes):
                next_seqs = seqs.copy()
                next_seqs[i] = j, a

                A_js[i][j].add(a)
                dfs(h_prime, p, next_seqs, us)

    dfs(game.root_node, 1, [None for _ in P], [0 for _ in P])

    SFP = partial(SequenceFormPolytope, kernel)
    sfps = tuple(starmap(SFP, zip(A_js, p_js)))
    dimensions = tuple(sfp.column_count for sfp in sfps)

    if game.is_two_player and game.is_zero_sum:
        type_ = TwoPlayerZeroSumExtensiveFormGame
        payoffs = lil_array(dimensions, dtype=dtype)

        for seqs, u in raw_payoffs[0].items():
            indices = []

            for sfp, seq in zip(sfps, seqs):
                indices.append(sfp.column(seq))

            payoffs[tuple(indices)] = u

        payoffs = scipy.sparse.csr_array(payoffs)
    else:
        raise NotImplementedError

    return type_(kernel, payoffs, sfps)


@singledispatch
def to_extensive_form_game(kernel, game):
    """Convert a given game to an extensive-form game.

    :param game: Game.
    :return: Extensive-form game.
    """
    if isinstance(game, NormalFormGame):
        game = _nfg2efg(kernel, game)
    elif isinstance(game, BlackBoxGame):
        game = _bbg2efg(kernel, game)
    else:
        raise ValueError('unknown game')

    return game


@to_extensive_form_game.register
def _(game: Game):
    return to_extensive_form_game(game.kernel, game)
