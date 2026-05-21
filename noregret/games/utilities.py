"""Module for utilities."""
from collections import defaultdict
from functools import partial, singledispatch
from itertools import starmap

from ordered_set import OrderedSet
from scipy.sparse import lil_array

from noregret.games.black_box import BlackBoxGame
from noregret.games.extensive_form.games import (
    ExtensiveFormGame,
    TwoPlayerExtensiveFormGame,
    TwoPlayerZeroSumExtensiveFormGame,
)
from noregret.games.games import Game
from noregret.games.normal_form.games import (
    NormalFormGame,
    TwoPlayerNormalFormGame,
    TwoPlayerZeroSumNormalFormGame,
)
from noregret.sequence_form_polytopes import SequenceFormPolytope


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
def to_extensive_form(kernel, game):
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


@to_extensive_form.register
def _(game: Game):
    return to_extensive_form(game.kernel, game)
