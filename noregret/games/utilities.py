"""Module for game utilities."""
from collections import defaultdict
from functools import partial
from itertools import starmap

from ordered_set import OrderedSet
from scipy.sparse import lil_array

from noregret.games.black_box import BlackBoxGame
from noregret.games.extensive_form import (
    ExtensiveFormGame,
    TwoPlayerExtensiveFormGame,
    TwoPlayerZeroSumExtensiveFormGame,
)
from noregret.games.normal_form import (
    NormalFormGame,
    TwoPlayerNormalFormGame,
    TwoPlayerZeroSumNormalFormGame,
)
from noregret.sequence_form_polytopes import SequenceFormPolytope


def _nfg2efg(ker, game, decision_points='p{}'.format):
    np = ker.numpy
    scipy = ker.scipy
    dtype = ker.data_type

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

    for i, A in enumerate(game.actions):
        j = decision_points(i)
        sfp = SequenceFormPolytope(ker, {j: A}, {j: None})

        sfps.append(sfp)

    sfps = tuple(sfps)

    return type_(ker, payoffs, sfps)


def _bbg2efg(ker, game):
    np = game.kernel.numpy
    dtype = game.kernel.data_type
    P = range(game.player_count)
    A_js = [defaultdict(OrderedSet) for _ in P]
    p_js = [{} for _ in P]
    raw_payoffs = defaultdict(int)

    def dfs(h, p, seqs, us):
        A, h_primes = game.actions_and_children(h)
        i = game.player(h)
        us = us + game.utilities(h)

        if not A:
            raw_payoffs[tuple(seqs)] += p * us
        elif i is None:
            p_primes = game.chance_probabilities(h)

            for h_prime, p_prime in zip(h_primes, p_primes):
                dfs(h_prime, p_prime * p, seqs, us)
        else:
            j = game.information_set(h)
            p_j = seqs[i]
            p_js[i][j] = p_j

            for a, h_prime in zip(A, h_primes):
                next_seqs = seqs.copy()
                next_seqs[i] = j, a

                A_js[i][j].add(a)
                dfs(h_prime, p, next_seqs, us)

    dfs(game.root_node, 1, [None for _ in P], np.zeros(len(P), dtype))

    scipy = ker.scipy
    dtype = ker.data_type
    SFP = partial(SequenceFormPolytope, ker)
    sfps = tuple(starmap(SFP, zip(A_js, p_js)))
    dimensions = tuple(sfp.column_count for sfp in sfps)

    if game.is_two_player and game.is_zero_sum:
        type_ = TwoPlayerZeroSumExtensiveFormGame
        payoffs = lil_array(dimensions, dtype=dtype)

        for seqs, us in raw_payoffs.items():
            indices = []

            for sfp, seq in zip(sfps, seqs):
                indices.append(sfp.column(seq))

            payoffs[tuple(indices)] = us[0]

        payoffs = scipy.sparse.csr_array(payoffs)
    else:
        raise NotImplementedError

    return type_(ker, payoffs, sfps)


def to_extensive_form_game(kernel, game):
    """Convert a given game to an extensive-form game.

    :param kernel: Kernel.
    :param game: Game.
    :return: Extensive-form game.
    """
    if isinstance(game, NormalFormGame):
        game = _nfg2efg(kernel, game)
    elif isinstance(game, BlackBoxGame):
        game = _bbg2efg(kernel, game)
    else:
        raise NotImplementedError('unknown game')

    return game
